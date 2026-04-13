from __future__ import annotations
"""
Fine-tune RNAGenesis for RNA tertiary structure prediction.

Usage:
    python train.py <npz_dir> <out_dir> \\
        --encoder_path /path/to/RNAGenesis \\
        --lst train.lst \\
        --val_lst val.lst \\
        --channels 64 \\
        --n_blocks 12 \\
        --max_epochs 30 \\
        --gpu 0
"""

import json
import os
import random
import sys
import warnings
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from data.dataset import RNAStructureDataset
from model.network import RNAGenesisStructurePredictor
from utils.loss import geometry_loss, distance_correlation


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_parser():
    parser = ArgumentParser(
        description="Fine-tune RNAGenesis for RNA tertiary structure prediction",
        formatter_class=RawTextHelpFormatter,
    )
    # Positional
    parser.add_argument("npz_dir", help="Directory containing NPZ training files")
    parser.add_argument("out_dir", help="Output directory for checkpoints and logs")

    # Data
    parser.add_argument("--lst", default=None, help="Text file listing training entry IDs")
    parser.add_argument("--val_lst", default=None, help="Text file listing validation entry IDs")
    parser.add_argument("--msa_cutoff", type=int, default=200, help="Max MSA rows per sample (default 200)")
    parser.add_argument("--crop_size", type=int, default=200, help="Max sequence length; longer entries are cropped (default 200)")

    # Encoder
    parser.add_argument(
        "--encoder_path",
        type=str,
        default=None,
        help="Path to pretrained RNAGenesis encoder (HuggingFace format). "
             "If not provided, the model runs on MSA features only.",
    )
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder weights during training")
    parser.add_argument("--encoder_dim", type=int, default=1280, help="Encoder hidden size (default 1280)")
    parser.add_argument("--encoder_proj_dim", type=int, default=32, help="Projection dim for outer product (default 32)")

    # Architecture
    parser.add_argument("--channels", type=int, default=64, help="Trunk hidden dimension (default 64)")
    parser.add_argument("--n_blocks", type=int, default=12, help="Number of Evoformer blocks (default 12)")

    # Optimisation
    parser.add_argument("--init_lr", type=float, default=5e-4, help="Initial learning rate (default 5e-4)")
    parser.add_argument("--encoder_lr_factor", type=float, default=0.5, help="LR multiplier for encoder (default 0.5)")
    parser.add_argument("--milestones", nargs="+", default=[10, 15, 20, 25], help="Epochs to halve the LR")
    parser.add_argument("--max_epochs", type=int, default=30, help="Training epochs (default 30)")
    parser.add_argument("--batch_size", type=int, default=1, help="Gradient accumulation batch size (default 1)")
    parser.add_argument("--num_recycle", type=int, default=3, help="Recycling iterations during training (default 3)")

    # Misc
    parser.add_argument("--checkpoint_path", default=None, help="Resume from this checkpoint")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index; -1 for CPU")
    parser.add_argument("--cpu", type=int, default=4, help="Number of CPU threads (default 4)")
    parser.add_argument("--warning", action="store_true", help="Print per-sample warnings")

    return parser


# ---------------------------------------------------------------------------
# Training / validation loop
# ---------------------------------------------------------------------------

def run_epoch(
    model,
    dataloader,
    optimizer,
    device,
    is_training: bool,
    msa_cutoff: int,
    num_recycle: int,
    batch_size: int = 1,
    log_file: str | None = None,
):
    model.train(is_training)
    for p in model.parameters():
        p.requires_grad_(is_training)

    metrics = {"loss": [], "dist_corr": []}
    n_oom = 0

    pbar = tqdm(dataloader, leave=False)
    for i, sample in enumerate(pbar):
        if not sample:
            continue

        msa = sample["aln"].long().to(device)             # (1, N, L)
        res_id = sample["idx"].long().to(device)           # (1, L)
        conf_mask1d = sample["conf_mask"].long().to(device)
        conf_mask2d = sample["conf_mask_2d"].long().to(device)
        native_geom = sample["labels"]
        input_ids = sample.get("input_ids")
        if input_ids is not None:
            input_ids = input_ids.to(device)

        recycles = random.choice(range(4)) if is_training else num_recycle

        try:
            preds = model(
                msa,
                input_ids=input_ids,
                res_id=res_id,
                num_recycle=recycles,
                msa_cutoff=msa_cutoff,
                is_training=is_training,
            )
            loss = geometry_loss(
                preds["geoms"], native_geom, device, conf_mask1d, conf_mask2d
            )
            loss_val = float(loss.detach())
            if np.isnan(loss_val):
                warnings.warn("NaN loss — skipping sample")
                continue

            if is_training:
                loss.backward()
                if (i + 1) % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            metrics["loss"].append(loss_val)

            if not is_training and "distance" in sample:
                pred_c3 = preds["geoms"]["distance"].get("C3'")
                native_c3 = sample["distance"].get("C3'")
                if pred_c3 is not None and native_c3 is not None:
                    dc = distance_correlation(
                        pred_c3.detach().cpu().numpy(),
                        native_c3.cpu().numpy(),
                    )
                    metrics["dist_corr"].append(dc)

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                n_oom += 1
                try:
                    del preds, loss
                except NameError:
                    pass
                torch.cuda.empty_cache()
                continue
            raise

        desc = f"{'train' if is_training else 'val'} loss={loss_val:.3f}"
        pbar.set_description(desc)

    return metrics, n_oom


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log(path: str, msg: str) -> None:
    print(msg)
    with open(path, "a") as fh:
        fh.write(msg + "\n")


def save_config(path: str, cfg: dict) -> None:
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = get_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.set_num_threads(args.cpu)
    device = torch.device("cuda:0") if torch.cuda.is_available() and args.gpu >= 0 else torch.device("cpu")

    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    log_file = os.path.join(args.out_dir, "training.log")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Load encoder ----
    encoder = None
    tokenizer = None
    if args.encoder_path:
        print(f"Loading RNAGenesis encoder from {args.encoder_path} ...")
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_path, trust_remote_code=True)
        encoder = AutoModel.from_pretrained(
            args.encoder_path, trust_remote_code=True, torch_dtype=torch.float32
        ).to(device)
        encoder.eval()

    # ---- Data lists ----
    if args.lst:
        train_ids = open(args.lst).read().splitlines()
    else:
        train_ids = [f[:-4] for f in os.listdir(args.npz_dir) if f.endswith(".npz")]

    if args.val_lst:
        val_ids = open(args.val_lst).read().splitlines()
    else:
        val_ids = random.sample(train_ids, min(100, len(train_ids) // 10))
        train_ids = [x for x in train_ids if x not in set(val_ids)]

    # ---- Datasets ----
    shared_kw = dict(
        npz_dir=args.npz_dir,
        tokenizer=tokenizer,
        max_length=args.crop_size,
        msa_cutoff=args.msa_cutoff,
        show_warnings=args.warning,
    )
    train_set = RNAStructureDataset(train_ids, random_crop=True, subsample_msa=True, **shared_kw)
    val_set = RNAStructureDataset(val_ids, random_crop=False, subsample_msa=False, **shared_kw)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    # ---- Model ----
    model = RNAGenesisStructurePredictor(
        encoder=encoder,
        encoder_dim=args.encoder_dim,
        dim=args.channels,
        depth=args.n_blocks,
        encoder_proj_dim=args.encoder_proj_dim,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        state = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(state.get("state_dict", state))
        print(f"Resumed from checkpoint: {args.checkpoint_path}")

    # ---- Optimiser ----
    param_groups = [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.init_lr}
    ]
    if encoder is not None and not args.freeze_encoder:
        # Give encoder a lower learning rate
        enc_params = [p for p in model.encoder_extractor.encoder.parameters() if p.requires_grad]
        trunk_params = [p for name, p in model.named_parameters()
                        if p.requires_grad and "encoder_extractor.encoder" not in name]
        param_groups = [
            {"params": trunk_params, "lr": args.init_lr},
            {"params": enc_params, "lr": args.init_lr * args.encoder_lr_factor},
        ]

    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(m) for m in args.milestones],
        gamma=0.5,
    )

    cfg = {
        "channels": args.channels, "n_blocks": args.n_blocks,
        "encoder_path": args.encoder_path, "lr": args.init_lr,
        "crop_size": args.crop_size, "msa_cutoff": args.msa_cutoff,
    }
    save_config(os.path.join(args.out_dir, "config.json"), cfg)
    log(log_file, f"Train: {len(train_set)}, Val: {len(val_set)}, epochs={args.max_epochs}, lr={args.init_lr}")

    best_corr = -1.0

    for epoch in range(args.max_epochs):
        t0 = time()
        train_metrics, n_oom_t = run_epoch(
            model, train_loader, optimizer, device,
            is_training=True, msa_cutoff=args.msa_cutoff,
            num_recycle=args.num_recycle, batch_size=args.batch_size,
            log_file=log_file,
        )
        val_metrics, n_oom_v = run_epoch(
            model, val_loader, optimizer, device,
            is_training=False, msa_cutoff=args.msa_cutoff,
            num_recycle=args.num_recycle, batch_size=1,
        )
        scheduler.step()

        mean_train_loss = np.mean(train_metrics["loss"]) if train_metrics["loss"] else float("nan")
        mean_val_loss = np.mean(val_metrics["loss"]) if val_metrics["loss"] else float("nan")
        mean_val_corr = np.mean(val_metrics["dist_corr"]) if val_metrics["dist_corr"] else float("nan")
        elapsed = time() - t0

        msg = (
            f"\n--- Epoch {epoch + 1}/{args.max_epochs}  ({elapsed:.0f}s) ---\n"
            f"  train loss : {mean_train_loss:.4f}  (OOM: {n_oom_t})\n"
            f"  val   loss : {mean_val_loss:.4f}  corr: {mean_val_corr:.4f}  (OOM: {n_oom_v})\n"
            f"  lr         : {optimizer.param_groups[0]['lr']:.6f}\n"
        )
        log(log_file, msg)

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        flag = f"ep{epoch:03d}_corr{mean_val_corr:.4f}"
        torch.save(ckpt, os.path.join(ckpt_dir, f"model_{flag}.pt"))

        if mean_val_corr > best_corr:
            best_corr = mean_val_corr
            torch.save(ckpt, os.path.join(ckpt_dir, "best_model.pt"))
            log(log_file, f"  → new best corr: {best_corr:.4f}")

    log(log_file, f"\nTraining complete. Best val corr: {best_corr:.4f}")


if __name__ == "__main__":
    main()
