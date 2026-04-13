from __future__ import annotations
"""
Run RNAGenesis structure prediction on one or more RNA sequences.

Outputs an NPZ file per sequence containing predicted distance distributions
and contact probabilities, which can then be passed to fold.py for 3-D
structure generation via PyRosetta.

Usage:
    python predict.py \\
        -i <msa.a3m> \\
        -o <output.npz> \\
        --encoder_path /path/to/RNAGenesis \\
        --checkpoint /path/to/best_model.pt \\
        --gpu 0
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from model.network import RNAGenesisStructurePredictor
from model.config import N_BINS, GEOMETRY_TARGETS
from utils.misc import parse_a3m, parse_ss_file


IDX_TO_BASE = {0: "A", 1: "U", 2: "C", 3: "G", 4: "-"}


# ---------------------------------------------------------------------------
# Sliding-window prediction for long RNAs
# ---------------------------------------------------------------------------

def predict_with_crops(
    model,
    msa: np.ndarray,
    input_ids: torch.Tensor | None,
    device: torch.device,
    window: int = 100,
    shift: int = 50,
    msa_cutoff: int = 200,
) -> dict:
    """
    Predict geometry using overlapping windows for sequences longer than 2×window.

    Args:
        model:      RNAGenesisStructurePredictor in eval mode.
        msa:        (N, L) integer MSA.
        input_ids:  (1, L) token IDs for the full sequence.
        device:     Target device.
        window:     Crop half-width.
        shift:      Grid step size.
        msa_cutoff: Max MSA rows.

    Returns:
        Aggregated prediction dict with keys 'distance' and 'contact'.
    """
    L = msa.shape[-1]
    n_dist_bins = N_BINS["2D"]["distance"]
    dist_atoms = GEOMETRY_TARGETS["2D"]["distance"]

    accum = {
        "contact": torch.zeros((L, L), device=device),
        "distance": {a: torch.zeros((L, L, n_dist_bins), device=device) for a in dist_atoms},
    }
    count_2d = torch.zeros((L, L), device=device)

    msa_t = torch.from_numpy(msa).to(device)
    grids = np.arange(0, L - window + shift, shift)
    res_idx_full = torch.arange(L, device=device).view(1, L)

    for gi in range(len(grids)):
        for gj in range(gi, len(grids)):
            s1, e1 = grids[gi], min(grids[gi] + window, L)
            s2, e2 = grids[gj], min(grids[gj] + window, L)
            sel = np.zeros(L, dtype=bool)
            sel[s1:e1] = True
            sel[s2:e2] = True

            sub_msa = msa_t[:, sel]
            # Remove sequences that are mostly gaps in this crop
            gap_frac = (sub_msa == 4).float().mean(-1)
            sub_msa = sub_msa[gap_frac < 0.7]
            if sub_msa.shape[0] == 0:
                continue
            sub_msa = sub_msa.unsqueeze(0)               # (1, N', L')
            sub_res_id = res_idx_full[:, sel]             # (1, L')

            sub_ids = None
            if input_ids is not None:
                sub_ids = input_ids[:, sel]

            seq_str = "".join(IDX_TO_BASE.get(int(b), "A") for b in msa[0][sel])
            print(f"  crop {s1}-{e1} / {s2}-{e2}  msa={sub_msa.shape}")

            with torch.no_grad():
                out = model(
                    sub_msa,
                    input_ids=sub_ids,
                    res_id=sub_res_id,
                    num_recycle=3,
                    msa_cutoff=msa_cutoff,
                    is_training=False,
                )

            sub_idx = np.where(sel)[0]
            idx2d = np.ix_(sub_idx, sub_idx)
            geoms = out["geoms"]
            accum["contact"][np.ix_(sub_idx, sub_idx)] += geoms["contact"].detach()
            for atom in dist_atoms:
                if atom in geoms["distance"]:
                    accum["distance"][atom][np.ix_(sub_idx, sub_idx)] += (
                        geoms["distance"][atom].detach()
                    )
            count_2d[np.ix_(sub_idx, sub_idx)] += 1

    # Normalise by overlap count
    count_2d = count_2d.clamp(min=1)
    accum["contact"] /= count_2d
    for atom in dist_atoms:
        accum["distance"][atom] /= count_2d.unsqueeze(-1)

    return {k: v.cpu() for k, v in accum.items() if not isinstance(v, dict)} | \
           {"distance": {k: v.cpu() for k, v in accum["distance"].items()}}


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict_single(
    model,
    tokenizer,
    msa: np.ndarray,
    device: torch.device,
    window: int = 100,
    shift: int = 50,
    msa_cutoff: int = 200,
) -> dict:
    """Run full prediction for a single MSA."""
    L = msa.shape[-1]
    seq_str = "".join(IDX_TO_BASE.get(int(b), "A") for b in msa[0])

    # Tokenise the query sequence
    input_ids = None
    if tokenizer is not None:
        token_ids = tokenizer.convert_tokens_to_ids(seq_str)
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    if L > 2 * window:
        print(f"Long sequence (L={L}): using sliding-window prediction")
        return predict_with_crops(
            model, msa, input_ids, device, window=window, shift=shift, msa_cutoff=msa_cutoff
        )

    msa_t = torch.from_numpy(msa).to(device).unsqueeze(0)
    res_id = torch.arange(L, device=device).view(1, L)

    with torch.no_grad():
        out = model(
            msa_t,
            input_ids=input_ids,
            res_id=res_id,
            num_recycle=3,
            msa_cutoff=msa_cutoff,
            is_training=False,
        )

    geoms = out["geoms"]
    result = {
        "contact": geoms["contact"].cpu(),
        "distance": {k: v.cpu() for k, v in geoms["distance"].items()},
        "ss": out["ss"].cpu(),
        "contact_mask": out["contact_mask"].cpu(),
    }
    return result


def save_prediction(pred: dict, out_path: str) -> None:
    """Write prediction to an NPZ file."""
    arrays = {}
    arrays["contact"] = pred["contact"].numpy()
    for atom, val in pred["distance"].items():
        arrays["dist_" + atom.replace("'", "p")] = val.numpy()
    if "ss" in pred:
        arrays["ss_pred"] = pred["ss"].numpy()
    if "contact_mask" in pred:
        arrays["contact_mask"] = pred["contact_mask"].numpy()
    np.savez_compressed(out_path, **arrays)
    print(f"Saved prediction: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_parser():
    parser = argparse.ArgumentParser(
        description="Predict RNA inter-residue distances using RNAGenesis"
    )
    parser.add_argument("-i", "--msa", required=True, help="Input MSA file (.a3m or .afa)")
    parser.add_argument("-o", "--out", required=True, help="Output NPZ file path")
    parser.add_argument("--encoder_path", default=None, help="Path to pretrained RNAGenesis encoder")
    parser.add_argument("--checkpoint", required=True, help="Trained structure predictor checkpoint (.pt)")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=12)
    parser.add_argument("--encoder_dim", type=int, default=1280)
    parser.add_argument("--encoder_proj_dim", type=int, default=32)
    parser.add_argument("--window", type=int, default=100, help="Sliding window size (default 100)")
    parser.add_argument("--shift", type=int, default=50, help="Sliding window shift (default 50)")
    parser.add_argument("--msa_cutoff", type=int, default=200)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--cpu", type=int, default=4)
    return parser


def main():
    args = get_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_num_threads(args.cpu)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # ---- Load encoder ----
    encoder, tokenizer = None, None
    if args.encoder_path:
        print(f"Loading RNAGenesis encoder from {args.encoder_path} ...")
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_path, trust_remote_code=True)
        encoder = AutoModel.from_pretrained(
            args.encoder_path, trust_remote_code=True, torch_dtype=torch.float32
        ).to(device)
        encoder.eval()

    # ---- Build model ----
    model = RNAGenesisStructurePredictor(
        encoder=encoder,
        encoder_dim=args.encoder_dim,
        dim=args.channels,
        depth=args.n_blocks,
        encoder_proj_dim=args.encoder_proj_dim,
        freeze_encoder=True,
    ).to(device)
    model.eval()

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("state_dict", state))
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ---- Parse MSA ----
    print(f"Parsing MSA: {args.msa}")
    msa = parse_a3m(args.msa)
    print(f"  MSA shape: {msa.shape}")

    # ---- Predict ----
    pred = predict_single(model, tokenizer, msa, device, args.window, args.shift, args.msa_cutoff)

    # ---- Save ----
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_prediction(pred, args.out)


if __name__ == "__main__":
    main()
