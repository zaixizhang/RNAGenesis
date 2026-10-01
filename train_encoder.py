"""Stage-one RNAGenesis MLM training. Launch with python or torchrun."""

import argparse
from contextlib import nullcontext
from datetime import timedelta
from itertools import islice
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler

from pretraining.data import (
    MLMCollator, PreparedRNADataset, RNATokenizer, build_line_index, corpus_identity,
)
from pretraining.model import EncoderConfig, RNAEncoderForMaskedLM


def learning_rate(update, settings):
    """One-based optimizer update: first=initial, warmup=peak, last=minimum."""
    warmup, total = settings["warmup_steps"], settings["max_steps"]
    if warmup > 1 and update <= warmup:
        fraction = (update - 1) / (warmup - 1)
        return settings["initial_lr"] + fraction * (settings["peak_lr"] - settings["initial_lr"])
    start = max(warmup, 1)
    fraction = min(max((update - start) / max(total - start, 1), 0.0), 1.0)
    return settings["min_lr"] + 0.5 * (settings["peak_lr"] - settings["min_lr"]) * (1 + math.cos(math.pi * fraction))


class SlicedBatchSampler:
    """Skip already-consumed batches without reading or corrupting their data."""

    def __init__(self, sampler, start, stop):
        self.sampler, self.start, self.stop = sampler, start, stop

    def __iter__(self):
        return islice(iter(self.sampler), self.start, self.stop)

    def __len__(self):
        return max(0, self.stop - self.start)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(device):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def atomic_save(value, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def validate_settings(config, world_size):
    settings = config["training"]
    required = {
        "global_batch_size", "micro_batch_size", "max_steps", "warmup_steps",
        "initial_lr", "peak_lr", "min_lr", "weight_decay", "adam_betas", "adam_eps",
        "max_grad_norm", "precision", "gradient_checkpointing", "mask_probability",
        "mask_replace_probability", "random_replace_probability", "seed", "num_workers",
        "log_every", "save_every", "keep_last_checkpoints", "eval_every", "eval_batches",
    }
    if set(settings) != required:
        raise ValueError("Training config keys mismatch: missing=%s, unknown=%s" % (required - set(settings), set(settings) - required))
    for key in ("global_batch_size", "micro_batch_size", "max_steps", "log_every", "save_every", "keep_last_checkpoints", "eval_every", "eval_batches"):
        if settings[key] < 1:
            raise ValueError("%s must be positive" % key)
    if not 0 <= settings["warmup_steps"] < settings["max_steps"]:
        raise ValueError("warmup_steps must be nonnegative and smaller than max_steps")
    if settings["global_batch_size"] % (settings["micro_batch_size"] * world_size):
        raise ValueError("global_batch_size must be divisible by micro_batch_size * world_size")
    if settings["precision"] not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if not 0 < settings["min_lr"] <= settings["initial_lr"] <= settings["peak_lr"]:
        raise ValueError("Require 0 < min_lr <= initial_lr <= peak_lr")
    if settings["num_workers"] < 0 or settings["max_grad_norm"] <= 0:
        raise ValueError("num_workers must be nonnegative and max_grad_norm positive")
    if config["model"].get("vocab_size", 20) != len(RNATokenizer.tokens):
        raise ValueError("Model vocabulary must match RNATokenizer")


def make_loader(dataset, settings, world_size, rank, start=0, training=True):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=training, seed=settings["seed"], drop_last=True)
    sampler.set_epoch(dataset.epoch)
    batches = BatchSampler(sampler, settings["micro_batch_size"], drop_last=training)
    accumulation = settings["global_batch_size"] // (settings["micro_batch_size"] * world_size)
    stop = (len(batches) // accumulation) * accumulation if training else min(len(batches), settings["eval_batches"])
    if stop == 0:
        raise ValueError("Corpus is too small for one %s batch; lower batch size or provide more sequences" % ("global training" if training else "validation"))
    collator = MLMCollator(settings["mask_probability"], settings["mask_replace_probability"], settings["random_replace_probability"])
    # A separate generator prevents worker initialization from consuming model RNG
    # when a loader is reconstructed after checkpoint resume.
    generator = torch.Generator().manual_seed(settings["seed"] + rank + dataset.epoch)
    return DataLoader(dataset, batch_sampler=SlicedBatchSampler(batches, start, stop),
                      collate_fn=collator, num_workers=settings["num_workers"],
                      pin_memory=torch.cuda.is_available(), generator=generator), stop


def evaluate(model, dataset, settings, device, world_size, rank):
    loader, _ = make_loader(dataset, settings, world_size, rank, training=False)
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=settings["precision"] == "bf16"):
                output = model(**batch)
            valid = batch["labels"].ne(-100)
            correct = (output["logits"].argmax(-1).eq(batch["labels"]) & valid).sum()
            totals += torch.stack((output["loss_sum"].detach().double(), output["masked_tokens"].double(), correct.double()))
    if world_size > 1:
        dist.all_reduce(totals)
    model.train()
    return {"validation_loss": (totals[0] / totals[1]).item(), "validation_accuracy": (totals[2] / totals[1]).item(), "validation_masked_tokens": int(totals[1].item())}


def save_checkpoint(model, optimizer, config, identity, validation_identity, step, epoch, batch_cursor, output_dir, device, world_size, rank):
    local_rng = rng_state(device)
    states = [None] * world_size if rank == 0 else None
    if world_size > 1:
        dist.gather_object(local_rng, states, dst=0)
    else:
        states = [local_rng]
    if rank == 0:
        path = output_dir / ("checkpoint-%08d.pt" % step)
        atomic_save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "config": config, "corpus": identity, "validation_corpus": validation_identity,
            "step": step, "epoch": epoch, "batch_cursor": batch_cursor,
            "world_size": world_size, "rng_states": states,
        }, path)
        print(json.dumps({"checkpoint": str(path), "step": step}), flush=True)
        # Remove only this trainer's older checkpoints, after the new atomic save.
        checkpoints = sorted(output_dir.glob("checkpoint-????????.pt"))
        for old_path in checkpoints[:-config["training"]["keep_last_checkpoints"]]:
            if old_path.stem[len("checkpoint-"):].isdigit():
                old_path.unlink()
    if world_size > 1:
        dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pretraining/encoder.json")
    parser.add_argument("--train_data", required=True, help="USER-SPECIFIED prepared TXT: one uppercase RNA per line")
    parser.add_argument("--validation_data", help="Optional, separately prepared validation TXT")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", help="Path to a checkpoint-XXXXXXXX.pt from this trainer")
    parser.add_argument("--micro_batch_size", type=int, help="Override per-device batch size, preserving global batch size")
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--stop_after_steps", type=int, help="Exit after this total update count while preserving the configured LR schedule")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main(args):
    world_size, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    if use_cuda:
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    settings = config["training"]
    for key in ("micro_batch_size", "num_workers"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    validate_settings(config, world_size)
    if use_cuda and settings["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 requires a supported GPU; select fp32 in the training config")
    if args.stop_after_steps is not None and args.stop_after_steps < 1:
        raise ValueError("stop_after_steps must be positive")
    if world_size > 1:
        dist.init_process_group("nccl" if use_cuda else "gloo", timeout=timedelta(hours=6))
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise ValueError("Output directory is nonempty; use --resume or a new output directory")
    # All ranks check before rank zero creates files in the shared output directory.
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in (args.train_data, args.validation_data):
            if path:
                print("Indexing prepared data: %s" % path, flush=True)
                build_line_index(path, output_dir / "data-index")
    if world_size > 1:
        dist.barrier()
    identity = corpus_identity(args.train_data)
    validation_identity = corpus_identity(args.validation_data) if args.validation_data else None
    model_config = EncoderConfig(**config["model"])
    dataset = PreparedRNADataset(args.train_data, output_dir / "data-index", model_config.max_length, settings["seed"])
    validation = PreparedRNADataset(args.validation_data, output_dir / "data-index", model_config.max_length, settings["seed"] + 1000000) if args.validation_data else None
    # Reject undersized data before allocating the full encoder.
    make_loader(dataset, settings, world_size, rank)
    seed_everything(settings["seed"])
    model = RNAEncoderForMaskedLM(model_config, settings["gradient_checkpointing"]).to(device)
    # AdamW decay on weight matrices, excluding biases and LayerNorm parameters.
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": settings["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=settings["initial_lr"], betas=tuple(settings["adam_betas"]), eps=settings["adam_eps"])
    step, epoch, cursor = 0, 0, 0
    resume_state = None
    if args.resume:
        # Full training state is a local trusted checkpoint, including Python/NumPy RNG.
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_config = state["config"]
        # Worker count does not change deterministic per-sample corruption.
        saved_config["training"]["num_workers"] = settings["num_workers"]
        if saved_config != config or state["world_size"] != world_size:
            raise ValueError("Resume requires the same model, training settings and world size (num_workers may differ)")
        if state["corpus"] != identity or state["validation_corpus"] != validation_identity:
            raise ValueError("Resume data paths, sizes and modification times must match the saved corpus")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, epoch, cursor = state["step"], state["epoch"], state["batch_cursor"]
        resume_state = state["rng_states"][rank]
        del state
    train_model = DistributedDataParallel(model, device_ids=[local_rank] if use_cuda else None, broadcast_buffers=False) if world_size > 1 else model
    seed_everything(settings["seed"] + rank)
    if resume_state is not None:
        restore_rng(resume_state, device)
    accumulation = settings["global_batch_size"] // (settings["micro_batch_size"] * world_size)
    if rank == 0:
        (output_dir / "run_config.json").write_text(json.dumps({"config": config, "corpus": identity, "validation_corpus": validation_identity, "world_size": world_size, "gradient_accumulation_steps": accumulation, "torch_version": torch.__version__}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()), "sequences": len(dataset), "world_size": world_size, "gradient_accumulation_steps": accumulation}), flush=True)
    end_step = min(settings["max_steps"], args.stop_after_steps or settings["max_steps"])
    if step >= end_step:
        raise ValueError("Checkpoint has already reached the requested stopping step")
    train_model.train()
    while step < end_step:
        dataset.epoch = epoch
        loader, usable_batches = make_loader(dataset, settings, world_size, rank, start=cursor)
        batches = iter(loader)
        while cursor < usable_batches and step < end_step:
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(2, dtype=torch.float64, device=device)
            lr = learning_rate(step + 1, settings)
            for group in optimizer.param_groups:
                group["lr"] = lr
            for micro_step in range(accumulation):
                batch = {key: value.to(device, non_blocking=True) for key, value in next(batches).items()}
                synchronization = train_model.no_sync() if world_size > 1 and micro_step + 1 < accumulation else nullcontext()
                with synchronization:
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=settings["precision"] == "bf16"):
                        result = train_model(**batch)
                    # Normalize once across ALL masked tokens in the global batch,
                    # not independently per microbatch or per distributed worker.
                    result["loss_sum"].backward()
                totals += torch.stack((result["loss_sum"].detach().double(), result["masked_tokens"].double()))
                cursor += 1
            if world_size > 1:
                dist.all_reduce(totals)
            if not torch.isfinite(totals).all() or totals[1] <= 0:
                raise FloatingPointError("Nonfinite loss or no masked tokens; checkpoint not advanced")
            scale = world_size / totals[1].item()  # DDP averages gradients across ranks.
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            step += 1
            metrics = {"step": step, "epoch": epoch, "loss": (totals[0] / totals[1]).item(), "lr": lr, "masked_tokens": int(totals[1].item()), "grad_norm": float(norm)}
            if validation is not None and (step % settings["eval_every"] == 0 or step == end_step):
                metrics.update(evaluate(model, validation, settings, device, world_size, rank))
            if rank == 0 and (step % settings["log_every"] == 0 or step == end_step or "validation_loss" in metrics):
                line = json.dumps(metrics)
                print(line, flush=True)
                with open(output_dir / "metrics.jsonl", "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            if step % settings["save_every"] == 0 or step == end_step:
                save_checkpoint(model, optimizer, config, identity, validation_identity, step, epoch, cursor, output_dir, device, world_size, rank)
        del batches, loader
        if cursor == usable_batches:
            epoch, cursor = epoch + 1, 0
    if rank == 0:
        model.save_pretrained(output_dir / "encoder")
        RNATokenizer().save_pretrained(output_dir / "encoder")
        print("Encoder exported to %s" % (output_dir / "encoder"), flush=True)
    if world_size > 1:
        dist.barrier()


if __name__ == "__main__":
    try:
        main(parse_args())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
