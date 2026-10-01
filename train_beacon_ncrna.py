"""Load a pretrained RNAGenesis encoder and fine-tune it for ncRNA classification."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from finetuning.beacon_ncrna import (
    RNAGenesisForNcRNAClassification, load_pretrained_encoder, validate_task_config,
)
from finetuning.data import ClassificationDataset, classification_collator
from pretraining.data import corpus_identity
from pretraining.model import EncoderConfig, RNAEncoderForMaskedLM
from train_encoder import atomic_save


def validate_training_config(config):
    validate_task_config(config["task"])
    settings = config["training"]
    for key in ("epochs", "batch_size", "learning_rate", "weight_decay", "optimizer", "scheduler", "warmup_ratio"):
        if settings[key] is None:
            raise ValueError("Set training.%s in your run config" % key)
    for key in ("epochs", "batch_size", "gradient_accumulation_steps"):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError("training.%s must be a positive integer" % key)
    if settings["learning_rate"] <= 0 or settings["weight_decay"] < 0:
        raise ValueError("learning_rate must be positive; weight_decay must be nonnegative")
    if settings["optimizer"] not in ("adam", "adamw") or settings["scheduler"] not in ("constant", "linear", "cosine"):
        raise ValueError("Choose optimizer adam/adamw and scheduler constant/linear/cosine")
    if not 0 <= settings["warmup_ratio"] < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if settings["precision"] not in ("fp32", "bf16") or settings["num_workers"] < 0:
        raise ValueError("Use fp32/bf16 precision and a nonnegative worker count")
    if settings["max_grad_norm"] <= 0 or settings["adam_eps"] <= 0:
        raise ValueError("max_grad_norm and adam_eps must be positive")
    if len(settings["adam_betas"]) != 2 or any(not 0 <= beta < 1 for beta in settings["adam_betas"]):
        raise ValueError("adam_betas must contain two values in [0, 1)")
    for name, value in (("pretrained_encoder", config["pretrained_encoder"]),
                        ("data.train", config["data"]["train"]),
                        ("data.validation", config["data"]["validation"]),
                        ("output_dir", config["output_dir"])):
        if not value:
            raise ValueError("Supply %s in the config or the corresponding command-line argument" % name)


def learning_rate_factor(update, total, warmup, schedule):
    if warmup and update <= warmup:
        return update / warmup
    if schedule == "constant":
        return 1.0
    start = max(warmup, 1)
    progress = (update - start) / max(1, total - start)
    progress = min(max(progress, 0), 1)
    return 1 - progress if schedule == "linear" else (1 + math.cos(math.pi * progress)) / 2


def make_loader(dataset, batch_size, workers, seed, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, collate_fn=classification_collator,
                      generator=torch.Generator().manual_seed(seed),
                      pin_memory=torch.cuda.is_available(), drop_last=False)


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items() if key != "ids"}


@torch.no_grad()
def evaluate(model, loader, device, precision="fp32", prediction_path=None):
    model.eval()
    count, loss_sum = 0, 0.0
    confusion = torch.zeros(13, 13, dtype=torch.long)
    records = []
    labeled = None
    for batch in loader:
        tensors = move_batch(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            output = model(**tensors)
        probabilities = output["logits"].float().softmax(-1).cpu()
        predicted = probabilities.argmax(-1)
        labeled = "labels" in batch
        if labeled:
            loss_sum += output["loss_sum"].item()
            labels = batch["labels"]
            confusion += torch.bincount(labels * 13 + predicted, minlength=169).reshape(13, 13)
        count += len(predicted)
        if prediction_path:
            for row, identifier in enumerate(batch["ids"]):
                record = {"id": identifier, "prediction": int(predicted[row])}
                if labeled:
                    record["label"] = int(batch["labels"][row])
                record.update({"probability_%d" % c: float(probabilities[row, c]) for c in range(13)})
                records.append(record)
    if count == 0:
        raise ValueError("Evaluation dataset is empty")
    metrics = {"samples": count}
    if labeled:
        true_positive = confusion.diag().double()
        denominator = confusion.sum(0) + confusion.sum(1)
        per_class_f1 = 2 * true_positive / denominator.clamp_min(1)
        metrics.update(loss=loss_sum / count, accuracy=float(true_positive.sum() / count),
                       macro_f1=float(per_class_f1.mean()), confusion_matrix=confusion.tolist())
    if prediction_path:
        with open(prediction_path, "w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return metrics


def train(config, args, device):
    validate_training_config(config)
    settings = config["training"]
    if device.type == "cuda" and settings["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; select fp32")
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Output directory is nonempty; use --resume or a new directory")
    torch.manual_seed(settings["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(settings["seed"])
    resume = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if resume is not None:
        if resume["device_type"] != device.type:
            raise ValueError("Resume requires the same device type as the saved run")
        saved_config = resume["run_config"]
        saved_config["training"]["num_workers"] = settings["num_workers"]
        if config != saved_config:
            raise ValueError("Resume requires the same run config (num_workers may change)")
        encoder = RNAEncoderForMaskedLM(EncoderConfig(**resume["encoder_config"]))
    else:
        encoder = load_pretrained_encoder(config["pretrained_encoder"])
    train_data = ClassificationDataset(config["data"]["train"], encoder.config.max_length)
    validation_data = ClassificationDataset(config["data"]["validation"], encoder.config.max_length)
    identities = {name: corpus_identity(config["data"][name]) for name in ("train", "validation")}
    if resume is not None and identities != resume["data_identity"]:
        raise ValueError("Training or validation data changed since the checkpoint")
    model = RNAGenesisForNcRNAClassification(encoder, config["task"], settings["gradient_checkpointing"]).to(device)
    if resume is not None:
        model.load_state_dict(resume["model_state"])
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[settings["optimizer"]]
    optimizer = optimizer_class(parameters, lr=settings["learning_rate"],
                                betas=tuple(settings["adam_betas"]), eps=settings["adam_eps"],
                                weight_decay=settings["weight_decay"])
    epoch, update, best_accuracy = 0, 0, -1.0
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer_state"])
        epoch, update, best_accuracy = resume["next_epoch"], resume["update"], resume["best_accuracy"]
        torch.set_rng_state(resume["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(resume["cuda_rng_state"], device)
        if not (output / "best.pt").exists():
            raise ValueError("Keep best.pt beside last.pt when resuming")
        del resume
    stop_epoch = min(settings["epochs"], args.stop_after_epochs or settings["epochs"])
    if epoch >= stop_epoch:
        raise ValueError("Checkpoint has already reached the requested stopping epoch")
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    total_batches = math.ceil(len(train_data) / settings["batch_size"])
    updates_per_epoch = math.ceil(total_batches / settings["gradient_accumulation_steps"])
    total_updates = updates_per_epoch * settings["epochs"]
    warmup = math.ceil(total_updates * settings["warmup_ratio"])
    if warmup >= total_updates:
        raise ValueError("Warmup consumes the entire run; reduce warmup_ratio")
    print(json.dumps({"trainable_parameters": sum(p.numel() for p in parameters),
                      "total_parameters": sum(p.numel() for p in model.parameters()),
                      "training_samples": len(train_data), "validation_samples": len(validation_data)}), flush=True)
    validation_loader = make_loader(validation_data, settings["batch_size"], settings["num_workers"], settings["seed"])
    while epoch < stop_epoch:
        loader = make_loader(train_data, settings["batch_size"], settings["num_workers"], settings["seed"] + epoch, shuffle=True)
        iterator = iter(loader)
        model.train()
        epoch_loss, epoch_count = 0.0, 0
        for window_start in range(0, len(loader), settings["gradient_accumulation_steps"]):
            optimizer.zero_grad(set_to_none=True)
            window_count = 0
            for _ in range(min(settings["gradient_accumulation_steps"], len(loader) - window_start)):
                tensors = move_batch(next(iterator), device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=settings["precision"] == "bf16"):
                    result = model(**tensors)
                if not torch.isfinite(result["loss_sum"]):
                    raise FloatingPointError("Nonfinite training loss")
                result["loss_sum"].backward()
                window_count += tensors["labels"].numel()
                epoch_loss += result["loss_sum"].detach().item()
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(window_count)
            torch.nn.utils.clip_grad_norm_(parameters, settings["max_grad_norm"], error_if_nonfinite=True)
            update += 1
            lr = settings["learning_rate"] * learning_rate_factor(update, total_updates, warmup, settings["scheduler"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            epoch_count += window_count
        epoch += 1
        validation_metrics = evaluate(model, validation_loader, device, settings["precision"])
        metrics = {"epoch": epoch, "update": update, "learning_rate": lr,
                   "train_loss": epoch_loss / epoch_count, "validation": validation_metrics}
        print(json.dumps(metrics), flush=True)
        with open(output / "metrics.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        state = {**model.export_state(), "precision": settings["precision"]}
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            atomic_save({**state, "epoch": epoch, "validation": validation_metrics}, output / "best.pt")
        atomic_save({**state, "run_config": config, "data_identity": identities, "device_type": device.type,
                     "optimizer_state": optimizer.state_dict(), "next_epoch": epoch,
                     "update": update, "best_accuracy": best_accuracy,
                     "torch_rng_state": torch.get_rng_state(),
                     "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}, output / "last.pt")
    # Read test examples only after training and validation-based model selection.
    if epoch == settings["epochs"] and config["data"]["test"]:
        best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(best["model_state"])
        test_data = ClassificationDataset(config["data"]["test"], encoder.config.max_length)
        test_loader = make_loader(test_data, settings["batch_size"], settings["num_workers"], settings["seed"])
        test_metrics = evaluate(model, test_loader, device, settings["precision"], output / "test_predictions.csv")
        (output / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"test": test_metrics}), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate", "predict"), default="train")
    parser.add_argument("--config", default="configs/finetuning/beacon_ncrna.json")
    parser.add_argument("--pretrained_encoder")
    parser.add_argument("--train_data")
    parser.add_argument("--validation_data")
    parser.add_argument("--test_data")
    parser.add_argument("--output_dir")
    parser.add_argument("--resume", help="Resume last.pt at its next epoch")
    parser.add_argument("--stop_after_epochs", type=int, help="Stop after this total epoch count without changing the LR schedule")
    parser.add_argument("--checkpoint", help="Fine-tuned best.pt for evaluation or prediction")
    parser.add_argument("--data", help="User-provided CSV for evaluate/predict mode")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_precision", choices=("fp32", "bf16"), help="Defaults to checkpoint precision on CUDA, fp32 on CPU")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main(args):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Launch this single-device task example with python, not torchrun")
    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.mode == "train":
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        for key in ("pretrained_encoder", "output_dir"):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        for key, attribute in (("train", "train_data"), ("validation", "validation_data"), ("test", "test_data")):
            if getattr(args, attribute) is not None:
                config["data"][key] = getattr(args, attribute)
        if args.stop_after_epochs is not None and args.stop_after_epochs < 1:
            raise ValueError("stop_after_epochs must be positive")
        train(config, args, device)
    else:
        if not args.checkpoint or not args.data or not args.output_dir or args.eval_batch_size < 1:
            raise ValueError("Supply --checkpoint, --data, --output_dir and a positive --eval_batch_size")
        model = RNAGenesisForNcRNAClassification.from_checkpoint(
            args.checkpoint, attention_backend="sdpa" if device.type == "cpu" else None).to(device)
        precision = args.eval_precision or (model.inference_precision if device.type == "cuda" else "fp32")
        if device.type == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support bf16; use --eval_precision fp32 with the SDPA/auto backend")
        dataset = ClassificationDataset(args.data, model.encoder.config.max_length, require_labels=args.mode == "evaluate")
        loader = make_loader(dataset, args.eval_batch_size, 0, 0)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        metrics = evaluate(model, loader, device, precision, prediction_path=output / "predictions.csv")
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main(parse_args())
