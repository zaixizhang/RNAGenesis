"""BEACON ncRNA classification: LoRA encoder, mean readout, and MLP head."""

from dataclasses import asdict
import math
from pathlib import Path

import torch
from torch import nn

from pretraining.model import EncoderConfig, RNAEncoderForMaskedLM


class LowRankUpdate(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, dropout):
        super().__init__()
        self.a = nn.Linear(in_features, rank, bias=False)
        self.b = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, x):
        return self.b(self.a(self.dropout(x))) * self.scale


class LoRALinear(nn.Module):
    def __init__(self, base, config):
        super().__init__()
        self.base = base
        self.update = LowRankUpdate(base.in_features, base.out_features,
                                    config["rank"], config["alpha"], config["dropout"])

    def forward(self, x):
        return self.base(x) + self.update(x)


class LoRAQKV(nn.Module):
    """Adapt individual Q/K/V projections inside the encoder's fused linear."""

    def __init__(self, base, config):
        super().__init__()
        self.base = base
        self.updates = nn.ModuleDict({
            name: LowRankUpdate(base.in_features, base.out_features // 3,
                                config["rank"], config["alpha"], config["dropout"])
            for name in ("query", "key", "value") if name in config["targets"]
        })

    def forward(self, x):
        parts = self.base(x).chunk(3, dim=-1)
        return torch.cat([
            value + self.updates[name](x) if name in self.updates else value
            for name, value in zip(("query", "key", "value"), parts)
        ], dim=-1)


def validate_task_config(config):
    if config["num_classes"] != 13 or config["readout"] != "mean" or config["adaptation"] != "lora":
        raise ValueError("This BEACON task uses 13 classes, mean readout, and LoRA adaptation")
    head, lora = config["head"], config["lora"]
    for name, value in (("head.hidden_sizes", head["hidden_sizes"]),
                        ("head.activation", head["activation"]), ("head.dropout", head["dropout"]),
                        ("lora.rank", lora["rank"]), ("lora.alpha", lora["alpha"]),
                        ("lora.dropout", lora["dropout"]), ("lora.targets", lora["targets"])):
        if value is None:
            raise ValueError("Set task.%s in your run config" % name)
    if not isinstance(head["hidden_sizes"], list) or any(not isinstance(n, int) or n <= 0 for n in head["hidden_sizes"]):
        raise ValueError("head.hidden_sizes must be a list of positive widths (or [] for a linear head)")
    if head["activation"] not in ("relu", "gelu", "tanh"):
        raise ValueError("head.activation must be relu, gelu, or tanh")
    if not isinstance(lora["rank"], int) or lora["rank"] <= 0 or lora["alpha"] <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    allowed = {"query", "key", "value", "attention_output", "ffn_gate_up", "ffn_down"}
    if not lora["targets"] or set(lora["targets"]) - allowed or len(set(lora["targets"])) != len(lora["targets"]):
        raise ValueError("lora.targets must contain unique entries from %s" % sorted(allowed))
    if not 0 <= head["dropout"] < 1 or not 0 <= lora["dropout"] < 1:
        raise ValueError("Dropout probabilities must be in [0, 1)")


class RNAGenesisForNcRNAClassification(nn.Module):
    def __init__(self, encoder, task_config, gradient_checkpointing=True):
        super().__init__()
        validate_task_config(task_config)
        self.encoder, self.task_config = encoder, task_config
        self.encoder.gradient_checkpointing = gradient_checkpointing
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        lora = task_config["lora"]
        for layer in self.encoder.layers:
            if set(lora["targets"]) & {"query", "key", "value"}:
                layer.attention.qkv = LoRAQKV(layer.attention.qkv, lora)
            for name, parent, attribute in (
                ("attention_output", layer.attention, "out"),
                ("ffn_gate_up", layer, "gate_up"), ("ffn_down", layer, "down"),
            ):
                if name in lora["targets"]:
                    setattr(parent, attribute, LoRALinear(getattr(parent, attribute), lora))
        head = task_config["head"]
        activation = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[head["activation"]]
        widths = [encoder.config.hidden_size] + head["hidden_sizes"]
        modules = []
        for before, after in zip(widths[:-1], widths[1:]):
            modules.extend([nn.Linear(before, after), activation(), nn.Dropout(head["dropout"])])
        modules.append(nn.Linear(widths[-1], task_config["num_classes"]))
        self.classifier = nn.Sequential(*modules)

    def forward(self, input_ids, attention_mask=None, labels=None):
        if attention_mask is None:
            attention_mask = input_ids.ne(self.encoder.config.pad_token_id)
        mask = attention_mask.bool()
        if not mask.any(dim=1).all():
            raise ValueError("Each sequence must contain at least one nucleotide")
        hidden = self.encoder.encode(input_ids, mask)
        # Exclude padding; no CLS readout and no MLM corruption during fine-tuning.
        pooled = (hidden.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        logits = self.classifier(pooled)
        output = {"logits": logits}
        if labels is not None:
            output["loss_sum"] = nn.functional.cross_entropy(logits.float(), labels, reduction="sum")
            output["loss"] = output["loss_sum"] / labels.numel()
        return output

    def export_state(self):
        return {"encoder_config": asdict(self.encoder.config), "task_config": self.task_config,
                "model_state": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, path, attention_backend=None):
        # Classifier checkpoints contain only tensors and JSON-compatible metadata.
        state = torch.load(path, map_location="cpu", weights_only=True)
        config = EncoderConfig(**state["encoder_config"])
        if attention_backend is not None:
            config.attention_backend = attention_backend
        encoder = RNAEncoderForMaskedLM(config)
        model = cls(encoder, state["task_config"])
        model.load_state_dict(state["model_state"])
        model.inference_precision = state.get("precision", "fp32")
        return model


def load_pretrained_encoder(path):
    path = Path(path)
    if path.is_dir():
        return RNAEncoderForMaskedLM.from_pretrained(path)
    # Stage-one full training checkpoints also contain Python/NumPy RNG state.
    state = torch.load(path, map_location="cpu", weights_only=False)
    encoder = RNAEncoderForMaskedLM(EncoderConfig(**state["config"]["model"]))
    encoder.load_state_dict(state["model"])
    return encoder
