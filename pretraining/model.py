"""Bidirectional RNA encoder with hybrid nucleotide / convolutional embeddings."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


@dataclass
class EncoderConfig:
    vocab_size: int = 20
    hidden_size: int = 1280
    num_layers: int = 32
    num_heads: int = 20
    intermediate_size: int = 3413
    max_length: int = 1024
    conv_kernel_sizes: list = field(default_factory=lambda: [3, 5, 7, 9, 11, 13])
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    attention_backend: str = "auto"
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    initializer_range: float = 0.02
    pad_token_id: int = 0

    def __post_init__(self):
        if self.hidden_size % self.num_heads or (self.hidden_size // self.num_heads) % 2:
            raise ValueError("hidden_size must be divisible by num_heads with even head dimension")
        if min(self.num_layers, self.max_length, self.intermediate_size) < 1:
            raise ValueError("Layer count, length, and intermediate size must be positive")
        if not self.conv_kernel_sizes or any(k < 1 or k % 2 == 0 for k in self.conv_kernel_sizes):
            raise ValueError("Convolution kernels must be positive odd integers")
        if self.attention_backend not in ("auto", "sdpa", "flash_attention_2"):
            raise ValueError("attention_backend must be auto, sdpa, or flash_attention_2")


class RotaryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = config.attention_dropout
        self.backend = config.attention_backend
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, attention_mask):
        batch, length, hidden = x.shape
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        # Compute the positional phase in fp32 even under mixed precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            phase = torch.outer(torch.arange(length, device=x.device).float(), self.inv_freq.float())
            cos, sin = phase.cos().to(q.dtype), phase.sin().to(q.dtype)

        def rotate(t):
            even, odd = t[..., 0::2], t[..., 1::2]
            return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

        q, k = rotate(q), rotate(k)
        flash_eligible = x.is_cuda and q.dtype in (torch.bfloat16, torch.float16)
        if self.backend == "flash_attention_2" and (not flash_eligible or flash_attn_varlen_func is None):
            raise RuntimeError("flash_attention_2 requires flash-attn on CUDA with bf16/fp16 activations")
        if self.backend != "sdpa" and flash_eligible and flash_attn_varlen_func is not None:
            # Remove padded tokens before FlashAttention and restore their positions
            # afterwards. This also enables flash kernels on PyTorch 2.1 with padding.
            indices = attention_mask.flatten().nonzero(as_tuple=False).flatten()
            lengths = attention_mask.sum(dim=1, dtype=torch.int32)
            cumulative = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
            packed = [t.transpose(1, 2).reshape(batch * length, self.num_heads, self.head_dim)[indices] for t in (q, k, v)]
            result = flash_attn_varlen_func(
                *packed, cumulative, cumulative, length, length,
                dropout_p=self.dropout if self.training else 0.0, causal=False,
            )
            unpacked = result.new_zeros(batch * length, self.num_heads, self.head_dim).index_copy(0, indices, result)
            return self.out(unpacked.reshape(batch, length, hidden))
        # True means an allowed key. SDPA selects its best available backend.
        attended = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask[:, None, None, :].bool(),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.out(attended.transpose(1, 2).reshape(batch, length, hidden))


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = RotaryAttention(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gate_up = nn.Linear(config.hidden_size, 2 * config.intermediate_size)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x, attention_mask):
        x = x + self.dropout(self.attention(self.attention_norm(x), attention_mask))
        gate, up = self.gate_up(self.ffn_norm(x)).chunk(2, dim=-1)
        x = x + self.dropout(self.down(F.silu(gate) * up))
        return x * attention_mask.unsqueeze(-1)


class RNAEncoderForMaskedLM(nn.Module):
    """Returns token-aligned embeddings and MLM logits in batch-first layout."""

    def __init__(self, config, gradient_checkpointing=False):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.convolutions = nn.ModuleList([
            nn.Conv1d(config.hidden_size, config.hidden_size, k, padding=k // 2)
            for k in config.conv_kernel_sizes
        ])
        self.embedding_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.apply(self._initialize)

    def _initialize(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding):
                with torch.no_grad():
                    module.weight[self.config.pad_token_id].zero_()

    def forward(self, input_ids, attention_mask=None, labels=None):
        if input_ids.ndim != 2 or input_ids.shape[1] > self.config.max_length:
            raise ValueError("Expected [batch, length] input within the configured context length")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        attention_mask = attention_mask.bool()
        # Corruption occurs in the collator, BEFORE any CNN sees the sequence.
        embedding = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        conv_input = embedding.transpose(1, 2)
        hybrid = embedding
        for conv in self.convolutions:
            hybrid = hybrid + conv(conv_input).transpose(1, 2)
        x = self.embedding_dropout(self.embedding_norm(hybrid)) * attention_mask.unsqueeze(-1)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, attention_mask, use_reentrant=False)
            else:
                x = layer(x, attention_mask)
        hidden = self.final_norm(x) * attention_mask.unsqueeze(-1)
        logits = self.lm_head(hidden)
        result = {"last_hidden_state": hidden, "logits": logits}
        if labels is not None:
            count = labels.ne(-100).sum()
            loss_sum = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum")
            result.update(loss_sum=loss_sum, masked_tokens=count, loss=loss_sum / count.clamp_min(1))
        return result

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(asdict(self.config), indent=2) + "\n", encoding="utf-8")
        torch.save(self.state_dict(), directory / "model.pt")

    @classmethod
    def from_pretrained(cls, directory, map_location="cpu"):
        directory = Path(directory)
        config = EncoderConfig(**json.loads((directory / "config.json").read_text(encoding="utf-8")))
        model = cls(config)
        model.load_state_dict(torch.load(directory / "model.pt", map_location=map_location, weights_only=True))
        return model
