from __future__ import annotations
"""
Neural network building blocks for the RNA structure prediction trunk.
Implements Evoformer-style MSA and pair representation update layers.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.utils.checkpoint import checkpoint

from .dropout import DropoutRowwise, DropoutColumnwise


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------

class Symmetrise(nn.Module):
    """Symmetrise a pairwise tensor by averaging with its transpose."""

    def __init__(self, pattern: str):
        super().__init__()
        self.pattern = pattern

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + Rearrange(self.pattern)(x)) / 2.0


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Residual convolution blocks
# ---------------------------------------------------------------------------

class ResConvBlock(nn.Module):
    """2-D residual convolutional block with optional dilation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.15,
        norm: type = nn.InstanceNorm2d,
    ):
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            norm(in_channels, affine=True),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation),
            nn.Dropout(dropout),
            norm(out_channels, affine=True),
            nn.ELU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=pad, dilation=dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ---------------------------------------------------------------------------
# Relative positional encoding
# ---------------------------------------------------------------------------

class RelativePositionEncoding(nn.Module):
    """Adds relative positional bias to pair representations."""

    def __init__(self, dim: int, max_relative: int = 32):
        super().__init__()
        self.max_relative = max_relative
        n_bins = 2 * max_relative + 1
        self.linear = nn.Linear(n_bins, dim)

    def forward(self, z: torch.Tensor, res_id: torch.Tensor) -> torch.Tensor:
        # z: (B, L, L, D), res_id: (B, L)
        device = z.device
        bdy = torch.tensor(self.max_relative, device=device)
        d = res_id[:, :, None] - res_id[:, None, :]       # (B, L, L)
        d = torch.clamp(d, -bdy, bdy)
        bin_vals = torch.arange(-self.max_relative, self.max_relative + 1, device=device)
        d_onehot = (d[..., None] == bin_vals).float()      # (B, L, L, n_bins)
        return z + self.linear(d_onehot)


# ---------------------------------------------------------------------------
# Triangle multiplicative update
# ---------------------------------------------------------------------------

class TriangleMultiplication(nn.Module):
    """Triangle multiplicative update (outgoing or incoming)."""

    def __init__(self, dim: int, direction: str = "outgoing"):
        super().__init__()
        assert direction in ("outgoing", "incoming")
        self.direction = direction
        self.norm = nn.LayerNorm(dim)
        self.proj_ab = nn.Linear(dim, dim * 2)
        self.gate_ab = nn.Sequential(nn.Linear(dim, dim * 2), nn.Sigmoid())
        self.gate_out = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.proj_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.norm(z)
        ab = self.proj_ab(z) * self.gate_ab(z)             # (B, L, L, 2D)
        a, b = ab.chunk(2, dim=-1)                          # (B, L, L, D) each
        gate = self.gate_out(z)
        if self.direction == "outgoing":
            prod = torch.einsum("bikd,bjkd->bijd", a, b)
        else:
            prod = torch.einsum("bkid,bkjd->bijd", a, b)
        return gate * self.proj_out(prod)


class TriangleAttention(nn.Module):
    """Triangle attention over starting or ending node."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, direction: str = "starting"):
        super().__init__()
        assert direction in ("starting", "ending")
        self.direction = direction
        self.heads = heads
        inner_dim = heads * dim_head
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_bias = nn.Linear(dim, heads, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.scale = dim_head ** -0.5

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, L, L, D)
        if self.direction == "ending":
            z = z.transpose(1, 2)

        B, L, _, D = z.shape
        z_norm = self.norm(z)
        qkv = self.to_qkv(z_norm).chunk(3, dim=-1)        # each (B, L, L, h*dh)
        q, k, v = [rearrange(t, "b i j (h d) -> b i h j d", h=self.heads) for t in qkv]
        bias = rearrange(self.to_bias(z_norm), "b i j h -> b i h () j")

        dots = torch.einsum("bihqd,bihjd->bihqj", q, k) * self.scale + bias
        attn = dots.softmax(dim=-1)
        out = torch.einsum("bihqj,bihjd->bihqd", attn, v)
        out = rearrange(out, "b i h j d -> b i j (h d)")
        out = self.to_out(out)

        if self.direction == "ending":
            out = out.transpose(1, 2)
        return out


# ---------------------------------------------------------------------------
# MSA attention
# ---------------------------------------------------------------------------

class MSARowAttentionWithPairBias(nn.Module):
    """Row-wise MSA attention gated by pair representation."""

    def __init__(
        self,
        dim: int,
        dim_pair: int,
        heads: int = 8,
        dim_head: int = 32,
        dropout: float = 0.0,
        tie_row_attn: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.tie_row_attn = tie_row_attn
        inner_dim = heads * dim_head
        self.norm_msa = nn.LayerNorm(dim)
        self.norm_pair = nn.LayerNorm(dim_pair)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_pair_bias = nn.Linear(dim_pair, heads, bias=False)
        self.to_gate = nn.Sequential(nn.Linear(dim, inner_dim), nn.Sigmoid())
        self.to_out = nn.Linear(inner_dim, dim)
        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)

    def _attn_row(self, m: torch.Tensor, pair_bias: torch.Tensor, tie_attn_dim=None, return_attn=False):
        # m: (B, R, L, D),  pair_bias: (B, H, L, L)
        B, R, L, D = m.shape
        qkv = self.to_qkv(self.norm_msa(m)).chunk(3, dim=-1)
        q, k, v = [rearrange(t, "b r l (h d) -> b r h l d", h=self.heads) for t in qkv]
        g = rearrange(self.to_gate(self.norm_msa(m)), "b r l (h d) -> b r h l d", h=self.heads)

        dots = (q * self.scale) @ k.transpose(-2, -1)                    # (B, R, H, L, L)
        dots = dots + pair_bias.unsqueeze(1)                             # broadcast over R

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v                                                    # (B, R, H, L, d)
        out = rearrange(out * g, "b r h l d -> b r l (h d)")
        out = self.to_out(out)
        if return_attn:
            return out, attn.mean(dim=1)                                 # mean over rows
        return out

    def forward(self, m: torch.Tensor, z: torch.Tensor, return_attn: bool = False):
        pair_bias = rearrange(
            self.to_pair_bias(self.norm_pair(z)), "b i j h -> b h i j"
        )
        attn_fn = partial(self._attn_row, pair_bias=pair_bias, return_attn=return_attn)
        if m.requires_grad:
            out = checkpoint(attn_fn, m)
        else:
            out = attn_fn(m)
        return out


# ---------------------------------------------------------------------------
# Outer product mean (MSA → pair)
# ---------------------------------------------------------------------------

class OuterProductMean(nn.Module):
    """Compute outer product mean from MSA representation → pair features."""

    def __init__(self, dim_msa: int, dim_pair: int, d_proj: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(dim_msa)
        self.proj = nn.Linear(dim_msa, d_proj)
        self.out = nn.Linear(d_proj * d_proj, dim_pair)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        # m: (B, R, L, D)
        m = self.proj(self.norm(m))                        # (B, R, L, d)
        R = m.shape[1]
        outer = torch.einsum("brid,brjc->bijcd", m, m) / R
        outer = rearrange(outer, "b i j c d -> b i j (c d)")
        return self.out(outer)


# ---------------------------------------------------------------------------
# Pair update via triangle operations
# ---------------------------------------------------------------------------

class TriUpdate(nn.Module):
    """Pair representation update using triangle multiplications and attentions."""

    def __init__(self, dim: int, dropout_rate_pair: float = 0.0):
        super().__init__()
        self.tri_out = TriangleMultiplication(dim, "outgoing")
        self.tri_in = TriangleMultiplication(dim, "incoming")
        self.tri_attn_start = TriangleAttention(dim, direction="starting")
        self.tri_attn_end = TriangleAttention(dim, direction="ending")
        self.ff = FeedForward(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.norm5 = nn.LayerNorm(dim)
        self.drop_row = DropoutRowwise(dropout_rate_pair)
        self.drop_col = DropoutColumnwise(dropout_rate_pair)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z + self.drop_row(self.tri_out(self.norm1(z)))
        z = z + self.drop_row(self.tri_in(self.norm2(z)))
        z = z + self.drop_row(self.tri_attn_start(self.norm3(z)))
        z = z + self.drop_col(self.tri_attn_end(self.norm4(z)))
        z = z + self.ff(self.norm5(z))
        return z


# ---------------------------------------------------------------------------
# MSA column-wise attention update
# ---------------------------------------------------------------------------

class MSAColumnAttention(nn.Module):
    """Column-wise attention over MSA."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 32):
        super().__init__()
        inner_dim = heads * dim_head
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_gate = nn.Sequential(nn.Linear(dim, inner_dim), nn.Sigmoid())
        self.to_out = nn.Linear(inner_dim, dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        # m: (B, R, L, D)
        B, R, L, D = m.shape
        m_t = m.transpose(1, 2)                           # (B, L, R, D)
        m_norm = self.norm(m_t)
        qkv = self.to_qkv(m_norm).chunk(3, dim=-1)
        q, k, v = [rearrange(t, "b l r (h d) -> b l h r d", h=self.heads) for t in qkv]
        g = rearrange(self.to_gate(m_norm), "b l r (h d) -> b l h r d", h=self.heads)

        dots = (q * self.scale) @ k.transpose(-2, -1)
        attn = dots.softmax(dim=-1)
        out = attn @ v                                     # (B, L, H, R, d)
        out = rearrange(out * g, "b l h r d -> b r l (h d)")
        return self.to_out(out)


# ---------------------------------------------------------------------------
# Pair → MSA update
# ---------------------------------------------------------------------------

class PairToMSA(nn.Module):
    """Update MSA representation from pair features."""

    def __init__(self, dim_msa: int, dim_pair: int, heads: int = 8):
        super().__init__()
        self.norm_pair = nn.LayerNorm(dim_pair)
        self.norm_msa = nn.LayerNorm(dim_msa)
        self.pair_to_bias = nn.Linear(dim_pair, heads)
        self.proj_v = nn.Linear(dim_msa, dim_msa // heads * heads)
        self.proj_out = nn.Linear(dim_msa, dim_msa)
        self.ff = FeedForward(dim_msa)
        self.norm_out = nn.LayerNorm(dim_msa)
        self.heads = heads

    def forward(self, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # z: (B, L, L, Dp), m: (B, R, L, Dm)
        pair = (z + z.transpose(1, 2)) / 2.0
        bias = rearrange(
            self.pair_to_bias(self.norm_pair(pair)).softmax(dim=-2),
            "b i j h -> b h i j"
        )                                                  # (B, H, L, L)
        v = rearrange(
            self.proj_v(self.norm_msa(m)),
            "b r l (h d) -> b h r l d", h=self.heads
        )                                                  # (B, H, R, L, d)
        # bias: (B, H, L, L) — attn over positions for each head
        out = torch.einsum("bhij,bhrjd->bhrid", bias, v)  # (B, H, R, L, d)
        out = rearrange(out, "b h r l d -> b r l (h d)")
        out = self.proj_out(out)
        m = m + out
        return m + self.ff(self.norm_out(m))


# ---------------------------------------------------------------------------
# Single Evoformer block
# ---------------------------------------------------------------------------

class EvoformerBlock(nn.Module):
    """One block of the Evoformer: MSA ↔ pair co-evolution."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 32,
        msa_tie_row_attn: bool = False,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ):
        super().__init__()
        prenorm = partial(PreNorm, dim)
        self.msa_row_attn = prenorm(
            MSARowAttentionWithPairBias(
                dim=dim, dim_pair=dim, heads=heads, dim_head=dim_head,
                dropout=attn_dropout, tie_row_attn=msa_tie_row_attn
            )
        )
        self.msa_ff = prenorm(FeedForward(dim=dim, dropout=ff_dropout))
        self.outer_product = OuterProductMean(dim_msa=dim, dim_pair=dim, d_proj=dim // 2)
        self.pair_update = TriUpdate(dim=dim, dropout_rate_pair=attn_dropout)
        self.pair_to_msa = PairToMSA(dim_msa=dim, dim_pair=dim, heads=heads)

    def forward(self, msa: torch.Tensor, pair: torch.Tensor, return_attn: bool = False):
        if return_attn:
            m_update, attn_map = self.msa_row_attn(msa, pair, return_attn=True)
            attn_map = rearrange(attn_map, "b h i j -> b i j h")
        else:
            m_update = self.msa_row_attn(msa, pair)
        msa = msa + m_update
        msa = msa + self.msa_ff(msa)
        pair = pair + self.outer_product(msa)
        pair = self.pair_update(pair)
        msa = self.pair_to_msa(pair, msa)
        if return_attn:
            return msa, pair, attn_map
        return msa, pair
