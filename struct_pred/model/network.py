from __future__ import annotations
"""
RNA tertiary structure prediction network.
Uses the RNAGenesis pretrained encoder to produce per-residue embeddings,
converts them to pairwise features via outer product mean, and refines them
through an Evoformer-style trunk to predict inter-residue distances and contacts.
"""

from collections import defaultdict

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from .config import GEOMETRY_TARGETS, N_BINS
from .modules import (
    EvoformerBlock,
    RelativePositionEncoding,
    ResConvBlock,
    Symmetrise,
)


# ---------------------------------------------------------------------------
# DCA / MSA statistics helpers
# ---------------------------------------------------------------------------

def _reweight_sequences(msa1hot: torch.Tensor, cutoff: float = 0.8) -> torch.Tensor:
    """Compute sequence weights via effective sequence count."""
    id_min = msa1hot.shape[1] * cutoff
    id_mtx = torch.tensordot(msa1hot, msa1hot, [[1, 2], [1, 2]])
    w = 1.0 / (id_mtx > id_min).float().sum(dim=-1)
    return w


def _msa_to_pssm(msa1hot: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    beff = w.sum()
    f_i = (w[:, None, None] * msa1hot).sum(0) / beff + 1e-9  # (L, 5)
    h_i = (-f_i * f_i.log()).sum(1)                            # (L,)
    return torch.cat([f_i, h_i[:, None]], dim=1)               # (L, 6)


def _fast_dca(msa1hot: torch.Tensor, weights: torch.Tensor, penalty: float = 4.5) -> torch.Tensor:
    """Fast approximate DCA via inverse covariance."""
    nr, nc, ns = msa1hot.shape
    try:
        x = msa1hot.view(nr, nc * ns)
    except RuntimeError:
        x = msa1hot.contiguous().view(nr, nc * ns)
    num_points = weights.sum() - weights.mean().sqrt()
    mean = (x * weights[:, None]).sum(0, keepdim=True) / num_points
    x = (x - mean) * weights[:, None].sqrt()
    cov = x.T @ x / num_points
    cov_reg = cov + torch.eye(nc * ns, device=x.device) * penalty / weights.sum().sqrt()
    inv_cov = torch.inverse(cov_reg)
    x1 = inv_cov.view(nc, ns, nc, ns)
    x2 = x1.permute(0, 2, 1, 3)
    features = x2.reshape(nc, nc, ns * ns)
    x3 = (x1[:, :-1, :, :-1] ** 2).sum((1, 3)).sqrt() * (1 - torch.eye(nc, device=x.device))
    apc = x3.sum(0, keepdim=True) * x3.sum(1, keepdim=True) / x3.sum()
    contacts = (x3 - apc) * (1 - torch.eye(nc, device=x.device))
    return torch.cat([features, contacts[:, :, None]], dim=2)   # (L, L, 26)


def compute_msa_features(msa: torch.Tensor) -> torch.Tensor:
    """
    Compute 2-D pairwise MSA statistics (DCA + 1-D features tiled into 2-D).

    Args:
        msa: (N, L) integer-encoded MSA (0=A,1=U,2=C,3=G,4=gap)

    Returns:
        f2d: (1, L, L, 46)  — batch dim added
    """
    nrow, ncol = msa.shape
    if nrow == 1:
        msa = msa.repeat(2, 1)
        nrow = 2
    msa1hot = (torch.arange(5, device=msa.device) == msa[..., None].long()).float()
    w = _reweight_sequences(msa1hot, 0.8)
    # f1d_seq: (L, 4)  — one-hot of query sequence (4 bases)
    # f1d_pssm: (L, 6) — per-position PSSM (5 freqs + entropy) = 6 dims
    # f1d: (L, 10)  →  tiled to (L, L, 10) for each axis → 20 dims total
    # f2d_dca: (L, L, 26) from fast DCA + APC contacts
    # total 2D input: 20 + 26 = 46 channels
    f1d_seq = msa1hot[0, :, :4]                                # (L, 4)
    f1d_pssm = _msa_to_pssm(msa1hot, w)                        # (L, 6)
    f1d = torch.cat([f1d_seq, f1d_pssm], dim=1)                # (L, 10)
    f2d_dca = _fast_dca(msa1hot, w)                            # (L, L, 26)
    # Tile 1-D features into 2-D: (L,10) → (L,L,10) by repeating along each axis
    f2d = torch.cat(
        [f1d[:, None, :].expand(-1, ncol, -1),   # (L, L, 10) — row features
         f1d[None, :, :].expand(ncol, -1, -1),   # (L, L, 10) — col features
         f2d_dca],                                # (L, L, 26)
        dim=-1,
    ).unsqueeze(0)                                              # (1, L, L, 46)
    return f2d


# ---------------------------------------------------------------------------
# Input embedder
# ---------------------------------------------------------------------------

class PairInputEmbedder(nn.Module):
    """
    Embed MSA statistics + RNAGenesis pair features into a common pair
    representation and produce an initial MSA (token embedding) track.

    Args:
        dim:      Output dimension.
        pair_in:  Number of input pair feature channels (46 DCA + k encoder).
    """

    def __init__(self, dim: int = 64, pair_in: int = 47):
        super().__init__()
        self.norm_pair = nn.InstanceNorm2d(pair_in, affine=True)
        self.pair_proj = nn.Sequential(
            nn.ELU(inplace=True),
            nn.Conv2d(pair_in, dim, 1),
        )
        self.token_emb = nn.Embedding(5, dim)

    def forward(
        self,
        msa: torch.Tensor,
        pair_in: torch.Tensor,
        msa_cutoff: int = 500,
    ):
        """
        Args:
            msa:      (B, N, L) integer MSA, B=1.
            pair_in:  (B, L, L, C) pre-computed pair features.
            msa_cutoff: max MSA rows to embed.

        Returns:
            dict with keys 'pair' (B,L,L,dim) and 'msa' (B,N,L,dim).
        """
        # Pair branch
        x = self.norm_pair(pair_in.permute(0, 3, 1, 2))        # (B, C, L, L)
        pair = self.pair_proj(x).permute(0, 2, 3, 1)           # (B, L, L, dim)
        # MSA branch — embed token indices
        m = self.token_emb(msa[:, :msa_cutoff].long())          # (B, N, L, dim)
        return {"pair": pair, "msa": m}


# ---------------------------------------------------------------------------
# Recycling embedder
# ---------------------------------------------------------------------------

class RecyclingEmbedder(nn.Module):
    """Add recycled pair and single representations to the current iteration."""

    def __init__(self, dim: int = 64, single_in: int = 38):
        super().__init__()
        self.proj_single = nn.Linear(single_in, dim)
        self.norm_pair = nn.LayerNorm(dim)
        self.norm_single = nn.LayerNorm(dim)

    def forward(self, prev: dict):
        pair = self.norm_pair(prev["pair"])
        single = self.norm_single(prev["single"])
        return single, pair


# ---------------------------------------------------------------------------
# Pair transformer trunk
# ---------------------------------------------------------------------------

class PairTransformer(nn.Module):
    """
    Evoformer-style trunk: alternating MSA self-attention and pair update.

    Args:
        dim:   Hidden dimension.
        depth: Number of Evoformer blocks.
    """

    def __init__(
        self,
        dim: int = 64,
        depth: int = 12,
        heads: int = 8,
        dim_head: int = 32,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ):
        super().__init__()
        self.rel_pos = RelativePositionEncoding(dim)
        self.blocks = nn.ModuleList([
            EvoformerBlock(
                dim=dim, heads=heads, dim_head=dim_head,
                attn_dropout=attn_dropout, ff_dropout=ff_dropout,
            )
            for _ in range(depth)
        ])

    def forward(
        self,
        pair: torch.Tensor,
        msa_emb: torch.Tensor,
        res_id: torch.Tensor | None = None,
        return_msa: bool = True,
        return_attn: bool = False,
    ):
        """
        Args:
            pair:    (B, L, L, D)
            msa_emb: (B, N, L, D) or (B, L, D) — will be unsqueezed if needed
            res_id:  (B, L) residue indices for relative positional encoding

        Returns:
            (pair, msa)  or  (pair, msa, attn_maps)
        """
        if msa_emb.dim() == 3:
            msa_emb = msa_emb.unsqueeze(1)               # (B, 1, L, D)

        if res_id is not None:
            pair = self.rel_pos(pair, res_id)

        attn_maps = []
        for i, block in enumerate(self.blocks):
            is_last = i == len(self.blocks) - 1
            outputs = block(msa_emb, pair, return_attn=(return_attn and is_last))
            msa_emb, pair = outputs[0], outputs[1]
            if return_attn and is_last:
                attn_maps = outputs[2]

        out = [pair]
        if return_msa:
            out.append(msa_emb)
        if return_attn:
            out.append(attn_maps)
        return tuple(out)


# ---------------------------------------------------------------------------
# Output geometry heads
# ---------------------------------------------------------------------------

class GeometryHead(nn.Module):
    """
    Convert pair representation to predicted inter-residue geometry distributions.
    Outputs distance distributions and binary contact probabilities.
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        self.norm = nn.Sequential(
            nn.InstanceNorm2d(dim, affine=True),
            nn.ELU(inplace=True),
        )
        # Distance heads (symmetric)
        dist_atoms = GEOMETRY_TARGETS["2D"]["distance"]
        n_dist_bins = N_BINS["2D"]["distance"]
        self.dist_heads = nn.ModuleDict({
            a: nn.Sequential(
                Symmetrise("b i j d->b j i d"),
                nn.Linear(dim, n_dist_bins),
            )
            for a in dist_atoms
        })
        # Contact head
        self.contact_head = nn.Sequential(
            Symmetrise("b i j d->b j i d"),
            nn.Linear(dim, 1),
        )

    def forward(self, pair: torch.Tensor) -> dict:
        # pair: (B, L, L, dim)
        pair = rearrange(
            self.norm(rearrange(pair, "b i j d -> b d i j")),
            "b d i j -> b i j d",
        )
        preds = {"distance": {}, "contact": None}
        for atom, head in self.dist_heads.items():
            preds["distance"][atom] = head(pair).softmax(-1).squeeze(0)
        preds["contact"] = self.contact_head(pair).sigmoid().squeeze()
        return preds


# ---------------------------------------------------------------------------
# RNAGenesis encoder feature extractor
# ---------------------------------------------------------------------------

class EncoderFeatureExtractor(nn.Module):
    """
    Wraps the RNAGenesis encoder and produces pair features via outer product mean.

    The encoder's hidden states (L, B, H) are projected down and converted to
    a (B, L, L, pair_dim) pairwise representation.

    Args:
        encoder:      Pretrained xTrimoPGLMModel instance.
        encoder_dim:  Hidden size of the encoder (1280 for RNAGenesis).
        proj_dim:     Projection dimension before outer product (default 32).
        pair_dim:     Output pair feature dimension.
        freeze:       If True, encoder weights are frozen during fine-tuning.
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int = 1280,
        proj_dim: int = 32,
        pair_dim: int = 1,
        freeze: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Linear(encoder_dim, proj_dim)
        self.pair_proj = nn.Linear(proj_dim * proj_dim, pair_dim)

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor, is_training: bool = True) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) integer token IDs.
            is_training: enables gradient through encoder if True.

        Returns:
            pair_feat: (B, L, L, pair_dim)
        """
        with torch.set_grad_enabled(is_training):
            outputs = self.encoder(input_ids)
            # last_hidden_state: (L, B, H) — seq-first convention
            hidden = outputs.last_hidden_state           # (L, B, H)
            hidden = hidden.permute(1, 0, 2)             # (B, L, H)

        feat = self.proj(hidden)                         # (B, L, proj_dim)
        # Outer product → (B, L, L, proj_dim^2)
        outer = torch.einsum("bid,bjc->bijcd", feat, feat)
        outer = rearrange(outer, "b i j c d -> b i j (c d)")
        return self.pair_proj(outer)                     # (B, L, L, pair_dim)


# ---------------------------------------------------------------------------
# Full structure predictor
# ---------------------------------------------------------------------------

class RNAGenesisStructurePredictor(nn.Module):
    """
    End-to-end RNA tertiary structure predictor built on the RNAGenesis encoder.

    The model:
      1. Extracts per-residue embeddings from the RNAGenesis encoder.
      2. Computes outer product mean → initial pair features.
      3. Concatenates with DCA-derived MSA pair statistics.
      4. Refines through an Evoformer-style trunk with recycling.
      5. Predicts inter-residue distance distributions, contacts, and a
         confidence mask.

    Args:
        encoder:       Pretrained xTrimoPGLMModel.
        encoder_dim:   RNAGenesis hidden size (1280).
        dim:           Trunk hidden dimension.
        depth:         Number of Evoformer blocks.
        encoder_proj_dim: Projection dim used in outer product (default 32 → 1 channel).
        freeze_encoder: Freeze encoder weights during training.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        encoder_dim: int = 1280,
        dim: int = 64,
        depth: int = 12,
        encoder_proj_dim: int = 32,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        # Encoder pair feature extractor (optional — can be None for MSA-only mode)
        self.encoder_extractor: EncoderFeatureExtractor | None = None
        if encoder is not None:
            self.encoder_extractor = EncoderFeatureExtractor(
                encoder=encoder,
                encoder_dim=encoder_dim,
                proj_dim=encoder_proj_dim,
                pair_dim=1,                              # 1 extra channel concatenated with DCA
                freeze=freeze_encoder,
            )

        # MSA DCA (46 channels) + optional encoder pair feat (1 channel)
        pair_in = 47 if encoder is not None else 46
        self.input_embedder = PairInputEmbedder(dim=dim, pair_in=pair_in)
        self.recycling_embedder = RecyclingEmbedder(dim=dim)
        self.trunk = PairTransformer(dim=dim, depth=depth)
        self.geometry_head = GeometryHead(dim=dim)

        # Secondary structure auxiliary head
        self.ss_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            Symmetrise("b i j d->b j i d"),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Dropout(0.1),
            nn.Linear(dim, 1),
        )

        # Contact mask head (for confidence / masking)
        self.mask_head = nn.Sequential(
            Rearrange("b i j d -> b d i j"),
            *[ResConvBlock(dim, dim) for _ in range(4)],
            Rearrange("b d i j -> b i j d"),
            nn.ELU(),
            nn.Linear(dim, 1),
            Symmetrise("b i j d->b j i d"),
            nn.Sigmoid(),
        )

        # Estogram head (distance error / confidence)
        self.estogram_head = nn.Sequential(
            Rearrange("b i j d -> b d i j"),
            *[ResConvBlock(dim, dim) for _ in range(4)],
            Rearrange("b d i j -> b i j d"),
            nn.ELU(),
            nn.Linear(dim, 19),
            Symmetrise("b i j d->b j i d"),
            nn.Softmax(dim=-1),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pair_features(
        self,
        msa: torch.Tensor,
        input_ids: torch.Tensor | None,
        is_training: bool,
    ) -> torch.Tensor:
        """Build 2-D pair feature tensor from MSA stats + encoder (if available)."""
        with torch.no_grad():
            f2d = compute_msa_features(msa[0])           # (1, L, L, 46)

        if self.encoder_extractor is not None and input_ids is not None:
            enc_pair = self.encoder_extractor(input_ids, is_training=is_training)
            f2d = torch.cat([f2d.to(enc_pair.device), enc_pair], dim=-1)  # (1, L, L, 47)

        return f2d

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        msa: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        res_id: torch.Tensor | None = None,
        num_recycle: int = 3,
        msa_cutoff: int = 500,
        is_training: bool = False,
    ) -> dict:
        """
        Args:
            msa:       (B, N, L) integer MSA (AUGC-gap encoded, B=1).
            input_ids: (B, L) integer token IDs for the RNAGenesis encoder.
                       If None the encoder branch is skipped.
            res_id:    (B, L) residue indices for relative positional encoding.
            num_recycle: Number of recycling iterations.
            msa_cutoff:  Maximum MSA rows used in attention.
            is_training: Enables gradient through encoder.

        Returns:
            dict with keys:
              'geoms'        — distance / contact predictions
              'ss'           — secondary structure logits (L, L, 1) after sigmoid
              'contact_mask' — per-residue-pair confidence (L, L)
              'estogram'     — per-pair distance error distribution (L, L, 19)
        """
        pair_in = self._build_pair_features(msa, input_ids, is_training)
        reprs_prev = None

        for c in range(1 + num_recycle):
            with torch.set_grad_enabled(is_training):
                with torch.cuda.amp.autocast(enabled=False):
                    reprs = self.input_embedder(msa, pair_in, msa_cutoff=msa_cutoff)

                    if reprs_prev is None:
                        reprs_prev = {
                            "pair": torch.zeros_like(reprs["pair"]),
                            "single": torch.zeros_like(reprs["msa"][:, 0]),
                        }

                    rec_single, rec_pair = self.recycling_embedder(reprs_prev)
                    reprs["msa"] = reprs["msa"] + rec_single.unsqueeze(1)
                    reprs["pair"] = reprs["pair"] + rec_pair

                    is_last = c == num_recycle
                    trunk_out = self.trunk(
                        reprs["pair"],
                        msa_emb=reprs["msa"],
                        res_id=res_id,
                        return_msa=True,
                        return_attn=is_last,
                    )
                    if is_last:
                        pair_repr, msa_repr, _ = trunk_out
                    else:
                        pair_repr, msa_repr = trunk_out

                    reprs_prev = {
                        "single": msa_repr[:, 0].detach(),
                        "pair": pair_repr.detach(),
                    }

        outputs = {
            "geoms": self.geometry_head(pair_repr),
            "ss": self.ss_head(pair_repr).sigmoid(),
            "contact_mask": self.mask_head(pair_repr).squeeze(-1),
            "estogram": self.estogram_head(pair_repr),
        }
        return outputs
