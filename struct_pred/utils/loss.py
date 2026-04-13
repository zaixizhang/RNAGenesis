from __future__ import annotations
"""
Loss functions for RNA structure geometry prediction.
"""

import numpy as np
import torch


def _binary_cross_entropy(pred: torch.Tensor, real: torch.Tensor, mask2d=None) -> torch.Tensor:
    """Binary cross-entropy with NaN masking and optional confidence mask."""
    pred, real = pred.squeeze(), real.squeeze()
    nan_mask = torch.isnan(real)
    pred, real = pred[~nan_mask], real[~nan_mask]
    bce = -((real * torch.log(pred + 1e-7)) + ((1 - real) * torch.log(1 - pred + 1e-7)))
    if bce.numel() == 0 or not (bce > -1e-4).all():
        return torch.tensor(0.0, device=pred.device)
    if mask2d is not None:
        mask2d = mask2d.squeeze()[~nan_mask]
        bce = bce * (1 - mask2d)
        denom = (1 - mask2d).sum()
        return bce.sum() / denom.clamp(min=1)
    return bce.mean()


def _cross_entropy_softmax(
    pred: torch.Tensor,
    real: torch.Tensor,
    mask1d=None,
    mask2d=None,
) -> torch.Tensor:
    """Cross-entropy for soft-label distributions with NaN masking."""
    pred, real = pred.squeeze(), real.squeeze()
    if pred.shape != real.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs real {real.shape}")
    ndim = pred.dim()
    nan_idx = torch.isnan(real[..., 0])
    pred, real = pred[~nan_idx], real[~nan_idx]
    loss = -(real * torch.log(pred + 1e-7)).sum(-1).squeeze()
    if ndim == 1 and mask1d is not None:
        mask1d = mask1d.squeeze()[~nan_idx]
        return (loss * (1 - mask1d)).sum() / (1 - mask1d).sum().clamp(min=1)
    if ndim == 2 and mask2d is not None:
        mask2d = mask2d.squeeze()[~nan_idx]
        return (loss * (1 - mask2d)).sum() / (1 - mask2d).sum().clamp(min=1)
    return loss.mean()


def geometry_loss(
    pred_geom: dict,
    native_geom: dict,
    device: torch.device,
    conf_mask1d=None,
    conf_mask2d=None,
) -> torch.Tensor:
    """
    Combined geometry loss summing distance cross-entropies and contact BCE.

    Args:
        pred_geom:   Model output dict with keys 'distance' (dict) and 'contact'.
        native_geom: Ground-truth dict matching the same keys.
        device:      Target device for native tensors.
        conf_mask1d: (L,) confidence mask — 1 = low-confidence residue (ignore).
        conf_mask2d: (L, L) confidence mask — 1 = low-confidence pair (ignore).

    Returns:
        Scalar loss tensor.
    """
    loss = torch.tensor(0.0, device=device)
    for key in native_geom:
        if key == "contact":
            loss = loss + _binary_cross_entropy(
                pred_geom["contact"].float(),
                native_geom["contact"].to(device),
                conf_mask2d,
            )
        else:
            n_atoms = max(len(native_geom[key]), 1)
            for atom in native_geom[key]:
                # Skip 1-D targets when there's a 1-D confidence mask
                if conf_mask1d is not None and pred_geom[key][atom].squeeze().dim() <= 2:
                    continue
                loss = loss + _cross_entropy_softmax(
                    pred_geom[key][atom].float(),
                    native_geom[key][atom].to(device),
                    conf_mask1d,
                    conf_mask2d,
                ) / n_atoms
    return loss


# ---------------------------------------------------------------------------
# Evaluation metric
# ---------------------------------------------------------------------------

def distance_correlation(pred: np.ndarray, dist: np.ndarray, sep: int = 12) -> float:
    """
    Pearson correlation between predicted and native C3' distances for long-range pairs.

    Args:
        pred:  (L, L, bins) or (L, L) predicted distribution / distance.
        dist:  (L, L) native distance matrix.
        sep:   Minimum sequence separation to consider.

    Returns:
        Correlation coefficient.
    """
    L = dist.shape[-1]

    if pred.ndim == 3:
        # Convert soft distribution to expected distance
        bin_vals = np.linspace(3.5, 40.5, pred.shape[-1])
        pred_d = (pred * bin_vals).sum(-1)
    else:
        pred_d = pred

    idx = np.triu_indices(L, k=sep)
    p, r = pred_d[idx], dist[idx]
    mask = ~np.isnan(r)
    p, r = p[mask], r[mask]
    if len(p) < 2:
        return 0.0
    cov = np.mean((p - p.mean()) * (r - r.mean()))
    denom = np.std(p) * np.std(r)
    return float(cov / denom) if denom > 1e-8 else 0.0
