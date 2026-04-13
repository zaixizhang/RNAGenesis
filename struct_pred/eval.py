from __future__ import annotations
"""
Evaluate predicted RNA tertiary structures against native PDB structures.

Metrics:
  - Distance correlation (C3' Pearson r, long-range ≥12 nt)
  - Contact precision (top L/5, L/2, L pairs at <8 Å)
  - TM-score and RMSD (via US-align, if available)

Usage:
    python eval.py \\
        --npz_dir   <pred_npz_dir> \\
        --native_dir <native_npz_dir> \\
        --lst       test.lst \\
        --out       results.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.loss import distance_correlation


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def contact_precision(pred_contact: np.ndarray, native_dist: np.ndarray, top_k_frac: float = 0.2, cutoff: float = 8.0, min_sep: int = 12) -> float:
    """
    Precision of top-k predicted contacts against native distance cutoff.

    Args:
        pred_contact: (L, L) predicted contact probability.
        native_dist:  (L, L) native pairwise distance matrix.
        top_k_frac:   Fraction of L pairs to evaluate.
        cutoff:       Distance cutoff for a 'true' contact (Å).
        min_sep:      Minimum sequence separation.

    Returns:
        Precision in [0, 1].
    """
    L = pred_contact.shape[0]
    k = max(1, int(L * top_k_frac))

    # Mask short-range and NaN
    mask = np.ones((L, L), dtype=bool)
    for i in range(L):
        mask[i, max(0, i - min_sep + 1): i + min_sep] = False
    np.fill_diagonal(mask, False)
    mask &= ~np.isnan(native_dist)

    # Upper triangle only
    iu = np.triu_indices(L, k=1)
    up_mask = np.zeros((L, L), dtype=bool)
    up_mask[iu] = True
    mask &= up_mask

    if mask.sum() == 0:
        return float("nan")

    idx = np.argwhere(mask)
    probs = pred_contact[mask]
    top_idx = idx[np.argsort(-probs)[:k]]
    native_contacts = native_dist[top_idx[:, 0], top_idx[:, 1]] < cutoff
    return float(native_contacts.mean())


def evaluate_entry(pred_npz: str, native_npz: str) -> dict:
    """
    Compute all evaluation metrics for a single prediction.

    Args:
        pred_npz:   Path to predicted NPZ (from predict.py).
        native_npz: Path to native NPZ (from prepare_data / process_pdb).

    Returns:
        dict of metric name → value.
    """
    pred = np.load(pred_npz, allow_pickle=True)
    native = np.load(native_npz, allow_pickle=True)

    metrics: dict[str, float] = {}

    # Distance correlation on C3'
    pred_c3_key = "dist_C3p"  # predict.py replaces ' with p
    if pred_c3_key in pred and "C3'" in native:
        pred_c3 = pred[pred_c3_key]         # (L, L, bins)
        native_c3 = native["C3'"]           # (L, L)
        L = min(pred_c3.shape[0], native_c3.shape[0])
        metrics["dist_corr_C3p"] = distance_correlation(pred_c3[:L, :L], native_c3[:L, :L])

    # Contact precision variants
    if "contact" in pred and "contact" in native:
        pc = pred["contact"]
        nc_mat = native.get("all", None)
        if nc_mat is None and "C3'" in native:
            nc_mat = native["C3'"]
        if nc_mat is not None:
            L = min(pc.shape[0], nc_mat.shape[0])
            for top_k in (0.2, 0.5, 1.0):
                key = f"prec_top{int(top_k * 100)}L"
                metrics[key] = contact_precision(pc[:L, :L], nc_mat[:L, :L], top_k_frac=top_k)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate RNA structure predictions")
    parser.add_argument("--npz_dir", required=True, help="Directory with predicted NPZ files")
    parser.add_argument("--native_dir", required=True, help="Directory with native NPZ files")
    parser.add_argument("--lst", default=None, help="Entry ID list; defaults to all entries in npz_dir")
    parser.add_argument("--out", default="results.csv", help="Output CSV file")
    args = parser.parse_args()

    if args.lst:
        entries = open(args.lst).read().splitlines()
    else:
        entries = [f[:-4] for f in os.listdir(args.npz_dir) if f.endswith(".npz")]

    rows = []
    for eid in sorted(entries):
        pred_path = os.path.join(args.npz_dir, f"{eid}.npz")
        native_path = os.path.join(args.native_dir, f"{eid}.npz")
        if not os.path.isfile(pred_path) or not os.path.isfile(native_path):
            print(f"[Skip] {eid}: file not found")
            continue
        try:
            m = evaluate_entry(pred_path, native_path)
            m["entry"] = eid
            rows.append(m)
            parts = ", ".join(f"{k}={v:.3f}" for k, v in m.items() if k != "entry")
            print(f"  {eid}: {parts}")
        except Exception as e:
            print(f"[Error] {eid}: {e}")

    if not rows:
        print("No results to write.")
        return

    all_keys = ["entry"] + [k for k in rows[0] if k != "entry"]
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Summary statistics
    print(f"\n--- Summary ({len(rows)} entries) ---")
    for key in all_keys:
        if key == "entry":
            continue
        vals = [r[key] for r in rows if not np.isnan(r.get(key, float("nan")))]
        if vals:
            print(f"  {key}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  n={len(vals)}")

    print(f"\nResults written to: {args.out}")


if __name__ == "__main__":
    main()
