from __future__ import annotations
"""
Dataset class for RNA tertiary structure prediction fine-tuning.

Each sample is stored as an NPZ file containing:
  aln   — (N, L) integer MSA (0=A,1=U,2=C,3=G,4=gap)
  ss    — (L, L) secondary structure contact matrix (half-matrix)
  conf  — optional (L,) per-residue confidence scores
  P, C3', C1', C4, N1, CiNj, PiNj  — (L, L) inter-residue distance matrices
"""

import os
import random
import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ..utils.misc import parse_a3m


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IDX_TO_BASE = {0: "A", 1: "U", 2: "C", 3: "G"}
BASE_TO_IDX = {v: k for k, v in IDX_TO_BASE.items()}


def _subsample_msa(msa: np.ndarray, limit: int = 200) -> np.ndarray:
    """Randomly subsample an MSA to at most *limit* sequences."""
    n = msa.shape[0]
    if n <= 10:
        return msa
    k = min(limit, int(10 ** np.random.uniform(np.log10(n)) - 10))
    if k <= 0:
        return msa[:1]
    indices = sorted(random.sample(range(1, n), k) + [0])
    return msa[indices]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RNAStructureDataset(Dataset):
    """
    PyTorch dataset for RNA tertiary structure prediction.

    Args:
        entries:          List of PDB/entry IDs (stems of NPZ file names).
        npz_dir:          Directory containing NPZ files.
        tokenizer:        RNAGenesis tokenizer (converts RNA sequence → token IDs).
        max_length:       Maximum sequence length (longer sequences are cropped).
        msa_cutoff:       Maximum number of MSA rows kept per sample.
        random_crop:      Crop randomly (train) vs from the start (eval).
        subsample_msa:    Randomly subsample MSA rows.
        show_warnings:    Emit warnings for malformed samples.
    """

    def __init__(
        self,
        entries: list[str],
        npz_dir: str,
        tokenizer=None,
        max_length: int = 300,
        msa_cutoff: int = 200,
        random_crop: bool = True,
        subsample_msa: bool = True,
        show_warnings: bool = False,
    ):
        self.entries = entries
        self.npz_dir = npz_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.msa_cutoff = msa_cutoff
        self.random_crop = random_crop
        self.subsample_msa = subsample_msa and random_crop
        self.show_warnings = show_warnings

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx):
        pid = self.entries[idx]
        npz_path = os.path.join(self.npz_dir, f"{pid}.npz")

        if not os.path.isfile(npz_path):
            if self.show_warnings:
                warnings.warn(f"Missing file: {npz_path}")
            return {}

        raw = np.load(npz_path, allow_pickle=True)

        # --- MSA ---
        aln = raw.get("aln")
        if aln is None or len(aln.shape) < 2:
            return {}
        if len(aln.shape) == 3:
            if aln.shape[0] == 1:
                aln = aln[0]
            else:
                if self.show_warnings:
                    warnings.warn(f"{pid}: unexpected MSA shape {aln.shape}")
                return {}

        # --- sequence ---
        seq_arr = aln[0]                                  # (L,)

        # Guard: skip if all C3' coords are NaN (no usable backbone)
        if "C3'" in raw and np.isnan(raw["C3'"]).all():
            return {}

        # --- secondary structure ---
        if "ss" not in raw:
            return {}
        ss = raw["ss"] + raw["ss"].T                      # symmetrise

        # --- confidence ---
        L = seq_arr.shape[-1]
        if "conf" in raw:
            conf_mask = (raw["conf"] <= 0.5).astype(np.int64)
            if (~conf_mask.astype(bool)).sum() < 10:
                return {}
            conf_mask_2d = (conf_mask[None, :] | conf_mask[:, None]).astype(np.int64)
        else:
            conf_mask = np.zeros(L, dtype=np.int64)
            conf_mask_2d = np.zeros((L, L), dtype=np.int64)

        # --- spatial crop for long sequences ---
        idx_arr = np.arange(L)
        if L > self.max_length:
            try:
                pair_mask = np.min(
                    np.stack([raw["P"], raw["N1"], raw["PiNj"], raw["PiNj"].T]),
                    axis=0,
                ) < 12
                seed = random.choice(idx_arr) if self.random_crop else 0
                crop = idx_arr[pair_mask[seed]]
                while len(crop) < self.max_length:
                    expanded = idx_arr[(pair_mask[crop]).max(0).astype(bool)]
                    if len(expanded) == len(crop) or len(expanded) > self.max_length:
                        break
                    crop = expanded
            except Exception:
                crop = idx_arr[: self.max_length]
        else:
            crop = idx_arr

        # --- parse distance labels ---
        try:
            labels = self._parse_labels(raw, L, crop)
        except Exception as e:
            if "shape" in str(e) and self.show_warnings:
                warnings.warn(f"{pid}: label shape error")
            elif "shape" not in str(e):
                raise
            return {}

        # --- subsample MSA ---
        if self.subsample_msa:
            aln = _subsample_msa(aln, self.msa_cutoff)
        else:
            aln = aln[: self.msa_cutoff]

        # --- build sequence string for tokenizer ---
        seq_arr_cropped = seq_arr[crop]
        seq_str = "".join(IDX_TO_BASE.get(int(b), "A") for b in seq_arr_cropped)

        # --- tokenize ---
        input_ids = None
        if self.tokenizer is not None:
            token_ids = self.tokenizer.convert_tokens_to_ids(seq_str)
            input_ids = torch.tensor(token_ids, dtype=torch.long)

        sample = {
            "pid": pid,
            "seq_arr": torch.tensor(seq_arr_cropped, dtype=torch.long),
            "aln": torch.tensor(aln[:, crop], dtype=torch.long),
            "ss": torch.tensor(ss[crop][:, crop], dtype=torch.float32).unsqueeze(-1),
            "conf_mask": torch.tensor(conf_mask[crop], dtype=torch.long),
            "conf_mask_2d": torch.tensor(conf_mask_2d[crop][:, crop], dtype=torch.long),
            "idx": torch.tensor(crop, dtype=torch.long),
            "labels": labels,
            # Raw cropped distances (for eval metrics)
            "distance": {
                k: torch.tensor(raw[k][crop][:, crop], dtype=torch.float32)
                for k in ("P", "C3'", "C1'", "C4", "N1")
                if k in raw
            },
        }
        if input_ids is not None:
            sample["input_ids"] = input_ids
        return sample

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_labels(raw: dict, L: int, crop: np.ndarray) -> dict:
        """
        Convert raw NPZ distance arrays to binned probability distributions.

        Distance bins: 3–40 Å in 1 Å steps (38 bins).
        """
        from ..model.config import N_BINS, DIST_MIN, DIST_MAX, DIST_BIN_SIZE

        n_bins = N_BINS["2D"]["distance"]
        dist_atoms = ["C3'", "P", "N1", "C4", "C1'", "CiNj", "PiNj"]
        bin_edges = np.linspace(DIST_MIN, DIST_MAX, n_bins)

        labels: dict = {"distance": {}, "contact": None}

        for atom in dist_atoms:
            if atom not in raw:
                continue
            d = raw[atom][crop][:, crop].astype(np.float32)
            # Convert continuous distance → one-hot bin
            lc = len(crop)
            one_hot = np.zeros((lc, lc, n_bins), dtype=np.float32)
            valid = ~np.isnan(d)
            bin_idx = np.clip(
                np.round((d - DIST_MIN) / DIST_BIN_SIZE).astype(int), 0, n_bins - 1
            )
            one_hot[valid, bin_idx[valid]] = 1.0
            # Fill NaN positions with uniform prior
            one_hot[~valid] = 1.0 / n_bins
            labels["distance"][atom] = torch.tensor(one_hot)

        if "contact" in raw:
            labels["contact"] = torch.tensor(
                raw["contact"][crop][:, crop].astype(np.float32)
            )
        elif "all" in raw:
            contact = (raw["all"][crop][:, crop] < 8).astype(np.float32)
            labels["contact"] = torch.tensor(contact)

        return labels
