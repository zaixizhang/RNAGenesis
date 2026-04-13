from __future__ import annotations
"""
Convert PDB files to the NPZ format expected by RNAStructureDataset.

Usage:
    python process_pdb.py <pdb_file> <out_dir> [--seq <fasta_or_seq>]
"""

import os
import argparse
from collections import defaultdict

import numpy as np
from Bio import PDB
from scipy.spatial.distance import cdist


# Residue name normalisation (handles modified nucleotides)
def _load_residue_map(path: str | None = None) -> dict[str, str]:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "non_res2res_list")
    if not os.path.isfile(path):
        return defaultdict(lambda: "N")
    return dict(line.split()[:2] for line in open(path))


RESIDUE_MAP = _load_residue_map()

BACKBONE_ATOMS = [
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'",
    "O2'", "C1'", "N1", "N2", "C2", "O2", "O4", "N3", "C4", "N4", "C5",
    "C6", "C8", "O6", "N9", "N7", "N8", "N6",
]


def get_atom_positions(pdb_path: str, seq: str | None = None) -> tuple[dict, list, str]:
    """
    Extract 3-D coordinates of backbone atoms from a PDB file.

    Returns:
        xyzs:      dict {atom: (L, 3) float array with NaN for missing atoms}
        chain_idx: list of chain indices per residue
        pdb_seq:   sequence string read from PDB
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("", pdb_path)[0]
    chains = list(structure.get_chains())
    L = sum(len(ch.child_list) for ch in chains)

    xyzs = {atom: np.full((L, 3), np.nan) for atom in BACKBONE_ATOMS}
    chain_idx, pdb_seq_list = [], []

    _map = defaultdict(lambda: "N")
    _map.update(RESIDUE_MAP)

    for ci, chain in enumerate(chains):
        residues = [r for r in chain if not PDB.is_aa(r)]
        for res in residues:
            pos = len(pdb_seq_list)
            for atom in BACKBONE_ATOMS:
                try:
                    xyzs[atom][pos] = res[atom].coord
                except KeyError:
                    pass
            pdb_seq_list.append(_map[res.resname.strip()])
            chain_idx.append(ci)

    pdb_seq = "".join(pdb_seq_list)
    if seq and seq != pdb_seq:
        raise ValueError(
            f"Sequence mismatch in {pdb_path}:\n  given : {seq}\n  in PDB: {pdb_seq}"
        )
    return xyzs, chain_idx, pdb_seq


def compute_labels(pdb_path: str, seq: str | None = None) -> tuple[dict, dict, list]:
    """
    Compute inter-residue geometry labels from a PDB file.

    Returns:
        labels:   dict of (L, L) distance and contact arrays
        xyzs:     dict of (L, 3) coordinate arrays
        chain_idx: list of chain indices
    """
    xyzs, chain_idx, seq = get_atom_positions(pdb_path, seq)
    L = len(seq)
    seq_arr = np.array(list(seq))

    # Purine (A/G) use N9; pyrimidine (C/U) use N1
    is_purine = (seq_arr == "A") | (seq_arr == "G")
    n9 = np.where(is_purine[:, None], xyzs["N9"], np.nan * np.zeros((L, 3)))
    n1 = np.where(~is_purine[:, None], xyzs["N1"], np.nan * np.zeros((L, 3)))
    xyzs["N1/9"] = np.where(is_purine[:, None], n9, n1)

    # Purine C2; pyrimidine C4
    c2 = np.where(is_purine[:, None], xyzs["C2"], np.nan * np.zeros((L, 3)))
    c4 = np.where(~is_purine[:, None], xyzs["C4"], np.nan * np.zeros((L, 3)))
    xyzs["C4/2"] = np.where(is_purine[:, None], c2, c4)

    labels: dict[str, np.ndarray] = {}
    for atom in ("C3'", "C1'", "P"):
        labels[atom] = cdist(xyzs[atom], xyzs[atom])
    labels["N1"] = cdist(xyzs["N1/9"], xyzs["N1/9"])
    labels["C4"] = cdist(xyzs["C4/2"], xyzs["C4/2"])
    labels["CiNj"] = cdist(xyzs["C4'"], xyzs["N1/9"])
    labels["PiNj"] = cdist(xyzs["P"], xyzs["N1/9"])

    all_dist = np.full((L, L), np.nan)
    contact = np.zeros((L, L), dtype=np.float32)
    all_coords = np.stack([xyzs[a] for a in BACKBONE_ATOMS], axis=1)  # (L, A, 3)

    for i in range(L):
        for j in range(i + 1, L):
            ci = all_coords[i]
            cj = all_coords[j]
            mask_i = ~np.isnan(ci).any(1)
            mask_j = ~np.isnan(cj).any(1)
            if mask_i.any() and mask_j.any():
                d = cdist(ci[mask_i], cj[mask_j]).min()
                all_dist[i, j] = all_dist[j, i] = d
                contact[i, j] = contact[j, i] = float(d < 8.0)

    labels["all"] = all_dist
    labels["contact"] = contact
    return labels, xyzs, chain_idx


def pdb_to_npz(pdb_path: str, out_dir: str, seq: str | None = None):
    """
    Process a single PDB file and write label / coordinate NPZ files.

    Output structure:
        out_dir/label/<name>.npz  — inter-residue distances + contacts
        out_dir/xyz/<name>.npz   — raw 3-D coordinates
    """
    from .pdb2seq import pdb_to_sequence  # optional helper

    if seq is None:
        try:
            seq = pdb_to_sequence(pdb_path)
        except Exception:
            print(f"[Warning] Could not extract sequence from {pdb_path}; skipping.")
            return None

    name = os.path.splitext(os.path.basename(pdb_path))[0]
    label_dir = os.path.join(out_dir, "label")
    xyz_dir = os.path.join(out_dir, "xyz")
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(xyz_dir, exist_ok=True)

    try:
        labels, xyzs, chain_idx = compute_labels(pdb_path, seq)
    except Exception as e:
        print(f"[Error] {pdb_path}: {e}")
        return None

    np.savez_compressed(os.path.join(label_dir, f"{name}.npz"), **labels)
    np.savez_compressed(os.path.join(xyz_dir, f"{name}.npz"), **xyzs)
    print(f"Saved: {name} (L={len(seq)})")
    return labels, xyzs, chain_idx


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDB files to NPZ format.")
    parser.add_argument("pdb", help="Input PDB file or directory")
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument("--seq", default=None, help="Expected sequence (for validation)")
    args = parser.parse_args()

    if os.path.isdir(args.pdb):
        pdbs = [
            os.path.join(args.pdb, f)
            for f in os.listdir(args.pdb)
            if f.endswith(".pdb")
        ]
    else:
        pdbs = [args.pdb]

    for pdb_file in sorted(pdbs):
        pdb_to_npz(pdb_file, args.out_dir, args.seq)
