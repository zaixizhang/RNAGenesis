"""
Generate 3-D RNA structure from predicted inter-residue distance distributions
using PyRosetta energy minimisation.

Usage:
    python fold.py \\
        -npz  <predicted_distances.npz> \\
        -fa   <sequence.fasta> \\
        -out  <output.pdb> \\
        -tmp  /tmp/fold_<name> \\
        -nm   5 \\
        --cpu 8

Requires: pyrosetta (install from https://www.pyrosetta.org/)
"""

import argparse
import math
import os
import random
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_fasta(path: str) -> str:
    """Read first sequence from a FASTA file."""
    seq = ""
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if seq:
                    break
                continue
            seq += line.strip()
    return seq.replace("T", "U")


def load_predictions(npz_path: str) -> dict:
    """
    Load the NPZ file written by predict.py and reconstruct the full prediction
    dict expected by the constraint-building routines.

    Returns:
        {'distance': {atom: (L,L,bins) array}, 'contact': (L,L) array}
    """
    raw = np.load(npz_path, allow_pickle=True)
    dist_atoms = ["P", "C3'", "C1'", "C4", "N1"]
    distances = {}
    for atom in dist_atoms:
        # Key convention: dist_<atom> with apostrophes replaced by 'p'
        key = "dist_" + atom.replace("'", "p")
        if key in raw:
            distances[atom] = raw[key]
    return {"distance": distances, "contact": raw.get("contact")}


# ---------------------------------------------------------------------------
# Rosetta constraint file generation
# ---------------------------------------------------------------------------

def _dist_spline(prob_row: np.ndarray, atom: str, i: int, j: int, spline_dir: str) -> str | None:
    """Write a spline file and return the AtomPair constraint string."""
    dist_bins = np.linspace(3.5, 40.5, len(prob_row))
    # Negative log probability as restraint potential
    potential = -np.log(prob_row.clip(min=1e-7))
    potential -= potential.min()

    # Map to Rosetta spline format: integer x-axis from 2 to 42 Å
    xs = np.arange(2, 43, dtype=float)
    ys = np.interp(xs, dist_bins, potential)

    sfile = os.path.join(spline_dir, f"{atom}_{i+1}.{j+1}.txt")
    with open(sfile, "w") as fh:
        fh.write("x_axis" + "".join(f"\t{x:.2f}" for x in xs) + "\n")
        fh.write("y_axis" + "".join(f"\t{y:.4f}" for y in ys) + "\n")

    # Resolve atom name
    if atom == "N1":
        a1, a2 = "N1", "N1"   # will be remapped downstream to N9 for purines
    elif atom == "C4":
        a1, a2 = "C4", "C4"
    else:
        a1 = a2 = atom

    cst = (
        f"AtomPair {a1} {i+1} {a2} {j+1} SPLINE EPotential {sfile} 1.0 0.0 1.0\n"
    )
    return cst


def _contact_spline(prob: float, seq: str, i: int, j: int, spline_dir: str) -> list[str]:
    """Write contact spline files for multiple atoms and return constraint strings."""
    sep = abs(i - j)
    if sep <= 12:
        w = 7.0
    elif sep <= 24:
        w = 6.0
    else:
        w = 5.0

    xs = list(range(4)) + list(np.linspace(3.5, 149.5, 147))

    csts = []
    for atom in ["P", "C3'", "C1'", "C4", "N1"]:
        # Contact energy function
        cont_cutoff = {"P": 15, "C3'": 20, "C1'": 20, "C4": 20, "N1": 20}[atom]
        ys = []
        for d in xs:
            if d <= cont_cutoff:
                e = -w
            elif d <= 35:
                e = -0.5 * w * (1 - np.sin(np.pi * (d - 0.5 - (cont_cutoff + 35) / 2) / (35 - cont_cutoff)))
            elif d <= 110:
                e = 0.5 * w * (1 + np.sin(np.pi * (d - 0.5 - (110 + 35) / 2) / (110 - 35)))
            else:
                e = w
            ys.append(e - w)

        sfile = os.path.join(spline_dir, f"cont_{atom}_{i+1}.{j+1}.txt")
        with open(sfile, "w") as fh:
            fh.write("x_axis" + "".join(f"\t{x:.2f}" for x in xs) + "\n")
            fh.write("y_axis" + "".join(f"\t{y:.4f}" for y in ys) + "\n")

        a1 = a2 = atom
        if atom == "N1" and seq[i] in "AG":
            a1 = "N9"
        if atom == "N1" and seq[j] in "AG":
            a2 = "N9"
        if atom == "C4" and seq[i] in "AG":
            a1 = "C2"
        if atom == "C4" and seq[j] in "AG":
            a2 = "C2"

        csts.append(
            f"AtomPair {a1} {i+1} {a2} {j+1} SPLINE EPotential {sfile} 1.0 0.0 1.0\n"
        )
    return csts


def predictions_to_constraints(pred: dict, seq: str, tmp_dir: str, dist_cutoff: float = 0.45, contact_cutoff: float = 0.6) -> tuple[str, str]:
    """
    Convert predicted distributions to Rosetta constraint files.

    Returns:
        (dist_cst_path, contact_cst_path)
    """
    spline_dist_dir = os.path.join(tmp_dir, "splines_dist")
    spline_cont_dir = os.path.join(tmp_dir, "splines_cont")
    os.makedirs(spline_dist_dir, exist_ok=True)
    os.makedirs(spline_cont_dir, exist_ok=True)

    dist_cst_path = os.path.join(tmp_dir, "cstfile_dist.txt")
    cont_cst_path = os.path.join(tmp_dir, "cstfile_cont.txt")

    L = len(seq)
    with open(dist_cst_path, "w") as dist_fh, open(cont_cst_path, "w") as cont_fh:
        # Distance restraints
        for atom, dist_mat in pred["distance"].items():
            for i in range(L):
                for j in range(i + 6, L):
                    prob_row = dist_mat[i, j]
                    # Only add if the distribution is confident enough (entropy cutoff)
                    expected_d = (prob_row * np.linspace(3.5, 40.5, len(prob_row))).sum()
                    if expected_d < 3.0 or expected_d > 40.0:
                        continue
                    if prob_row.max() < dist_cutoff:
                        continue
                    cst = _dist_spline(prob_row, atom, i, j, spline_dist_dir)
                    if cst:
                        dist_fh.write(cst)

        # Contact restraints
        if pred.get("contact") is not None:
            contact = pred["contact"]
            for i in range(L):
                for j in range(i + 1, L):
                    if contact[i, j] >= contact_cutoff:
                        csts = _contact_spline(contact[i, j], seq, i, j, spline_cont_dir)
                        for cst in csts:
                            cont_fh.write(cst)

    return dist_cst_path, cont_cst_path


# ---------------------------------------------------------------------------
# PyRosetta folding
# ---------------------------------------------------------------------------

def fold_with_rosetta(
    npz_path: str,
    fasta_path: str,
    out_pdb: str,
    tmp_dir: str,
    n_models: int = 5,
    dist_cutoff: float = 0.45,
    n_cpu: int = 4,
):
    """
    Run PyRosetta-based 3-D folding from predicted restraints.

    Args:
        npz_path:    Path to the NPZ file from predict.py.
        fasta_path:  Path to FASTA file with the RNA sequence.
        out_pdb:     Output PDB file path.
        tmp_dir:     Temporary directory for intermediate files.
        n_models:    Number of decoy structures to generate.
        dist_cutoff: Minimum probability threshold for distance restraints.
        n_cpu:       Number of parallel folding workers.
    """
    try:
        from pyrosetta import init, create_score_function, MoveMap
        from pyrosetta.rosetta import core, protocols, basic
        from pyrosetta.rosetta.protocols.minimization_packing import MinMover
        from pyrosetta.rosetta.core.import_pose import RNA_HelixAssembler
        import concurrent.futures
    except ImportError:
        raise ImportError(
            "PyRosetta is required for 3-D folding. "
            "Install from https://www.pyrosetta.org/ and activate the appropriate conda environment."
        )

    os.makedirs(tmp_dir, exist_ok=True)
    seq = read_fasta(fasta_path)
    pred = load_predictions(npz_path)

    print("Building Rosetta constraint files ...")
    dist_cst, cont_cst = predictions_to_constraints(pred, seq, tmp_dir, dist_cutoff)

    # Combine into a single constraint file
    all_cst_path = os.path.join(tmp_dir, "cstfile.txt")
    with open(all_cst_path, "w") as fh:
        fh.write(open(dist_cst).read())
        fh.write(open(cont_cst).read())

    init(
        "-mute all -hb_cen_soft -relax:dualspace true -relax:default_repeats 3 "
        "-default_max_cycles 200 -detect_disulf -detect_disulf_tolerance 3.0"
    )

    sf = create_score_function("ref2015")
    sf.set_weight(core.scoring.atom_pair_constraint, 9.0)
    sf.set_weight(core.scoring.dihedral_constraint, 4.0)
    sf.set_weight(core.scoring.angle_constraint, 4.0)
    sf.set_weight(core.scoring.fa_rep, 9.0)
    sf.set_weight(core.scoring.rna_sugar_close, 9.0)
    sf.set_weight(core.scoring.fa_intra_rep, 9.0)
    sf.set_weight(core.scoring.rna_base_pair, 9.0)
    sf.set_weight(core.scoring.rna_base_stack, 9.0)

    def fold_one(_):
        mmap = MoveMap()
        mmap.set_bb(True)
        mmap.set_chi(True)
        mmap.set_jump(True)
        min_mover = MinMover(mmap, sf, "lbfgs_armijo_nonmonotone", 1e-4, True)
        min_mover.max_iter(10000)

        assembler = RNA_HelixAssembler()
        pose = assembler.build_init_pose(seq.lower(), "")

        # Apply constraints
        cst_mover = protocols.constraint_movers.ConstraintSetMover()
        cst_mover.add_constraints(True)
        cst_mover.constraint_file(all_cst_path)
        cst_mover.apply(pose)

        for _ in range(5):
            min_mover.apply(pose)

        return pose

    print(f"Generating {n_models} decoys with {n_cpu} workers ...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpu) as ex:
        futures = [ex.submit(fold_one, i) for i in range(n_models)]
        poses = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Select lowest-energy model
    scores = {p: sf(p) for p in poses}
    best = min(scores, key=scores.__getitem__)
    best.dump_pdb(out_pdb)
    print(f"Best model energy: {scores[best]:.2f}")
    print(f"Saved: {out_pdb}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fold RNA from predicted inter-residue distances via PyRosetta")
    parser.add_argument("-npz", required=True, help="Predicted distances NPZ file (from predict.py)")
    parser.add_argument("-fa", required=True, help="Input FASTA file")
    parser.add_argument("-out", required=True, help="Output PDB file")
    parser.add_argument("-tmp", default="/tmp/rna_fold", help="Temporary working directory")
    parser.add_argument("-nm", type=int, default=5, help="Number of decoys (default 5)")
    parser.add_argument("-dcut", type=float, default=0.45, help="Distance restraint probability cutoff (default 0.45)")
    parser.add_argument("--cpu", type=int, default=4, help="Parallel folding workers (default 4)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fold_with_rosetta(
        npz_path=args.npz,
        fasta_path=args.fa,
        out_pdb=args.out,
        tmp_dir=args.tmp,
        n_models=args.nm,
        dist_cutoff=args.dcut,
        n_cpu=args.cpu,
    )


if __name__ == "__main__":
    main()
