from __future__ import annotations
"""
Parsing utilities for MSA and secondary structure files.
"""

import string
import numpy as np


# ---------------------------------------------------------------------------
# MSA parsing
# ---------------------------------------------------------------------------

def parse_a3m(filename: str, limit: int = 20000, rm_query_gap: bool = True) -> np.ndarray:
    """
    Parse an A3M-format multiple sequence alignment.

    Args:
        filename:    Path to the A3M file.
        limit:       Maximum number of sequences to load.
        rm_query_gap: Remove columns that are gaps in the query.

    Returns:
        msa: (N, L) uint8 array with encoding A=0, U=1, C=2, G=3, gap=4.
    """
    seqs = []
    # Lowercase letters indicate insertions in A3M — strip them
    _strip_inserts = str.maketrans("", "", string.ascii_lowercase)

    with open(filename, "r") as fh:
        n = 0
        for line in fh:
            line = line.rstrip()
            if not line or line[0] == ">":
                continue
            seq = (
                line
                .replace("W", "A").replace("R", "A").replace("Y", "C")
                .replace("E", "A").replace("I", "A").replace("P", "G")
                .replace("T", "U")
                .translate(_strip_inserts)
            )
            seqs.append(seq)
            n += 1
            if n == limit:
                break

    alphabet = np.array(list("AUCG-"), dtype="|S1").view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype="|S1").view(np.uint8)
    for i, base in enumerate(alphabet):
        msa[msa == base] = i
    msa[msa > 4] = 4                                     # unknown → gap

    if rm_query_gap:
        return msa[:, msa[0] < 4]
    return msa


# ---------------------------------------------------------------------------
# Secondary structure parsing
# ---------------------------------------------------------------------------

def dot_bracket_to_matrix(ss_seq: str) -> np.ndarray:
    """
    Convert dot-bracket notation to a binary contact matrix.

    Supports standard (), [], {}, <> brackets and uppercase/lowercase letter pairs.
    """
    n = len(ss_seq)
    mat = np.zeros((n, n), dtype=np.float32)
    stacks = {"(": [], "[": [], "{": [], "<": []}
    closing = {")": "(", "]": "[", "}": "{", ">": "<"}
    alpha_stacks: dict[str, list] = {ch: [] for ch in string.ascii_lowercase}

    for i, s in enumerate(ss_seq):
        if s in stacks:
            stacks[s].append(i)
        elif s in closing:
            partner = stacks[closing[s]].pop()
            mat[i, partner] = mat[partner, i] = 1.0
        elif s.isupper():
            alpha_stacks[s.lower()].append(i)
        elif s.islower():
            partner = alpha_stacks[s].pop()
            mat[i, partner] = mat[partner, i] = 1.0
        elif s in ".,_:-":
            continue
        else:
            raise ValueError(f"Unrecognised dot-bracket character: {s!r}")

    unmatched = sum(len(v) for v in stacks.values()) + sum(len(v) for v in alpha_stacks.values())
    if unmatched > 0:
        raise ValueError("Provided dot-bracket notation has unmatched brackets.")
    return mat


def parse_ct(ct_file: str, length: int | None = None) -> np.ndarray:
    """
    Parse a .ct (connectivity table) secondary structure file.

    Returns:
        mat: (L, L) binary contact matrix.
    """
    lines = open(ct_file).readlines()
    if length is None:
        length = int(lines[0].split()[0])
    mat = np.zeros((length, length), dtype=np.float32)
    for line in lines:
        items = line.split()
        if (
            len(items) >= 6
            and items[0].isnumeric()
            and items[2].isnumeric()
            and items[3].isnumeric()
            and items[4].isnumeric()
        ):
            i = int(items[0]) - 1
            j = int(items[4]) - 1
            if j >= 0:
                mat[i, j] = mat[j, i] = 1.0
    return mat


def parse_ss_file(path: str, fmt: str = "dot_bracket") -> np.ndarray:
    """
    Unified secondary structure file parser.

    Args:
        path: Path to SS file.
        fmt:  One of 'dot_bracket' or 'ct'.

    Returns:
        mat: (L, L) binary contact matrix.
    """
    if fmt == "dot_bracket":
        ss_seq = open(path).read().strip()
        return dot_bracket_to_matrix(ss_seq)
    elif fmt == "ct":
        return parse_ct(path)
    else:
        raise ValueError(f"Unknown secondary structure format: {fmt!r}")
