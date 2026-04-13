from __future__ import annotations
"""
Download and curate the tertiary structure training / evaluation dataset.

The source data (NPZ files with MSA, secondary structure, and inter-residue
geometry labels) is hosted on Zenodo.  This script:
  1. Downloads the Zenodo archive.
  2. Extracts the relevant task split.
  3. Verifies file integrity.
  4. Writes train / val / test list files.

Zenodo DOI: 10.5281/zenodo.16754363
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile

ZENODO_DOI = "10.5281/zenodo.16754363"
ZENODO_BASE_URL = "https://zenodo.org/api/records"

# Sub-archive name inside the Zenodo deposit that holds the tertiary structure task data
TERTIARY_ARCHIVE_NAME = "TSP_data.zip"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _resolve_zenodo_files(doi: str) -> list[dict]:
    """Query the Zenodo REST API and return the list of file records."""
    import urllib.request
    import json as _json

    record_id = doi.split(".")[-1]
    url = f"{ZENODO_BASE_URL}/{record_id}"
    with urllib.request.urlopen(url) as resp:
        data = _json.loads(resp.read())
    return data["files"]


def _download_file(url: str, dest: str, chunk_size: int = 1 << 20) -> None:
    """Stream-download a URL to *dest* with a progress indicator."""
    import urllib.request

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            fh.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100 * downloaded / total
                print(f"\r  {pct:.1f}%  ({downloaded >> 20} / {total >> 20} MB)", end="", flush=True)
    print()


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def download_dataset(out_dir: str, force: bool = False) -> str:
    """
    Download the Zenodo archive to *out_dir* and return the local path.

    Args:
        out_dir: Destination directory.
        force:   Re-download even if the file already exists.

    Returns:
        Path to the downloaded archive.
    """
    print(f"Resolving Zenodo deposit: {ZENODO_DOI}")
    files = _resolve_zenodo_files(ZENODO_DOI)

    target = next(
        (f for f in files if f["key"] == TERTIARY_ARCHIVE_NAME), None
    )
    if target is None:
        # Fall back: take the first zip that looks like task data
        target = next(
            (f for f in files if "TSP" in f["key"] or "tertiary" in f["key"].lower()),
            files[0],
        )
        print(f"[Warning] {TERTIARY_ARCHIVE_NAME!r} not found; using {target['key']!r}")

    dest = os.path.join(out_dir, target["key"])
    if os.path.isfile(dest) and not force:
        print(f"Archive already exists: {dest}")
        return dest

    print(f"Downloading {target['key']}  ({target['size'] >> 20} MB) ...")
    _download_file(target["links"]["self"], dest)

    expected_md5 = target.get("checksum", "").replace("md5:", "")
    if expected_md5:
        actual_md5 = _md5(dest)
        if actual_md5 != expected_md5:
            raise RuntimeError(
                f"MD5 mismatch for {dest}:\n  expected {expected_md5}\n  got      {actual_md5}"
            )
        print("  MD5 verified ✓")
    return dest


def extract_and_curate(archive_path: str, out_dir: str) -> str:
    """
    Extract the downloaded archive and return the path to the NPZ directory.

    The Zenodo deposit organises tertiary structure data under a folder named
    'TSP' (Tertiary Structure Prediction).  After extraction we expect:
        out_dir/
          npz/        — per-entry .npz files (MSA + SS + geometry labels)
          train.lst   — list of training entry IDs
          val.lst     — list of validation entry IDs
          test.lst    — list of test entry IDs

    Args:
        archive_path: Path to the downloaded zip archive.
        out_dir:      Root output directory.

    Returns:
        Path to the directory containing the NPZ files.
    """
    print(f"Extracting {archive_path} ...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(out_dir)
    print("  Extraction complete.")

    # Locate the NPZ sub-directory
    npz_dir = None
    for root, dirs, files in os.walk(out_dir):
        if any(f.endswith(".npz") for f in files):
            npz_dir = root
            break

    if npz_dir is None:
        raise RuntimeError(f"No .npz files found after extracting to {out_dir}")

    print(f"NPZ files located at: {npz_dir}")
    return npz_dir


def build_split_lists(npz_dir: str, out_dir: str, val_frac: float = 0.05) -> None:
    """
    Build train / val / test split list files.

    If the Zenodo deposit already provides split files, they are used as-is.
    Otherwise a random 95/5 train/val split is created (no held-out test set).

    Args:
        npz_dir:  Directory containing .npz files.
        out_dir:  Where to write the list files.
        val_frac: Fraction of entries to hold out for validation.
    """
    import random

    # Check for pre-existing split files
    predefined = {}
    for split in ("train", "val", "test"):
        for fname in (f"{split}.lst", f"{split}.txt", f"{split}_ids.txt"):
            candidate = os.path.join(out_dir, fname)
            if os.path.isfile(candidate):
                predefined[split] = candidate
                break

    if predefined:
        print("Pre-existing split files detected:")
        for split, path in predefined.items():
            n = len(open(path).read().splitlines())
            print(f"  {split}: {n} entries  ({path})")
        return

    # Build splits from available npz files
    all_ids = sorted(f[:-4] for f in os.listdir(npz_dir) if f.endswith(".npz"))
    random.shuffle(all_ids)
    n_val = max(1, int(len(all_ids) * val_frac))
    val_ids = all_ids[:n_val]
    train_ids = all_ids[n_val:]

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        path = os.path.join(out_dir, f"{split}.lst")
        with open(path, "w") as fh:
            fh.write("\n".join(ids) + "\n")
        print(f"  {split}: {len(ids)} entries  →  {path}")

    meta = {"total": len(all_ids), "train": len(train_ids), "val": len(val_ids)}
    with open(os.path.join(out_dir, "dataset_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download and curate the RNA tertiary structure dataset from Zenodo."
    )
    parser.add_argument(
        "out_dir",
        help="Root output directory (will be created if it does not exist).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the archive even if it already exists.",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.05,
        help="Fraction of entries to hold out for validation (default 0.05).",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download; assume archive already in out_dir.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.skip_download:
        archive = download_dataset(args.out_dir, force=args.force)
    else:
        archives = [
            os.path.join(args.out_dir, f)
            for f in os.listdir(args.out_dir)
            if f.endswith(".zip")
        ]
        if not archives:
            raise RuntimeError(f"No .zip archive found in {args.out_dir}")
        archive = archives[0]
        print(f"Using existing archive: {archive}")

    npz_dir = extract_and_curate(archive, args.out_dir)
    build_split_lists(npz_dir, args.out_dir, val_frac=args.val_frac)

    print("\nDataset ready.")
    print(f"  NPZ dir : {npz_dir}")
    print(f"  Lists   : {args.out_dir}/<train|val>.lst")


if __name__ == "__main__":
    main()
