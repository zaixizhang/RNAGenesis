from __future__ import annotations
"""
Download and prepare the RNA tertiary structure dataset.

The curated dataset (rnagenesis_struct_pred_data.zip, ~2.4 GB) is hosted on
Google Drive:
  https://drive.google.com/drive/folders/1R5lLvKCWWrAcLHVQjV8lT4hmkXJa1xhO

Contents after extraction:
  train/npz/        3,633 NPZ files  (MSA + SS + inter-residue geometry labels)
  train/train.lst   3,452 training entry IDs
  train/val.lst       181 validation entry IDs
  train/npz.lst     full entry list
  train/BPfold_SS.csv  secondary structure predictions
  test/20_RNA_Puzzles/ 20 RNA Puzzle structures (MSA + native PDB + SS)
  test/CASP15_RNAs/    12 CASP15 RNA targets  (MSA + SS + PDB reference)

Each NPZ file contains:
  aln        (N, L)  integer MSA  (A=0, U=1, C=2, G=3, gap=4)
  ss         (L, L)  secondary structure contact matrix
  P, C3', C1', C4, N1, CiNj, PiNj  (L, L)  inter-residue distances (Å)
  contact    (L, L)  all-atom contact matrix (< 8 Å)

Usage:
    pip install gdown
    python data/prepare_data.py /path/to/output_dir
"""

import argparse
import json
import os
import zipfile

GDRIVE_FOLDER_ID = "1R5lLvKCWWrAcLHVQjV8lT4hmkXJa1xhO"
GDRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"
ARCHIVE_NAME = "rnagenesis_struct_pred_data.zip"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_with_gdown(out_dir: str) -> str:
    """Download the zip archive from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required to download from Google Drive.\n"
            "Install it with:  pip install gdown"
        )

    dest = os.path.join(out_dir, ARCHIVE_NAME)
    if os.path.isfile(dest):
        print(f"Archive already present: {dest}")
        return dest

    print(f"Downloading dataset from Google Drive folder ...")
    print(f"  {GDRIVE_FOLDER_URL}")

    # Download all files in the folder; the folder contains only the zip.
    downloaded = gdown.download_folder(
        id=GDRIVE_FOLDER_ID,
        output=out_dir,
        quiet=False,
        use_cookies=False,
    )

    # gdown returns list of downloaded paths; find our zip
    if downloaded:
        for path in downloaded:
            if path.endswith(".zip"):
                if path != dest:
                    os.rename(path, dest)
                return dest

    # Fallback: look for any zip in out_dir
    for fname in os.listdir(out_dir):
        if fname.endswith(".zip"):
            found = os.path.join(out_dir, fname)
            if found != dest:
                os.rename(found, dest)
            return dest

    raise RuntimeError(
        f"Download completed but {ARCHIVE_NAME} was not found in {out_dir}.\n"
        f"Try downloading manually from:\n  {GDRIVE_FOLDER_URL}\n"
        f"and place the zip in {out_dir}."
    )


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def extract(archive_path: str, out_dir: str) -> str:
    """Extract the zip archive and return the path to the data root."""
    print(f"Extracting {os.path.basename(archive_path)} ...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(out_dir)

    # The zip extracts to rnagenesis_struct_pred_data/
    data_root = os.path.join(out_dir, "rnagenesis_struct_pred_data")
    if not os.path.isdir(data_root):
        # Find wherever the npz files landed
        for root, dirs, files in os.walk(out_dir):
            if any(f.endswith(".npz") for f in files):
                data_root = os.path.dirname(root)
                break

    print(f"Data root: {data_root}")
    return data_root


# ---------------------------------------------------------------------------
# Verify + summarise
# ---------------------------------------------------------------------------

def verify(data_root: str) -> None:
    """Print a summary of the extracted dataset."""
    npz_dir = os.path.join(data_root, "train", "npz")
    train_lst = os.path.join(data_root, "train", "train.lst")
    val_lst = os.path.join(data_root, "train", "val.lst")

    n_npz = len([f for f in os.listdir(npz_dir) if f.endswith(".npz")]) if os.path.isdir(npz_dir) else 0
    n_train = len(open(train_lst).read().splitlines()) if os.path.isfile(train_lst) else 0
    n_val = len(open(val_lst).read().splitlines()) if os.path.isfile(val_lst) else 0

    test_sets = []
    test_dir = os.path.join(data_root, "test")
    if os.path.isdir(test_dir):
        test_sets = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]

    print("\nDataset summary")
    print("---------------")
    print(f"  NPZ files (train+val) : {n_npz}")
    print(f"  Train entries         : {n_train}")
    print(f"  Val entries           : {n_val}")
    print(f"  Test sets             : {', '.join(test_sets)}")
    print(f"\nNPZ directory : {npz_dir}")
    print(f"Train list    : {train_lst}")
    print(f"Val list      : {val_lst}")

    meta = {
        "npz_dir": npz_dir,
        "train_lst": train_lst,
        "val_lst": val_lst,
        "n_train": n_train,
        "n_val": n_val,
        "test_sets": test_sets,
    }
    meta_path = os.path.join(data_root, "dataset_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Metadata      : {meta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare the RNAGenesis structure prediction dataset."
    )
    parser.add_argument(
        "out_dir",
        help="Directory where the dataset will be downloaded and extracted.",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download; assume the zip is already in out_dir.",
    )
    parser.add_argument(
        "--keep_zip",
        action="store_true",
        help="Keep the zip archive after extraction (default: delete it).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.skip_download:
        # Look for an existing zip in out_dir
        zips = [f for f in os.listdir(args.out_dir) if f.endswith(".zip")]
        if not zips:
            raise FileNotFoundError(
                f"No zip file found in {args.out_dir}. "
                f"Download manually from:\n  {GDRIVE_FOLDER_URL}"
            )
        archive = os.path.join(args.out_dir, zips[0])
        print(f"Using existing archive: {archive}")
    else:
        archive = _download_with_gdown(args.out_dir)

    data_root = extract(archive, args.out_dir)

    if not args.keep_zip:
        os.remove(archive)
        print(f"Removed archive: {archive}")

    verify(data_root)
    print("\nReady for training.")


if __name__ == "__main__":
    main()
