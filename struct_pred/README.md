# RNAGenesis — RNA Tertiary Structure Prediction

This module fine-tunes the [RNAGenesis](https://huggingface.co/Zaixi/RNAGenesis) pretrained encoder for RNA tertiary structure prediction. It predicts inter-residue distance distributions and contact probabilities, which are then used to generate 3-D atomic models via PyRosetta energy minimisation.

## Overview

The model takes an RNA sequence (with optional multiple sequence alignment) and predicts:
- **Inter-residue distances** for 7 atom pairs (C3′, P, N1, C4, C1′, CiNj, PiNj) as 38-bin distributions
- **All-atom contact probabilities**
- **Secondary structure** (auxiliary head)
- **Per-pair confidence mask**

### Architecture

```
RNA sequence
    │
    ▼
RNAGenesis encoder (1280-dim, 32 layers)
    │  outer product mean
    ▼
Pair features (L×L×1)  ──┐
                          ├──► concat (L×L×47) ──► Evoformer trunk (12 blocks, recycling ×3)
MSA DCA features (L×L×46)┘                                │
                                              ┌────────────┴────────────┐
                                              ▼                         ▼
                                    Distance heads              Contact head
                                  (7 atoms × 38 bins)           (binary)
```

The MSA branch computes standard DCA + PSSM statistics (46 channels). The encoder branch projects per-residue RNAGenesis embeddings to pairwise features via outer product mean (1 channel). Both are concatenated and embedded into a shared `dim`-dimensional space before entering the Evoformer trunk.

## Installation

```bash
# Core dependencies (add to the existing rnagenesis environment)
pip install einops biopython scipy

# For 3-D structure generation only
conda install -c https://levinthal:paradox@conda.rosettacommons.org -c conda-forge pyrosetta
```

## Data Preparation

Download the training dataset from Zenodo (DOI: [10.5281/zenodo.16754363](https://doi.org/10.5281/zenodo.16754363), ~3.7 GB):

```bash
python data/prepare_data.py /path/to/data
```

This downloads the tertiary structure task archive, extracts NPZ files (each containing MSA, secondary structure, and inter-residue geometry labels), and writes `train.lst` / `val.lst` split files.

To convert your own PDB files into the NPZ format:

```bash
python data/process_pdb.py /path/to/pdbs/ /path/to/output/
```

Each output NPZ contains:
| Key | Shape | Description |
|-----|-------|-------------|
| `aln` | (N, L) | Integer MSA (A=0, U=1, C=2, G=3, gap=4) |
| `ss` | (L, L) | Secondary structure contact matrix |
| `P`, `C3'`, `C1'`, `C4`, `N1`, `CiNj`, `PiNj` | (L, L) | Inter-residue distances (Å) |
| `conf` | (L,) | Per-residue confidence (optional) |

## Training

```bash
python train.py /path/to/data/npz /path/to/output \
    --encoder_path /path/to/RNAGenesis \
    --lst  /path/to/data/train.lst \
    --val_lst /path/to/data/val.lst \
    --channels 64 \
    --n_blocks 12 \
    --max_epochs 30 \
    --gpu 0
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--encoder_path` | — | Path to pretrained RNAGenesis (HuggingFace format) |
| `--freeze_encoder` | False | Freeze encoder weights; only train the trunk |
| `--channels` | 64 | Trunk hidden dimension |
| `--n_blocks` | 12 | Number of Evoformer blocks |
| `--crop_size` | 200 | Max sequence length (longer entries are spatially cropped) |
| `--msa_cutoff` | 200 | Max MSA rows per sample |
| `--num_recycle` | 3 | Recycling iterations |
| `--init_lr` | 5e-4 | Learning rate for trunk |
| `--encoder_lr_factor` | 0.5 | LR multiplier for encoder (if not frozen) |

Checkpoints are saved to `<out_dir>/checkpoints/` each epoch; the best model by validation distance correlation is saved as `best_model.pt`.

## Inference

### Step 1 — Predict inter-residue distances

```bash
python predict.py \
    -i  sequence.a3m \
    -o  predictions/seq.npz \
    --encoder_path /path/to/RNAGenesis \
    --checkpoint   /path/to/best_model.pt \
    --gpu 0
```

For sequences longer than 200 nt the prediction runs with an overlapping sliding window (controlled by `--window` and `--shift`).

### Step 2 — Generate 3-D structure (requires PyRosetta)

```bash
python fold.py \
    -npz predictions/seq.npz \
    -fa  sequence.fasta \
    -out models/seq.pdb \
    -nm  5 \
    --cpu 8
```

This converts the predicted distributions to Rosetta spline restraints and runs energy minimisation, returning the lowest-energy decoy.

## Evaluation

```bash
python eval.py \
    --npz_dir    predictions/ \
    --native_dir data/npz/ \
    --lst        test.lst \
    --out        results.csv
```

Reports per-entry and mean values for:
- **Distance correlation** — Pearson r between predicted and native C3′ distances (sequence separation ≥ 12)
- **Contact precision** — precision of top-L/5, top-L/2, top-L predicted contacts at < 8 Å

## File Structure

```
struct_pred/
├── train.py              # Fine-tuning script
├── predict.py            # Inference → distance NPZ
├── fold.py               # PyRosetta 3-D folding
├── eval.py               # Evaluation metrics
├── requirements.txt
├── model/
│   ├── config.py         # Geometry bin definitions
│   ├── dropout.py        # Structured dropout
│   ├── modules.py        # Evoformer building blocks
│   └── network.py        # RNAGenesisStructurePredictor
├── data/
│   ├── dataset.py        # RNAStructureDataset
│   ├── process_pdb.py    # PDB → NPZ label converter
│   └── prepare_data.py   # Zenodo data downloader
└── utils/
    ├── misc.py           # MSA / SS file parsers
    └── loss.py           # Geometry loss, distance correlation
```

## Citation

If you use this code, please cite the RNAGenesis paper:

```bibtex
@article{zhang2024rna,
  title={RNAGenesis: Foundation Model for Enhanced RNA Sequence Generation and Structural Insights},
  author={Zhang, Zaixi and Chao, Linlin and Jin, Ruofan and Zhang, Yikun and Zhou, Guowei and
          Yang, Yujie and Yang, Yukang and Huang, Kaixuan and Yang, Qirong and Xu, Ziyao and
          Zhang, Xiaoming and Cong, Le and Wang, Mengdi},
  journal={bioRxiv},
  pages={2024--12},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
