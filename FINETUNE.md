# Fine-tuning the RNAGenesis latent diffusion model

This tutorial shows how to **fine-tune** (or train from scratch) the latent
diffusion model that powers RNAGenesis sequence generation. It complements the
existing inference instructions in the main `README.md`.

RNAGenesis generates sequences with a two-stage latent diffusion pipeline:

```
RNA sequence  --(frozen RNA-FM encoder + Q-former)-->  latent z
        latent z   <--(latent DDIM diffusion model)-->   Gaussian noise
        latent z  --(ProGen decoder)-->  RNA sequence
```

Only the **diffusion model** (the denoiser that maps noise <-> latent) is trained
here. The sequence auto-encoder (`EncDec` = RNA-FM encoder + Q-former + ProGen
decoder) stays **frozen** and is loaded from the released checkpoint. This is the
same `EncDec` object that `generation.py` already loads via `--enc_dec_file`, and
the diffusion checkpoint produced here plugs straight back into `generation.py`
via `--dm_file`.

There are two use cases, both handled by the single script `train_diffusion.py`:

| Use case | When | Key flags |
|----------|------|-----------|
| **Fine-tune** an existing diffusion model on a new corpus (e.g. a target RNA family, UTRs, aptamers, your own sequences) | You have a released RNAGenesis diffusion checkpoint and want to adapt it | `--pretrained_ckpts <dm_file>` (+ usually `--lr_warmup_steps`, few `--num_epochs`) |
| **Train from scratch** a new diffusion model on a frozen auto-encoder | You retrained / swapped the auto-encoder, or want a fresh model | `--model_config_name_or_path configs/diffusion/config_n.json` |

> The fine-tuning workflow below is the one most reviewers / users will want: it
> reuses the released checkpoints and only continues training on new data.

---

## 0. Where the files go

This `finetune_tutorial/` folder mirrors the RNAGenesis repository layout. Copy
its contents into the **repository root** (next to `generation.py`), merging into
the existing `data/`, `configs/`, and `models/` folders:

```
RNAGenesis/
├── generation.py                         # (already present) inference
├── train_diffusion.py                    # <-- ADD: (pre-)training / fine-tuning
├── util.py                               # (already present) must expose the cache vars (see §1)
├── FINETUNE.md                           # <-- ADD: this tutorial
├── data/
│   ├── dataset_builder.py                # <-- ADD: builds the HF dataset from a .txt file
│   └── fasta2txt.py                      # <-- ADD: FASTA -> one-sequence-per-line .txt
├── configs/diffusion/
│   └── config_n.json                     # <-- ADD: denoiser architecture (train-from-scratch only)
└── models/
        ├── autoencoder/encdec.py             # (already present, used by generation.py)
        └── diffusion_models/
                ├── pipeline_ddim.py              # (already present, used by generation.py)
                ├── transformer.py / util.py      # (already present) denoiser + model registry
                └── config_ddim.json              # <-- ADD: DDIM noise scheduler config
```

`train_diffusion.py` imports the **same modules `generation.py` already imports**
(`models.autoencoder.encdec.EncDec`, `models.diffusion_models.pipeline_ddim`,
`models.diffusion_models.util.get_model`), plus `data.dataset_builder` for the
training set. Nothing in the model code changes — this only adds the training
entry point.

---

## 1. Environment & one-time setup

1. Use the same conda environment you use for inference (`environment.yml`). The
   extra packages the training loop needs — `accelerate`, `diffusers`, and
   (optionally) `wandb`/`tensorboard` — are already part of the inference stack.

2. **Configure the cache paths in `util.py`.** `train_diffusion.py` and
   `data/dataset_builder.py` read three module-level constants from `util.py`:

   ```python
   ProGenPath     = "models/autoencoder/decoder/progen_configs"  # tokenizer location, keep as-is
   DATA_CACHE_DIR = "/your/path/.cache/huggingface/datasets"     # <-- change to a writable dir
   XDG_CACHE_HOME = "/your/path/.cache"                          # <-- change to a writable dir
   ```

   Point the two `*_CACHE*` paths at a directory you can write to (the HuggingFace
   `datasets` cache for the tokenized corpus). If your `util.py` does not define
   `XDG_CACHE_HOME` / `DATA_CACHE_DIR` yet, add them.

3. Configure `accelerate` once (single-GPU is fine):

   ```bash
   accelerate config        # or: accelerate config default
   ```

4. Make sure you have the released checkpoints downloaded (same ones used for
   inference): the **auto-encoder** checkpoint (the `--enc_dec_file` you pass to
   `generation.py`) and, for fine-tuning, the **diffusion** checkpoint (the
   `--dm_file` you pass to `generation.py`).

---

## 2. Prepare your fine-tuning data

The training corpus is a **plain text file with one RNA sequence per line**
(letters `A G C U`; `T` is auto-converted to `U`). No headers, no labels.

```
GGGCUAUUAGCUCAGUUGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAAUUCAGCAUAGCCCA
AUGGCGAGCACCUUUGUGGCCAAGCUGAUCGAGAACGGCAAGUACAAGGUG
...
```

If your sequences are in FASTA, convert them with the provided helper (it also
filters by length and replaces `T`->`U`):

```bash
python data/fasta2txt.py \
        --fasta_file my_sequences.fasta \
        --folder data/my_corpus \
        --min 30 --max 769 --replace
# -> writes data/my_corpus/my_sequences_min30max769.txt
```

Guidelines:
- **Length**: keep sequences within the model's max length (the released models
  were trained with `max_seq_len 960` at generation; the ncRNA corpus was
  filtered to `< 769` nt). Filter overly long sequences with `--max`.
- **Alphabet**: only `A/G/C/U` (after `T`->`U`). Drop sequences with `N` or other
  characters beforehand.
- **Size**: fine-tuning works with as few as a few thousand sequences; the script
  automatically holds out 10% as a validation split (`train_test_split`,
  `seed=42`) inside `build_dataset_rna`.

---

## 3. Fine-tune the diffusion model

This is the main command. It loads the released diffusion checkpoint via
`--pretrained_ckpts` and continues training it on your corpus, keeping the
auto-encoder frozen.

```bash
accelerate launch train_diffusion.py \
        --train_data        data/my_corpus/my_sequences_min30max769.txt \
        --output            exps/my_finetune/diffusion-finetuned \
        --encdec_checkpoint <PATH_TO_RELEASED_AUTOENCODER>   `# == generation.py --enc_dec_file` \
        --pretrained_ckpts  <PATH_TO_RELEASED_DIFFUSION>     `# == generation.py --dm_file` \
        --data_type rna \
        --train_batch_size 64 \
        --gradient_accumulation_steps 4 \
        --num_epochs 1 \
        --learning_rate 1e-4 \
        --lr_warmup_steps 50 \
        --save_all_epochs
```

What each important flag does:

| Flag | Meaning |
|------|---------|
| `--train_data` | one or more `.txt` corpora (space-separated for multiple). |
| `--output` | directory where the fine-tuned pipeline is written. |
| `--encdec_checkpoint` | **frozen** auto-encoder; latents are computed with `encdec.get_latent(...)`. Must match the one used at generation. |
| `--pretrained_ckpts` | the diffusion checkpoint to fine-tune. The denoiser weights (and EMA, if present) are loaded from `<dir>/unet`. Omit this flag to instead train from scratch. |
| `--data_type rna` | selects the RNA tokenization path; asserts the auto-encoder is an RNA model. |
| `--num_epochs` | fine-tuning usually needs only **1–5** epochs. |
| `--learning_rate` / `--lr_warmup_steps` | a short warmup (e.g. 50 steps) avoids destroying the pretrained weights at the start. |
| `--train_batch_size` / `--gradient_accumulation_steps` | effective batch = `batch_size × grad_accum × num_gpus`. Lower `train_batch_size` if you hit OOM. |
| `--save_all_epochs` | save a checkpoint per epoch under `output/epoch-{i}/` (otherwise only the final model is saved to `output/`). |
| `--use_ema` | (optional) keep an exponential moving average of the weights; the released models were trained with EMA, and EMA state is reloaded from the pretrained ckpt if present. |

**Reference recipe** (the UTR fine-tune used for the released models — fine-tune
the ncRNA-pretrained diffusion model on a UTR corpus for 1 epoch with a 50-step
warmup):

```bash
accelerate launch train_diffusion.py \
        --train_data data/UTR/Fivespecies_...minNonemax769.txt \
        --output     exps/.../diffusion/ae-2-dm-10-finetune-dm-1-1e-4-warm-up-50 \
        --encdec_checkpoint exps/.../autoencoder/vocab-clean-epoch-2 \
        --pretrained_ckpts  exps/.../diffusion/ae-2-dm-10 \
        --data_type rna --train_batch_size 64 --gradient_accumulation_steps 4 \
        --num_epochs 1 --lr_warmup_steps 50
```

### How the training step works (for reference)

For each batch, `train_diffusion.py`:
1. encodes the sequences to latents with the **frozen** auto-encoder
   (`clean_images = encdec.get_latent(input_ids, attention_mask)`), under
   `torch.no_grad()`;
2. samples a random timestep `t` and adds Gaussian noise with the DDIM scheduler
   (forward diffusion);
3. predicts the noise with the denoiser and minimizes the MSE loss
   (`prediction_type="epsilon"` by default, see `models/diffusion_models/config_ddim.json`);
4. saves a `DDIMPipeline1D` to `--output`, which is exactly the format
   `generation.py --dm_file` expects.

---

## 4. (Optional) Train the diffusion model from scratch

If you (re)trained the auto-encoder or want a brand-new denoiser, drop
`--pretrained_ckpts` and pass the architecture config instead:

```bash
accelerate launch train_diffusion.py \
        --train_data data/my_corpus/my_sequences_min30max769.txt \
        --output     exps/my_run/diffusion-from-scratch \
        --encdec_checkpoint <PATH_TO_AUTOENCODER> \
        --model_config_name_or_path configs/diffusion/config_n.json \
        --data_type rna \
        --train_batch_size 64 --gradient_accumulation_steps 4 \
        --num_epochs 10 \
        --save_model_epochs 5 --save_all_epochs
```

`config_n.json` defines the Transformer denoiser; `in_channels` is overwritten
automatically to match the auto-encoder's latent dimension
(`encdec.qt.config.hidden_size`), so you don't have to edit it by hand.

---

## 5. Generate with your fine-tuned model

Point `generation.py` at the fine-tuned diffusion directory — everything else is
unchanged from the inference README:

```bash
python generation.py \
        --enc_dec_file <PATH_TO_AUTOENCODER> \
        --dm_file      exps/my_finetune/diffusion-finetuned \
        --batch_size 128 --batch_num 10 \
        --eos_token "2" --do_sample --top_p 0.95 --top_k 0 --max_seq_len 960 \
        --superfolder generation/my_finetune --mid_folder unconditional
```

(If you saved per-epoch checkpoints with `--save_all_epochs`, pass the specific
epoch dir, e.g. `--dm_file exps/my_finetune/diffusion-finetuned/epoch-0`.)

For **guided / property-conditioned** generation, fine-tune on the target domain
first (as above), then follow the guided-generation section of the main README,
using your fine-tuned `--dm_file`.

---

## 6. Tips & troubleshooting

- **OOM**: lower `--train_batch_size` and raise `--gradient_accumulation_steps`
  to keep the effective batch size constant; or set `--mixed_precision bf16`
  (Ampere+) / `fp16`.
- **`data_type` assertion error**: `--data_type` must match the auto-encoder you
  load (use `rna`). The script asserts `encdec.data_type == args.data_type`.
- **Resume an interrupted run**: checkpoints are saved every
  `--checkpointing_steps`; relaunch with `--resume_from_checkpoint latest`.
- **Reproducibility**: pass `--seed <n>` (default 42); `--full_deterministic`
  forces deterministic kernels (slower, single-GPU debugging only).
- **Logging**: defaults to Weights & Biases (`--logger wandb`); use
  `--logger tensorboard` for offline logs under `--output/logs`.
- **Validation**: after training, you can quantify the held-out likelihood with
  `test_nll.py` (NLL = ELBO of the diffusion model + VAE reconstruction NLL) if
  you ship that evaluation script; otherwise inspect the training/val loss curves.

---

## Provenance

`train_diffusion.py`, `data/dataset_builder.py`, `data/fasta2txt.py`,
`configs/diffusion/config_n.json`, and `models/diffusion_models/config_ddim.json`
implement the same latent-diffusion training procedure used to produce the
released RNAGenesis checkpoints. They operate on the **existing** RNAGenesis model
code (`models/autoencoder/encdec.py`, `models/diffusion_models/`) that
`generation.py` already relies on — this tutorial only adds the training entry
point and its data utilities, no changes to the model definitions.
