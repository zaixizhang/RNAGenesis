# Stage-one RNA encoder pretraining

`train_encoder.py` trains the RNAGenesis sequence encoder from scratch using
masked RNA modeling. It loads a prepared corpus supplied by the user. The
encoder, hybrid N-gram embeddings, and MLM head are optimized together.

## Install

Use Python 3.8+ with PyTorch 2.1+ and NumPy. The repository's PyTorch 2.1.2
environment supports this entry point. No Hugging Face download or pretrained
checkpoint is needed. Multi-GPU training uses `torchrun` and PyTorch DDP on Linux
with CUDA/NCCL. The unit tests run on CPU.

For FlashAttention-2 on A100 GPUs, install a `flash-attn` version compatible
with your PyTorch/CUDA environment, for example on a Linux CUDA development host:

```bash
pip install 'flash-attn>=2.3,<3' --no-build-isolation
```

`attention_backend: "auto"` uses FlashAttention-2 for CUDA bf16 inputs when the
package is installed, and otherwise uses PyTorch SDPA. Set the backend to
`"flash_attention_2"` to require that kernel, or `"sdpa"` to explicitly select
PyTorch attention. Padded positions are removed and restored around the
FlashAttention-2 call.

## Prepared data

Set `--train_data` to **your prepared dataset location**. There is no built-in
dataset path. The input is an uncompressed ASCII `.txt` file containing one
uppercase RNA sequence per line, without headers or empty lines. Accepted bases
are `ACGU` and IUPAC codes `RYSWKMBDHVN`. Supply a separate prepared validation
file with `--validation_data` if desired; the trainer does not split the corpus.

```text
ACGUACGUACGU
GGGAAACCCUUU
```

The loader caches a compact byte-offset index under the output directory to
support shuffled random access without holding the corpus in RAM. It does not
change or copy the sequences. RNAs longer than 1,024 nt are randomly cropped to
a contiguous 1,024-nt window each epoch. Shorter RNAs are used in full and padded
within each batch. Crops and masks are determined by seed, epoch, and row index,
independently of DataLoader worker count.

## Architecture and objective

- Single-nucleotide embeddings of width 1,280, including IUPAC and special tokens.
- Six parallel stride-one, same-padded Conv1D layers with kernels
  `3, 5, 7, 9, 11, 13`; their outputs are summed with the nucleotide embeddings.
- 32 bidirectional Transformer blocks, 20 attention heads, RoPE, pre-LayerNorm,
  SwiGLU FFNs of intermediate width 3,413, and residual connections.
- FlashAttention-2 with a PyTorch scaled dot-product attention fallback.
- Hidden and attention dropout are both 0.1. The MLM head is an untied linear
  projection from the final token embeddings to the 20-token vocabulary.

Each nucleotide is independently selected with probability `0.30`. By default,
every selected nucleotide is replaced with `<mask>`; at least one position is
selected per sequence. Padding is excluded. **Masking happens before both the
convolutional embeddings and the Transformer**, so the original selected bases
cannot leak through the CNN. The objective is cross-entropy on selected positions
only, averaged across all selected tokens in the global accumulated batch.

`mask_replace_probability` and `random_replace_probability` control replacement
of selected positions; any remaining probability leaves the selected input
unchanged. Random replacements are drawn from `A/C/G/U`. For example, `0.8` and
`0.1` configure an 80/10/10 replacement scheme. These are separate from the 30%
position-selection probability.

## Training settings

The configuration is [`configs/pretraining/encoder.json`](configs/pretraining/encoder.json).

| Setting | Value |
|---|---|
| Optimizer | AdamW, betas `(0.9, 0.999)`, epsilon `1e-8` |
| Weight decay | 0.01 on weight matrices; no decay on biases or LayerNorm parameters |
| Global batch size | 512 sequences |
| Context length | 1,024 nucleotides; no extra start/end tokens |
| Optimization steps | 500,000 |
| Warmup | Linear: `1e-5` on update 1 to `1e-4` on update 10,000 |
| Decay | Cosine: `1e-4` to `1e-6` on update 500,000 |
| Precision | bf16 autocast; fp32 parameters and optimizer state |
| Gradient clipping | Global norm 1.0 |
| Activation checkpointing | Enabled |

Gradient accumulation is computed automatically:

```text
accumulation_steps = 512 / (micro_batch_size * world_size)
```

The denominator must divide 512 exactly. Incomplete global batches at each epoch
boundary are dropped. On 16 GPUs, the default microbatch of 4 gives 8 accumulation
steps and exactly 512 sequences per optimizer update. Adjust `--micro_batch_size`
to fit GPU memory while retaining the global batch size.

## Launch

Replace the uppercase placeholders with your own paths. On one machine with 16
GPUs:

```bash
torchrun --standalone --nproc_per_node=16 train_encoder.py \
    --config configs/pretraining/encoder.json \
    --train_data /PATH/TO/YOUR/PREPARED_RNA.txt \
    --output_dir /PATH/TO/YOUR/PRETRAIN_RUN
```

For two machines with 8 GPUs each, run the following on both machines, setting
`NODE_RANK` to `0` or `1` and `MASTER_ADDR` to the first machine's reachable address:

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port=29500 train_encoder.py \
    --config configs/pretraining/encoder.json \
    --train_data /SHARED/PATH/TO/PREPARED_RNA.txt \
    --output_dir /SHARED/PATH/TO/PRETRAIN_RUN
```

Data and output paths must refer to the same shared files on all machines.
For a single GPU, use `python train_encoder.py` with the same arguments. To run
a small CPU experiment, copy the JSON config, reduce the architecture and batch
sizes, set `precision` to `fp32`, and pass `--device cpu`.

Optional flags:

```text
--validation_data /PATH/TO/PREPARED_VALIDATION.txt
--micro_batch_size 2
--num_workers 4
--stop_after_steps 10000
```

`--stop_after_steps` stops at the specified total update count, saves state, and
exports the encoder without changing the full learning-rate schedule. Validation
uses fixed crops/masks, reports token-weighted cross-entropy and masked-position
accuracy, and evaluates up to `eval_batches` local batches per rank. Validation
drops up to `world_size - 1` tail sequences to avoid duplicate distributed samples.

## Checkpoints and resume

The output directory contains:

```text
run_config.json                 # resolved config, dataset identity, world size
metrics.jsonl                   # training and optional validation metrics
data-index/                     # cached line offsets
checkpoint-00005000.pt          # full training state
encoder/config.json             # exported model architecture
encoder/model.pt                # exported encoder and MLM head weights
encoder/vocab.json              # fixed token-to-ID mapping
```

Full checkpoints are written atomically every `save_every` updates and at a
requested stop. The most recent three full checkpoints are retained (configure
`keep_last_checkpoints` to change this). They include optimizer state, per-rank RNG state, epoch, and the
next data position. A nonempty output directory requires `--resume`.

```bash
torchrun --standalone --nproc_per_node=16 train_encoder.py \
    --config configs/pretraining/encoder.json \
    --train_data /PATH/TO/YOUR/PREPARED_RNA.txt \
    --output_dir /PATH/TO/YOUR/PRETRAIN_RUN \
    --resume /PATH/TO/YOUR/PRETRAIN_RUN/checkpoint-00005000.pt
```

Resume requires the same model/training settings, world size, microbatch size,
and corpus identity (absolute path, size, and modification time). DataLoader
worker count may change. Pass the same validation path when resuming a run that
uses validation. For continuation after a planned stop, omit `--stop_after_steps`
or increase it. GPU floating-point nondeterminism can still affect bitwise results.

## Load the trained encoder

```python
import torch
from pretraining.data import RNATokenizer
from pretraining.model import RNAEncoderForMaskedLM

tokenizer = RNATokenizer()
model = RNAEncoderForMaskedLM.from_pretrained("/PATH/TO/PRETRAIN_RUN/encoder")
model.eval()
input_ids = torch.tensor([tokenizer.encode("ACGUACGU")])
with torch.no_grad():
    result = model(input_ids)
embeddings = result["last_hidden_state"]  # [batch, length, 1280]
logits = result["logits"]               # [batch, length, 20]
```

These exports use the `pretraining.model` loader and token vocabulary shown
above. They are separate from the existing Hugging Face encoder format and from
the `EncDec` checkpoints consumed by `train_diffusion.py`. Stage two trains the
Q-Former and causal decoder using the stage-one encoder's token embeddings.

## Tests

```bash
python -m unittest discover -s tests -p test_pretraining.py -v
```

Tests cover masking and padding, epoch cropping, padding-invariant embeddings,
token-normalized gradient accumulation, checkpointed backward passes, model
export/reload, the learning-rate endpoints, and uninterrupted versus resumed
CPU training (including validation and an epoch boundary).
