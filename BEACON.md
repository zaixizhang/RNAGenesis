# BEACON ncRNA classification with a pretrained RNAGenesis encoder

This end-to-end example loads the stage-one encoder, attaches a prediction head,
and trains the model on user-provided classification data:

```text
RNA sequence -> pretrained encoder with LoRA -> mean readout -> MLP -> 13 logits
```

The task follows the RNAGenesis Methods: sequence-level prediction uses average
readout rather than CLS, followed by an MLP; ncRNA classification has 13 output
classes and uses LoRA fine-tuning. The encoder architecture and vocabulary come
directly from the pretrained checkpoint. The original encoder weights stay
frozen while the LoRA updates and the prediction head receive gradients.

## Configuration and paths

Start from [`configs/finetuning/beacon_ncrna.json`](configs/finetuning/beacon_ncrna.json).
All data paths are empty and must be supplied by the user. There are no bundled
training/validation/test datasets, automatic splits, fixed sample counts, or
automatic test-subset selection.

Set the run-specific fields before training:

| Configuration field | Meaning |
|---|---|
| `task.head.hidden_sizes` | List of MLP hidden widths; `[]` gives a single linear classifier |
| `task.head.activation` | `relu`, `gelu`, or `tanh` between hidden layers |
| `task.head.dropout` | Dropout after each hidden activation |
| `task.lora.rank`, `alpha`, `dropout` | Low-rank adaptation dimensions, scale, and dropout |
| `task.lora.targets` | Projections to adapt; options listed below |
| `training.epochs`, `batch_size` | Total epochs and per-step microbatch size |
| `training.learning_rate`, `weight_decay` | Optimizer settings |
| `training.optimizer` | `adam` or `adamw` |
| `training.scheduler` | `constant`, `linear`, or `cosine` |
| `training.warmup_ratio` | Fraction of optimizer updates used for linear warmup; zero disables warmup |

These fields are explicit run inputs. Replace the `null` values with your run
settings before launch; the trainer checks them before allocating the encoder.

Supported LoRA targets are `query`, `key`, `value`, `attention_output`,
`ffn_gate_up`, and `ffn_down`. Q/K/V updates are applied independently inside the
fused QKV projection. Each update is scaled by `alpha / rank`, with its output
matrix initialized to zero so the encoder initially matches the pretrained model.

General execution settings are also visible in the JSON: random seed, precision,
worker count, gradient clipping, Adam betas/epsilon, activation checkpointing,
and gradient accumulation. They can be set for the intended run. The effective
batch size is `batch_size * gradient_accumulation_steps`, except for the final
partial batch of an epoch. Losses are normalized by the actual number of
sequences, including partial batches. All supplied training examples are used.

## Input format

Supply separate CSV files. Training and validation require `sequence,label`
columns; `id` is optional. Labels are integers from `0` through `12`. Use the
same class-to-index mapping across all files.

```csv
id,sequence,label
example_a,ACGUACGU,0
example_b,GGGAAACCCUUU,12
```

Sequences must already be uppercase RNA (`ACGU` and IUPAC codes
`RYSWKMBDHVN`). The loader pads within each batch and does not apply MLM masking.
It rejects sequences longer than the pretrained encoder's context rather than
silently cropping or truncating classification examples. Prepare those inputs
according to the intended task protocol before loading them.

## Train

The dependencies are the same PyTorch/NumPy environment as stage-one pretraining.
No PEFT, Transformers, or online model download is required. The example runs on
one GPU or CPU; gradient accumulation and activation checkpointing control memory
usage. Use `python`, not `torchrun`, for this task entry point.

After setting the run fields in the JSON, supply the paths either in that file
or with the command below. Command-line paths override the JSON paths.

```bash
python train_beacon_ncrna.py \
    --config configs/finetuning/beacon_ncrna.json \
    --pretrained_encoder /PATH/TO/PRETRAIN_RUN/encoder \
    --train_data /PATH/TO/YOUR/TRAIN.csv \
    --validation_data /PATH/TO/YOUR/VALIDATION.csv \
    --output_dir /PATH/TO/YOUR/FINETUNE_RUN
```

`--pretrained_encoder` accepts either the `encoder/` export from
`train_encoder.py` or a full `checkpoint-XXXXXXXX.pt` produced by that trainer.
The loader restores those weights before adding LoRA and the prediction head.

An optional `--test_data /PATH/TO/YOUR/TEST.csv` evaluates the selected model
after training finishes. It never participates in optimization or checkpoint
selection. Validation accuracy selects `best.pt`; the earliest epoch wins ties.
The script trains for the configured number of epochs without early stopping.

The output directory contains:

```text
run_config.json         # resolved run settings and paths
metrics.jsonl           # per-epoch training loss and validation metrics
best.pt                 # self-contained model selected by validation accuracy
last.pt                 # model, optimizer and RNG state for epoch-boundary resume
test_metrics.json       # only when a test path is supplied
test_predictions.csv    # labels, predictions and all 13 class probabilities
```

The model exports include the frozen encoder, LoRA parameters, classification
head, and architecture, so prediction does not need the original pretraining
files. Full checkpoints are correspondingly larger than adapter-only exports.

For a planned interruption, `--stop_after_epochs N` saves and stops after the
specified total epoch count while retaining the original learning-rate schedule.
Resume with the same run configuration and paths:

```bash
python train_beacon_ncrna.py \
    --config /PATH/TO/YOUR/FILLED_RUN_CONFIG.json \
    --resume /PATH/TO/YOUR/FINETUNE_RUN/last.pt
```

Keep `best.pt` and `last.pt` in the run output directory. Training/validation
file paths, sizes and modification times must match on resume. The DataLoader
worker count may change. GPU floating-point operations may not be bitwise
deterministic. Resume uses the same device type (CPU or CUDA) as the saved run.

## Evaluate or predict separately

Evaluate a labeled CSV using the saved classifier:

```bash
python train_beacon_ncrna.py --mode evaluate \
    --checkpoint /PATH/TO/YOUR/FINETUNE_RUN/best.pt \
    --data /PATH/TO/YOUR/EVALUATION.csv \
    --output_dir /PATH/TO/YOUR/EVALUATION_OUTPUT
```

For unlabeled prediction, use `--mode predict` and a CSV containing `sequence`
and optionally `id`. Both modes write `predictions.csv`; evaluation also reports
cross-entropy, accuracy, macro-F1 over all 13 classes, and the confusion matrix
(rows are ground truth, columns are predictions). Accuracy is the primary ncRNA
classification metric. `--eval_batch_size` controls inference batch size.
CUDA inference uses the checkpoint's saved precision; CPU inference uses fp32
and PyTorch SDPA. `--eval_precision` can explicitly select fp32 or bf16.

## Correctness checks

```bash
python -m unittest discover -s tests -p test_beacon_ncrna.py -v
```

The tests use small synthetic inputs. They check loading of pretrained weights,
zero-initialized LoRA equivalence, frozen base weights, adapter/head updates,
padding-aware pooling, configuration validation, training/validation/test flow,
partial-batch gradient accumulation, exact CPU resume, and standalone prediction.
