# Qwen LoRA fine-tuning

Pairwise and pointwise share one trainer, model loader, data pipeline, and
checkpoint implementation. Both import their unchanged prompts from
[`prompts/`](../prompts/README.md).

**Pairwise training is implemented. Pointwise training is a scaffold:** its
numeric-text output is agreed, but its loss and checkpoint-selection rule are
undecided. Both functions raise `NotImplementedError` in
[`pointwise_finetune/task.py`](pointwise_finetune/task.py). No MSE, cross-entropy,
regression head, or weight-error metric has been chosen for pointwise.

## Setup and defaults

Use Python 3.11 and install the repository's single dependency file:

```bash
python -m pip install -r requirements.txt
```

Run commands from the repository root. The saved pair files are under `pairs/`;
extract the dataset under `amazon_data_10k/`. See the parent
[README](../README.md) for dataset and platform setup.

Start pairwise training with all defaults:

```bash
python -m finetune.train
```

| Setting | Default | Optional override |
|---|---|---|
| Task | Pairwise | `--task pointwise` (loss/selection pending) |
| Train / dev pairs | `pairs/train_pairs.jsonl`, `pairs/dev_pairs.jsonl` | `--train`, `--dev` |
| Records | `amazon_data_10k/` | `--root` |
| Output | `finetune/output/<task>/run1` | `--output` |
| Device / dtype | Auto; CUDA BF16 when supported, otherwise FP32 | `--device`, `--dtype` |
| Epochs / gradient accumulation | 3 / 8 | `--epochs`, `--grad-accum` |
| Learning rate / weight decay | 0.0001 / 0.01 | `--lr`, `--weight-decay` |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 | `--rank`, `--alpha`, `--dropout` |
| Maximum image pixels | 262144 | `--max-pixels` |
| Seed | 42 | `--seed` |
| Gradient checkpointing | Enabled | `--no-checkpointing` |
| Save interval | 100 optimizer updates | `--save-every` |

Only specify overrides when needed. `python -m finetune.train --help` lists all
options. A new run needs a fresh output directory.

Validate metadata without loading model weights:

```bash
python -m finetune.train --dry-run
python -m finetune.train --task pointwise --dry-run
```

Add `--check-images` to also open the records. Pointwise dry-run works before
its loss is implemented; actual pointwise training reaches the placeholder
after model and run setup.

For a small pilot:

```bash
python -m finetune.train --output finetune/output/pairwise/pilot \
  --limit-train 128 --limit-dev 32 --max-steps 20
```

`--limit-train` counts pairs for pairwise and unique objects for pointwise.
`--limit-dev` counts pairs; `--max-steps` counts optimizer updates. Without
limits, one epoch visits every training example once. Accumulation averages
8 examples per update by default, including the correct smaller final group.

## Code layout

```text
finetune/
  __init__.py                 task selection
  data.py                     pair validation, record loading, object extraction
  model.py                    Qwen, LoRA, device selection, input processing
  checkpoint.py               adapter saves and resumable training state
  train.py                    shared training loop
  evaluate.py                 shared evaluation entry point
  pairwise_finetune/task.py    pairwise prompts, loss, predictions, metrics
  pointwise_finetune/task.py   numeric prediction; loss/selection placeholders
  tests/
```

Data is validated once in `read_pairs`; downstream code uses those rows.
Opening a record checks that its ID, dimensions, and weight match the metadata.
Training also rejects train/dev object overlap. JSON writers, hashing, numeric
parsing, and prompt builders reuse existing repository helpers.

The shared loop records `examples`, `loss`, and `examples_seen` for either task;
`count_unit` in the run config identifies pairs or objects.

## Pairwise behavior

Qwen receives two images and their dimensions. Gold weights and labels stay
outside the prompt. The label supplies the training target:

```text
p_A = softmax([logit_A, logit_B])[0]
target = 1 if label == "A" else 0
loss = mean((p_A - target) ** 2)
```

Only the first answer position contributes to this binary probability MSE.
An optional answer string must begin with A/B matching the label; trailing
explanations are ignored. Predict A when `p_A > 0.5`, otherwise B. Dev accuracy,
then lower dev MSE, selects the best pairwise adapter.

LoRA updates language-attention q/k/v/o projections while the base model and
vision encoder stay frozen. Pointwise inference generates a numeric weight for
each unique object using the existing baseline grammar, then compares weights
to produce pair predictions. Its training objective remains undecided.

## Resume and evaluate

```bash
python -m finetune.train --resume finetune/output/pairwise/run1
```

Resume loads the run's latest checkpoint and its original arguments. It restores
the adapter, processor, optimizer, scheduler, RNG state, and epoch/example cursor.
Results stay in the directory passed to `--resume`, even if that run was moved
or copied from its original location.
Checkpoints are saved initially, at the save interval, before dev, and after dev.
Only two completed checkpoints are retained; the latest pointer is updated
atomically. Unsaved work is replayed, and an interrupted dev pass is rerun without
repeating completed training updates.

Data, prompt, model, preprocessing, device/dtype, and training settings must
match. Source-file hashes and package-version equality are no longer required.
The data root and save interval may be overridden when resuming. Completed runs
add no updates. Use one process per output directory.

`best/` and `last/` contain inference adapters, not optimizer state. Evaluate one:

```bash
python -m finetune.evaluate \
  --checkpoint finetune/output/pairwise/run1/best \
  --pairs pairs/test_pairs.jsonl \
  --output finetune/output/pairwise/run1/test
```

The task and dtype come from checkpoint metadata by default. Add
`--subset pairs/test_volume_wrong_pairs.jsonl` to score a slice using the same
predictions. Select settings on train/dev before evaluating test.

## Checks

```bash
python -m unittest discover -s tests
python -m unittest discover -s finetune/tests
```

Tests cover prompts, validation, gradients, partial accumulation, subset scoring,
and interrupted training recovery. Recovery tests compare stochastic toy runs
including weights, optimizer/scheduler state, RNG state, and history. They do
not establish pretrained Qwen training quality; model integration tests require
the full dependency stack.
