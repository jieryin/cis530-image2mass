# Pairwise probing

`extract.py` runs the frozen zero-shot Qwen3-VL-2B on each pair with the
unchanged pairwise prompt, loaded and preprocessed as in the zero-shot
baseline (`qwen_strong_baseline.model.QwenPredictor`). At the last prompt
token, whose next token is the answer, it saves the hidden state of every
layer and the A/B logits. `probe.py` fits one logistic regression per layer
to predict whether A is heavier:

```text
h_l  = hidden state of layer l at the last prompt token (l = 0..28, 2,048 values)
p_A  = sigmoid(w_l . standardize(h_l) + b_l)
loss = BCE(p_A, y) + (lambda / 2) * ||w_l||^2,   y = 1 if label == "A"
```

Standardization uses train statistics. Predict A when `p_A > 0.5`, otherwise B.
Dev accuracy, then lower dev log-loss, selects `lambda` for each layer and then
the layer. Two rows use the same pairs for comparison:

- `dims_only`: the same probe on the sorted log dimensions of A and B.
- `model_answer`: A when the A logit exceeds the B logit.

## Setup and defaults

Use the shared environment in the [root README](../README.md#environment-setup)
and run from the repository root with records under `amazon_data_10k/`.
Extraction needs a CUDA GPU with BF16; probing runs on CPU.

| Setting | Default | Optional override |
|---|---|---|
| Extracted splits | `dev test train` | `--splits` |
| Pair files | `pairs/<split>_pairs.jsonl`, `pairs/<split>_volume_wrong_pairs.jsonl` | `--pairs-dir` |
| Records | `amazon_data_10k/` | `--root` |
| Features | `probing/features/` | `--features` |
| Results | `probing/output/` | `--output` |
| Saved zero-shot predictions | `qwen_strong_baseline/pairwise/output/` | `--zero-shot-dir` |
| L2 grid | 1e-4, 1e-3, 1e-2, 1e-1, 1 | `L2_GRID` in `probe.py` |

For a small pilot:

```bash
python -m probing.extract --splits dev --limit 8 \
  --features probing/pilot/features --output probing/pilot/output
```

Extract all splits, then fit the probes:

```bash
python -m probing.extract
python -m probing.probe
```

Extraction skips a split whose feature file already holds the same pairs. Each
split reports `zero_shot_agreement`, the share of saved zero-shot predictions
that its A/B logits reproduce; train has no saved predictions and reports
`null`. Features take about 2.4 GB.

## Code layout

```text
probing/
  extract.py   hidden states and A/B logits from frozen zero-shot Qwen
  probe.py     per-layer probes, dims-only probe, model answer
```

## Outputs

| Path | Content |
|---|---|
| `probing/features/<split>.npz` | `pair_ids`, `features` [pairs, 29, 2048] float32, `ab_logits` [pairs, 2]; git-ignored |
| `probing/output/extract_<split>.json` | pair count, feature shape, time, zero-shot agreement, model metadata |
| `probing/output/layers.json` | per layer: L2, train/dev/test accuracy, dev/test volume-wrong accuracy, dev log-loss |
| `probing/output/summary.json` | selected layer, `dims_only`, and `model_answer` in the same format |
| `probing/output/<dev,test>/preds.jsonl` | selected probe predictions with `p_a` |
| `probing/output/<dev,test>/dims_preds.jsonl` | `dims_only` predictions |
| `probing/output/<dev,test>/scores.json`, `subset_scores.json` | selected probe on the full split and the volume-wrong subset |

Prediction files use the `score.py` format:

```bash
python score.py pairs/test_pairs.jsonl probing/output/test/preds.jsonl
```
