# Pairwise fine-tuning with answer-only MSE

Fine-tune `Qwen/Qwen3-VL-2B-Instruct` with LoRA on the existing image2mass pair files. Each example contains two RGB images and their physical dimensions. The supervised task is to choose the heavier object, A or B. Gold weights, labels, object IDs, filenames, and explanations are not model inputs.

This is a separate training workflow from the frozen zero-shot baseline in `qwen_strong_baseline/`. It uses the same base model and saved splits. The saved training set has 8,000 pairs; dev and test each have 1,000. The optional volume-wrong subset is an additional evaluation slice, not a training split.

## Objective and answer format

Let `z_A` and `z_B` be the logits for the A and B answer tokens at the first assistant answer position, after the complete user prompt and its two images. Define:

```text
p_A = softmax([z_A, z_B])[0]
y   = 1 if label == "A" else 0
loss = mean((p_A - y) ** 2)
```

This is binary probability MSE (the binary Brier score), conditioned on the two allowed labels. It does not regress token IDs or physical weight differences. Conditional A/B normalization is part of the method: `p_A` is not the unrestricted vocabulary probability of the token A.

Only this first answer decision contributes to the loss. There is no cross-entropy, rationale-token loss, or EOS loss. The trainer does not append a gold answer or a gold explanation to the model context. Evaluation uses the same prompt and the same A/B probability calculation, so it does not depend on unconstrained text generation or parsing a generated explanation.

Predict A when `p_A > 0.5`; otherwise predict B, including an exact tie.

The existing `label` field supplies the target. If an optional `answer` string is supplied, it must begin with a standalone A or B that agrees with the label. A following explanation is ignored. For example, `A. Its dimensions suggest a larger object.` supplies only A; `After considering the images, the answer is A.` is rejected. This keeps the answer position the same during training and evaluation and prevents an explanation from revealing the target in the input.

## Environment

Use Python 3.11 in an isolated environment. Install PyTorch 2.10.0 and
torchvision 0.25.0 for your training environment, then install the pinned
project dependencies:

```bash
python -m pip install -r pairwise_finetune/requirements.txt
```

The model and dataset are downloaded separately. Run the commands below from
the repository root. `--device auto` selects an available accelerator, falling
back to CPU; check the selected device in the run metadata before a full run.

## Data validation

Extract the project's [10k dataset archive](https://github.com/jieryin/cis530-image2mass/releases/download/v1.0.0/amazon_data_10k.tar.gz) so that each pair's `path_a` and `path_b` resolves under `amazon_data_10k/`. If the records are elsewhere, replace every `--root amazon_data_10k` with the actual record root. Retain the saved pair files and split assignments.

Check metadata without loading model weights:

```bash
python -m pairwise_finetune.train \
  --train pairs/train_pairs.jsonl \
  --dev pairs/dev_pairs.jsonl \
  --root amazon_data_10k \
  --output pairwise_finetune/output/metadata-check \
  --dry-run
```

Add `--check-images` to also decode and validate the image records. Metadata-only validation does not prove that records have been downloaded or that a model can run. A dry run is not a training result.

## Pilot and full training

Start with a separate small pilot:

```bash
python -m pairwise_finetune.train \
  --train pairs/train_pairs.jsonl \
  --dev pairs/dev_pairs.jsonl \
  --root amazon_data_10k \
  --output pairwise_finetune/output/pilot \
  --device auto \
  --epochs 3 \
  --grad-accum 8 \
  --rank 8 \
  --max-pixels 262144 \
  --max-steps 20 \
  --limit-train 128 \
  --limit-dev 32
```

Confirm that forward, backward, and optimizer updates finish, the loss stays finite, adapter parameters receive gradients, and the saved checkpoint can be evaluated. Pilot dev scores cover only the limited dev sample and must not be reported as full-split results.

`--max-steps 20` limits optimizer updates, not individual pair passes. With `--grad-accum 8`, each full update accumulates eight pair losses.

For a full run, use a new output directory and omit all pilot limits:

```bash
python -m pairwise_finetune.train \
  --train pairs/train_pairs.jsonl \
  --dev pairs/dev_pairs.jsonl \
  --root amazon_data_10k \
  --output pairwise_finetune/output/run1 \
  --device auto \
  --epochs 3 \
  --grad-accum 8 \
  --rank 8 \
  --max-pixels 262144
```

The microbatch size is always one pair (two images). Gradient accumulation combines multiple pairs into each optimizer update; it does not remove their forward/backward work. Gradient checkpointing is enabled by default to reduce activation memory. LoRA updates the selected language-model projections while the base model and vision encoder remain frozen.

`--max-pixels` bounds image preprocessing and can change the visual information presented to the model. It is a training configuration choice, not necessarily the same preprocessing as the zero-shot baseline's unmodified processor defaults. Select this setting and other hyperparameters using train/dev only, record them with the experiment, and keep them fixed for test.

The run saves `run.json`, `history.jsonl`, `best/` and `last/` PEFT adapters, and per-epoch dev predictions and scores. `best/` is selected by higher dev accuracy, breaking accuracy ties with lower dev MSE. These adapters require the original base model for inference; they are not standalone copies of the full Qwen model. The initial workflow does not resume optimizer state. Each run requires a new output directory, and the pilot and full experiment outputs must stay separate.

## Final test and volume-wrong evaluation

After choosing the configuration and checkpoint on dev, evaluate the best checkpoint on the complete test split:

```bash
python -m pairwise_finetune.evaluate \
  --checkpoint pairwise_finetune/output/run1/best \
  --pairs pairs/test_pairs.jsonl \
  --root amazon_data_10k \
  --output pairwise_finetune/output/run1/test \
  --device auto \
  --subset pairs/test_volume_wrong_pairs.jsonl
```

Omit `--subset` if no subset file is available. When supplied, it selects a slice of predictions from the same complete evaluation; it does not train or change the model. The strict project scorer in `score.py` supplies accuracy, macro-F1, per-class metrics, confusion matrix, and coverage. MSE is an optimization/diagnostic measure; pairwise accuracy remains the primary comparison metric.

Evaluation reloads the checkpoint's saved processor. Its default `--dtype auto`
preserves the training dtype recorded in that checkpoint; an explicit dtype
override is recorded in the evaluation metadata. `preds.jsonl` and `scores.json`
cover the full input file; `subset_scores.json` covers only the requested slice.

Do not tune on test or on the test volume-wrong slice. Report the subset size beside its accuracy, because it is deliberately selected for a baseline's mistakes and is not representative of the complete test distribution. Keep the full test predictions for comparison with existing baselines.

## Implementation checks

Run the existing tests and the new data, objective, and workflow checks with:

```bash
python -m unittest discover -s tests
```

With the training dependencies installed, the suite includes a tiny randomly
initialized Qwen3-VL: two images, checkpointed backward, frozen base/vision
parameters, LoRA updates, and adapter save/reload. A separate workflow check
verifies partial gradient accumulation groups and exact subset scoring. Without
the training dependencies, tensor/model tests are skipped rather than claiming
a model pass.

The full 116-test suite passed during implementation. The pinned tokenizer was
also checked: A and B are distinct single continuation tokens after the
assistant prefix. These are software checks, not pretrained 2B training results
or final test accuracy.
