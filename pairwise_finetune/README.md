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

Use Python 3.11 and a separate environment for this workflow. The model and dataset are downloaded separately; installing the packages does not download either. Run the commands below from the repository root.

### Apple Silicon / MPS

```bash
python3.11 -m venv .venv-pairwise
source .venv-pairwise/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0
python -m pip install -r pairwise_finetune/requirements.txt
python -c "import torch; print(torch.__version__); print('MPS available:', torch.backends.mps.is_available())"
```

Use `--device mps` in the commands below. MPS must be available; CPU execution is not a substitute for a successful GPU training pilot. This workflow uses ordinary LoRA, with no bitsandbytes or 4-bit quantization dependency.

The default `--dtype auto` uses float32 on MPS for compatibility. This consumes more memory and can be slower than BF16. An explicit `--dtype bfloat16` is an optional, separately validated configuration; completing a small tensor operation does not guarantee that every model forward/backward operation is supported. Use a new pilot output directory when comparing dtypes.

### NVIDIA / CUDA

On Linux, create and activate a Python 3.11 environment as above. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv-pairwise
.\.venv-pairwise\Scripts\Activate.ps1
```

Then install the CUDA wheels before the remaining requirements:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r pairwise_finetune/requirements.txt
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Use `--device cuda` instead of `--device mps` in every training/evaluation command. The multiline commands below use Bash continuation (`\`); on PowerShell, enter them on one line or replace continuation characters with PowerShell backticks. The pinned PyTorch wheel combinations follow the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v2100).

On CUDA, `--dtype auto` selects BF16 when the GPU supports it and float32 otherwise. The RTX 4090 supports the BF16 path.

The common dependency pins match the existing Qwen workflow, adding PEFT 0.17.1. The [PEFT package requirements](https://pypi.org/project/peft/0.17.1/) are compatible with these version constraints. A successful installation is not evidence that the full model has completed a forward/backward pass on either device; run the pilot below.

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
  --output pairwise_finetune/output/pilot-mps \
  --device mps \
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
  --device mps \
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
  --device mps \
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

On an Apple Silicon Mac, explicitly exercise the tiny model on MPS:

```bash
PAIRWISE_TEST_DEVICE=mps python -m unittest discover -s tests -p test_pairwise_finetune_model.py
```

During implementation, the full 116-test suite passed on CPU and the tiny
two-image model's forward/backward and adapter roundtrip passed on this M4 Pro's
MPS device. The real pinned tokenizer was also checked: A and B are distinct
single continuation tokens after the assistant prefix. These are software
checks, not pretrained 2B training results or final test accuracy.

## Hardware and run planning

For this project's 2B model and 8,000 training pairs, **M4 Pro with 48GB unified memory is a reasonable LoRA pilot target**. This is a feasibility assessment, not a measured full-training memory or runtime result. Mac system memory is shared with the OS and other applications. The existing repository baseline is CUDA-specific, so its installation and timings do not establish MPS compatibility for training.

An available **RTX 4090 is recommended for repeated formal experiments**, particularly to keep the CUDA software path consistent with the existing baseline. NVIDIA specifies [24GB of GPU memory](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/). No new GPU purchase is required merely by the number of pairs. Full-parameter Adam training has a different memory budget and is outside this LoRA workflow.

Measure warmed-up training throughput and memory during a representative pilot before estimating total time. CUDA logs peak allocated memory; MPS logs current tensor and driver allocations, not a peak measurement. For a microbatch of one pair, a rough epoch estimate is `8,000 × measured seconds per pair`, plus dev evaluation and checkpoint writes. If timing optimizer updates instead, account for the accumulation count and any shorter final update. Model download/loading is additional. The zero-shot baseline's inference time and memory are not training estimates.

### 本地运行提示

- 这台 M4 Pro / 48GB 可以先跑 LoRA 小实验；当前还没有完整训练的实测速度，不能保证固定几小时完成。
- 先安装 Mac 版本依赖，执行 `--dry-run --check-images`，再跑单独的 pilot，确认梯度更新和保存/读取 checkpoint 都正常。Mac 默认 float32；它比 BF16 更占内存，实际速度要测。
- 正式训练使用新的输出目录，去掉 `--max-steps`、`--limit-train` 和 `--limit-dev`。MSE 只监督第一个 A/B 答案；解释文字不进入输入，也不算 loss。
- 如果已有 4090，正式多组实验优先用它：安装 CUDA 版本 PyTorch，把命令中的 `--device mps` 改成 `--device cuda`。不需要因为 8,000 对数据就专门购买显卡。
- 模型选择只看 dev；配置确定后再跑完整 test 和 volume-wrong 子集，保留结果与实际耗时。
