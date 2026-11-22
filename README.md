# CIS530: Object Mass Estimation

## Environment setup

Python 3.11 is the supported project version. Create and activate a virtual environment, then install the required dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pairwise mass comparison task

Project documentation, milestone materials, and code for a reproducible 10,000-record subset of the Amazon image2mass dataset.

| Split | Original source | Records |
|---|---|---:|
| train | train_data | 8,000 |
| dev | val_data | 1,000 |
| test | test_data | 1,000 |

Inputs are the RGB image, physical dimensions, and supplied thickness mask. The target is the listed weight. Sampling uses seed 42 and preserves the original split assignments and record bytes.

The generated pair files are stored in the `pairs/` directory:

- `pairs/train_pairs.jsonl`
- `pairs/dev_pairs.jsonl`
- `pairs/test_pairs.jsonl`


## Downloads

- Project 10k subset: [amazon_data_10k.tar.gz](https://github.com/jieryin/cis530-image2mass/releases/download/v1.0.0/amazon_data_10k.tar.gz).
- Full original download: [amazon_data.zip on Google Drive](https://drive.google.com/file/d/1EnL_sSfZBgxUiL63W6UHWIsp00-rID_A/view?usp=drive_link).

The 10k archive is distributed as a Release asset rather than committed to Git. The full original data stays on Google Drive.

## Files

- `data.md`: dataset description and input fields.
- `CIS530_Milestone1.pdf`: milestone submission.
- `amazon_data_10k/prepare_10k.py`: extraction code.
- `amazon_data_10k/manifest.csv`: selected records and checksums.
- `amazon_data_10k/sampling.json`: sampling parameters and statistics.
- `amazon_data_10k/README.md`: extraction instructions.
- [`simple_baseline/simple-baseline.md`](simple_baseline/simple-baseline.md): simple baseline methods, usage, and results.
- [`openai_strong_baseline/strong-baseline.md`](openai_strong_baseline/strong-baseline.md): OpenAI pairwise and pointwise methods, usage, and results.
- [`qwen_strong_baseline/strong-baseline.md`](qwen_strong_baseline/strong-baseline.md): local Qwen3-VL pairwise and pointwise methods, installation, resumable inference, and results.
- `SOURCE_DATA_TERMS.txt`: terms supplied with the original dataset.

## Reproduce the subset

Place the original source download in `amazon_data/`, then run with Python 3.11 or later:

```sh
python amazon_data_10k/prepare_10k.py --source amazon_data --output amazon_data_10k_reproduced --seed 42
```

The script uses only the Python standard library. Add `--dry-run` to preview the selection. Existing outputs are never overwritten.

## Run the simple baseline

The simple baseline supports two methods:

- `majority`: predicts the most frequent A/B label in the training set for every test example.
- `volume`: predicts that the object with the larger bounding-box volume is heavier.

Run the training-majority baseline from the repository root:

```bash
python simple_baseline/simple-baseline.py \
  --method majority \
  --train pairs/train_pairs.jsonl \
  --test pairs/test_pairs.jsonl \
  --output simple_baseline/simple_preds.jsonl
```

To run the volume baseline:

```bash
python simple_baseline/simple-baseline.py \
  --method volume \
  --train pairs/train_pairs.jsonl \
  --test pairs/test_pairs.jsonl \
  --output simple_baseline/volume_preds.jsonl
```

## Run the strong baseline

The OpenAI baseline uses `gpt-6.1-sol` with RGB images and dimensions. Pairwise mode directly predicts A or B; pointwise mode estimates each object's weight in pounds, then compares the estimates. Ground-truth weights and labels are used only for evaluation.

Extract the dataset records under `amazon_data_10k/`. For new API predictions, create `openai_strong_baseline/.env` from [`.env.example`](openai_strong_baseline/.env.example) and fill in `OPENAI_API_KEY`. Then, from the repository root:

```bash
python -m pip install openai==3.51.0
python openai_strong_baseline/runner.py --mode both --pairs pairs/dev_pairs.jsonl
```

Replace `dev_pairs.jsonl` with `train_pairs.jsonl` or `test_pairs.jsonl` to evaluate another split. Add `--dry-run` to check inputs without API calls. Compatible cached predictions are reused; completed runs automatically save predictions and JSON/text scores under `openai_strong_baseline/<mode>/output/<split>/`.

See [strong-baseline.md](openai_strong_baseline/strong-baseline.md) for prompts, settings, output formats, and commands for all three splits.

## Run the local Qwen strong baseline

The Qwen baseline uses `Qwen/Qwen3-VL-2B-Instruct` with the same zero-shot prompts, RGB images, and physical dimensions as the OpenAI baseline. Pairwise mode directly predicts A/B; pointwise mode estimates each unique object's weight once and compares the estimates, selecting B for ties. Model inputs exclude gold weights, labels, filenames, IDs, and thickness masks. No API key is required.

Use an isolated Python 3.11 environment with a CUDA GPU supporting BF16. Install the pinned CUDA PyTorch wheels first, followed by the Qwen dependencies:

```bash
python -m pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r qwen_strong_baseline/requirements.txt
python qwen_strong_baseline/runner.py --mode both --pairs pairs/dev_pairs.jsonl --dry-run
python qwen_strong_baseline/runner.py --mode both --pairs pairs/dev_pairs.jsonl --max-new-pairs 20 --max-new-objects 20
```

These commands expect dataset record files under `amazon_data_10k/`. If the dataset is elsewhere, pass `--root PATH` to every command. For the local D: checkout, use `--root ../amazon_data_10k`; the [Qwen method documentation](qwen_strong_baseline/strong-baseline.md#usage) gives complete Windows commands and cache paths.

The model revision and configuration are pinned: BF16, batch size 1, greedy decoding with at most 32 new tokens, and the standard model processor. Pointwise numeric output is constrained during token generation; the exact prompts are preserved. A dry-run validates inputs without loading a model or writing outputs. Pilot limits count only new inferences; remove the limits after checking the pilot to complete dev, then run `pairs/test_pairs.jsonl` with the frozen configuration. Compatible caches resume automatically. Complete predictions and strict scorer reports are saved under `qwen_strong_baseline/<mode>/output/<split>/`; incomplete pilot outputs do not receive full evaluation scores.

See [Qwen strong-baseline.md](qwen_strong_baseline/strong-baseline.md) for exact prompts, installation, input validation, generation and failure policies, development diagnostics, and verified dev/test measurements.

## Pairwise LoRA fine-tuning (Extension 1)

The new [`pairwise_finetune/`](pairwise_finetune/README.md) workflow fine-tunes
`Qwen/Qwen3-VL-2B-Instruct` with language-attention LoRA. It uses answer-only
MSE between the A/B-normalized probability of A and the binary gold label.
Only the first answer decision is supervised; prompts and trailing explanations
do not contribute token losses. Mac MPS and NVIDIA CUDA have separate setup
instructions, and the existing zero-shot baselines remain available.

Validate the saved train/dev pair metadata without downloading model weights:

```bash
python -m pairwise_finetune.train --dry-run
```

See the [fine-tuning guide](pairwise_finetune/README.md) for installation,
pilot/full runs, and evaluation on both all test pairs and
`pairs/test_volume_wrong_pairs.jsonl`. No fine-tuned accuracy results are claimed
until a full training and evaluation run has been completed.

## Run the evaluation script

Score the majority-baseline predictions from the repository root:

```bash
python score.py \
  pairs/test_pairs.jsonl \
  simple_baseline/simple_preds.jsonl
```

The primary evaluation metric is accuracy. The scorer also reports macro-F1, per-class precision, recall and F1, a confusion matrix, and prediction coverage.

For the complete evaluation protocol and file-format requirements, see [`scoring.md`](scoring.md).

The saved strong-baseline predictions can also be scored without an API key:

```bash
python score.py pairs/test_pairs.jsonl openai_strong_baseline/pairwise/output/test/preds.jsonl
python score.py pairs/test_pairs.jsonl openai_strong_baseline/pointwise/output/test/preds.jsonl
```

## Baseline results

Accuracy on the same 1,000 dev pairs and 1,000 test pairs:

| Baseline | Dev accuracy | Test accuracy |
|---|---:|---:|
| Majority class | 50.4% | 49.3% |
| Volume rule | 82.1% | 83.2% |
| OpenAI pairwise | 91.8% | 91.1% |
| OpenAI pointwise | 92.4% | 92.1% |
| Qwen3-VL pairwise | 54.4% | 53.9% |
| Qwen3-VL pointwise* | 67.1% | 65.0% |

Pointwise results measure A/B comparison accuracy after comparing predicted weights. Full strong-baseline score reports and predictions are stored under each mode's `output/<split>/` folder.

*Qwen pointwise uses the `positive_json_number_v1` numeric generation constraint with the unchanged user prompt. Its configuration was frozen after dev validation and before test. All four Qwen outputs cover exactly 1,000 pairs with zero invalid outputs; both methods score below the volume rule on dev and test. See the [complete comparison](qwen_strong_baseline/results.md) for macro-F1, timing, memory, and tie counts.

Revalidate the saved outputs and regenerate the comparison without model inference:

```bash
python qwen_strong_baseline/summarize_results.py --require-complete
```

## Source

Standley et al. (2017), [image2mass](https://github.com/tstandley/image2mass). The original dataset's terms remain applicable; see `SOURCE_DATA_TERMS.txt`.
