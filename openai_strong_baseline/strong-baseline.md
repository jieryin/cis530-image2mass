# Strong baselines

[`runner.py`](runner.py) evaluates OpenAI's `gpt-6.1-sol` model as a pretrained vision-language baseline for object mass comparison. We use zero-shot prompts, with no fine-tuning or labeled examples in the requests. Unlike the simple volume rule, the model can use the image to infer material and construction as well as the dimensions to judge scale.

**Pairwise** (`--mode pairwise`) sends two objects, labeled A and B, with each object's dimensions followed by its image. The model returns only `A` or `B` to identify the heavier object. The exact prompt and response parsing are in [`pairwise/predict.py`](pairwise/predict.py).

**Pointwise** (`--mode pointwise`) asks for one positive numeric weight estimate in pounds for each unique object. The runner reuses that estimate wherever the object appears, then compares the two predicted weights locally:

```python
if predicted_weight_a > predicted_weight_b:
    prediction = "A"
else:
    prediction = "B"
```

Equal predicted weights therefore select B. The prompt is in [`pointwise/predict.py`](pointwise/predict.py), and the comparisons are made by `save_results()` in `runner.py`. `--mode both` runs both approaches with separate outputs and a shared concurrency limit.

## Input and output

The runner reuses the existing pair files produced by [`make_pairs.py`](../make_pairs.py), loads the corresponding `.pklz` records with [`load_record.py`](../load_record.py), and evaluates predictions with [`score.py`](../score.py). It does not regenerate or reorder the saved pairs.

The model receives **RGB images and three physical dimensions in inches**. The images are sent as PNG image inputs, not file paths. IDs are used locally to match records and predictions. Ground-truth weights, gold A/B labels, filenames, and thickness masks are excluded from the requests. [`api.py`](api.py) handles image encoding and asynchronous Responses API calls.

Both approaches write one prediction per pair in the format accepted directly by `score.py`:

```json
{"pair_id": "test-00000", "pred": "B"}
```

Pointwise also writes one estimated weight per unique object:

```json
{"id": 6852, "weight_lb": 0.13}
```

Outputs are separated by method and split:

```text
openai_strong_baseline/
├── pairwise/output/{train,dev,test}/
│   ├── preds.jsonl
│   ├── scores.json
│   └── scores.txt
└── pointwise/output/{train,dev,test}/
    ├── weight_preds.jsonl
    ├── preds.jsonl
    ├── scores.json
    └── scores.txt
```

Each run also creates local `cache.json` and `run.json` files for resuming and recording configuration, input checksums, raw output text, usage, and progress. These local files and `.env` are excluded from Git; the predictions and score reports are included.

## Usage

Run the commands from the repository root with Python 3.11. Download and extract the [10k dataset subset](../README.md#downloads) so the records are under `amazon_data_10k/train/`, `dev/`, and `test/`.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt openai==3.51.0
```

Create `openai_strong_baseline/.env` from [`.env.example`](.env.example) if it does not already exist, then fill in your own `OPENAI_API_KEY`. Keep the key local. An exported `OPENAI_API_KEY` takes precedence over the file. Model selection follows `--model`, then an exported `OPENAI_MODEL`, then `.env`, then the default model.

The reported runs used `gpt-6.1-sol`, **high** reasoning effort, **high** image detail, a maximum of **4,096 output tokens per request**, and **16 concurrent requests**. The output limit includes reasoning tokens; it is not a dollar budget. The 16-request limit is shared when running both modes.

Check the inputs without API calls or output writes:

```sh
python openai_strong_baseline/runner.py --mode both --pairs pairs/dev_pairs.jsonl --dry-run
```

Generate predictions and automatically save scores for each split:

```sh
for split in train dev test; do
  python openai_strong_baseline/runner.py --mode both --pairs "pairs/${split}_pairs.jsonl" || break
done
```

Use `--mode pairwise` or `--mode pointwise` to run only one approach. The runner infers the split from the record paths. Custom inputs can specify `--split train`, `--split dev`, or `--split test`.

Rerunning the same command reuses compatible local cache entries and requests only missing predictions. A cached smoke subset can be extended to the full split; the completed run replaces that split's prediction and score reports. Fresh API calls need not give identical predictions, and the runner does not send a generation seed. To change the model or prompt settings, select a new output location with `--output-dir`, which creates `<directory>/<mode>/output/<split>/`.

The committed prediction files can be scored **without an API key or new model calls**:

```sh
python score.py pairs/test_pairs.jsonl openai_strong_baseline/pairwise/output/test/preds.jsonl
python score.py pairs/test_pairs.jsonl openai_strong_baseline/pointwise/output/test/preds.jsonl
```

The offline tests cover image/dimension inputs, exclusion of gold metadata, asynchronous execution, cache reuse, split separation, tie handling, and compatibility with the scorer:

```sh
python -m unittest discover -s tests -v
```

## Sample output

The first three pointwise test predictions are:

```json
{"pair_id": "test-00000", "pred": "B"}
{"pair_id": "test-00001", "pred": "B"}
{"pair_id": "test-00002", "pred": "B"}
```

Scoring the full prediction file gives:

```text
$ python score.py pairs/test_pairs.jsonl openai_strong_baseline/pointwise/output/test/preds.jsonl
Examples:  1000
Correct:   921
Coverage:  1.0000
Accuracy:  0.9210
Macro-F1:  0.9210
...
```

## Score

We evaluated the same saved pairs as the simple baselines, using the unchanged scorer. Accuracy is the primary metric; the reports also include macro-F1, per-class precision/recall/F1, a confusion matrix, and coverage. The pointwise scores below measure **A/B comparison accuracy**, not exact numerical weight accuracy.

| Split | Evaluation pairs | Unique objects predicted by pointwise |
|---|---:|---:|
| Train | 8,000 | 6,945 |
| Dev | 1,000 | 869 |
| Test | 1,000 | 882 |

The 10,000 pairs reference 8,696 distinct objects. The remaining 1,304 dataset objects do not appear in the saved pairs and are not needed for this evaluation. The training split is evaluated in the same zero-shot way; no fitting takes place on it.

| Baseline | Train accuracy | Dev accuracy | Test accuracy |
|---|---:|---:|---:|
| Majority class | 0.5012 | 0.5040 | 0.4930 |
| Volume rule | 0.8353 | 0.8210 | 0.8320 |
| GPT-6.1 Sol, pairwise | 0.9154 | 0.9180 | 0.9110 |
| GPT-6.1 Sol, pointwise | 0.9199 | 0.9240 | 0.9210 |

Both strong baselines have full coverage on all splits. On test, pairwise gets **911/1,000** correct (macro-F1 **0.9110**), improving over volume by **7.9 percentage points**. Pointwise gets **921/1,000** correct (macro-F1 **0.9210**), improving over volume by **8.9 points**. Pointwise's advantage over pairwise is 10 test pairs; this is an observed difference, not a claim of statistical significance.

The complete test reports are saved as [pairwise scores](pairwise/output/test/scores.txt) and [pointwise scores](pointwise/output/test/scores.txt), with corresponding JSON reports in the same folders.
