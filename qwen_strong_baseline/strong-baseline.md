# Strong baselines: Qwen3-VL

[`runner.py`](runner.py) evaluates `Qwen/Qwen3-VL-2B-Instruct` as a local, zero-shot vision-language baseline for object mass comparison. There is no fine-tuning, optimizer, or labeled example in the prompt. Both methods use the existing pair files and the unchanged [`score.py`](../score.py).

The repository baseline is commit `7455cbcbbff047d544b9020ca953c62c9b2ce0e5`, matching the execution guide. No baseline interface differences were found at implementation time; the existing loader, saved pairs, prompts, and scorer are reused.

**Pairwise** (`--mode pairwise`) sends objects A and B with each object's dimensions followed by its image. The model returns `A` or `B`. [`prompts/pairwise.py`](../prompts/pairwise.py) builds the message; [`pairwise/predict.py`](pairwise/predict.py) accepts only one of those labels after trimming surrounding whitespace.

**Pointwise** (`--mode pointwise`) estimates a positive weight in pounds for each unique object ID. An object's estimate is reused in every pair containing that object. [`pointwise/predict.py`](pointwise/predict.py) converts these estimates to A/B predictions locally:

```python
prediction = "A" if predicted_weight_a > predicted_weight_b else "B"
```

Equal estimates select B. `--mode both` runs the two methods sequentially while sharing one loaded model. These are two methods for the same pairwise classification task; numerical weight estimation error is a different metric.

## Input and output

Records are decoded with the existing [`load_record.py`](../load_record.py). The model receives only **RGB image content and three physical dimensions in inches**, in a single user message. Ground-truth weights, gold labels, thickness masks, object IDs, pair IDs, filenames, and paths are excluded from model messages. IDs and paths are used locally for validation and result association.

The unchanged prompt constants and message builders are imported directly from the shared [`prompts/`](../prompts/README.md) package, also used by the OpenAI baseline and pairwise fine-tuning. Importing them does not construct an API client or issue a request. No OpenAI API calls or API keys are needed for this baseline.

The exact pairwise prompt is:

```text
Compare objects A and B using their images and physical dimensions in inches. Consider each object's size, likely material, and construction to decide which is heavier. Return A if object A is heavier, otherwise B. Return only A or B, with no explanation.
```

Its content order is prompt, A dimensions, A image, B dimensions, B image. Each dimensions block uses the same wording and JSON representation as the OpenAI baseline:

```python
f"Object {side}: dimensions in inches = {json.dumps(dims)}. Image follows."
```

The exact pointwise prompt is:

```text
Estimate the product's weight in pounds from its image and physical dimensions in inches. Consider its size, likely material, and construction, and give your best estimate. Return only one positive number, with no units or explanation.
```

Its content order is prompt, dimensions, image. The dimensions block is:

```python
f"Physical dimensions in inches = {json.dumps(dims)}. Image follows."
```

Qwen's official chat template and `AutoProcessor` convert these messages and actual in-memory images to model inputs. The image processor uses its standard settings from the pinned model revision. There is no custom crop, resize policy, additional object description, or use of a filename as image input. The effective processor settings and chat-template checksum are recorded in the run metadata.

Outputs are separated by method and split:

```text
qwen_strong_baseline/
├── pairwise/output/{dev,test}/
│   ├── preds.jsonl
│   ├── scores.json
│   ├── scores.txt
│   ├── cache.json
│   └── run.json
└── pointwise/output/{dev,test}/
    ├── weight_preds.jsonl
    ├── preds.jsonl
    ├── scores.json
    ├── scores.txt
    ├── cache.json
    └── run.json
```

Both `preds.jsonl` files contain exactly one `{"pair_id": "...", "pred": "A"}` or `B` row per saved pair when complete. The pointwise `weight_preds.jsonl` contains `{"id": ..., "weight_lb": ...}` rows for objects whose estimates are available. Partial pilot weight files are progress artifacts; they are not complete pairwise evaluation outputs.

`cache.json` retains raw response text, parsed values, timing, diagnostics, errors, input file checksums, and configuration. `run.json` records progress, the immutable model revision, dependencies, effective processor settings, GPU information, and per-method timing and memory summaries. Images, model tensors, weights, and API keys are not included in these artifacts.

## Usage

Use Python 3.11 and an isolated environment. The local execution machine has an NVIDIA GeForce RTX 4090 with BF16 support. CUDA must be available in PyTorch; the runner refuses CPU fallback and requires all model parameters on CUDA.

The shared pinned dependencies are in the repository-root [`requirements.txt`](../requirements.txt). From the repository root, these PowerShell commands create an environment on D: alongside this checkout:

```powershell
python -m venv ..\.venv
..\.venv\Scripts\python.exe -m pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:HF_HOME = 'D:\angie\cis530\.huggingface'
..\.venv\Scripts\python.exe -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

This checkout is `D:\angie\cis530\cis530-image2mass`; its existing complete dataset is `D:\angie\cis530\amazon_data_10k`. Therefore the commands below use **`--root ../amazon_data_10k`**. The runner's default is the `amazon_data_10k` directory inside the repository; use that default only when the actual record files are there. Do not resample the dataset or regenerate the saved pairs.

The Hugging Face cache is kept under the D: directory specified by `HF_HOME`. Model files are downloaded automatically if absent. Loading and download time are excluded from reported inference timing. The model and processor are pinned to revision:

```text
89644892e4d85e24eaac8bacfd4f463576704203
```

Once the complete model snapshot is cached, inference also supports offline execution with `$env:HF_HUB_OFFLINE = '1'`. Keep `HF_HOME` pointed to the same D: cache.

The inference configuration is BF16, batch size 1, CUDA device 0, SDPA attention, greedy decoding (`do_sample=False`, `num_beams=1`), and `max_new_tokens=32`. Pointwise uses the deterministic `positive_json_number_v1` generation policy described below. The model is loaded once per process, frozen with `requires_grad_(False)`, kept in `eval()` mode, and called inside `torch.inference_mode()`. Parameter version counters are checked to detect changes. Model loading uses `Qwen3VLForConditionalGeneration` and `AutoProcessor`.

The environment pins PyTorch 2.10.0 with CUDA 12.8 and torchvision 0.25.0 using the [official installation index](https://pytorch.org/get-started/previous-versions/#v2100). An initial dev-only pilot under PyTorch 2.9.1 showed a vision-side slowdown: a local profile attributed 21.66 of 22.13 seconds to `torch.conv3d`. This is consistent with the [reported PyTorch 2.9.x Qwen3-VL Conv3d regression](https://github.com/pytorch/pytorch/issues/174051). Those six pairwise outputs and timings were archived outside the repository at `D:\angie\cis530\qwen_pilot_torch291`, were not scored, and are not used in the final cache. The dependency change is for execution performance; the model revision, prompts, decoding, and preprocessing policy remain fixed, and no test results were used to choose it.

Validate all dataset files, image decoding, manifest checksums, and pair/record consistency without loading the model:

```powershell
..\.venv\Scripts\python.exe qwen_strong_baseline/validate_data.py --root ../amazon_data_10k --json-output qwen_strong_baseline/data_validation.json
```

Check dev inputs and required inference counts without loading a model, performing inference, or writing result files:

```powershell
..\.venv\Scripts\python.exe qwen_strong_baseline/runner.py --mode both --pairs pairs/dev_pairs.jsonl --root ../amazon_data_10k --dry-run
```

Run a dev pilot with at most 20 new inferences for each method:

```powershell
..\.venv\Scripts\python.exe qwen_strong_baseline/runner.py --mode both --pairs pairs/dev_pairs.jsonl --root ../amazon_data_10k --max-new-pairs 20 --max-new-objects 20
```

Inspect raw responses, parsing, CUDA execution, parameter checks, per-method mean time, peak allocated GPU memory, and cache recovery. A pilot does not produce a complete accuracy score. Pilot limits restrict newly computed items, so cached items remain available for a subsequent complete run:

```powershell
..\.venv\Scripts\python.exe qwen_strong_baseline/runner.py --mode both --pairs pairs/dev_pairs.jsonl --root ../amazon_data_10k
```

After dev execution and any dev-only format/configuration diagnosis, freeze the configuration and run test once:

```powershell
..\.venv\Scripts\python.exe qwen_strong_baseline/runner.py --mode both --pairs pairs/test_pairs.jsonl --root ../amazon_data_10k
```

Rerunning an identical command loads compatible cached predictions and performs only missing inferences. Configuration, prompt, dependency/source checksums, pair-file checksums, and input-record checksums must match. An incompatible or failed cache is rejected. Changing the prompt, model revision, decoding, preprocessing, or implementation requires a new `--output-dir`; keep earlier failed artifacts as evidence. Do not adjust the configuration using test scores.

Invalid output is saved with an error and aborts that run. Pairwise accepts only trimmed `A` or `B`; pointwise accepts only one finite positive number. Explanations, units, nonpositive values, nonfinite values, and truncated outputs are errors. There is no fallback label, silent skip, gold-based repair, or repeated retry of the same deterministic response. Diagnose formatting or truncation on dev, record a uniform policy, and use a new output directory if the configuration changes.

The pointwise-only numeric generation policy, [`positive_json_number_v1`](numeric_format.py), uses Transformers' `prefix_allowed_tokens_fn` to restrict generated prefixes to a positive JSON number. EOS is permitted only after a complete finite positive number. This is a constraint applied before token selection, not a repair that extracts a number from an invalid response. The single user prompt, image/dimensions order, image preprocessing, model revision, BF16 configuration, greedy selection, and 32-token limit are preserved. Pairwise generation remains unconstrained. Both methods completed dev validation with these sources. [`frozen_config.json`](frozen_config.json) records their configuration and dev results before test execution; test uses that frozen configuration.

On Windows, atomic output replacement may briefly fail because another process holds the file. The runner retries only the same storage write, with a bounded limit; it does not rerun model inference or change the saved response. A persistent storage error still stops execution. Storage failures are recorded separately from invalid model output.

Final pair predictions and score reports are written only after every required item for that method is available and no errors remain. Complete outputs are evaluated with the original strict scorer, which requires coverage 1.0 and rejects duplicate, missing, unknown, or invalid predictions. The weight estimates are not passed directly to the scorer:

```powershell
..\.venv\Scripts\python.exe score.py pairs/test_pairs.jsonl qwen_strong_baseline/pairwise/output/test/preds.jsonl --json-output qwen_strong_baseline/pairwise/output/test/scores.json
..\.venv\Scripts\python.exe score.py pairs/test_pairs.jsonl qwen_strong_baseline/pointwise/output/test/preds.jsonl --json-output qwen_strong_baseline/pointwise/output/test/scores.json
```

Offline tests use fake predictors and do not download model weights or use GPU inference:

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

All 82 offline regression tests passed for the final inference sources. These checks cover the existing baselines and scorer as well as prompt consistency, exclusion of gold metadata, image/dimensions order, pointwise deduplication and ties, invalid outputs, compatible resume, incompatible cache rejection, complete-output scoring, bounded storage retry, and positive-number grammar behavior.

## Dev pilot

The final-policy dev pilot completed 20 new pairwise and 20 new pointwise inferences on the RTX 4090. Pointwise used `positive_json_number_v1`; pairwise was unconstrained. [`pilot_summary.json`](pilot_summary.json) contains these measured results:

| Method | Completed inferences | Inference seconds | Mean seconds/inference | Peak allocated GPU memory (bytes) | Invalid outputs |
|---|---:|---:|---:|---:|---:|
| Pairwise | 20 | 2.692195 | 0.134610 | 4,458,623,488 | 0 |
| Pointwise | 20 | 3.353867 | 0.167693 | 4,353,143,808 | 0 |

CUDA execution, frozen evaluation mode, zero trainable parameters, unchanged parameter version counters, and valid output parsing were confirmed. Times exclude model download/loading. The pilot has incomplete split coverage and was not scored; no pilot accuracy was used for model selection. Its compatible cache entries can be extended by the full dev run. Earlier unconstrained pilot artifacts are archived separately and are not part of these measurements.

## Development diagnostics

[`development_failures.json`](development_failures.json) retains the actual development failures and raw failed responses. All occurred on dev before test execution:

- The PyTorch 2.9.1 pilot was stopped after six pairwise predictions because of the profiled Conv3d slowdown. The pinned environment was upgraded to PyTorch 2.10.0 without changing model inputs.
- A later dev run encountered a Windows atomic-cache replacement error after 510 pairwise predictions. Its model response was valid `B`; this was a storage failure. The run was archived at `D:\angie\cis530\qwen_interrupted_windows_lock`, and bounded storage retry was added and tested.
- Unconstrained pointwise generation stopped after 54 valid object estimates when object 5754 returned `1.24, 3.09, 7.52` (17 generated tokens, not truncated). The response was rejected intact, and no pointwise score was produced. The associated completed pairwise dev run had accuracy 0.5440 and macro-F1 0.421613; it is a development result, not an entry in the final result table. Both methods' artifacts were archived at `D:\angie\cis530\qwen_dev_unconstrained`.

The pointwise formatting failure motivated a uniform numeric constraint applied during generation. No gold weight, first-number extraction, default prediction, or post hoc repair was used. Fresh caches were used for both final methods because inference source hashes changed; old predictions were not carried into the final run. The completed final dev and test results are reported below.

## Sample output

The first three actual pointwise test pair predictions are:

```json
{"pair_id": "test-00000", "pred": "B"}
{"pair_id": "test-00001", "pred": "B"}
{"pair_id": "test-00002", "pred": "B"}
```

The first two saved pointwise test weight estimates are:

```json
{"id": 6852, "weight_lb": 1.5}
{"id": 7125, "weight_lb": 1.5}
```

Scoring the complete pointwise test file reports 1,000 examples, 650 correct, coverage 1.0000, accuracy 0.6500, and macro-F1 0.6388.

## Score

All four final dev/test outputs are complete, contain exactly the corresponding 1,000 pair IDs, and were rechecked with the unchanged strict scorer. Each has coverage 1.0 and zero invalid outputs. The unified baseline comparison is in [`results.md`](results.md), with machine-readable metrics and provenance in [`results.json`](results.json). Archived development runs are excluded from the final result tables. [`frozen_config.json`](frozen_config.json) was written after dev validation and before test execution; no prompt, decoding, preprocessing, or dependency changes were selected from test scores.

| Baseline | Dev accuracy | Dev macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| Majority class | 0.5040 | 0.3351 | 0.4930 | 0.3302 |
| Volume rule | 0.8210 | 0.8210 | 0.8320 | 0.8320 |
| OpenAI pairwise | 0.9180 | 0.9180 | 0.9110 | 0.9110 |
| OpenAI pointwise | 0.9240 | 0.9240 | 0.9210 | 0.9210 |
| Qwen3-VL pairwise | 0.5440 | 0.4216 | 0.5390 | 0.4350 |
| Qwen3-VL pointwise* | 0.6710 | 0.6608 | 0.6500 | 0.6388 |

*Qwen pointwise uses the `positive_json_number_v1` generation constraint. It shares the exact user prompt with OpenAI, but its output-control mechanism and computation budget differ. Simple dev results are recomputed from the existing deterministic baseline functions; saved test predictions and OpenAI predictions are re-scored without new model requests.

| Split | Pairwise inferences required | Pointwise unique objects required | Evaluation pairs |
|---|---:|---:|---:|
| Dev | 1,000 | 869 | 1,000 |
| Test | 1,000 | 882 | 1,000 |

| Method | Split | Accuracy | Macro-F1 | Coverage | Invalid outputs | Pointwise tie pairs |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-VL pairwise | Dev | 0.5440 | 0.4216 | 1.0000 | 0 | N/A |
| Qwen3-VL pointwise | Dev | 0.6710 | 0.6608 | 1.0000 | 0 | 296 |
| Qwen3-VL pairwise | Test | 0.5390 | 0.4350 | 1.0000 | 0 | N/A |
| Qwen3-VL pointwise | Test | 0.6500 | 0.6388 | 1.0000 | 0 | 344 |

| Method | Split | Completed inferences | Inference seconds | Mean seconds/inference | Peak allocated GPU memory (bytes) |
|---|---|---:|---:|---:|---:|
| Pairwise | Dev | 1,000 | 112.738292 | 0.112738 | 4,459,322,368 |
| Pointwise | Dev | 869 | 145.034988 | 0.166899 | 4,353,428,992 |
| Pairwise | Test | 1,000 | 115.223482 | 0.115223 | 4,459,089,920 |
| Pointwise | Test | 882 | 156.317883 | 0.177231 | 4,353,428,992 |

On dev, pointwise gets 671/1,000 correct, 12.7 percentage points above pairwise's 544/1,000. On test, pointwise gets 650/1,000 correct versus pairwise's 539/1,000, a difference of 11.1 points. Both Qwen methods remain below the volume rule and the saved OpenAI results on both splits. These are observed differences, not claims of statistical significance. The models and computation budgets differ, so the comparison does not isolate prompting.

The final dev/test runs contain **3,751 inferences** (2,000 pairwise and 1,751 unique-object pointwise) and **529.314644 seconds** of recorded inference time, approximately 8.82 minutes. This total includes the compatible final-policy pilot predictions reused by dev, and excludes superseded development runs, downloads, model loading, input validation, storage writes, and scoring.

To verify all four saved prediction files and regenerate the comparison and factual error-summary artifacts without loading a model:

```powershell
..\.venv\Scripts\python.exe qwen_strong_baseline/summarize_results.py --require-complete
```

The original scorer CLI was also rerun for each method/split, and compatible-cache dry-runs passed for both dev and test.

Train is not run for this baseline. The saved training split could be evaluated with the same zero-shot method, but doing so would not train Qwen.

Timing is measured separately for pairwise and pointwise. CUDA is synchronized before and after each inference; timing includes processor work, input transfer, generation, and decoding, and excludes model download/load, dataset validation, cache writes, and scoring. Cached per-item times are retained across resumed runs. Peak GPU memory refers to PyTorch's peak allocated bytes, with the peak counter reset before each call, rather than total memory shown by `nvidia-smi`.

The existing OpenAI baseline uses `gpt-6.1-sol` with high reasoning effort, high image detail, up to 4,096 output tokens per request, and 16 concurrent requests. Qwen uses local greedy decoding with up to 32 new tokens and batch size 1; its pointwise output grammar is an additional generation constraint. Identical prompts do not imply equal computation budgets or identical output-control mechanisms; results should be described with those execution differences.

## Error analysis

The dev pairwise model predicts B for 956 of 1,000 pairs, although the gold labels are nearly balanced (496 A, 504 B). Its low A recall (0.0847) explains why macro-F1 is lower than accuracy. Pointwise has 296 equal-estimate pairs; the required tie=B rule is correct for 148 and incorrect for 148. These counts are observations, not evidence of a particular internal model mechanism.

The following real dev errors were checked against decoded RGB images and the saved predictions. The IDs identify examples locally and were not supplied to the model. Gold weights are dataset labels; descriptions below are visual observations. Because the model returned no explanation, proposed causes remain hypotheses.

- **dev-00000, scale/unit error:** A appears to be a heated styling brush and B a powder bottle. Pointwise estimates 1.5 lb and 71 lb, versus gold 0.8 lb and 0.16 lb, so both methods choose the wrong B. B's image visibly labels its contents `2.5 oz (71 g)`. The match between the printed 71 g and the 71 lb prediction is observed; copying the number without unit conversion is a plausible hypothesis.
- **dev-00001, construction and tie:** A is a safe-like box with dimensions 13 × 13 × 9 inches; B is a bottle with dimensions 4.75 × 13 × 4.75 inches. Gold weights are 11.95 lb and 0.88 lb, but both estimates are 10 lb, making pointwise choose B incorrectly. Confusion about construction or density is a hypothesis; material composition was not verified.
- **dev-00004, mass scale:** A looks like a network switch and B a fabric-looking headband. Estimates are 2.5 lb and 12 lb, versus gold 0.55 lb and 0.13 lb. Both methods choose B incorrectly. Misinterpreting material, contents, or physical scale could explain the reversal, but the images do not establish which cause applies.
- **dev-00012, equal estimates:** A appears to be a small camera-like device and B a mouse. Both estimates are 1.5 lb despite gold weights of 1.87 lb and 0.6 lb, so pointwise's tie=B decision is wrong. This illustrates information lost by coarse repeated numeric estimates; pairwise also chooses B.

[`error_analysis.json`](error_analysis.json) records factual mismatch and equal-weight examples. Original image copies are not committed. This method documentation describes the experiment; no course report, Overleaf project, or LaTeX document is produced.
