# Qwen3-VL-2B pointwise mass fine-tuning

Selected `full` epoch 3 at optimizer step 2607 by highest dev pair accuracy, then lowest dev object log-MSE. Default best dev accuracy exceeded both baselines; retain one seed and stop tuning.

## Training and reproducibility

- Source commit: `d65197f1ca61232231573b78011799932bea3537`. Base model: `Qwen/Qwen3-VL-2B-Instruct`, revision `89644892e4d85e24eaac8bacfd4f463576704203`. Device `cuda`, dtype `bfloat16`, SDPA attention.
- 6,945 unique training objects per epoch, 1,000 dev pairs; 3 epochs, microbatch 1, gradient accumulation 8, seed 42, maximum image pixels 262,144.
- Language attention LoRA rank 8, alpha 16, dropout 0.05; trainable scalar mass head. AdamW LR 0.0001, weight decay 0.01, 3% warmup then linear decay, gradient clipping at 1.0.
- 3,213,313/2,130,747,394 trainable/total parameters. The base model and vision encoder are frozen.
- Each object supplies one RGB image and dimensions. The mass head predicts softplus(linear(final assistant-prefix hidden state)); training minimizes MSE against log1p(gold pounds). Evaluation converts with expm1 and compares the two predicted weights; ties select B.
- Resumable state saved every 25 optimizer updates and at stage boundaries. The most recent two complete checkpoints contain the LoRA adapter, mass head, processor, optimizer, scheduler, RNG state, and training cursor. `best/` and `last/` are inference adapters.

## Data verification

Data verification passed: True. Matched 10,000 of 10,000 records against the source manifest; 0 missing, 0 mismatched.
Split counts: {"dev": 1000, "test": 1000, "train": 8000}. Release archive checksum matches: True; manifest matches the checked origin/main snapshot: True.
Manifest SHA-256: `cd406ee4778a678b0c4f3de8183d6a728d3ed3ca2175a2f9e6a8bed659d13418`. Archive SHA-256: `aa17aeb12ed1b67f4492cba6820a90b4dd9d55632f2291ef4868c2c3f1e2634e`.

## Pilot and reload

Pilot: 20 updates, 160 objects seen. Mean loss in the first/last five updates: 0.940779/0.824047. Mean update time: 2.51 seconds.
Peak pilot tensor allocation: 4.13 GiB.
Pre-clipping gradient norms: 130.025 to 2864.05; all finite: True.

Independent reload verification:

```json
{
  "passed": true,
  "controlled_resume_from_step": 10,
  "final_step": 20,
  "matching_pair_predictions": 32,
  "matching_object_weights": 63,
  "mass_head_and_lora_saved": true,
  "optimizer_scheduler_rng_loaded": true,
  "loss_first_5_mean": 0.9407787386739983,
  "loss_last_5_mean": 0.8240473400115661,
  "gradient_norm_min": 130.0251922607422,
  "gradient_norm_max": 2864.047607421875,
  "peak_allocated_gib": 4.1295084953308105,
  "mean_update_seconds": 2.5111061400093604,
  "projected_full_training_seconds": 6800.779029225325,
  "expected_full_updates": 2607
}
```

Operational recovery: a Windows console-close event stopped the initial process chain at update 128. Training resumed from the durable update-125 checkpoint, restoring optimizer, scheduler, RNG and example cursor; 3 unsaved updates were replayed. The launcher and all Python children now use the windowless interpreter. Wall times below include the failed training attempt, but exclude the idle repair interval.

## Development selection

| Run | Epoch | Mean train log-MSE | Dev correct | Dev accuracy | Dev macro F1 | Object log-MSE | MAE (lb) | RMSE (lb) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 1 | 0.253538 | 869/1000 | 86.9% | 0.8690 | 0.112001 | 0.8857 | 1.6558 |
| full | 2 | 0.121910 | 884/1000 | 88.4% | 0.8840 | 0.090267 | 0.7639 | 1.3940 |
| full | 3 | 0.083924 | 886/1000 | 88.6% | 0.8860 | 0.084456 | 0.7363 | 1.2949 |

Selection uses dev only. The test set is evaluated after selecting the final checkpoint.

## Final test results

| Slice | Pairs | Baseline correct | Tuned correct | Baseline accuracy | Tuned accuracy | Change | Tuned macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full test | 1000 | 650 | 895 | 65.0% | 89.5% | +24.5 pp | 0.8950 |
| Volume-rule errors | 168 | 73 | 95 | 43.5% | 56.5% | +13.1 pp | 0.5647 |
| Remaining test | 832 | 577 | 800 | 69.4% | 96.2% | +26.8 pp | 0.9615 |

Full-test object regression: 882 unique objects, log-MSE 0.080158, MAE 0.7077 lb, RMSE 1.3210 lb.
Volume-error subset regression: 277 unique objects, log-MSE 0.116502, MAE 0.7163 lb, RMSE 1.4014 lb.

The baseline is the repository's saved zero-shot pointwise numeric-generation pipeline. Fine-tuning uses a scalar regression head and a fixed image-pixel budget. These results compare complete pipelines; they do not isolate LoRA training from the output-head and preprocessing changes.

Matched outcomes: 593 retained correct, 302 corrected, 57 regressed, and 48 still wrong. The net change is +245 correct pairs out of 1000.

## Improvements and remaining errors

| Slice | Pairs | Baseline accuracy | Tuned accuracy | Change | Corrected | Regressed | Still wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gold answer A | 507 | 46.7% | 89.2% | +42.4 pp | 235 | 20 | 35 |
| Gold answer B | 493 | 83.8% | 89.9% | +6.1 pp | 67 | 37 | 13 |
| Volume-rule errors | 168 | 43.5% | 56.5% | +13.1 pp | 59 | 37 | 36 |
| Remaining test | 832 | 69.4% | 96.2% | +26.8 pp | 243 | 20 | 12 |
| Gold weight ratio < 1.5 | 83 | 51.8% | 63.9% | +12.0 pp | 28 | 18 | 12 |
| Gold weight ratio >= 1.5 | 917 | 66.2% | 91.8% | +25.6 pp | 274 | 39 | 36 |
| Both gold weights < 1 lb | 268 | 55.2% | 83.2% | +28.0 pp | 96 | 21 | 24 |
| At least one gold weight >= 1 lb | 732 | 68.6% | 91.8% | +23.2 pp | 206 | 36 | 24 |

The largest accuracy gain among these slices is for Gold answer A: +42.4 percentage points across 507 pairs. The lowest tuned accuracy is 56.5% for Volume-rule errors (168 pairs).

Of 105 remaining pair errors, 73 are in the volume-rule-error subset. 48 pairs are wrong in both pipelines, while 57 were correct in the baseline and became wrong.

The test contains 1000 pairs built from 882 unique objects. 605 objects appear in more than one pair, so pair errors are correlated. These overlapping slices are descriptive comparisons; they are not independent statistical tests or evidence for visual/material causes.

## Largest object weight errors

| Object ID | Product | Gold (lb) | Predicted (lb) | Absolute error (lb) | Signed error (lb) | Wrong pairs / appearances |
|---|---|---:|---:|---:|---:|---:|
| 6953 | Margarita Glasses boys | 11 | 0.62091 | 10.379 | -10.379 | 0 / 5 |
| 6841 | Electronics Marine Electronics Subwoofers special | 11.2 | 3.2789 | 7.9211 | -7.9211 | 0 / 2 |
| 6642 | Home, Garden & Pets Home Appliances Small Appliances Deep Fryers skinny | 11 | 3.8938 | 7.1062 | -7.1062 | 1 / 4 |
| 6691 | shades glass | 8 | 1.0272 | 6.9728 | -6.9728 | 1 / 2 |
| 7887 | pedestal  | 3 | 9.6371 | 6.6371 | +6.6371 | 0 / 1 |
| 7543 | solar thermal system big | 11.65 | 5.0784 | 6.5716 | -6.5716 | 1 / 4 |
| 7511 | rain gauge sturdy | 12 | 5.5776 | 6.4224 | -6.4224 | 0 / 3 |
| 7067 | Ice-Cream Machines steel | 12 | 5.6654 | 6.3346 | -6.3346 | 1 / 2 |

Mean signed object error is -0.088314 lb; a positive value means overestimation. Object metrics count each unique object once, while the pair table measures ordering accuracy.

Strong ordering accuracy does not guarantee accurate absolute mass. For example, object 6953 has gold weight 11 lb and predicted weight 0.6209 lb, yet only 0 of its 5 comparisons are wrong. Both regression error and pair accuracy should therefore be retained in future evaluations.

## Historical pairwise fine-tuning comparison

| Slice | Pairs | Prior pairwise accuracy | Pointwise accuracy | Pointwise minus pairwise |
|---|---:|---:|---:|---:|
| Full test | 1000 | 88.1% | 89.5% | +1.4 pp |
| Volume-rule errors | 168 | 51.2% | 56.5% | +5.4 pp |

The prior pairwise predictions are matched on the same test pair IDs. This historical comparison is added after pointwise selection and is not used for tuning or checkpoint selection.

## Matched cases

| Change | Pair | Gold | Baseline → tuned | Object A (gold → predicted lb) | Object B (gold → predicted lb) | Hard subset |
|---|---|---|---|---|---|---|
| corrected | test-00035 | A | B → A | buckle fat (0.95 → 1.135) | Home, Garden & Pets Pet Supplies Houses & Habitats Accessories Kennel Food & Water Cups common (0.41 → 0.5068) | True |
| corrected | test-00075 | A | B → A | hand axe special (1.95 → 0.8075) | penny bank well made (1.15 → 0.6042) | True |
| regressed | test-00108 | B | B → A | sleigh old (0.13 → 0.6065) | Drive Sockets translucent (0.25 → 0.218) | True |
| regressed | test-00115 | B | B → A | meter kitchen (1.25 → 1.01) | cantle powerful (2 → 0.3722) | True |
| still_wrong | test-00011 | A | B → B | flag glass (0.2 → 0.2888) | pliers smart (0.15 → 0.4724) | True |
| still_wrong | test-00014 | A | B → B | stapler metal (0.9 → 0.6295) | savings bank yellow (0.2 → 0.9258) | True |

All matched cases and predicted/gold object weights are in [test_cases.csv](test_cases.csv). Full per-object predictions are in [test/weights.jsonl](test/weights.jsonl).

## Elapsed time

- Pilot total, including the intentional interruption and resumed segment: 0h 1m 19s. This sums active stage times and excludes the gap between processes.
- pilot-interrupt: 0h 0m 38s; exit code 75.
- pilot resumed segment: 0h 0m 41s; exit code 0.
- reload: 0h 0m 12s; exit code 0.
- full: 1h 55m 8s; exit code 0.
- test: 0h 1m 44s; exit code 0.

- Full-training wall time from first launch to completion, including the repair/approval wait: 2h 31m 18s.
- End-to-end pilot, recovery, full training, test and initial report generation: 2h 34m 38s.
- The pilot projected 1h 53m 20s; active full-stage intervals totaled 1h 55m 8s, 1.6% above that estimate.

The 1h 55m full-stage total includes model loading, checkpoint saves, three dev evaluations and the failed initial attempt. The longer calendar times include the idle repair/approval interval. No additional hyperparameter configuration or seed was run, and test results were not used for selection.

## Running and recovering on this Windows machine

The source checkout used for this experiment is `D:\angie\cis530\pointwise-source`. Run the following in PowerShell to launch or recover the workflow:

```powershell
& 'D:\angie\cis530\.venv\Scripts\python.exe' 'D:\angie\cis530\pointwise-results\start.py'
```

The supervisor and model workers use pythonw.exe without console windows. The supervisor prevents automatic sleep during active work, holds a single-instance lock, updates supervisor_status.json every 30 seconds and retries failed workflows up to three times. Training saves every 25 updates; latest_checkpoint.json identifies the complete resumable state. Completed stages are skipped on restart.

Locking the screen or closing a separate log-viewing window can leave training running. Keep the machine powered; manual sleep, logout, shutdown or reboot can stop the job. After a reboot, run start.py again to resume the last durable checkpoint. best/ and last/ are inference adapters; use the full run directory with its latest_checkpoint.json for training recovery.

To inspect progress:

```powershell
& 'D:\angie\cis530\.venv\Scripts\python.exe' 'D:\angie\cis530\pointwise-results\status.py'
Get-Content 'D:\angie\cis530\pointwise-results\full.stdout.log' -Tail 5 -Wait
```

Checkpoint recovery restores the adapter, mass head, optimizer, scheduler, RNG and example cursor. Unsaved updates are replayed; absolute uninterrupted execution cannot be guaranteed across power loss or operating-system termination.

## Artifacts and downloads

The readable report, predictions, per-update history and run scripts are committed beside this file. The complete archive also contains the pilot and full-run best/last adapters, mass heads, processors, and resumable checkpoints with optimizer, scheduler and RNG state.

- [Experiment release](https://github.com/jieryin/cis530-image2mass/releases/tag/pointwise-qwen3-vl-2b-seed42-20261125)
- [Complete results ZIP](https://github.com/jieryin/cis530-image2mass/releases/download/pointwise-qwen3-vl-2b-seed42-20261125/pointwise-qwen3-vl-2b-seed42-results.zip)
- [Artifact file manifest](https://github.com/jieryin/cis530-image2mass/releases/download/pointwise-qwen3-vl-2b-seed42-20261125/artifact_manifest.csv)
- [SHA256SUMS](https://github.com/jieryin/cis530-image2mass/releases/download/pointwise-qwen3-vl-2b-seed42-20261125/SHA256SUMS)

Check the ZIP and manifest against SHA256SUMS after download. For recovery, extract the archive and use the run directory with its latest_checkpoint.json; best/ and last/ are inference artifacts. The base Qwen model and source dataset are referenced by revision/checksum and are not duplicated in the results archive.
