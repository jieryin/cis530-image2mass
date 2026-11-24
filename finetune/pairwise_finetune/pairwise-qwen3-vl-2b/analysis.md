# Fine-tuning Qwen3-VL-2B to compare object weights

Given two product images and their dimensions, the model predicts which object is heavier. The checkpoint selected on the development set (dev) scored **88.1% on 1,000 test pairs**, up from **53.9%** for zero-shot Qwen. On the 168 pairs where the larger bounding box predicted the wrong answer, accuracy rose from **44.0% to 51.2%**.

## Training setup

- **Model and hardware.** We trained `Qwen/Qwen3-VL-2B-Instruct` on one RTX 4090 using CUDA, BF16, and scaled dot-product attention. The training code was at commit `981b8bf`. We loaded model revision `89644892e4d85e24eaac8bacfd4f463576704203` from the local cache and installed PEFT 0.17.1 from source.
- **Data and schedule.** Each of three epochs covered 8,000 training pairs and ended with an evaluation on 1,000 dev pairs. A microbatch contained one pair. Accumulating gradients over eight pairs gave 1,000 optimizer updates per epoch. We used seed 42 and capped each image at 262,144 pixels.
- **Parameters.** LoRA trained 3,211,264 of 2,130,743,296 parameters in the language model's query, key, value, and output attention projections. The base weights and vision encoder were frozen. LoRA rank was 8, alpha was 16, and dropout was 0.05.
- **Optimization.** AdamW used a peak learning rate of 1e-4, weight decay of 0.01, a 3% warmup, and linear decay. We clipped the gradient norm at 1.0 and enabled gradient checkpointing.
- **Objective.** At the first answer position, we applied softmax to the logits for A and B. The loss was the mean squared error (MSE) between the probability of A and the binary label for whether A was heavier.

## Pre-training checks and runtime

A separate 20-update pilot processed 160 pairs using its own learning-rate schedule. Mean loss was 0.2601 over the first five updates and 0.1591 over the last five. Gradient norms before clipping ranged from 0.626 to 6.116. Peak GPU memory allocated to PyTorch tensors was 4.24 GiB. The pilot answered 22 of 32 dev pairs correctly. Reloading the saved adapter in a fresh process reproduced all 32 predicted labels and probabilities exactly.

| Measurement | Result |
|---|---:|
| Pilot runtime | 1 min 11 s |
| Independent checkpoint reload and evaluation | 11 s |
| Mean time per pilot update | 2.53 s |
| Projected time for 3,000 updates and three dev evaluations | 2 h 11 min 43 s |
| Measured wall time for the completed training run | 2 h 11 min 54 s |
| Time spent on optimizer updates, including forward and backward passes | 2 h 6 min 23 s |
| Mean time per full-run update | 2.53 s |
| Peak PyTorch tensor memory during the full run | 4.24 GiB |
| Final test evaluation | 1 min 54 s |

The full-run timing covers the successful run. The earlier interrupted attempt completed 57 updates. Its runtime and the gap before restart are excluded.

## Checkpoint selection and tuning decision

We selected the checkpoint with the highest dev accuracy, using lower dev MSE to break a tie. The hard dev subset contains 179 pairs on which the bounding-box volume rule fails.

| Epoch | Training MSE | Dev correct | Dev accuracy | Dev macro F1 | Dev MSE | Hard dev correct |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.1094 | 867/1,000 | 86.7% | 0.867 | 0.0942 | 86/179 (48.0%) |
| 2 | 0.0772 | 871/1,000 | 87.1% | 0.871 | 0.0924 | 94/179 (52.5%) |
| 3 | 0.0635 | 882/1,000 | 88.2% | 0.882 | 0.0897 | 100/179 (55.9%) |

**Retain the default configuration and the epoch 3 checkpoint (update 3,000).** Dev accuracy increased and dev MSE fell at every epoch. Epoch 3 also had the highest hard-subset dev accuracy and exceeded both full-dev baselines, zero-shot Qwen (54.4%) and the volume rule (82.1%). We therefore stopped tuning after this run. A single seed and three epochs leave variation across seeds and the effect of longer training unmeasured.

## Final test results

We evaluated the selected checkpoint once on the full test set and scored the predefined subsets using those same predictions. The zero-shot comparison uses the repository's saved predictions for the same Qwen model. The baseline generated answers with standard image preprocessing. The fine-tuned pipeline used A/B logits and a fixed pixel cap. The measured difference therefore includes preprocessing and decoding as well as fine-tuning.

| Test set | Pairs | Zero-shot correct | Fine-tuned correct | Zero-shot accuracy | Fine-tuned accuracy | Gain (percentage points) | Fine-tuned macro F1 | Fine-tuned MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full test | 1,000 | 539 | 881 | 53.9% | 88.1% | +34.2 | 0.881 | 0.0856 |
| Volume rule incorrect (hard subset) | 168 | 74 | 86 | 44.0% | 51.2% | +7.1 | 0.507 | 0.3304 |
| Volume rule correct | 832 | 465 | 795 | 55.9% | 95.6% | +39.7 | 0.956 | 0.0362 |

## Which predictions changed?

Of the 1,000 test pairs, fine-tuning corrected 400 baseline errors and introduced 58 new errors. Both pipelines answered 481 pairs correctly and missed 61. Of the 400 corrections, 397 involved pairs where A was heavier, consistent with a reduction in the zero-shot model's tendency to choose B.

| Group | Pairs | Corrected | Newly incorrect | Incorrect in both | Fine-tuned accuracy |
|---|---:|---:|---:|---:|---:|
| Correct answer A | 507 | 397 | 0 | 55 | 89.2% |
| Correct answer B | 493 | 3 | 58 | 6 | 87.0% |
| Volume rule incorrect | 168 | 51 | 39 | 43 | 51.2% |
| Volume rule correct | 832 | 349 | 19 | 18 | 95.6% |
| Heavier-to-lighter weight ratio < 1.5 | 83 | 32 | 18 | 16 | 59.0% |
| Heavier-to-lighter weight ratio ≥ 1.5 | 917 | 368 | 40 | 45 | 90.7% |

The hard subset accounted for 82 of the 119 remaining errors (68.9%). Correcting 51 pairs while introducing 39 errors yielded a net gain of 12 on that subset. The other 832 pairs accounted for 330 of the 342 net additional correct answers. If the goal of a follow-up run is better performance on hard pairs, one experiment would be to oversample training pairs where the volume rule fails. Use both full-dev and hard-dev accuracy to judge that trade-off and keep the test set reserved for final evaluation.

### Example pairs

Within each outcome, the table shows two examples chosen by prioritizing hard pairs and then the fine-tuned model's confidence in its answer. Product names come from the source records. Weights are in pounds, and P(A) is the fine-tuned model's probability that A is heavier.

| Outcome | Pair ID | Object A (lb) | Object B (lb) | Zero-shot → fine-tuned | Correct answer | P(A) |
|---|---|---|---|---|---|---:|
| Corrected | test-00561 | mixing faucet ceramic (4.0) | Chucks newest (3.1) | B → A | A | 0.9941 |
| Corrected | test-00436 | cache solid (0.25) | butter knife skiny (0.15) | B → A | A | 0.9876 |
| Incorrect in both | test-00852 | clutch bag quality (1.1) | Sports & Outdoors Hunting & Fishing Hunting Air Guns discount (0.63) | B → B | A | 0.0180 |
| Incorrect in both | test-00905 | Dessert Bowls light (5.0) | matrix black (2.0) | B → B | A | 0.0180 |
| Newly incorrect | test-00395 | pepper spray well made (0.33) | opener wonderful (0.44) | B → A | B | 0.9876 |
| Newly incorrect | test-00135 | pew thick (1.73) | black kettle (2.5) | B → A | B | 0.9669 |

All 1,000 pairs and predictions are available in [test_cases.csv](test_cases.csv). The local training directory also contains the best and last adapters, resumable checkpoints, per-update loss, and dev predictions for each epoch.
