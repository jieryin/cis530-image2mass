# Baseline results

Re-scored with the unchanged strict scorer.

| Baseline | Dev accuracy | Dev macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| Majority class | 0.5040 | 0.3351 | 0.4930 | 0.3302 |
| Volume rule | 0.8210 | 0.8210 | 0.8320 | 0.8320 |
| OpenAI pairwise | 0.9180 | 0.9180 | 0.9110 | 0.9110 |
| OpenAI pointwise | 0.9240 | 0.9240 | 0.9210 | 0.9210 |
| Qwen pairwise | 0.5440 | 0.4216 | 0.5390 | 0.4350 |
| Qwen pointwise | 0.6710 | 0.6608 | 0.6500 | 0.6388 |

| Qwen mode / split | Status | Inferences | Seconds | Mean seconds | Peak allocated GiB | Invalid outputs | Tied pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| pairwise / dev | complete | 1000/1000 | 112.738 | 0.113 | 4.153 | 0 | n/a |
| pairwise / test | complete | 1000/1000 | 115.223 | 0.115 | 4.153 | 0 | n/a |
| pointwise / dev | complete | 869/869 | 145.035 | 0.167 | 4.054 | 0 | 296 |
| pointwise / test | complete | 882/882 | 156.318 | 0.177 | 4.054 | 0 | 344 |

- Pointwise metrics measure A/B accuracy after comparing weights, with B on ties.
- Simple dev predictions are recomputed in memory using the existing baseline functions; no baseline outputs are overwritten.
- OpenAI high-reasoning API runs and local Qwen Instruct use different compute budgets.
- Qwen timing is copied from run.json and excludes model downloads/loading.

Factual mismatch and equal-weight examples are saved in `error_analysis.json` for image review.
