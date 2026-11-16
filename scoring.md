# Scoring Protocol

## Task

Each example contains two objects, A and B. The system predicts which object is heavier. `A` means that object A is heavier, and `B` means that object B is heavier. The object pairs are constructed within the original train, development, and test splits, and every retained pair has a heavier-to-lighter mass ratio of at least 1.25.

All systems, including rule-based baselines, pointwise mass estimators, pairwise vision-language models, and API models, are converted to the same A/B prediction format before scoring.

## File formats

Gold and prediction files use JSON Lines: one JSON object per non-empty line. File order is not significant because records are aligned by `pair_id`.

The gold file must contain `pair_id` and `label`. Additional fields are allowed and ignored by the scorer.

```json
{"pair_id": "test-00000", "label": "B"}
```

The prediction file must contain exactly one prediction for every gold `pair_id` using the field `pred`.

```json
{"pair_id": "test-00000", "pred": "B"}
```

Labels are case-sensitive and must be exactly `A` or `B`. Duplicate IDs, missing predictions, unknown IDs, malformed JSON, and invalid labels are errors. Invalid rows are never silently skipped.

## Metrics

The primary metric is accuracy:

```text
accuracy = number of correct predictions / number of gold examples
```

Accuracy is appropriate because every example has one correct label and the constructed pair sets are approximately balanced between A and B.

The scorer also reports:

- precision, recall, and F1 for each label;
- macro-F1, which gives A and B equal weight;
- a 2 x 2 confusion matrix;
- coverage, which must equal 1.0 under the strict validation policy.

For each label:

```text
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 × precision × recall / (precision + recall)
macro-F1 = (F1_A + F1_B) / 2
```

When a precision, recall, or F1 denominator is zero, the corresponding metric is defined as zero.

## Running the scorer

From the repository root:

```sh
python score.py pairs/test_pairs.jsonl simple_baseline/simple_preds.jsonl
```

To save the complete metric dictionary in machine-readable form:

```sh
python score.py \
  pairs/test_pairs.jsonl \
  simple_baseline/simple_preds.jsonl \
  --json-output results/majority_test.json
```

The scorer uses only the Python standard library and does not import model code. Python 3.11 is the supported project version.

## Example output

Running:

```sh
python score.py pairs/test_pairs.jsonl simple_baseline/simple_preds.jsonl
```

produces:

```text
Examples:  1000
Correct:   493
Coverage:  1.0000
Accuracy:  0.4930
Macro-F1:  0.3302

Confusion matrix (rows=gold, columns=prediction)
             Pred A   Pred B
Gold A            0      507
Gold B            0      493

Class  Precision  Recall      F1  Support
    A     0.0000  0.0000  0.0000      507
    B     0.4930  1.0000  0.6604      493
```

