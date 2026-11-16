"""Score A/B predictions for the image2mass pairwise comparison task.

Gold and prediction files use JSON Lines. Gold rows must contain ``pair_id``
and ``label``; prediction rows must contain ``pair_id`` and ``pred``. Labels
are exactly ``A`` or ``B``. Rows are aligned by pair_id, not file order.

Example:
    python score.py pairs/test_pairs.jsonl simple_baseline/simple_preds.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping


# Keeping the label vocabulary in one place prevents the CLI, metrics, and
# validation code from silently disagreeing about accepted outputs.
LABELS = ("A", "B")


class ScoringError(ValueError):
    """Raised when a gold or prediction file violates the scoring contract."""


def load_labels(path: Path, value_field: str) -> Dict[str, str]:
    """Load pair_id -> label from a JSONL file and validate every row."""
    # A dictionary makes scoring independent of row order and also gives us a
    # natural way to detect duplicate pair IDs before computing any metrics.
    values: Dict[str, str] = {}
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise ScoringError(f"cannot open {path}: {exc}") from exc

    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScoringError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ScoringError(f"{path}:{line_number}: each row must be a JSON object")
            if "pair_id" not in row or value_field not in row:
                raise ScoringError(
                    f"{path}:{line_number}: required fields are pair_id and {value_field}"
                )
            pair_id = row["pair_id"]
            value = row[value_field]
            if not isinstance(pair_id, str) or not pair_id:
                raise ScoringError(f"{path}:{line_number}: pair_id must be a non-empty string")
            if pair_id in values:
                raise ScoringError(f"{path}:{line_number}: duplicate pair_id {pair_id!r}")
            if value not in LABELS:
                raise ScoringError(
                    f"{path}:{line_number}: {value_field} must be A or B, got {value!r}"
                )
            values[pair_id] = value

    if not values:
        raise ScoringError(f"{path}: file contains no records")
    return values


def _preview(items: Iterable[str], limit: int = 5) -> str:
    items = sorted(items)
    shown = ", ".join(items[:limit])
    return shown + (", ..." if len(items) > limit else "")


def score(gold: Mapping[str, str], predictions: Mapping[str, str]) -> dict:
    """Validate ID coverage and compute accuracy, F1, and a confusion matrix."""
    # Require exactly one prediction for every gold item. Reporting a score on
    # a convenient subset would otherwise make different systems incomparable.
    missing = set(gold) - set(predictions)
    extra = set(predictions) - set(gold)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {len(missing)} prediction(s): {_preview(missing)}")
        if extra:
            details.append(f"found {len(extra)} unknown pair_id(s): {_preview(extra)}")
        raise ScoringError("; ".join(details))

    # Matrix orientation is fixed as rows=gold and columns=prediction. This is
    # stated explicitly in both terminal output and scoring.md.
    confusion = {gold_label: {pred_label: 0 for pred_label in LABELS} for gold_label in LABELS}
    for pair_id, gold_label in gold.items():
        confusion[gold_label][predictions[pair_id]] += 1

    total = len(gold)
    correct = sum(confusion[label][label] for label in LABELS)
    per_class = {}
    for label in LABELS:
        # For a binary task, the other label supplies FP/FN counts for the
        # current one. Zero denominators are defined as zero rather than NaN.
        other = LABELS[1] if label == LABELS[0] else LABELS[0]
        tp = confusion[label][label]
        fn = confusion[label][other]
        fp = confusion[other][label]
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "examples": total,
        "correct": correct,
        "coverage": len(predictions) / total,
        "accuracy": correct / total,
        "macro_f1": sum(per_class[label]["f1"] for label in LABELS) / len(LABELS),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def print_report(result: Mapping) -> None:
    """Print a stable, human-readable report."""
    print(f"Examples:  {result['examples']}")
    print(f"Correct:   {result['correct']}")
    print(f"Coverage:  {result['coverage']:.4f}")
    print(f"Accuracy:  {result['accuracy']:.4f}")
    print(f"Macro-F1:  {result['macro_f1']:.4f}")
    print()
    print("Confusion matrix (rows=gold, columns=prediction)")
    print("             Pred A   Pred B")
    for label in LABELS:
        row = result["confusion_matrix"][label]
        print(f"Gold {label}   {row['A']:10d}{row['B']:9d}")
    print()
    print("Class  Precision  Recall      F1  Support")
    for label in LABELS:
        metrics = result["per_class"][label]
        print(
            f"{label:>5}  {metrics['precision']:9.4f}  {metrics['recall']:6.4f}  "
            f"{metrics['f1']:6.4f}  {metrics['support']:7d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gold", type=Path, help="Gold JSONL with pair_id and label")
    parser.add_argument("predictions", type=Path, help="Prediction JSONL with pair_id and pred")
    parser.add_argument("--json-output", type=Path, help="Optionally save all metrics as JSON")
    args = parser.parse_args()

    try:
        gold = load_labels(args.gold, "label")
        predictions = load_labels(args.predictions, "pred")
        result = score(gold, predictions)
    except ScoringError as exc:
        parser.error(str(exc))

    print_report(result)
    if args.json_output:
        # The optional JSON report lets later experiment scripts build result
        # tables without parsing the human-readable terminal output.
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved JSON metrics to {args.json_output}")


if __name__ == "__main__":
    main()
