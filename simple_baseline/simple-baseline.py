"""
Simple baselines for pairwise mass comparison.
This file contains 2 simple baselines:
majority: finds the most frequent gold label ("A" or "B") among the training
pairs and predicts it for every pair in the evaluation file.
volume:   predicts that the object with the larger bounding-box volume
(product of its three listed dimensions) is heavier. Equal volumes fall back
to the majority label.

    python simple_baseline/simple-baseline.py --train pairs/train_pairs.jsonl --test pairs/test_pairs.jsonl --output simple_baseline/simple_preds.jsonl
    python simple_baseline/simple-baseline.py --method volume --train pairs/train_pairs.jsonl --test pairs/test_pairs.jsonl --output simple_baseline/volume_preds.jsonl

"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


LABELS = {"A", "B"}


def read_jsonl(path):
    """Read a pair file while rejecting malformed or duplicate examples."""
    rows = []
    pair_ids = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict) or "pair_id" not in row:
                raise ValueError(f"{path}:{line_number}: each row needs a pair_id")
            if row["pair_id"] in pair_ids:
                raise ValueError(f"{path}:{line_number}: duplicate pair_id {row['pair_id']!r}")
            pair_ids.add(row["pair_id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: file contains no records")
    return rows


def training_majority(rows):
    """Learn the constant prediction from training labels only."""
    labels = []
    for row in rows:
        if row.get("label") not in LABELS:
            raise ValueError("every training row must contain label A or B")
        labels.append(row["label"])
    counts = Counter(labels)
    # The lexicographic tie break makes an exactly balanced training file deterministic.
    return min(LABELS, key=lambda label: (-counts[label], label))


def bounding_box_volume(pair, side):
    """Return width x height x depth after validating one object's dimensions."""
    key = f"dims_{side}"
    dims = pair.get(key)
    if not isinstance(dims, list) or len(dims) != 3:
        raise ValueError(f"pair {pair['pair_id']!r}: {key} must contain three dimensions")
    try:
        dims = [float(value) for value in dims]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pair {pair['pair_id']!r}: {key} must be numeric") from exc
    if any(not math.isfinite(value) or value <= 0 for value in dims):
        raise ValueError(f"pair {pair['pair_id']!r}: {key} values must be positive and finite")
    return math.prod(dims)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True, help="Training pairs (JSON lines)")
    parser.add_argument("--test", type=Path, required=True, help="Pairs to predict (JSON lines)")
    parser.add_argument("--output", type=Path, required=True, help="Where to write predictions")
    parser.add_argument("--method", choices=("majority", "volume"), default="majority")
    args = parser.parse_args()

    # The majority label is always learned from train, never from dev/test.
    majority = training_majority(read_jsonl(args.train))
    test = read_jsonl(args.test)
    ties = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for pair in test:
            pred = majority
            if args.method == "volume":
                # Volume is a geometry-only heuristic: it deliberately ignores
                # RGB appearance, material, thickness masks, and gold weights.
                volume_a = bounding_box_volume(pair, "a")
                volume_b = bounding_box_volume(pair, "b")
                if volume_a != volume_b:
                    pred = "A" if volume_a > volume_b else "B"
                else:
                    ties += 1
            handle.write(json.dumps({"pair_id": pair["pair_id"], "pred": pred}) + "\n")
    print(f"method: {args.method}; majority label: {majority}; volume ties: {ties}; "
          f"wrote {len(test)} predictions to {args.output}")


if __name__ == "__main__":
    main()
