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


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True, help="Training pairs (JSON lines)")
    parser.add_argument("--test", type=Path, required=True, help="Pairs to predict (JSON lines)")
    parser.add_argument("--output", type=Path, required=True, help="Where to write predictions")
    parser.add_argument("--method", choices=("majority", "volume"), default="majority")
    args = parser.parse_args()

    majority = Counter(row["label"] for row in read_jsonl(args.train)).most_common(1)[0][0]
    test = read_jsonl(args.test)
    ties = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for pair in test:
            pred = majority
            if args.method == "volume":
                volume_a, volume_b = math.prod(pair["dims_a"]), math.prod(pair["dims_b"])
                if volume_a != volume_b:
                    pred = "A" if volume_a > volume_b else "B"
                else:
                    ties += 1
            handle.write(json.dumps({"pair_id": pair["pair_id"], "pred": pred}) + "\n")
    print(f"method: {args.method}; majority label: {majority}; volume ties: {ties}; "
          f"wrote {len(test)} predictions to {args.output}")


if __name__ == "__main__":
    main()