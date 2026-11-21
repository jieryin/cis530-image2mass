"""Write the pairs that the volume-rule simple baseline gets wrong.

The volume rule and its tie fallback are imported from
simple_baseline/simple-baseline.py, so the subset matches the reported
baseline. Rows are copied unchanged from the input pair file.

    python pairs/make_volume_wrong_pairs.py --train pairs/train_pairs.jsonl --pairs pairs/dev_pairs.jsonl --output pairs/dev_volume_wrong_pairs.jsonl
    python pairs/make_volume_wrong_pairs.py --train pairs/train_pairs.jsonl --pairs pairs/test_pairs.jsonl --output pairs/test_volume_wrong_pairs.jsonl
"""

import argparse
import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("simple_baseline", REPO_ROOT / "simple_baseline" / "simple-baseline.py")
simple = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(simple)


def volume_pred(pair, majority):
    """Same decision as simple-baseline.py --method volume."""
    volume_a = simple.bounding_box_volume(pair, "a")
    volume_b = simple.bounding_box_volume(pair, "b")
    if volume_a == volume_b:
        return majority
    return "A" if volume_a > volume_b else "B"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True, help="Training pairs, used only for the tie fallback label")
    parser.add_argument("--pairs", type=Path, required=True, help="Pair file to filter")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the volume-wrong pairs")
    args = parser.parse_args()

    majority = simple.training_majority(simple.read_jsonl(args.train))
    pairs = simple.read_jsonl(args.pairs)
    wrong = [pair for pair in pairs if volume_pred(pair, majority) != pair["label"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for pair in wrong:
            handle.write(json.dumps(pair) + "\n")
    share_a = sum(pair["label"] == "A" for pair in wrong) / len(wrong)
    print(f"{len(wrong)} of {len(pairs)} pairs are wrong under the volume rule "
          f"(label A share {share_a:.3f}); wrote {args.output}")


if __name__ == "__main__":
    main()