"""Build object pairs within each split of the 10k image2mass subset.
Pairs whose weights differ by less than 1:1.25 -min-ratio are skipped, 
no unordered pair repeats, and the sampled order decides which object is A. 
"""

import argparse
import csv
import json
import random
from pathlib import Path


def load_objects(manifest):
    """Load model-facing object metadata and group it by the preserved split."""
    objects = {}
    with Path(manifest).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            objects.setdefault(row["project_split"], []).append({
                "id": int(row["id"]),
                "path": row["subset_path"],
                "dims": [float(row[f"dimension_{i}_in"]) for i in (1, 2, 3)],
                "weight": float(row["weight_lb"]),
            })
    return {split: sorted(items, key=lambda item: item["id"]) for split, items in objects.items()}


def make_pairs(split, objects, count, min_ratio, seed):
    """Sample unique, sufficiently different-weight pairs within one split."""
    # Including the split in the seed gives each split an independent but fully
    # reproducible random stream.
    rng = random.Random(f"{seed}-{split}")
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < count:
        attempts += 1
        if attempts > 100 * count:
            raise RuntimeError(f"{split}: found only {len(pairs)} valid pairs")
        a, b = rng.sample(objects, 2)
        # frozenset treats (A, B) and (B, A) as the same underlying pair.
        key = frozenset((a["id"], b["id"]))
        if key in seen or max(a["weight"], b["weight"]) < min_ratio * min(a["weight"], b["weight"]):
            continue
        seen.add(key)
        # The sampled order is retained: it determines which object is A and
        # prevents the labels from acquiring a systematic position bias.
        pairs.append({"pair_id": f"{split}-{len(pairs):05d}", "id_a": a["id"], "id_b": b["id"],
                      "path_a": a["path"], "path_b": b["path"], "dims_a": a["dims"], "dims_b": b["dims"],
                      "weight_a": a["weight"], "weight_b": b["weight"],
                      "label": "A" if a["weight"] > b["weight"] else "B"})
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("amazon_data_10k/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("pairs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-ratio", type=float, default=1.25, help="Minimum heavier/lighter weight ratio")
    parser.add_argument("--train-pairs", type=int, default=8000)
    parser.add_argument("--dev-pairs", type=int, default=1000)
    parser.add_argument("--test-pairs", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} already exists; choose a new --output directory")

    objects = load_objects(args.manifest)
    counts = {"train": args.train_pairs, "dev": args.dev_pairs, "test": args.test_pairs}
    args.output.mkdir(parents=True)
    stats = {}
    for split, count in counts.items():
        pairs = make_pairs(split, objects[split], count, args.min_ratio, args.seed)
        with (args.output / f"{split}_pairs.jsonl").open("x", encoding="utf-8") as handle:
            handle.writelines(json.dumps(pair) + "\n" for pair in pairs)
        stats[split] = {"pairs": len(pairs),
                        "label_A_share": round(sum(p["label"] == "A" for p in pairs) / len(pairs), 4),
                        "unique_objects": len({p["id_a"] for p in pairs} | {p["id_b"] for p in pairs})}
        print(f"{split}: {stats[split]}")
    settings = {"seed": args.seed, "min_ratio": args.min_ratio, "manifest": str(args.manifest), "stats": stats}
    (args.output / "pairing.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
