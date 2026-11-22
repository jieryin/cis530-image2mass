"""Evaluate one adapter on full pairs and an optional volume-error subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from score import score
from .core import (MODEL_ID, REVISION, evaluate_rows, file_hash, load_model,
                   save_predictions, select_device, write_json)
from .data import read_pairs


def subset_metrics(rows, subset, predictions, probabilities):
    """Select predictions from the same full run, with strict subset identity."""
    full = {row["pair_id"]: row for row in rows}
    fields = ("id_a", "id_b", "path_a", "path_b", "weight_a", "weight_b", "dims_a", "dims_b", "label")
    for row in subset:
        parent = full.get(row["pair_id"])
        if parent is None or any(parent[field] != row[field] for field in fields):
            raise ValueError(f"Subset is not an exact slice of full pairs: {row['pair_id']}")
    selected = {row["pair_id"]: predictions[row["pair_id"]] for row in subset}
    gold = {row["pair_id"]: row["label"] for row in subset}
    metrics = score(gold, selected)
    metrics["mse"] = sum((probabilities[key] - (gold[key] == "A")) ** 2 for key in gold) / len(gold)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("amazon_data_10k"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset", type=Path)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}; choose a new --output")
    rows = read_pairs(args.pairs)
    subset = read_pairs(args.subset) if args.subset else None
    if subset:
        # Validate the subset before expensive model inference.
        subset_metrics(rows, subset, {r["pair_id"]: "B" for r in rows}, {r["pair_id"]: 0.5 for r in rows})
    checkpoint = json.loads((args.checkpoint / "training.json").read_text(encoding="utf-8"))
    config = checkpoint["config"]
    if config["model"] != MODEL_ID or config["revision"] != REVISION or config["answer_policy"] != "leading_choice_only_v1":
        raise ValueError("Checkpoint model or answer policy does not match this implementation")
    device, dtype = select_device(args.device, config["dtype"] if args.dtype == "auto" else args.dtype)
    model, processor, choices = load_model(device, dtype, adapter=args.checkpoint,
                                         max_pixels=config["max_pixels"])
    if list(choices) != config["answer_token_ids"]:
        raise ValueError("Checkpoint A/B token IDs differ from the tokenizer")
    predictions, probabilities, metrics = evaluate_rows(model, processor, choices, rows, args.root, device)
    save_predictions(args.output, predictions, probabilities, metrics)
    if subset:
        write_json(args.output / "subset_scores.json", subset_metrics(rows, subset, predictions, probabilities))
    write_json(args.output / "run.json", {"status": "complete", "checkpoint": str(args.checkpoint),
               "checkpoint_metadata_sha256": file_hash(args.checkpoint / "training.json"),
               "pairs": str(args.pairs), "pairs_sha256": file_hash(args.pairs),
               "subset": str(args.subset) if args.subset else None,
               "subset_sha256": file_hash(args.subset) if args.subset else None,
               "device": str(device), "dtype": str(dtype), "training_config": config})
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
