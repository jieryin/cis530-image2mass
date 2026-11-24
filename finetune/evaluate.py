"""Evaluate a task-specific adapter through the shared loading/data pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai_strong_baseline.runner import sha256, write_json
from . import TASK_NAMES, get_task
from .data import read_pairs, validate_subset
from .model import MODEL_ID, REVISION, load_model, select_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--task", choices=TASK_NAMES, help="Inferred from the checkpoint by default")
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
    checkpoint = json.loads((args.checkpoint / "training.json").read_text(encoding="utf-8"))
    config = checkpoint["config"]
    task_name = config.get("task", "pairwise")
    if args.task is not None and args.task != task_name:
        raise ValueError("Requested task does not match the checkpoint task")
    task = get_task(task_name)
    if config["model"] != MODEL_ID or config["revision"] != REVISION:
        raise ValueError("Checkpoint model does not match this implementation")
    if config.get("task_prompt", task.PROMPT) != task.PROMPT:
        raise ValueError("Checkpoint task prompt differs from the current prompt")
    rows = read_pairs(args.pairs, root=args.root)
    subset = read_pairs(args.subset, root=args.root) if args.subset else None
    if subset:
        validate_subset(rows, subset)
    device, dtype = select_device(args.device, config["dtype"] if args.dtype == "auto" else args.dtype)
    model, processor = load_model(
        device, dtype, adapter=args.checkpoint, max_pixels=config["max_pixels"],
        pointwise_regression=task_name == "pointwise",
    )
    context = task.setup(model, processor)
    inference_settings = task.metadata(context)
    for key in ("answer_policy", "answer_token_ids", "max_new_tokens"):
        if key in config and config[key] != inference_settings.get(key):
            raise ValueError(f"Checkpoint task setting differs: {key}")
    predictions, auxiliary, metrics = task.evaluate_rows(model, processor, context, rows, args.root, device)
    task.save_predictions(args.output, predictions, auxiliary, metrics)
    if subset:
        write_json(args.output / "subset_scores.json", task.subset_metrics(subset, predictions, auxiliary))
    write_json(args.output / "run.json", {"status": "complete", "task": task_name,
               "checkpoint": str(args.checkpoint),
               "checkpoint_metadata_sha256": sha256(args.checkpoint / "training.json"),
               "pairs": str(args.pairs), "pairs_sha256": sha256(args.pairs),
               "subset": str(args.subset) if args.subset else None,
               "subset_sha256": sha256(args.subset) if args.subset else None,
               "device": str(device), "dtype": str(dtype), "training_config": config})
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
