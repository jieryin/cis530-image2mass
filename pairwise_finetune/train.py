"""Train language-only LoRA with first-answer A/B probability MSE."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from .core import (MODEL_ID, REVISION, environment_metadata, evaluate_rows,
                   file_hash, load_model, memory_report, predict_probability,
                   prepare_inputs, save_predictions, select_device, synchronize,
                   write_json)
from .data import read_pairs, validate_disjoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("pairs/train_pairs.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("pairs/dev_pairs.jsonl"))
    parser.add_argument("--root", type=Path, default=Path("amazon_data_10k"))
    parser.add_argument("--output", type=Path, default=Path("pairwise_finetune/output/run1"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, help="Pilot limit, in optimizer updates")
    parser.add_argument("--limit-train", type=int, help="Seeded pilot subset size")
    parser.add_argument("--limit-dev", type=int, help="Seeded pilot subset size")
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate metadata without model/dependency downloads")
    parser.add_argument("--check-images", action="store_true", help="Also decode/validate records in a dry-run")
    args = parser.parse_args(argv)
    for name in ("epochs", "grad_accum", "rank", "alpha", "max_pixels", "max_steps", "limit_train", "limit_dev"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_pixels < 1024:
        parser.error("--max-pixels must be at least 1024 (one merged image token)")
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.dropout < 1 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Invalid learning rate, dropout, or weight decay")
    return args


def main(argv=None):
    args = parse_args(argv)
    train = read_pairs(args.train, root=args.root, check_images=args.check_images)
    dev = read_pairs(args.dev, root=args.root, check_images=args.check_images)
    # Validate the FULL source split before applying a pilot limit.
    validate_disjoint(train, dev)
    rng = random.Random(args.seed)
    if args.limit_train:
        train = rng.sample(train, min(args.limit_train, len(train)))
    if args.limit_dev:
        dev = rng.sample(dev, min(args.limit_dev, len(dev)))
    if args.dry_run:
        print(json.dumps({"status": "validated", "train_pairs": len(train), "dev_pairs": len(dev),
                          "images_checked": args.check_images, "model_loaded": False,
                          "loss": "MSE(softmax([z_A,z_B])[A], 1[label=A])",
                          "answer_position": "first assistant answer token"}, indent=2))
        return
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}; choose a new --output")
    # Check record presence before model downloads; each loaded record is also
    # checked against its pair metadata immediately before it is used.
    for row in train + dev:
        for side in ("a", "b"):
            if not (args.root / row[f"path_{side}"]).is_file():
                raise ValueError(f"Missing dataset record: {args.root / row[f'path_{side}']}")

    import torch
    import torch.nn.functional as F

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device, dtype = select_device(args.device, args.dtype)
    model, processor, choices = load_model(
        device, dtype, max_pixels=args.max_pixels, rank=args.rank, alpha=args.alpha,
        dropout=args.dropout, checkpointing=not args.no_checkpointing,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train) / args.grad_accum) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = min(total_steps, max(1, round(0.03 * total_steps)))

    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"model": MODEL_ID, "revision": REVISION, "device": str(device),
                   "dtype": str(dtype).removeprefix("torch."), "batch_size": 1,
                   "probability_space": "binary", "answer_token_ids": list(choices),
                   "answer_policy": "leading_choice_only_v1", "tie_label": "B",
                   "loss": "answer_probability_mse", "train_pairs": len(train), "dev_pairs": len(dev),
                   "train_sha256": file_hash(args.train), "dev_sha256": file_hash(args.dev),
                   "selection": "dev accuracy, then lower dev MSE",
                   "trainable_parameters": sum(p.numel() for p in trainable),
                   "total_parameters": sum(p.numel() for p in model.parameters()),
                   "versions": environment_metadata(), "seed_reproducibility": "seeded, not bitwise guaranteed across devices",
                   "source_sha256": {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}})
    # Persist actual preprocessing, not just requested limits, with the run.
    config["image_processor"] = processor.image_processor.to_dict()
    config["chat_template"] = processor.chat_template
    args.output.mkdir(parents=True)
    write_json(args.output / "run.json", {"status": "running", "config": config})
    step, seen, train_seconds = 0, 0, 0.0
    best_key = (-1.0, -float("inf"))
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        order = list(train)
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        epoch_loss, epoch_seen = 0.0, 0
        for start in range(0, len(order), args.grad_accum):
            group = order[start:start + args.grad_accum]
            optimizer.zero_grad(set_to_none=True)
            synchronize(device)
            started = time.perf_counter()
            group_loss = 0.0
            for row in group:
                inputs = prepare_inputs(processor, row, args.root, device)
                p_a = predict_probability(model, inputs, choices)
                target = torch.tensor([float(row["label"] == "A")], device=device)
                loss = F.mse_loss(p_a, target)
                if not bool(torch.isfinite(loss)):
                    raise ValueError(f"Non-finite loss at {row['pair_id']}")
                # Divide by the actual group size, including a short final group.
                (loss / len(group)).backward()
                group_loss += float(loss.detach().item())
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            synchronize(device)
            seconds = time.perf_counter() - started
            step += 1
            seen += len(group)
            epoch_seen += len(group)
            epoch_loss += group_loss
            train_seconds += seconds
            event = {"type": "train", "epoch": epoch, "optimizer_step": step,
                     "pairs": len(group), "mse": group_loss / len(group),
                     "seconds": seconds, "lr": scheduler.get_last_lr()[0], **memory_report(device)}
            with (args.output / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            if step == 1 or step % 10 == 0 or step == total_steps:
                print(json.dumps(event), flush=True)
            if step >= total_steps:
                break

        predictions, probabilities, metrics = evaluate_rows(model, processor, choices, dev, args.root, device)
        save_predictions(args.output / f"dev_epoch_{epoch}", predictions, probabilities, metrics)
        snapshot = {"config": config, "epoch": epoch, "optimizer_steps": step,
                    "epoch_complete": epoch_seen == len(train), "train_mse": epoch_loss / epoch_seen,
                    "dev": metrics, "optimizer_state_saved": False}
        model.save_pretrained(args.output / "last")
        processor.save_pretrained(args.output / "last")
        write_json(args.output / "last" / "training.json", snapshot)
        key = (metrics["accuracy"], -metrics["mse"])
        if key > best_key:
            best_key, best_metrics = key, metrics
            model.save_pretrained(args.output / "best")
            processor.save_pretrained(args.output / "best")
            write_json(args.output / "best" / "training.json", snapshot)
        print(json.dumps({"type": "dev", "epoch": epoch, **metrics}), flush=True)
        if step >= total_steps:
            break
    pilot = bool(args.limit_train or args.limit_dev or args.max_steps)
    write_json(args.output / "run.json", {"status": "pilot_complete" if pilot else "complete",
               "config": config, "optimizer_steps": step, "pairs_seen": seen,
               "train_seconds": train_seconds, "train_pairs_per_second": seen / train_seconds,
               "best_dev": best_metrics, "memory": memory_report(device)})


if __name__ == "__main__":
    main()
