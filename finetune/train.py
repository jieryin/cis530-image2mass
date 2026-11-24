"""Shared LoRA training loop; task modules supply examples, loss, and evaluation."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

from openai_strong_baseline.runner import atomic_write, sha256, write_json
from . import TASK_NAMES, get_task
from .model import MODEL_DEFAULTS, MODEL_ID, REVISION, load_model, memory_report, select_device, synchronize
from .data import read_pairs, validate_disjoint
from .checkpoint import load_checkpoint, restore_rng_state, save_adapter, save_checkpoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--task", choices=TASK_NAMES, default="pairwise")
    parser.add_argument("--train", type=Path, default=Path("pairs/train_pairs.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("pairs/dev_pairs.jsonl"))
    parser.add_argument("--root", type=Path, default=Path("amazon_data_10k"))
    parser.add_argument("--output", type=Path, help="Default: finetune/output/<task>/run1")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--rank", type=int, default=MODEL_DEFAULTS["rank"])
    parser.add_argument("--alpha", type=int, default=MODEL_DEFAULTS["alpha"])
    parser.add_argument("--dropout", type=float, default=MODEL_DEFAULTS["dropout"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-pixels", type=int, default=MODEL_DEFAULTS["max_pixels"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, help="Pilot limit, in optimizer updates")
    parser.add_argument("--limit-train", type=int, help="Seeded pilot training examples (pairs or unique objects)")
    parser.add_argument("--limit-dev", type=int, help="Seeded pilot dev pair count")
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate metadata without model/dependency downloads")
    parser.add_argument("--check-images", action="store_true", help="Also decode/validate records in a dry-run")
    parser.add_argument("--save-every", type=int, default=100, help="Save resumable state every N optimizer updates")
    parser.add_argument("--resume", type=Path, help="Continue the latest checkpoint in this run directory")
    args = parser.parse_args(argv)
    args._explicit = {token[2:].split("=", 1)[0].replace("-", "_")
                      for token in (sys.argv[1:] if argv is None else argv) if token.startswith("--")}
    for name in ("epochs", "grad_accum", "rank", "alpha", "max_steps", "limit_train", "limit_dev", "save_every"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_pixels < 1024:
        parser.error("--max-pixels must be at least 1024 (one merged image token)")
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.dropout < 1 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Invalid learning rate, dropout, or weight decay")
    return args


def resume_arguments(args, saved):
    """Recover the schedule while keeping writes inside the selected run."""
    paths = {"train", "dev", "root", "output"}
    overrides = {"root", "device", "dtype", "save_every", "dry_run", "check_images"}
    for name in vars(args).copy():
        if name.startswith("_") or name == "resume":
            continue
        original = Path(saved[name]) if name in paths else saved[name]
        if name == "output":
            original = args.resume.resolve()
        if name in args._explicit:
            current = getattr(args, name)
            equal = current.resolve() == original.resolve() if name in paths else current == original
            if name not in overrides and not equal:
                raise ValueError(f"Cannot change --{name.replace('_', '-')} when resuming")
        else:
            setattr(args, name, original)


def truncate_history(output, step):
    """Discard updates after the durable checkpoint, including a torn last line."""
    path = output / "history.jsonl"
    if not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            break
        if event["optimizer_step"] > step:
            break
        kept.append(json.dumps(event) + "\n")
    atomic_write(path, "".join(kept))


def finish_run(args, config, progress, device=None):
    pilot = bool(args.limit_train or args.limit_dev or args.max_steps)
    seconds = progress["train_seconds"]
    previous = args.output / "run.json"
    memory = memory_report(device) if device is not None else (
        json.loads(previous.read_text(encoding="utf-8")).get("memory", {}) if previous.is_file() else {})
    write_json(previous, {"status": "pilot_complete" if pilot else "complete",
               "config": config, "optimizer_steps": progress["optimizer_steps"],
               "examples_seen": progress["examples_seen"], "train_seconds": seconds,
               "train_examples_per_second": progress["examples_seen"] / seconds if seconds else 0,
               "best_dev": progress["best_metrics"],
               "memory": memory})


def main(argv=None):
    args = parse_args(argv)
    resume_path = metadata = state = None
    if args.resume:
        resume_path, metadata, state = load_checkpoint(args.resume)
        resume_arguments(args, metadata["config"])
    if args.output is None:
        args.output = Path("finetune/output") / args.task / "run1"
    task = get_task(args.task)
    train_pairs = read_pairs(args.train, root=args.root, check_images=args.check_images)
    dev = read_pairs(args.dev, root=args.root, check_images=args.check_images)
    # Validate the FULL source split before applying a pilot limit.
    validate_disjoint(train_pairs, dev)
    train = task.training_rows(train_pairs)
    rng = random.Random(args.seed)
    if args.limit_train:
        train = rng.sample(train, min(args.limit_train, len(train)))
    if args.limit_dev:
        dev = rng.sample(dev, min(args.limit_dev, len(dev)))
    if args.dry_run:
        print(json.dumps({"status": "validated", "task": task.NAME,
                          "train_examples": len(train), "count_unit": task.COUNT_UNIT,
                          "dev_pairs": len(dev),
                          "images_checked": args.check_images, "model_loaded": False,
                          **task.metadata(None)}, indent=2))
        return
    if args.output.exists() and not args.resume:
        raise ValueError(f"Output already exists: {args.output}; choose a new --output")
    import torch

    train_hash, dev_hash = sha256(args.train), sha256(args.dev)
    if metadata:
        saved = metadata["config"]
        for key, actual in {"model": MODEL_ID, "revision": REVISION, "train_sha256": train_hash,
                            "dev_sha256": dev_hash, "task_prompt": task.PROMPT}.items():
            if saved.get(key) != actual:
                raise ValueError(f"Resume mismatch: {key}; use the original data, model, and prompt")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device, dtype = select_device(args.device, args.dtype)
    if metadata and (str(device) != saved["device"] or str(dtype).removeprefix("torch.") != saved["dtype"]):
        raise ValueError("Resume requires the original device type and dtype")
    total_steps = math.ceil(len(train) / args.grad_accum) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    if metadata and metadata["progress"]["phase"] == "epoch_complete" and metadata["progress"]["optimizer_steps"] >= total_steps:
        finish_run(args, {**saved, "output": str(args.output.resolve())}, metadata["progress"])
        print("Run is already complete; no further optimizer updates.")
        return
    model, processor = load_model(
        device, dtype, max_pixels=args.max_pixels, rank=args.rank, alpha=args.alpha,
        dropout=args.dropout, checkpointing=not args.no_checkpointing,
        pointwise_regression=args.task == "pointwise",
        **({"adapter": resume_path, "trainable_adapter": True} if metadata else {}),
    )
    context = task.setup(model, processor)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    warmup = min(total_steps, max(1, round(0.03 * total_steps)))

    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()
              if not key.startswith("_") and key != "resume"}
    config.update({"model": MODEL_ID, "revision": REVISION, "device": str(device),
                   "dtype": str(dtype).removeprefix("torch."), "batch_size": 1,
                   "task": task.NAME, "count_unit": task.COUNT_UNIT,
                   "train_examples": len(train), "dev_pairs": len(dev),
                   "train_sha256": train_hash, "dev_sha256": dev_hash,
                   "trainable_parameters": sum(p.numel() for p in trainable),
                   "total_parameters": sum(p.numel() for p in model.parameters()),
                   "task_prompt": task.PROMPT,
                   **task.metadata(context)})
    # Persist actual preprocessing, not just requested limits, with the run.
    config["image_processor"] = processor.image_processor.to_dict()
    config["chat_template"] = processor.chat_template
    if metadata:
        for key in (*task.metadata(context), "image_processor", "chat_template"):
            if config[key] != saved[key]:
                raise ValueError(f"Resume mismatch: {key}")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
    args.output.mkdir(parents=True, exist_ok=bool(args.resume))
    write_json(args.output / "run.json", {"status": "running", "config": config})
    progress = metadata["progress"].copy() if metadata else {
        "phase": "train", "epoch": 1, "optimizer_steps": 0, "epoch_seen": 0,
        "epoch_loss": 0.0, "examples_seen": 0, "train_seconds": 0.0, "best_metrics": None,
    }
    if metadata:
        truncate_history(args.output, progress["optimizer_steps"])
        # Loading the model may consume RNG; restore only after initialization.
        restore_rng_state(state["rng"], device)
    else:
        save_checkpoint(args.output, model, processor, optimizer, scheduler,
                        config=config, progress=progress, device=device)
    step, seen, train_seconds = progress["optimizer_steps"], progress["examples_seen"], progress["train_seconds"]
    best_metrics = progress["best_metrics"]
    best_key = task.selection_key(best_metrics) if best_metrics else None
    first_epoch = progress["epoch"] + (progress["phase"] == "epoch_complete")
    for epoch in range(first_epoch, args.epochs + 1):
        order = list(train)
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        continuing = progress["phase"] == "train" and progress["epoch"] == epoch
        epoch_loss = progress["epoch_loss"] if continuing else 0.0
        epoch_seen = progress["epoch_seen"] if continuing else 0
        for start in range(epoch_seen, len(order), args.grad_accum):
            if step >= total_steps:
                break
            group = order[start:start + args.grad_accum]
            optimizer.zero_grad(set_to_none=True)
            synchronize(device)
            started = time.perf_counter()
            group_loss = 0.0
            for row in group:
                loss = task.training_loss(model, processor, context, row, args.root, device)
                if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise ValueError("The task must return a finite scalar training loss")
                # Divide by the actual group size, including a short final group.
                (loss / len(group)).backward()
                group_loss += float(loss.detach().item())
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            used_lr = optimizer.param_groups[0]["lr"]
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
                     "examples": len(group), "loss": group_loss / len(group),
                     "seconds": seconds, "lr": used_lr, **memory_report(device)}
            with (args.output / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            if step == 1 or step % 10 == 0 or step == total_steps:
                print(json.dumps(event), flush=True)
            progress = {"phase": "train", "epoch": epoch, "optimizer_steps": step,
                        "epoch_seen": epoch_seen, "epoch_loss": epoch_loss,
                        "examples_seen": seen, "train_seconds": train_seconds, "best_metrics": best_metrics}
            if step % args.save_every == 0 or epoch_seen == len(order) or step >= total_steps:
                save_checkpoint(args.output, model, processor, optimizer, scheduler,
                                config=config, progress=progress, device=device)
            if step >= total_steps:
                break

        predictions, auxiliary, metrics = task.evaluate_rows(model, processor, context, dev, args.root, device)
        task.save_predictions(args.output / f"dev_epoch_{epoch}", predictions, auxiliary, metrics)
        snapshot = {"config": config, "epoch": epoch, "optimizer_steps": step,
                    "epoch_complete": epoch_seen == len(train), "train_loss": epoch_loss / epoch_seen,
                    "dev": metrics, "optimizer_state_saved": False}
        save_adapter(args.output / "last", model, processor, snapshot)
        key = task.selection_key(metrics)
        if best_key is None or key > best_key:
            best_key, best_metrics = key, metrics
            save_adapter(args.output / "best", model, processor, snapshot)
        progress.update(phase="epoch_complete", best_metrics=best_metrics)
        save_checkpoint(args.output, model, processor, optimizer, scheduler,
                        config=config, progress=progress, device=device)
        print(json.dumps({"type": "dev", "epoch": epoch, **metrics}), flush=True)
        if step >= total_steps:
            break
    finish_run(args, config, progress, device)


if __name__ == "__main__":
    main()
