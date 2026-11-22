"""Atomic, bounded training checkpoints, including optimizer and RNG state."""

from __future__ import annotations

import json
import random
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from openai_strong_baseline.runner import write_json


FORMAT = "finetune_training_v1"
_CHECKPOINT_NAME = re.compile(r"checkpoint-\d+-(?:train|epoch_complete)-[0-9a-f]{32}\Z")


def _rng_state(device):
    import torch

    device = torch.device(device)
    state = {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
             "device_type": device.type}
    if device.type == "cuda":
        state["accelerator"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        state["accelerator"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state, device):
    """Restore generators after model/optimizer construction, before training."""
    import torch

    device = torch.device(device)
    if state["device_type"] != device.type:
        raise ValueError("Resume requires the same device type to restore RNG state")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["accelerator"], device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state["accelerator"])


def _metadata(path):
    metadata = json.loads((path / "training.json").read_text(encoding="utf-8"))
    if metadata.get("format") != FORMAT:
        raise ValueError(f"Not a resumable training checkpoint: {path}; best/last adapters alone cannot resume training")
    return metadata


def _prune_checkpoints(directory, current):
    """Retain two completed checkpoints; ignore unrelated or partial files."""
    completed = []
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_dir() or not _CHECKPOINT_NAME.fullmatch(path.name):
            continue
        try:
            metadata = _metadata(path)
        except (OSError, ValueError, AttributeError):
            continue
        created = metadata.get("created_at_ns")
        if type(created) is int and (path / "training_state.pt").is_file():
            completed.append((created, path))
    newest_other = sorted((entry for entry in completed if entry[1] != current), reverse=True)
    for _, path in newest_other[1:]:
        shutil.rmtree(path)


def save_adapter(path, model, processor, metadata):
    """Save the adapter, its processor, and task/training metadata together."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    write_json(path / "training.json", metadata)


def save_checkpoint(output, model, processor, optimizer, scheduler, *, config, progress, device):
    """Commit all state before publishing a latest pointer; retain two saves.

    Call only after a complete optimizer update, with no pending accumulated
    gradients. The progress cursor identifies the next example to process (or the
    pending dev evaluation). A failed save never replaces the previous pointer.
    """
    import torch

    output = Path(output).resolve()
    directory = output / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    phase, step = progress["phase"], progress["optimizer_steps"]
    name = f"checkpoint-{step:08d}-{phase}-{uuid.uuid4().hex}"
    destination = directory / name
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=directory))
    try:
        save_adapter(temporary, model, processor, {
            "format": FORMAT, "created_at_ns": time.time_ns(),
            "optimizer_state_saved": True, "config": config, "progress": progress,
        })
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "rng": _rng_state(device)}, temporary / "training_state.pt")
        temporary.replace(destination)
        write_json(output / "latest_checkpoint.json", {"checkpoint": f"checkpoints/{name}"})
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    _prune_checkpoints(directory, destination)
    return destination


def load_checkpoint(output):
    """Load the latest completed checkpoint in a run directory."""
    import torch

    output = Path(output).resolve()
    pointer = json.loads((output / "latest_checkpoint.json").read_text(encoding="utf-8"))
    path = output / pointer["checkpoint"]
    if (path.is_symlink() or path.resolve().parent != output / "checkpoints"
            or not _CHECKPOINT_NAME.fullmatch(path.name)):
        raise ValueError("Invalid latest checkpoint pointer: expected a completed checkpoint in this run")
    metadata = _metadata(path)
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    return path.resolve(), metadata, state
