"""Use the pinned local Qwen cache and instrument only the short pilot."""
import json
import math
import os
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parent / "pointwise-source"
sys.path.insert(0, str(REPO))
from finetune import model, train, evaluate

stage = sys.argv.pop(1)
import msvcrt
gpu_lock = (OUT / "gpu.lock").open("a+b")
if gpu_lock.seek(0, 2) == 0:
    gpu_lock.write(b"0")
    gpu_lock.flush()
gpu_lock.seek(0)
try:
    msvcrt.locking(gpu_lock.fileno(), msvcrt.LK_NBLCK, 1)
except OSError:
    raise SystemExit("Another pointwise model process owns gpu.lock; refusing a duplicate GPU job.")
snapshot = Path(os.environ["HF_HOME"]) / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots" / model.REVISION
if not snapshot.is_dir():
    raise FileNotFoundError(snapshot)
# Keep the canonical model ID in run metadata; only resolve weight loading locally.
model.MODEL_ID = str(snapshot)

if stage.startswith("pilot"):
    import torch
    original_clip = torch.nn.utils.clip_grad_norm_
    original_save = train.save_checkpoint

    def clip_and_record(parameters, *args, **kwargs):
        parameters = list(parameters)
        norm = original_clip(parameters, *args, **kwargs)
        value = float(norm)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid pilot gradient norm: {value}")
        with (OUT / "pilot_gradient.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"gradient_norm": value}) + "\n")
        return norm

    def save_and_interrupt(*args, **kwargs):
        path = original_save(*args, **kwargs)
        progress = kwargs["progress"]
        if stage == "pilot-interrupt" and progress["optimizer_steps"] == 10:
            (OUT / "pilot_interruption.json").write_text(json.dumps({
                "checkpoint": str(path), "optimizer_step": 10,
                "reason": "Controlled process exit after a durable checkpoint; resume must reach step 20.",
            }), encoding="utf-8")
            raise SystemExit(75)
        return path

    torch.nn.utils.clip_grad_norm_ = clip_and_record
    train.save_checkpoint = save_and_interrupt

if stage in {"reload", "test"}:
    evaluate.main()
else:
    train.main()
