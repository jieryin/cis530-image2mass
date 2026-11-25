"""Pilot, verify recovery/reload, train defaults, select on dev, then evaluate test."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent
REPO = PROJECT / "pointwise-source"
sys.path.insert(0, str(REPO))
from finetune.data import read_pairs, unique_objects
from finetune.checkpoint import load_checkpoint
from score import score


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def status(stage, **details):
    write(OUT / "workflow_status.json", {"stage": stage, "pid": os.getpid(), "time": time.time(), **details})
    print(json.dumps({"stage": stage, **details}), flush=True)


def environment():
    env = dict(os.environ, HF_HOME=str(PROJECT / ".huggingface"), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", PYTHONUTF8="1", PYTHONUNBUFFERED="1",
               TEMP=str(PROJECT / ".pip-cache"), TMP=str(PROJECT / ".pip-cache"),
               TORCH_HOME=str(PROJECT / ".torch"), CUDA_CACHE_PATH=str(PROJECT / ".cuda-cache"),
               TORCHINDUCTOR_CACHE_DIR=str(PROJECT / ".torchinductor"))
    return env


def call(stage, arguments, expected=(0,)):
    status(stage, status="running")
    started = time.monotonic()
    with (OUT / f"{stage}.stdout.log").open("a", encoding="utf-8") as stdout, (OUT / f"{stage}.stderr.log").open("a", encoding="utf-8") as stderr:
        code = subprocess.call([sys.executable, "-u", str(OUT / "run_model.py"), stage,
                                *map(str, arguments), "--root", str(PROJECT / "amazon_data_10k"),
                                "--device", "cuda"], cwd=REPO, env=environment(),
                               stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    timing_path = OUT / f"{stage}.time.json"
    old = read(timing_path) if timing_path.is_file() else {}
    write(timing_path, {"seconds": old.get("seconds", 0) + time.monotonic() - started,
                        "exit_code": code, "attempts": old.get("attempts", 0) + 1})
    if code not in expected:
        raise RuntimeError(f"{stage} exited {code}; see {OUT / (stage + '.stderr.log')}")
    return code


def complete_training(name):
    path = OUT / name / "run.json"
    return path.is_file() and read(path).get("status") in {"complete", "pilot_complete"}


def train_run(name):
    if complete_training(name):
        return
    directory = OUT / name
    if (directory / "latest_checkpoint.json").is_file():
        load_checkpoint(directory)  # Fail before spawning if the durable state is corrupt.
        args = ["--resume", directory, "--save-every", 5 if name == "pilot" else 25]
        stage = name
    elif directory.exists():
        raise RuntimeError(f"Incomplete initialization without a checkpoint: {directory}")
    else:
        args = ["--task", "pointwise", "--output", directory, "--save-every", 25]
        stage = name
        if name == "pilot":
            stage = "pilot-interrupt"
            args += ["--dev", OUT / "pilot_dev.jsonl", "--max-steps", 20, "--save-every", 5]
        elif name == "alternative":
            args += ["--lr", "0.00005"]
    if stage == "pilot-interrupt":
        call(stage, args, expected=(75,))
        checkpoint, metadata, state = load_checkpoint(directory)
        if metadata["progress"]["optimizer_steps"] != 10 or not state["optimizer"]["state"]:
            raise RuntimeError("Controlled interruption did not preserve step 10 optimizer state")
        del state
        call("pilot", ["--resume", directory, "--save-every", 5])
    else:
        call(stage, args)
    if not complete_training(name):
        raise RuntimeError(f"Training did not finish: {name}")


def evaluate(name, checkpoint, pairs, subset=None):
    output = OUT / name
    if (output / "run.json").is_file() and read(output / "run.json").get("status") == "complete":
        return
    # Keep incomplete evaluation attempts; each retry writes a new destination.
    attempt = 1
    pending = OUT / f".{name}-attempt-{attempt}"
    while pending.exists():
        attempt += 1
        pending = OUT / f".{name}-attempt-{attempt}"
    args = ["--checkpoint", checkpoint, "--pairs", pairs, "--output", pending]
    if subset:
        args += ["--subset", subset]
    call(name, args)
    if output.exists():
        raise RuntimeError(f"Evaluation output is incomplete and requires review: {output}")
    pending.replace(output)


def verify_pilot():
    status("pilot-verification", status="running")
    history = rows(OUT / "pilot/history.jsonl")
    norms = [r["gradient_norm"] for r in rows(OUT / "pilot_gradient.jsonl")]
    assert [h["optimizer_step"] for h in history] == list(range(1, 21)), "Recovery duplicated/lost updates"
    assert len(norms) == 20 and all(math.isfinite(n) and n > 0 for n in norms)
    assert all(math.isfinite(h["loss"]) for h in history)
    checkpoint, metadata, state = load_checkpoint(OUT / "pilot")
    assert metadata["progress"]["optimizer_steps"] == 20
    assert state["optimizer"]["state"] and state["scheduler"] and state["rng"]
    from safetensors import safe_open
    with safe_open(str(checkpoint / "adapter_model.safetensors"), framework="pt", device="cpu") as adapter:
        keys = list(adapter.keys())
        assert any("mass_head" in k for k in keys), "Missing mass head in checkpoint"
        assert any("lora_" in k for k in keys), "Missing LoRA in checkpoint"
    initial = {r["pair_id"]: r["pred"] for r in rows(OUT / "pilot/dev_epoch_1/preds.jsonl")}
    restored = {r["pair_id"]: r["pred"] for r in rows(OUT / "reload/preds.jsonl")}
    assert initial == restored and len(initial) == 32, "Reload changed pilot pair predictions"
    before = {r["id"]: r["weight_lb"] for r in rows(OUT / "pilot/dev_epoch_1/weights.jsonl")}
    after = {r["id"]: r["weight_lb"] for r in rows(OUT / "reload/weights.jsonl")}
    assert before.keys() == after.keys()
    assert all(math.isclose(v, after[k], rel_tol=1e-6, abs_tol=1e-7) for k, v in before.items()), "Reload changed mass predictions"
    train_objects = len(unique_objects(read_pairs(REPO / "pairs/train_pairs.jsonl")))
    dev_objects = len(unique_objects(read_pairs(REPO / "pairs/dev_pairs.jsonl")))
    update_seconds = statistics.mean(h["seconds"] for h in history)
    dev_metrics = read(OUT / "reload/scores.json")
    projected = math.ceil(train_objects / 8) * 3 * update_seconds + dev_metrics["elapsed_seconds"] / len(after) * dev_objects * 3
    result = {"passed": True, "controlled_resume_from_step": 10, "final_step": 20,
              "matching_pair_predictions": len(initial), "matching_object_weights": len(before),
              "mass_head_and_lora_saved": True, "optimizer_scheduler_rng_loaded": True,
              "loss_first_5_mean": statistics.mean(h["loss"] for h in history[:5]),
              "loss_last_5_mean": statistics.mean(h["loss"] for h in history[-5:]),
              "gradient_norm_min": min(norms), "gradient_norm_max": max(norms),
              "peak_allocated_gib": max(h["peak_allocated_bytes"] for h in history) / 2**30,
              "mean_update_seconds": update_seconds, "projected_full_training_seconds": projected,
              "expected_full_updates": math.ceil(train_objects / 8) * 3}
    write(OUT / "reload/verification.json", result)
    status("pilot-verified", **result)


def main():
    if (OUT / "stop.request").exists():
        raise RuntimeError("stop.request is present; workflow will not start new stages")
    if not (OUT / "pilot_dev.jsonl").is_file():
        # Exactly the same 32-pair dev slice as the established pairwise pilot.
        (OUT / "pilot_dev.jsonl").write_bytes((PROJECT / "pairwise-results/pilot_dev.jsonl").read_bytes())
    train_run("pilot")
    evaluate("reload", OUT / "pilot/best", OUT / "pilot_dev.jsonl")
    verify_pilot()
    # Pause here only if the actual pilot failed; successful verification authorizes defaults.
    if (OUT / "stop.request").exists():
        raise RuntimeError("stop.request prevents full training")
    train_run("full")
    dev = {r["pair_id"]: r["label"] for r in rows(REPO / "pairs/dev_pairs.jsonl")}
    baseline = {r["pair_id"]: r["pred"] for r in rows(REPO / "qwen_strong_baseline/pointwise/output/dev/preds.jsonl")}
    zero_accuracy = score(dev, baseline)["accuracy"]
    volume_accuracy = 1 - len(rows(REPO / "pairs/dev_volume_wrong_pairs.jsonl")) / len(dev)

    def result_key(name):
        metrics = read(OUT / name / "best/training.json")["dev"]
        return metrics["accuracy"], -metrics["object_log_mse"]

    names = ["full"]
    if result_key("full")[0] <= max(zero_accuracy, volume_accuracy):
        decision = (f"Default best dev accuracy did not exceed the stronger baseline ({max(zero_accuracy, volume_accuracy):.1%}); "
                    "run one predeclared alternative at LR 5e-5, seed 42, and select by dev accuracy then lower object log-MSE.")
        if (OUT / "stop.request").exists():
            raise RuntimeError("stop.request prevents alternative training")
        train_run("alternative")
        names.append("alternative")
    else:
        decision = "Default best dev accuracy exceeded both baselines; retain one seed and stop tuning."
    selected = max(names, key=result_key)
    write(OUT / "selection.json", {"selected": selected, "names": names, "decision": decision,
                                   "zero_shot_dev_accuracy": zero_accuracy, "volume_dev_accuracy": volume_accuracy})
    evaluate("test", OUT / selected / "best", REPO / "pairs/test_pairs.jsonl", REPO / "pairs/test_volume_wrong_pairs.jsonl")
    from report import report
    report()
    status("complete", selected=selected, report=str(OUT / "analysis.md"), test=read(OUT / "test/scores.json"))


if __name__ == "__main__":
    import msvcrt
    workflow_lock = (OUT / "workflow.lock").open("a+b")
    if workflow_lock.seek(0, 2) == 0:
        workflow_lock.write(b"0")
        workflow_lock.flush()
    workflow_lock.seek(0)
    try:
        msvcrt.locking(workflow_lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit("Another workflow still holds workflow.lock; no duplicate stages were launched.")
    main()
