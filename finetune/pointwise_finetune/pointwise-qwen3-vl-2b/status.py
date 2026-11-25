"""Print compact progress; --wait waits at most 55 seconds for useful changes."""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

out = Path(__file__).resolve().parent


def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def snapshot():
    supervisor = read(out / "supervisor_status.json")
    workflow = read(out / "workflow_status.json")
    stage = workflow.get("stage", "starting")
    result = {"stage": stage, "supervisor": supervisor.get("status"),
              "awake": supervisor.get("keep_awake"), "attempt": supervisor.get("attempt")}
    if supervisor.get("heartbeat_at"):
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(supervisor["heartbeat_at"])).total_seconds()
        result["heartbeat_age_seconds"] = round(age)
        if age > 90 and supervisor.get("status") in {"running", "starting", "retry_wait"}:
            result.update(supervisor="stale", awake="unknown")
    directory = out / stage
    history = directory / "history.jsonl"
    if history.is_file():
        events = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
        if events:
            event = events[-1]
            result.update(step=event["optimizer_step"], epoch=event["epoch"], loss=round(event["loss"], 4))
            recent = events[-min(50, len(events)):]
            result["recent_loss"] = round(sum(e["loss"] for e in recent) / len(recent), 4)
            result["seconds_per_update"] = round(sum(e["seconds"] for e in recent) / len(recent), 3)
    pointer = read(directory / "latest_checkpoint.json")
    if pointer.get("checkpoint"):
        checkpoint = directory / pointer["checkpoint"]
        progress = read(checkpoint / "training.json").get("progress", {})
        result["durable_step"] = progress.get("optimizer_steps")
        result["state_exists"] = (checkpoint / "training_state.pt").is_file()
    dev = []
    for path in sorted(directory.glob("dev_epoch_*/scores.json")):
        metrics = read(path)
        dev.append({"epoch": int(path.parent.name.rsplit("_", 1)[1]), "accuracy": metrics.get("accuracy"),
                    "log_mse": metrics.get("object_log_mse")})
    if dev:
        result["dev"] = dev
    if supervisor.get("status") in {"attention", "retry_wait"}:
        result["error_log"] = supervisor.get("attempts", [{}])[-1].get("stderr")
    return result


if __name__ == "__main__":
    if "--wait" in sys.argv:
        initial = snapshot()
        deadline = time.monotonic() + 55
        while time.monotonic() < deadline:
            current = snapshot()
            if current.get("stage") != initial.get("stage") or current.get("supervisor") in {"complete", "attention", "retry_wait", "stale"}:
                break
            time.sleep(min(5, max(0, deadline - time.monotonic())))
    print(json.dumps(snapshot(), ensure_ascii=False))
