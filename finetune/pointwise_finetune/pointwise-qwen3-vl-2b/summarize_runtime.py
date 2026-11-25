"""Capture immutable elapsed-time evidence from the completed run files."""
import json
from pathlib import Path
from datetime import datetime

out = Path(__file__).resolve().parent
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

active = read(out / "full.time.json")["seconds"]
projected = read(out / "reload/verification.json")["projected_full_training_seconds"]
supervisor = read(out / "supervisor_status.json")
assert supervisor["status"] == "complete" and not supervisor["keep_awake"]
full_wall = (out / "full/run.json").stat().st_mtime - (out / "full.stdout.log").stat().st_ctime
pipeline_wall = datetime.fromisoformat(supervisor["finished_at"]).timestamp() - (out / "supervisor.stdout.log").stat().st_ctime
summary = {"active_full_seconds": active, "full_wall_seconds": full_wall,
           "pipeline_wall_seconds": pipeline_wall, "pilot_projected_full_seconds": projected,
           "projection_error_percent": (active / projected - 1) * 100,
           "idle_repair_interval_seconds": full_wall - active,
           "completed_at_utc": supervisor["finished_at"], "keep_awake_released": True}
(out / "runtime_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
