"""Launch a detached supervisor; no terminal needs to remain open."""
import json
from pathlib import Path
import subprocess
import sys

out = Path(__file__).resolve().parent
pythonw = Path(sys.executable).with_name("pythonw.exe")
if not pythonw.is_file():
    raise FileNotFoundError(pythonw)
with (out / "supervisor.stdout.log").open("ab", buffering=0) as stdout, (out / "supervisor.stderr.log").open("ab", buffering=0) as stderr:
    process = subprocess.Popen([str(pythonw), "-u", str(out / "supervise.py")],
                               cwd=out.parent / "pointwise-source", stdin=subprocess.DEVNULL,
                               stdout=stdout, stderr=stderr, close_fds=True,
                               creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_BREAKAWAY_FROM_JOB)
print(json.dumps({"supervisor_pid": process.pid, "status_file": str(out / "supervisor_status.json")}))
