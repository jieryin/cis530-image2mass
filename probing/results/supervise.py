"""Run the pointwise workflow independently, with bounded recovery and heartbeats.

Launch this supervisor as a detached process. A stop.request file prevents new
attempts; it deliberately does not interrupt an optimizer update or evaluation.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


RESULTS = Path(__file__).resolve().parent
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def last_history_event(path):
    """Read a bounded tail, tolerating a concurrent, partially written line."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 16384))
            data = handle.read()
        lines = data.decode("utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                return {key: event[key] for key in
                        ("type", "epoch", "optimizer_step", "examples", "loss", "seconds", "lr")
                        if key in event}
    except OSError:
        pass
    return None


def progress_snapshot():
    """Expose durable progress without loading model or optimizer tensors."""
    snapshot = {}
    for stage in ("pilot", "full", "alternative"):
        directory = RESULTS / stage
        if not directory.is_dir():
            continue
        report = {}
        history = directory / "history.jsonl"
        event = last_history_event(history)
        if event is not None:
            report["last_update"] = event
            try:
                report["history_age_seconds"] = round(max(0, time.time() - history.stat().st_mtime), 1)
            except OSError:
                pass
        pointer = read_json(directory / "latest_checkpoint.json")
        if isinstance(pointer, dict) and isinstance(pointer.get("checkpoint"), str):
            checkpoint = (directory / pointer["checkpoint"]).resolve()
            # Only inspect a checkpoint inside this run's checkpoint directory.
            if checkpoint.parent == (directory / "checkpoints").resolve():
                metadata = read_json(checkpoint / "training.json")
                report["checkpoint"] = str(checkpoint)
                report["training_state_exists"] = (checkpoint / "training_state.pt").is_file()
                if isinstance(metadata, dict):
                    report["checkpoint_progress"] = metadata.get("progress")
        run = read_json(directory / "run.json")
        if isinstance(run, dict):
            report["run_status"] = run.get("status")
        snapshot[stage] = report
    return snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RESULTS.parent / "cis530-image2mass")
    parser.add_argument("--workflow", type=Path, default=RESULTS / "workflow.py")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args(argv)
    if os.name != "nt":
        parser.error("This supervisor requires Windows (msvcrt lock and keep-awake API).")
    if not 0 <= args.max_retries <= 3:
        parser.error("--max-retries must be between 0 and 3")
    source, workflow = args.source.resolve(), args.workflow.resolve()
    if not source.is_dir() or not workflow.is_file():
        parser.error(f"Source directory or workflow is missing: {source}; {workflow}")

    import msvcrt

    RESULTS.mkdir(parents=True, exist_ok=True)
    lock = (RESULTS / "supervisor.lock").open("a+b")
    lock.seek(0, os.SEEK_END)
    if lock.tell() == 0:
        lock.write(b"\0")
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        print("Another probing supervisor owns supervisor.lock; no job was started.", file=sys.stderr)
        return 2

    execution_state = ctypes.windll.kernel32.SetThreadExecutionState
    execution_state.argtypes = [ctypes.c_uint]
    execution_state.restype = ctypes.c_uint
    state = {"supervisor_pid": os.getpid(), "started_at": utc_now(),
             "python": sys.executable, "source": str(source), "workflow": str(workflow),
             "max_retries": args.max_retries, "attempts": []}
    stop_path = RESULTS / "stop.request"
    status_path = RESULTS / "supervisor_status.json"
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    child = None

    def heartbeat(status, **extra):
        awake = bool(execution_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
        state.update(status=status, heartbeat_at=utc_now(), keep_awake=awake,
                     stop_requested=stop_path.exists(), progress=progress_snapshot(), **extra)
        try:
            atomic_json(status_path, state)
        except OSError as error:
            print(f"Cannot write supervisor heartbeat: {error}", file=sys.stderr, flush=True)

    try:
        heartbeat("starting")
        for attempt in range(1, args.max_retries + 2):
            if stop_path.exists():
                heartbeat("stopped", finished_at=utc_now(),
                          message="stop.request prevents a new attempt; no child was interrupted.")
                return 0
            stdout_path = RESULTS / f"workflow.{run_tag}.attempt-{attempt:02d}.stdout.log"
            stderr_path = RESULTS / f"workflow.{run_tag}.attempt-{attempt:02d}.stderr.log"
            record = {"attempt": attempt, "started_at": utc_now(),
                      "stdout": str(stdout_path), "stderr": str(stderr_path)}
            state["attempts"].append(record)
            with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open("ab", buffering=0) as stderr:
                child = subprocess.Popen([sys.executable, "-u", str(workflow)], cwd=source,
                                         stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
                record["pid"] = child.pid
                heartbeat("running", attempt=attempt, child_pid=child.pid, exit_code=None,
                          message="Workflow is running.")
                next_heartbeat = time.monotonic() + 30
                stop_seen = False
                while child.poll() is None:
                    stop_present = stop_path.exists()
                    if time.monotonic() >= next_heartbeat or (stop_present and not stop_seen):
                        heartbeat("running_stop_requested" if stop_present else "running",
                                  message=("Current workflow continues; automatic retries are disabled."
                                           if stop_present else "Workflow is running."))
                        next_heartbeat = time.monotonic() + 30
                    stop_seen = stop_seen or stop_present
                    time.sleep(1)
                code = child.returncode
            record.update(exit_code=code, finished_at=utc_now())
            child = None
            if code == 0:
                heartbeat("complete", child_pid=None, exit_code=0, finished_at=utc_now(),
                          message="Workflow exited successfully.")
                return 0
            if stop_seen or stop_path.exists() or attempt > args.max_retries:
                heartbeat("attention", child_pid=None, exit_code=code, finished_at=utc_now(),
                          message=("Workflow failed; stop.request prevents a retry."
                                   if stop_seen or stop_path.exists() else "Workflow failed; retry budget exhausted."))
                return code if 0 < code < 256 else 1
            heartbeat("retry_wait", child_pid=None, exit_code=code,
                      message="Workflow failed; retrying from durable workflow stages in 30 seconds.")
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if stop_path.exists():
                    heartbeat("stopped", finished_at=utc_now(),
                              message="stop.request cancelled the pending retry.")
                    return 0
                time.sleep(min(1, max(0, deadline - time.monotonic())))
        return 1
    except Exception as error:
        heartbeat("attention", error=f"{type(error).__name__}: {error}",
                  child_pid=child.pid if child is not None and child.poll() is None else None,
                  finished_at=utc_now(),
                  message="Supervisor needs attention; any surviving child was left running.")
        raise
    finally:
        execution_state(ES_CONTINUOUS)
        state.update(keep_awake=False, keep_awake_released_at=utc_now())
        try:
            atomic_json(status_path, state)
        except OSError:
            pass
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())

