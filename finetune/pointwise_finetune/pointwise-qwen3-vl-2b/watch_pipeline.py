"""Read-only progress stream for the current Codex turn; training runs independently."""
import json
import time
from status import snapshot

while True:
    current = snapshot()
    print(json.dumps(current, ensure_ascii=False), flush=True)
    if current.get("supervisor") in {"complete", "attention", "stale", "stopped"}:
        break
    deadline = time.monotonic() + 55
    while time.monotonic() < deadline:
        time.sleep(min(5, max(0, deadline - time.monotonic())))
        updated = snapshot()
        if updated.get("stage") != current.get("stage") or updated.get("supervisor") != current.get("supervisor") or updated.get("dev") != current.get("dev"):
            break
