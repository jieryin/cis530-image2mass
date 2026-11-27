"""Start a detached monitoring process with persistent logs."""
from pathlib import Path
import json
import subprocess
import sys

out = Path(__file__).resolve().parent
pythonw = Path(sys.executable).with_name('pythonw.exe')
with (out/'supervisor.stdout.log').open('ab',buffering=0) as stdout, (out/'supervisor.stderr.log').open('ab',buffering=0) as stderr:
    child = subprocess.Popen([str(pythonw),'-u',str(out/'supervise.py'),'--source',str(out.parent/'cis530-image2mass')],
        cwd=out,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,close_fds=True,
        creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB)
print(json.dumps({'pid':child.pid,'status':str(out/'supervisor_status.json')}))
