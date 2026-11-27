"""Execute the documented probing commands without changing their experiment."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent
REPO = PROJECT / 'cis530-image2mass'
ENV = dict(os.environ, HF_HOME=str(PROJECT / '.huggingface'), HF_HUB_OFFLINE='1',
           TRANSFORMERS_OFFLINE='1', PYTHONUTF8='1', PYTHONUNBUFFERED='1',
           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
           CUDA_CACHE_PATH=str(PROJECT / '.cuda-cache'))

def write(path, data):
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)

def call(stage, module, arguments):
    done = OUT / (stage + '.time.json')
    if done.exists() and json.loads(done.read_text())['exit_code'] == 0:
        return
    command = [sys.executable, '-u', '-m', module, *map(str, arguments)]
    write(OUT / 'workflow_status.json', {'stage':stage, 'status':'running', 'command':command, 'time':time.time()})
    start = time.monotonic()
    with (OUT / (stage + '.stdout.log')).open('ab', buffering=0) as stdout, (OUT / (stage + '.stderr.log')).open('ab', buffering=0) as stderr:
        process = subprocess.Popen(command, cwd=REPO, env=ENV, stdin=subprocess.DEVNULL,
                                   stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        write(OUT / 'workflow_status.json', {'stage':stage, 'status':'running', 'pid':process.pid, 'command':command, 'time':time.time()})
        code = process.wait()
    write(done, {'exit_code':code, 'seconds':time.monotonic()-start, 'command':command})
    if code:
        raise RuntimeError(f'{stage} exited {code}; inspect {stage}.stderr.log')

def main():
    call('pilot', 'probing.extract', ['--splits','dev','--limit','8','--root',PROJECT/'amazon_data_10k',
                                     '--features',OUT/'pilot'/'features','--output',OUT/'pilot'/'output'])
    pilot = json.loads((OUT/'pilot'/'output'/'extract_dev.json').read_text())
    if pilot['feature_shape'] != [8,29,2048] or pilot['zero_shot_agreement'] != 1.0:
        raise RuntimeError(f'Pilot requires review: {pilot}')
    call('extract', 'probing.extract', ['--root',PROJECT/'amazon_data_10k',
                                      '--features',OUT/'features','--output',OUT/'output'])
    call('probe', 'probing.probe', ['--features',OUT/'features','--output',OUT/'output'])
    write(OUT/'workflow_status.json', {'stage':'complete','status':'complete','time':time.time()})

if __name__ == '__main__':
    main()
