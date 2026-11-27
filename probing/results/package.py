"""Archive all completed run artifacts under probing/results."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess

OUT = Path(__file__).resolve().parent
REPO = OUT.parent/'cis530-image2mass'

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    assert json.loads((OUT/'verification.json').read_text())['status'] == 'passed'
    assert json.loads((OUT/'supervisor_status.json').read_text())['status'] == 'complete'
    destination = REPO/'probing'/'results'
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(OUT,destination,ignore=shutil.ignore_patterns('__pycache__','*.pending'))
    summary = json.loads((destination/'output'/'summary.json').read_text())
    layers = json.loads((destination/'output'/'layers.json').read_text())
    agreement = json.loads((destination/'verification.json').read_text())['dev_test_zero_shot_agreement']
    with (destination/'layers.csv').open('w',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(layers[0]))
        writer.writeheader()
        writer.writerows(layers)
    table = ['| Method | Dev accuracy | Test accuracy | Dev volume-wrong | Test volume-wrong |',
             '|---|---:|---:|---:|---:|']
    for key in ('selected','dims_only','model_answer'):
        row=summary[key]
        table.append('| '+row['name']+' | '+' | '.join(f'{100*row[k]:.2f}%' for k in
            ('dev_accuracy','test_accuracy','dev_volume_wrong_accuracy','test_volume_wrong_accuracy'))+' |')
    readme = '''# Frozen Qwen3-VL-2B pairwise probing results

The experiment follows `../probing.md` with unchanged `extract.py` and `probe.py`.
The model, processor, prompts, BF16 precision, 29 layers, and five L2 values use
the repository defaults. Train statistics standardize each probe. Dev accuracy,
then dev log-loss, selects L2 and the final layer; test does not select settings.

'''+ '\n'.join(table)+f'''

Selected layer: {summary['selected']['layer']}; L2: {summary['selected']['l2']}.
All 8,000 train, 1,000 dev, and 1,000 test pairs are present. The pilot agrees
with the saved zero-shot predictions at 100%. Full dev/test agreement is
{100*agreement['dev']:.2f}% / {100*agreement['test']:.2f}%.
`agreement_details.json` records every mismatch and checks whether A/B logits
are tied: the documented extraction/probe rule selects B on ties, while greedy
generation selects the lower token ID A when these are the maximal logits.
`verification.json` records finite feature checks, exact pair alignment, layer
selection verification, and independent rescoring of full and volume-wrong sets.

## Artifacts

- `features/`: complete float32 states, pair IDs and A/B logits for all splits.
- `output/`: extraction metadata, all 29 layer results, summary, predictions and scores.
- `pilot/`: original eight-pair pilot states and metadata.
- `layers.csv`: the full per-layer table in CSV form.
- `*.log`, `*.time.json`, supervisor/workflow status and scripts: full execution record.
- `manifest.json`: byte sizes and SHA-256 hashes of every other archived file.

The large `.npz` files use Git LFS. Run `git lfs pull` after cloning to retrieve
the full feature arrays. They are retained rather than omitted from this delivery.
The monitoring scripts record their original local execution paths; reproduce
the experiment using the repository entry points below from the repository root.

```sh
python -m probing.extract --root ../amazon_data_10k --features probing/reproduced/features --output probing/reproduced/output
python -m probing.probe --features probing/reproduced/features --output probing/reproduced/output
python score.py pairs/test_pairs.jsonl probing/results/output/test/preds.jsonl
```

The detached supervisor kept the system awake, wrote 30-second heartbeats, and
allowed three bounded retries. Completed extraction splits are reusable. This
protects against terminal closure and transient process failures; a power loss
still requires restarting the supervisor and redoing any unfinished split.
'''
    (destination/'README.md').write_text(readme,encoding='utf-8')
    provenance = {'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                  'source_files':{str(p.relative_to(REPO)):sha256(p) for p in [REPO/'probing'/'extract.py',REPO/'probing'/'probe.py',REPO/'probing'/'probing.md',*sorted((REPO/'pairs').glob('*.jsonl'))]}}
    (destination/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    files=[{'path':p.relative_to(destination).as_posix(),'bytes':p.stat().st_size,'sha256':sha256(p)}
           for p in sorted(destination.rglob('*')) if p.is_file()]
    (destination/'manifest.json').write_text(json.dumps({'files':files,'total_bytes':sum(f['bytes'] for f in files)},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'destination':str(destination),'files':len(files)+1,'bytes':sum(f['bytes'] for f in files)}))

if __name__ == '__main__':
    main()
