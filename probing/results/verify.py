"""Validate completed outputs against the documented probing protocol."""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys

OUT = Path(__file__).resolve().parent
REPO = next((p for p in OUT.parents if (p/'probing'/'extract.py').is_file()), OUT.parent/'cis530-image2mass')
sys.path.insert(0,str(REPO))
import numpy as np
from finetune.data import read_pairs
from score import score

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def main():
    counts, agreements, agreement_details = {}, {}, {}
    for split, expected in [('train',8000),('dev',1000),('test',1000)]:
        pairs = read_pairs(REPO/'pairs'/f'{split}_pairs.jsonl')
        report = read(OUT/'output'/f'extract_{split}.json')
        with np.load(OUT/'features'/f'{split}.npz') as data:
            assert data['pair_ids'].tolist() == [p['pair_id'] for p in pairs]
            assert data['features'].shape == (expected,29,2048)
            assert data['features'].dtype == np.float32
            assert data['ab_logits'].shape == (expected,2)
            assert np.isfinite(data['features']).all() and np.isfinite(data['ab_logits']).all()
            if split != 'train':
                saved = {p['pair_id']:p['pred'] for p in rows(REPO/'qwen_strong_baseline'/'pairwise'/'output'/split/'preds.jsonl')}
                predicted = ['A' if a > b else 'B' for a,b in data['ab_logits']]
                agreements[split] = sum(saved[p['pair_id']]==v for p,v in zip(pairs,predicted))/expected
                assert report['zero_shot_agreement'] == agreements[split]
                mismatches = [{'pair_id':p['pair_id'],'saved_pred':saved[p['pair_id']],
                               'logit_pred':v,'a_logit':float(z[0]),'b_logit':float(z[1])}
                              for p,v,z in zip(pairs,predicted,data['ab_logits']) if saved[p['pair_id']] != v]
                agreement_details[split] = {'agreement':agreements[split], 'mismatches':mismatches,
                    'all_mismatches_are_ab_ties':all(p['a_logit']==p['b_logit'] for p in mismatches)}
        assert report['pairs'] == expected
        counts[split] = expected
    summary = read(OUT/'output'/'summary.json')
    layers = read(OUT/'output'/'layers.json')
    assert [r['layer'] for r in layers] == list(range(29))
    assert summary['selected'] == max(layers,key=lambda r:(r['dev_accuracy'],-r['dev_log_loss']))
    for split in ('dev','test'):
        gold = {p['pair_id']:p['label'] for p in read_pairs(REPO/'pairs'/f'{split}_pairs.jsonl')}
        preds = rows(OUT/'output'/split/'preds.jsonl')
        assert len(preds) == len(gold) and {p['pair_id'] for p in preds} == set(gold)
        assert all(p['pred'] == ('A' if p['p_a'] > 0.5 else 'B') and 0 <= p['p_a'] <= 1 for p in preds)
        predicted = {p['pair_id']:p['pred'] for p in preds}
        assert score(gold,predicted) == read(OUT/'output'/split/'scores.json')
        subset = {p['pair_id'] for p in read_pairs(REPO/'pairs'/f'{split}_volume_wrong_pairs.jsonl')}
        assert score({k:gold[k] for k in subset},{k:predicted[k] for k in subset}) == read(OUT/'output'/split/'subset_scores.json')
        dims = rows(OUT/'output'/split/'dims_preds.jsonl')
        assert len(dims) == len(gold) and {p['pair_id'] for p in dims} == set(gold)
        assert sum(p['pred']==gold[p['pair_id']] for p in dims)/len(gold) == summary['dims_only'][f'{split}_accuracy']
    result = {'status':'passed','counts':counts,'feature_shape_per_pair':[29,2048],
              'features_finite':True,'dev_test_zero_shot_agreement':agreements,
              'selection_uses_dev_only':True,'rescored_predictions_match':True,
              'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (OUT/'agreement_details.json').write_text(json.dumps(agreement_details,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
