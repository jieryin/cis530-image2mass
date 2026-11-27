# Frozen Qwen3-VL-2B pairwise probing results

The experiment follows `../probing.md` with unchanged `extract.py` and `probe.py`.
The model, processor, prompts, BF16 precision, 29 layers, and five L2 values use
the repository defaults. Train statistics standardize each probe. Dev accuracy,
then dev log-loss, selects L2 and the final layer; test does not select settings.

| Method | Dev accuracy | Test accuracy | Dev volume-wrong | Test volume-wrong |
|---|---:|---:|---:|---:|
| layer_19 | 84.30% | 85.10% | 41.90% | 41.67% |
| dims_only | 82.60% | 83.30% | 6.15% | 4.17% |
| model_answer | 53.50% | 53.00% | 50.28% | 44.05% |

Selected layer: 19; L2: 0.01.
All 8,000 train, 1,000 dev, and 1,000 test pairs are present. The pilot agrees
with the saved zero-shot predictions at 100%. Full dev/test agreement is
98.90% / 98.70%.
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
