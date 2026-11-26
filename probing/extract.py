"""Save every layer's answer-position hidden state from frozen zero-shot Qwen.

The model, processor defaults, prompt, and chat template are those of the
pairwise zero-shot baseline, so the A/B logits saved beside the features
should reproduce its saved predictions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from finetune.data import read_pairs
from finetune.model import REVISION, prepare_messages
from finetune.pairwise_finetune.task import answer_token_ids, build_messages_for_pair
from openai_strong_baseline.runner import write_json


def extract(predictor, pairs, root, choices):
    """Return [pairs, layers + 1, hidden] states and [pairs, 2] A/B logits."""
    torch = predictor.torch
    features = logits = None
    with torch.inference_mode():
        for index, pair in enumerate(pairs):
            inputs = prepare_messages(predictor.processor, build_messages_for_pair(pair, root), predictor.device)
            outputs = predictor.model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
            # Index 0 is the embedding output; the last prompt token predicts the answer.
            states = torch.stack([hidden[0, -1] for hidden in outputs.hidden_states])
            if features is None:
                features = np.empty((len(pairs), *states.shape), dtype=np.float32)
                logits = np.empty((len(pairs), 2), dtype=np.float32)
            features[index] = states.float().cpu().numpy()
            logits[index] = outputs.logits[0, -1, list(choices)].float().cpu().numpy()
            if (index + 1) % 500 == 0:
                print(f"  {index + 1}/{len(pairs)}", flush=True)
    return features, logits


def agreement(pair_ids, logits, path):
    """Fraction of saved zero-shot predictions that the A/B logits reproduce."""
    if not path.exists():
        return None
    saved = {row["pair_id"]: row["pred"] for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    preds = ["A" if a > b else "B" for a, b in logits]
    return sum(saved[key] == pred for key, pred in zip(pair_ids, preds)) / len(pair_ids)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["dev", "test", "train"])
    parser.add_argument("--pairs-dir", default="pairs")
    parser.add_argument("--root", default="amazon_data_10k")
    parser.add_argument("--features", default="probing/features")
    parser.add_argument("--output", default="probing/output")
    parser.add_argument("--zero-shot-dir", default="qwen_strong_baseline/pairwise/output")
    parser.add_argument("--limit", type=int, help="first N pairs per split, for a pilot")
    args = parser.parse_args(argv)

    from qwen_strong_baseline.model import QwenPredictor

    predictor = QwenPredictor(revision=REVISION)
    choices = answer_token_ids(predictor.processor)
    for split in args.splits:
        pairs = read_pairs(Path(args.pairs_dir) / f"{split}_pairs.jsonl", args.root)[:args.limit]
        pair_ids = [pair["pair_id"] for pair in pairs]
        path = Path(args.features) / f"{split}.npz"
        if path.exists():
            with np.load(path) as data:
                saved_ids = data["pair_ids"].tolist()
            if saved_ids != pair_ids:
                raise ValueError(f"{path} holds different pairs; move it or choose another --features")
            print(f"{split}: {path} exists, skipping")
            continue
        print(f"{split}: {len(pairs)} pairs", flush=True)
        started = time.perf_counter()
        features, logits = extract(predictor, pairs, args.root, choices)
        elapsed = time.perf_counter() - started
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{split}.tmp.npz")
        np.savez(temporary, pair_ids=np.array(pair_ids), features=features, ab_logits=logits)
        temporary.replace(path)
        report = {
            "split": split, "pairs": len(pairs), "feature_shape": list(features.shape),
            "elapsed_seconds": elapsed,
            "zero_shot_agreement": agreement(pair_ids, logits, Path(args.zero_shot_dir) / split / "preds.jsonl"),
            "answer_token_ids": list(choices), "model_metadata": predictor.metadata,
        }
        write_json(Path(args.output) / f"extract_{split}.json", report)
        print(f"{split}: done in {elapsed:.0f}s, zero-shot agreement {report['zero_shot_agreement']}", flush=True)


if __name__ == "__main__":
    main()
