"""Layer-wise linear probes for "is A heavier?" on frozen zero-shot Qwen states.

Each layer gets an L2-regularized logistic regression fit on train. The L2
strength and then the layer are chosen by dev accuracy, breaking ties with
lower dev log-loss; test is reported for every layer so the sweep shows dev
and test together. A probe on the listed dimensions alone is the control for
information the prompt already states, and the A/B logits at the same
position give the model's own answer.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from finetune.data import read_pairs
from openai_strong_baseline.runner import write_json, write_jsonl
from score import score


L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
SPLITS = ("train", "dev", "test")


def fit_logistic(x, y, l2):
    """Full-batch L-BFGS logistic regression; the bias is not penalized."""
    import torch
    import torch.nn.functional as F

    w = torch.zeros(x.shape[1], dtype=x.dtype, requires_grad=True)
    b = torch.zeros((), dtype=x.dtype, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ w + b, y) + 0.5 * l2 * w.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return w.detach(), b.detach()


def dims_features(pairs):
    """Sorted log dimensions of A then B; the volume rule is one linear direction."""
    return np.array([[math.log(d) for side in ("a", "b") for d in sorted(pair[f"dims_{side}"])]
                     for pair in pairs])


def load_split(features_dir, pairs_dir, split):
    pairs = read_pairs(Path(pairs_dir) / f"{split}_pairs.jsonl")
    with np.load(Path(features_dir) / f"{split}.npz") as data:
        pair_ids, features, logits = data["pair_ids"].tolist(), data["features"], data["ab_logits"]
    if pair_ids != [pair["pair_id"] for pair in pairs]:
        raise ValueError(f"{split}: features do not match {split}_pairs.jsonl; rerun extract without --limit")
    subset = set()
    if split != "train":
        subset = {pair["pair_id"] for pair in read_pairs(Path(pairs_dir) / f"{split}_volume_wrong_pairs.jsonl")}
    return pairs, features, logits, subset


def main(argv=None):
    import torch
    import torch.nn.functional as F

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="probing/features")
    parser.add_argument("--pairs-dir", default="pairs")
    parser.add_argument("--output", default="probing/output")
    args = parser.parse_args(argv)

    pairs, features, readout, subsets = {}, {}, {}, {}
    for split in SPLITS:
        pairs[split], features[split], readout[split], subsets[split] = load_split(args.features, args.pairs_dir, split)
    y = {s: torch.tensor([p["label"] == "A" for p in pairs[s]], dtype=torch.float64) for s in SPLITS}
    hard = {s: torch.tensor([p["pair_id"] in subsets[s] for p in pairs[s]]) for s in ("dev", "test")}

    def summarize(name, logits, l2=None):
        row = {"name": name, "l2": l2,
               "dev_log_loss": F.binary_cross_entropy_with_logits(logits["dev"], y["dev"]).item()}
        for s in logits:
            row[f"{s}_accuracy"] = ((logits[s] > 0) == (y[s] > 0.5)).double().mean().item()
        for s in hard:
            row[f"{s}_volume_wrong_accuracy"] = ((logits[s][hard[s]] > 0) == (y[s][hard[s]] > 0.5)).double().mean().item()
        return row

    def key(row):
        return row["dev_accuracy"], -row["dev_log_loss"]

    def probe(name, x):
        x = {s: torch.as_tensor(x[s], dtype=torch.float64) for s in SPLITS}
        mean, std = x["train"].mean(0), x["train"].std(0)
        x = {s: (x[s] - mean) / torch.where(std > 0, std, torch.ones_like(std)) for s in SPLITS}
        best = None
        for l2 in L2_GRID:
            w, b = fit_logistic(x["train"], y["train"], l2)
            logits = {s: x[s] @ w + b for s in SPLITS}
            row = summarize(name, logits, l2)
            if best is None or key(row) > key(best[0]):
                best = row, logits
        return best

    layers, selected = [], None
    for layer in range(features["train"].shape[1]):
        row, logits = probe(f"layer_{layer}", {s: features[s][:, layer] for s in SPLITS})
        layers.append({"layer": layer, **row})
        print(f"layer {layer:2d}  l2 {row['l2']:g}  dev {row['dev_accuracy']:.3f}  test {row['test_accuracy']:.3f}  "
              f"dev-vw {row['dev_volume_wrong_accuracy']:.3f}  test-vw {row['test_volume_wrong_accuracy']:.3f}", flush=True)
        if selected is None or key(row) > key(selected[0]):
            selected = layers[-1], logits
    dims = probe("dims_only", {s: dims_features(pairs[s]) for s in SPLITS})
    model_answer = summarize("model_answer", {s: torch.as_tensor(readout[s][:, 0] - readout[s][:, 1], dtype=torch.float64)
                                              for s in SPLITS})

    output = Path(args.output)
    for split in ("dev", "test"):
        gold = {p["pair_id"]: p["label"] for p in pairs[split]}
        for filename, logits in (("preds.jsonl", selected[1][split]), ("dims_preds.jsonl", dims[1][split])):
            write_jsonl(output / split / filename, (
                {"pair_id": p["pair_id"], "pred": "A" if z > 0 else "B", "p_a": torch.sigmoid(z).item()}
                for p, z in zip(pairs[split], logits)))
        preds = {p["pair_id"]: "A" if z > 0 else "B" for p, z in zip(pairs[split], selected[1][split])}
        write_json(output / split / "scores.json", score(gold, preds))
        write_json(output / split / "subset_scores.json",
                   score({k: gold[k] for k in subsets[split]}, {k: preds[k] for k in subsets[split]}))
    write_json(output / "layers.json", layers)
    write_json(output / "summary.json", {"selected": selected[0], "dims_only": dims[0],
                                         "model_answer": model_answer, "l2_grid": list(L2_GRID)})
    print(f"selected layer {selected[0]['layer']}: test {selected[0]['test_accuracy']:.3f}, "
          f"test-vw {selected[0]['test_volume_wrong_accuracy']:.3f}; dims-only test {dims[0]['test_accuracy']:.3f}, "
          f"test-vw {dims[0]['test_volume_wrong_accuracy']:.3f}; model answer test {model_answer['test_accuracy']:.3f}")


if __name__ == "__main__":
    main()
