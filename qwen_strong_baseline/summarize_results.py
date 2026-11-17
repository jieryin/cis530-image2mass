"""Re-score saved baselines and export a comparison with factual error examples.

This utility reads predictions, scores and run metadata only. It never imports
the model, requests predictions, or changes inference caches/configuration.
Missing Qwen results stay pending; --require-complete requires all four runs.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASELINE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score import ScoringError, load_labels, print_report, score


class ResultsError(ValueError):
    """Saved outputs disagree with the strict scoring or run contract."""


SPLITS = ("dev", "test")
NAMES = {
    "majority": "Majority class",
    "volume": "Volume rule",
    "openai_pairwise": "OpenAI pairwise",
    "openai_pointwise": "OpenAI pointwise",
    "qwen_pairwise": "Qwen pairwise",
    "qwen_pointwise": "Qwen pointwise",
}


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResultsError(f"Cannot read {path}: {exc}") from exc


def read_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise ResultsError(f"{path}:{number}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ResultsError(f"{path}:{number}: expected an object")
                rows.append(row)
    return rows


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def displayed_path(path, repo):
    path = Path(path).resolve()
    return path.relative_to(repo).as_posix() if path.is_relative_to(repo) else str(path)


def verified_scores(gold, prediction_path, scores_path=None):
    predictions = load_labels(prediction_path, "pred")
    actual = score(gold, predictions)
    if scores_path is not None and actual != read_json(scores_path):
        raise ResultsError(f"Saved scores differ from the unchanged scorer: {scores_path}")
    return actual, predictions


def load_weights(path, expected_ids):
    weights = {}
    for row in read_rows(path):
        object_id, weight = row.get("id"), row.get("weight_lb")
        if type(object_id) is not int or object_id in weights:
            raise ResultsError(f"Invalid/duplicate object ID in {path}: {object_id!r}")
        if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(weight) or weight <= 0):
            raise ResultsError(f"Invalid positive finite weight in {path}: {weight!r}")
        weights[object_id] = weight
    if set(weights) != expected_ids:
        raise ResultsError(f"Pointwise weight IDs do not exactly cover the objects: {path}")
    return weights


def simple_results(repo, pair_rows, gold):
    """Reuse the baseline's functions; reproduce missing dev predictions in memory."""
    source = repo / "simple_baseline" / "simple-baseline.py"
    spec = importlib.util.spec_from_file_location("image2mass_simple_results", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    majority = module.training_majority(module.read_jsonl(repo / "pairs" / "train_pairs.jsonl"))
    result = {"majority": {}, "volume": {}}
    for method in result:
        for split in SPLITS:
            predictions = {}
            for pair in pair_rows[split]:
                pred = majority
                if method == "volume":
                    a, b = (module.bounding_box_volume(pair, side) for side in ("a", "b"))
                    if a != b:
                        pred = "A" if a > b else "B"
                predictions[pair["pair_id"]] = pred
            metrics = score(gold[split], predictions)
            provenance = {"implementation": displayed_path(source, repo),
                          "implementation_sha256": sha256(source),
                          "training_majority_label": majority,
                          "source": "recomputed existing deterministic rule in memory"}
            if split == "test":
                filename = "simple_preds.jsonl" if method == "majority" else "volume_preds.jsonl"
                saved = repo / "simple_baseline" / filename
                if saved.exists():
                    saved_metrics, saved_predictions = verified_scores(gold[split], saved)
                    if saved_predictions != predictions:
                        raise ResultsError(f"Saved simple baseline differs from its existing rule: {saved}")
                    metrics = saved_metrics
                    provenance.update(source="saved predictions, validated against the existing rule",
                                      predictions=displayed_path(saved, repo),
                                      predictions_sha256=sha256(saved))
            result[method][split] = {"status": "complete", "scores": metrics,
                                     "provenance": provenance}
    return result


def qwen_runtime(run):
    keys = ("required_inferences", "completed_inferences", "remaining_inferences", "invalid_outputs",
            "inference_seconds", "mean_inference_seconds", "peak_memory_allocated_bytes", "tie_pairs",
            "timing_policy")
    return {key: run[key] for key in keys if key in run}


def qwen_result(repo, qwen_dir, mode, split, pairs, gold):
    folder = qwen_dir / mode / "output" / split
    run_path = folder / "run.json"
    run = read_json(run_path) if run_path.exists() else None
    final_paths = [folder / name for name in ("preds.jsonl", "scores.json", "scores.txt")]
    complete = isinstance(run, dict) and run.get("status") == "complete"
    if not complete:
        if any(path.exists() for path in final_paths):
            raise ResultsError(f"Incomplete/missing run metadata has final predictions or scores: {folder}")
        return ({"status": "pending", "scores": None,
                 "run_status": run.get("status") if isinstance(run, dict) else "not started",
                 "runtime": qwen_runtime(run) if isinstance(run, dict) else {},
                 "provenance": {"run": displayed_path(run_path, repo)}}, None, None, run)
    if not all(path.is_file() for path in final_paths):
        raise ResultsError(f"Complete run is missing predictions or score files: {folder}")
    metrics, predictions = verified_scores(gold, final_paths[0], final_paths[1])
    expected_text = io.StringIO()
    with contextlib.redirect_stdout(expected_text):
        print_report(metrics)
    if final_paths[2].read_text(encoding="utf-8") != expected_text.getvalue():
        raise ResultsError(f"Saved text score report differs from the unchanged scorer: {final_paths[2]}")
    expected_ids = {pair[f"id_{side}"] for pair in pairs for side in ("a", "b")}
    expected_requests = len(pairs) if mode == "pairwise" else len(expected_ids)
    expected_fields = {"mode": mode, "split": split, "pair_count": len(pairs),
                       "unique_objects": len(expected_ids), "required_inferences": expected_requests,
                       "completed_inferences": expected_requests, "remaining_inferences": 0,
                       "invalid_outputs": 0,
                       "pairs_sha256": sha256(repo / "pairs" / f"{split}_pairs.jsonl")}
    for field, expected in expected_fields.items():
        if run.get(field) != expected:
            raise ResultsError(f"{run_path}: {field}={run.get(field)!r}; expected {expected!r}")
    for field in ("inference_seconds", "mean_inference_seconds", "peak_memory_allocated_bytes"):
        value = run.get(field)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ResultsError(f"{run_path}: invalid {field}")
    if not math.isclose(run["mean_inference_seconds"], run["inference_seconds"] / expected_requests,
                        rel_tol=1e-9, abs_tol=1e-9):
        raise ResultsError(f"{run_path}: mean inference time is inconsistent")
    weights = None
    if mode == "pointwise":
        weights_path = folder / "weight_preds.jsonl"
        weights = load_weights(weights_path, expected_ids)
        converted = {pair["pair_id"]: ("A" if weights[pair["id_a"]] > weights[pair["id_b"]] else "B")
                     for pair in pairs}
        if predictions != converted:
            raise ResultsError(f"Pointwise A/B predictions do not match saved weights: {folder}")
        ties = sum(weights[p["id_a"]] == weights[p["id_b"]] for p in pairs)
        if run.get("tie_pairs") != ties:
            raise ResultsError(f"{run_path}: tie_pairs disagrees with saved weights")
    return ({"status": "complete", "scores": metrics, "runtime": qwen_runtime(run),
             "provenance": {"predictions": displayed_path(final_paths[0], repo),
                            "predictions_sha256": sha256(final_paths[0]),
                            "scores": displayed_path(final_paths[1], repo),
                            "run": displayed_path(run_path, repo)},
             "model": run.get("config", {}).get("model"),
             "revision": run.get("config", {}).get("revision")}, predictions, weights, run)


def error_examples(pairs, predictions, weights, limit):
    """Report measurements for manual review without attributing visual causes."""
    def example(pair):
        a, b = pair["id_a"], pair["id_b"]
        volume_a, volume_b = (math.prod(pair[f"dims_{side}"]) for side in ("a", "b"))
        row = {field: pair[field] for field in
               ("pair_id", "id_a", "id_b", "path_a", "path_b", "dims_a", "dims_b",
                "weight_a", "weight_b", "label")}
        row.update(pred=predictions[pair["pair_id"]],
                   volume_comparison="A" if volume_a > volume_b else "B" if volume_a < volume_b else "tie",
                   correct=predictions[pair["pair_id"]] == pair["label"])
        if weights is not None:
            row.update(predicted_weight_a_lb=weights[a], predicted_weight_b_lb=weights[b],
                       predicted_weight_tie=weights[a] == weights[b])
        return row
    wrong = [p for p in pairs if predictions[p["pair_id"]] != p["label"]]
    ties = [p for p in pairs if weights is not None and weights[p["id_a"]] == weights[p["id_b"]]]
    return {"mismatch_count": len(wrong), "mismatches": [example(p) for p in wrong[:limit]],
            "equal_predicted_weight_pair_count": len(ties) if weights is not None else None,
            "equal_predicted_weight_examples": [example(p) for p in ties[:limit]],
            "selection": "first matching pairs in the unchanged saved pair order",
            "interpretation": "Dimensions, labels, weights and ties are recorded facts; material/construction causes require image review."}


def build_summary(repo_root=REPO_ROOT, qwen_dir=BASELINE_DIR, *, require_complete=False, error_limit=5):
    repo, qwen_dir = Path(repo_root).resolve(), Path(qwen_dir).resolve()
    pair_rows = {split: read_rows(repo / "pairs" / f"{split}_pairs.jsonl") for split in SPLITS}
    gold = {split: load_labels(repo / "pairs" / f"{split}_pairs.jsonl", "label") for split in SPLITS}
    results = simple_results(repo, pair_rows, gold)
    diagnostics, runs = {}, {}
    for family in ("openai", "qwen"):
        for mode in ("pairwise", "pointwise"):
            key = f"{family}_{mode}"
            results[key] = {}
            for split in SPLITS:
                if family == "openai":
                    folder = repo / "openai_strong_baseline" / mode / "output" / split
                    pred_path, scores_path = folder / "preds.jsonl", folder / "scores.json"
                    if not pred_path.exists() and not scores_path.exists():
                        results[key][split] = {"status": "pending", "scores": None}
                        continue
                    if not pred_path.exists() or not scores_path.exists():
                        raise ResultsError(f"Incomplete saved OpenAI prediction/score files: {folder}")
                    metrics, _ = verified_scores(gold[split], pred_path, scores_path)
                    results[key][split] = {"status": "complete", "scores": metrics,
                                           "provenance": {"predictions": displayed_path(pred_path, repo),
                                                          "predictions_sha256": sha256(pred_path),
                                                          "scores": displayed_path(scores_path, repo)}}
                else:
                    row, predictions, weights, run = qwen_result(repo, qwen_dir, mode, split,
                                                                 pair_rows[split], gold[split])
                    results[key][split] = row
                    runs[(mode, split)] = run
                    if row["status"] == "complete":
                        diagnostics.setdefault(key, {})[split] = error_examples(pair_rows[split], predictions,
                                                                                weights, error_limit)
            if family == "qwen" and all(results[key][s]["status"] == "complete" for s in SPLITS):
                if runs[(mode, "dev")].get("config") != runs[(mode, "test")].get("config"):
                    raise ResultsError(f"Qwen {mode} dev/test configurations differ; test must use the frozen dev configuration")
    pending = [f"{mode}/{split}" for mode in ("qwen_pairwise", "qwen_pointwise") for split in SPLITS
               if results[mode][split]["status"] != "complete"]
    if require_complete and pending:
        raise ResultsError("Four complete Qwen dev/test results required; pending: " + ", ".join(pending))
    summary = {"version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
               "scorer": "unchanged score.py; exact coverage required; saved score dictionaries revalidated",
               "pair_files": {split: {"pairs": len(pair_rows[split]),
                                      "sha256": sha256(repo / "pairs" / f"{split}_pairs.jsonl")}
                              for split in SPLITS},
               "results": results, "qwen_complete": not pending,
               "notes": ["Pointwise metrics measure A/B accuracy after comparing weights, with B on ties.",
                         "Simple dev predictions are recomputed in memory using the existing baseline functions; no baseline outputs are overwritten.",
                         "OpenAI high-reasoning API runs and local Qwen Instruct use different compute budgets.",
                         "Qwen timing is copied from run.json and excludes model downloads/loading."]}
    return summary, {"version": 1, "example_limit_per_category": error_limit,
                     "results": diagnostics,
                     "note": "Ground truth appears here only for post-inference evaluation. No material or construction claims are inferred."}


def markdown(summary):
    lines = ["# Baseline results", "", "Re-scored with the unchanged strict scorer.", "",
             "| Baseline | Dev accuracy | Dev macro-F1 | Test accuracy | Test macro-F1 |",
             "|---|---:|---:|---:|---:|"]
    for key, name in NAMES.items():
        cells = []
        for split in SPLITS:
            row = summary["results"][key][split]
            cells += ([f"{row['scores']['accuracy']:.4f}", f"{row['scores']['macro_f1']:.4f}"]
                      if row["status"] == "complete" else ["pending", "pending"])
        lines.append("| " + " | ".join([name, *cells]) + " |")
    lines += ["", "| Qwen mode / split | Status | Inferences | Seconds | Mean seconds | Peak allocated GiB | Invalid outputs | Tied pairs |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for mode in ("pairwise", "pointwise"):
        for split in SPLITS:
            row = summary["results"][f"qwen_{mode}"][split]
            run = row.get("runtime", {})
            counts = (f"{run.get('completed_inferences', 0)}/{run['required_inferences']}"
                      if "required_inferences" in run else "pending")
            fmt = lambda key, divisor=1: (f"{run[key] / divisor:.3f}" if run.get(key) is not None else "pending")
            lines.append("| " + " | ".join([f"{mode} / {split}", row["status"], counts,
                                               fmt("inference_seconds"), fmt("mean_inference_seconds"),
                                               fmt("peak_memory_allocated_bytes", 1024 ** 3),
                                               str(run.get("invalid_outputs", "pending")),
                                               str(run.get("tie_pairs", "n/a" if mode == "pairwise" else "pending"))]) + " |")
    lines += ["", *[f"- {note}" for note in summary["notes"]], "",
              "Factual mismatch and equal-weight examples are saved in `error_analysis.json` for image review.", ""]
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--qwen-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, help="Summary files only (default: --qwen-dir)")
    parser.add_argument("--error-limit", type=int, default=5, help="First N mismatches and ties per complete Qwen run")
    parser.add_argument("--require-complete", action="store_true", help="Fail unless both Qwen modes completely cover dev and test")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.error_limit < 1:
        parser.error("--error-limit must be positive")
    try:
        summary, diagnostics = build_summary(args.repo_root, args.qwen_dir,
                                             require_complete=args.require_complete,
                                             error_limit=args.error_limit)
        target = (args.output_dir or args.qwen_dir).resolve()
        protected = {(Path(args.qwen_dir) / mode / "output" / split).resolve()
                     for mode in ("pairwise", "pointwise") for split in SPLITS}
        if any(target == path or target.is_relative_to(path) for path in protected):
            raise ResultsError("Summary output must be outside the per-mode inference output folders")
        target.mkdir(parents=True, exist_ok=True)
        (target / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (target / "results.md").write_text(markdown(summary), encoding="utf-8")
        (target / "error_analysis.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(markdown(summary))
        print(f"Saved results.json, results.md and error_analysis.json under {target}")
    except (ResultsError, ScoringError, OSError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
