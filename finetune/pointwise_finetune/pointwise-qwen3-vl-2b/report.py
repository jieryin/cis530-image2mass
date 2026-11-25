"""Create the final pointwise report after dev selection and test evaluation."""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def indexed(path, key="pair_id"):
    rows = lines(path)
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key} values in {path}")
    return result


def optional(path, default=None):
    return read(path) if Path(path).is_file() else default


def elapsed(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}h {seconds % 3600 // 60}m {seconds % 60}s"


def product(path):
    return Path(path).stem.partition("_")[2].replace("|", "/")


def atomic_text(path, text):
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def report(output=None, source=None):
    out = Path(output).resolve() if output else Path(__file__).resolve().parent
    repo = Path(source).resolve() if source else out.parent / "pointwise-source"
    sys.path.insert(0, str(repo))
    from score import score

    selection = read(out / "selection.json")
    selected = selection["selected"]
    names = selection.get("names", ["full"] + (["alternative"] if selected == "alternative" else []))
    if selected not in names or any(name not in {"full", "alternative"} for name in names):
        raise ValueError("Unexpected selected run in selection.json")
    best = read(out / selected / "best" / "training.json")
    cfg = best["config"]
    if cfg.get("task") != "pointwise":
        raise ValueError("The selected checkpoint is not pointwise")

    pairs = indexed(repo / "pairs" / "test_pairs.jsonl")
    hard = set(indexed(repo / "pairs" / "test_volume_wrong_pairs.jsonl"))
    baseline = indexed(repo / "qwen_strong_baseline" / "pointwise" / "output" / "test" / "preds.jsonl")
    predictions = indexed(out / "test" / "preds.jsonl")
    weights = indexed(out / "test" / "weights.jsonl", "id")
    if not pairs or set(pairs) != set(baseline) or set(pairs) != set(predictions) or not hard <= set(pairs):
        raise ValueError("Test, baseline, tuned predictions, and hard-subset IDs must match")
    object_ids = {row[f"id_{side}"] for row in pairs.values() for side in ("a", "b")}
    if object_ids != set(weights):
        raise ValueError("Test object weight predictions do not exactly cover the test objects")
    for row in weights.values():
        if not math.isfinite(row["weight_lb"]) or row["weight_lb"] <= 0:
            raise ValueError("Predicted weights must be positive and finite")
    objects = {}
    appearances, error_appearances = Counter(), Counter()
    for pair_id, row in pairs.items():
        for side in ("a", "b"):
            object_id = row[f"id_{side}"]
            gold_weight = row[f"weight_{side}"]
            if object_id in objects and objects[object_id]["gold"] != gold_weight:
                raise ValueError(f"Inconsistent gold weight for object {object_id}")
            objects[object_id] = {"gold": gold_weight, "product": product(row[f"path_{side}"]),
                                  "predicted": weights[object_id]["weight_lb"]}
            appearances[object_id] += 1
            error_appearances[object_id] += predictions[pair_id]["pred"] != row["label"]

    groups = Counter()
    cases = []
    category = {(True, True): "retained", (False, True): "corrected",
                (True, False): "regressed", (False, False): "still_wrong"}
    for pair_id, row in pairs.items():
        old, new, gold = baseline[pair_id]["pred"], predictions[pair_id]["pred"], row["label"]
        predicted_a, predicted_b = (weights[row[f"id_{side}"]]["weight_lb"] for side in ("a", "b"))
        if new != ("A" if predicted_a > predicted_b else "B"):
            raise ValueError(f"Pair label and object weights disagree: {pair_id}")
        change = category[old == gold, new == gold]
        groups[change] += 1
        cases.append(dict(pair_id=pair_id, change=change, hard_subset=pair_id in hard,
                          gold=gold, baseline=old, tuned=new,
                          id_a=row["id_a"], id_b=row["id_b"],
                          product_a=product(row["path_a"]), product_b=product(row["path_b"]),
                          gold_weight_a_lb=row["weight_a"], gold_weight_b_lb=row["weight_b"],
                          predicted_weight_a_lb=predicted_a, predicted_weight_b_lb=predicted_b))

    csv_path = out / "test_cases.csv"
    temporary = csv_path.with_name(csv_path.name + ".pending")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    os.replace(temporary, csv_path)

    def scores(ids, values):
        return score({i: pairs[i]["label"] for i in ids}, {i: values[i]["pred"] for i in ids})

    test_scores = read(out / "test" / "scores.json")
    actual_scores = scores(set(pairs), predictions)
    if (test_scores["examples"], test_scores["correct"]) != (actual_scores["examples"], actual_scores["correct"]):
        raise ValueError("Saved test scores disagree with test predictions")
    revision_result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                     capture_output=True, text=True, check=False,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    revision = revision_result.stdout.strip() if revision_result.returncode == 0 else "unavailable"
    md = ["# Qwen3-VL-2B pointwise mass fine-tuning", "",
          f"Selected `{selected}` epoch {best['epoch']} at optimizer step {best['optimizer_steps']} by highest dev pair accuracy, then lowest dev object log-MSE. {selection.get('decision', '')}",
          "", "## Training and reproducibility", "",
          f"- Source commit: `{revision}`. Base model: `{cfg['model']}`, revision `{cfg['revision']}`. Device `{cfg['device']}`, dtype `{cfg['dtype']}`, SDPA attention.",
          f"- {cfg['train_examples']:,} unique training objects per epoch, {cfg['dev_pairs']:,} dev pairs; {cfg['epochs']} epochs, microbatch 1, gradient accumulation {cfg['grad_accum']}, seed {cfg['seed']}, maximum image pixels {cfg['max_pixels']:,}.",
          f"- Language attention LoRA rank {cfg['rank']}, alpha {cfg['alpha']}, dropout {cfg['dropout']}; trainable scalar mass head. AdamW LR {cfg['lr']:g}, weight decay {cfg['weight_decay']:g}, 3% warmup then linear decay, gradient clipping at 1.0.",
          f"- {cfg['trainable_parameters']:,}/{cfg['total_parameters']:,} trainable/total parameters. The base model and vision encoder are frozen.",
          "- Each object supplies one RGB image and dimensions. The mass head predicts softplus(linear(final assistant-prefix hidden state)); training minimizes MSE against log1p(gold pounds). Evaluation converts with expm1 and compares the two predicted weights; ties select B.",
          f"- Resumable state saved every {cfg['save_every']} optimizer updates and at stage boundaries. The most recent two complete checkpoints contain the LoRA adapter, mass head, processor, optimizer, scheduler, RNG state, and training cursor. `best/` and `last/` are inference adapters.",
          ""]
    data_verification = optional(out / "data_verification.json")
    if data_verification:
        md.extend(["## Data verification", "",
                   f"Data verification passed: {data_verification.get('all_passed', 'unknown')}. Matched {data_verification.get('matched_records', 0):,} of {data_verification.get('records', 0):,} records against the source manifest; {data_verification.get('missing_count', 'unknown')} missing, {data_verification.get('mismatch_count', 'unknown')} mismatched.",
                   f"Split counts: {json.dumps(data_verification.get('split_counts', {}), sort_keys=True)}. Release archive checksum matches: {data_verification.get('archive_matches_release', 'unknown')}; manifest matches the checked origin/main snapshot: {data_verification.get('matches_current_origin_main_manifest', 'unknown')}.",
                   f"Manifest SHA-256: `{data_verification.get('manifest_sha256', 'unavailable')}`. Archive SHA-256: `{data_verification.get('archive_sha256', 'unavailable')}`.", ""])
    md.extend(["## Pilot and reload", ""])
    pilot_path = out / "pilot" / "history.jsonl"
    if pilot_path.is_file():
        pilot = [row for row in lines(pilot_path) if row.get("type") == "train"]
        if pilot:
            first, last = pilot[:5], pilot[-5:]
            md.append(f"Pilot: {len(pilot)} updates, {sum(r['examples'] for r in pilot)} objects seen. Mean loss in the first/last five updates: {statistics.mean(r['loss'] for r in first):.6f}/{statistics.mean(r['loss'] for r in last):.6f}. Mean update time: {statistics.mean(r['seconds'] for r in pilot):.2f} seconds.")
            peaks = [row["peak_allocated_bytes"] for row in pilot if "peak_allocated_bytes" in row]
            if peaks:
                md.append(f"Peak pilot tensor allocation: {max(peaks) / 2**30:.2f} GiB.")
    gradient_path = out / "pilot_gradient.jsonl"
    if gradient_path.is_file():
        gradients = lines(gradient_path)
        norms = [float(row if isinstance(row, (float, int)) else
                       row.get("gradient_norm", row.get("norm", row.get("grad_norm"))))
                 for row in gradients]
        if norms:
            md.append(f"Pre-clipping gradient norms: {min(norms):.6g} to {max(norms):.6g}; all finite: {all(math.isfinite(value) for value in norms)}.")
    verification = optional(out / "reload" / "verification.json", optional(out / "reload_verification.json"))
    if verification is not None:
        md.extend(["", "Independent reload verification:", "", "```json", json.dumps(verification, indent=2, ensure_ascii=False), "```"])
    recovery = optional(out / "recovery.json")
    if recovery:
        md.extend(["", f"Operational recovery: a Windows console-close event stopped the initial process chain at update {recovery['last_history_step_before_exit']}. Training resumed from the durable update-{recovery['durable_checkpoint_step']} checkpoint, restoring optimizer, scheduler, RNG and example cursor; {recovery['replayed_optimizer_updates']} unsaved updates were replayed. The launcher and all Python children now use the windowless interpreter. Wall times below include the failed training attempt, but exclude the idle repair interval."])
    md.extend(["", "## Development selection", "",
               "| Run | Epoch | Mean train log-MSE | Dev correct | Dev accuracy | Dev macro F1 | Object log-MSE | MAE (lb) | RMSE (lb) |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for name in names:
        history_path = out / name / "history.jsonl"
        history = lines(history_path) if history_path.is_file() else []
        for directory in sorted((out / name).glob("dev_epoch_*"), key=lambda path: int(path.name.rsplit("_", 1)[1])):
            dev = read(directory / "scores.json")
            epoch = int(directory.name.rsplit("_", 1)[1])
            chunk = [row for row in history if row.get("type") == "train" and row["epoch"] == epoch]
            seen = sum(row["examples"] for row in chunk)
            loss = f"{sum(row['loss'] * row['examples'] for row in chunk) / seen:.6f}" if seen else "n/a"
            md.append(f"| {name} | {epoch} | {loss} | {dev['correct']}/{dev['examples']} | {dev['accuracy']:.1%} | {dev['macro_f1']:.4f} | {dev['object_log_mse']:.6f} | {dev['object_mae_lb']:.4f} | {dev['object_rmse_lb']:.4f} |")
    md.extend(["", "Selection uses dev only. The test set is evaluated after selecting the final checkpoint.",
               "", "## Final test results", "",
               "| Slice | Pairs | Baseline correct | Tuned correct | Baseline accuracy | Tuned accuracy | Change | Tuned macro F1 |",
               "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for title, ids in [("Full test", set(pairs)), ("Volume-rule errors", hard), ("Remaining test", set(pairs) - hard)]:
        if not ids:
            continue
        old, new = scores(ids, baseline), scores(ids, predictions)
        md.append(f"| {title} | {len(ids)} | {old['correct']} | {new['correct']} | {old['accuracy']:.1%} | {new['accuracy']:.1%} | {(new['accuracy'] - old['accuracy']) * 100:+.1f} pp | {new['macro_f1']:.4f} |")
    md.extend(["", f"Full-test object regression: {test_scores['objects']} unique objects, log-MSE {test_scores['object_log_mse']:.6f}, MAE {test_scores['object_mae_lb']:.4f} lb, RMSE {test_scores['object_rmse_lb']:.4f} lb."])
    subset_scores = optional(out / "test" / "subset_scores.json")
    if subset_scores:
        md.append(f"Volume-error subset regression: {subset_scores['objects']} unique objects, log-MSE {subset_scores['object_log_mse']:.6f}, MAE {subset_scores['object_mae_lb']:.4f} lb, RMSE {subset_scores['object_rmse_lb']:.4f} lb.")
    md.extend(["", "The baseline is the repository's saved zero-shot pointwise numeric-generation pipeline. Fine-tuning uses a scalar regression head and a fixed image-pixel budget. These results compare complete pipelines; they do not isolate LoRA training from the output-head and preprocessing changes.",
               "", f"Matched outcomes: {groups['retained']} retained correct, {groups['corrected']} corrected, {groups['regressed']} regressed, and {groups['still_wrong']} still wrong. The net change is {groups['corrected'] - groups['regressed']:+d} correct pairs out of {len(cases)}."])
    md.extend(["", "## Improvements and remaining errors", "",
               "| Slice | Pairs | Baseline accuracy | Tuned accuracy | Change | Corrected | Regressed | Still wrong |",
               "|---|---:|---:|---:|---:|---:|---:|---:|"])
    slices = [("Gold answer A", [row for row in cases if row["gold"] == "A"]),
              ("Gold answer B", [row for row in cases if row["gold"] == "B"]),
              ("Volume-rule errors", [row for row in cases if row["hard_subset"]]),
              ("Remaining test", [row for row in cases if not row["hard_subset"]]),
              ("Gold weight ratio < 1.5", [row for row in cases if max(row["gold_weight_a_lb"], row["gold_weight_b_lb"]) < 1.5 * min(row["gold_weight_a_lb"], row["gold_weight_b_lb"])]),
              ("Gold weight ratio >= 1.5", [row for row in cases if max(row["gold_weight_a_lb"], row["gold_weight_b_lb"]) >= 1.5 * min(row["gold_weight_a_lb"], row["gold_weight_b_lb"])]),
              ("Both gold weights < 1 lb", [row for row in cases if max(row["gold_weight_a_lb"], row["gold_weight_b_lb"]) < 1]),
              ("At least one gold weight >= 1 lb", [row for row in cases if max(row["gold_weight_a_lb"], row["gold_weight_b_lb"]) >= 1])]
    slice_results = []
    for title, rows in slices:
        if not rows:
            continue
        count = len(rows)
        changes = Counter(row["change"] for row in rows)
        old_accuracy = sum(row["baseline"] == row["gold"] for row in rows) / count
        new_accuracy = sum(row["tuned"] == row["gold"] for row in rows) / count
        delta = new_accuracy - old_accuracy
        slice_results.append((title, count, new_accuracy, delta))
        md.append(f"| {title} | {count} | {old_accuracy:.1%} | {new_accuracy:.1%} | {delta * 100:+.1f} pp | {changes['corrected']} | {changes['regressed']} | {changes['still_wrong']} |")
    if slice_results:
        largest_gain = max(slice_results, key=lambda row: row[3])
        lowest = min(slice_results, key=lambda row: row[2])
        if largest_gain[3] > 0:
            md.extend(["", f"The largest accuracy gain among these slices is for {largest_gain[0]}: {largest_gain[3] * 100:+.1f} percentage points across {largest_gain[1]} pairs. The lowest tuned accuracy is {lowest[2]:.1%} for {lowest[0]} ({lowest[1]} pairs)."])
        else:
            md.extend(["", f"None of these slices improves over the baseline. The lowest tuned accuracy is {lowest[2]:.1%} for {lowest[0].lower()} ({lowest[1]} pairs)."])
    wrong = [row for row in cases if row["tuned"] != row["gold"]]
    hard_wrong = sum(row["hard_subset"] for row in wrong)
    md.extend(["", f"Of {len(wrong)} remaining pair errors, {hard_wrong} are in the volume-rule-error subset. {groups['still_wrong']} pairs are wrong in both pipelines, while {groups['regressed']} were correct in the baseline and became wrong.",
               "", f"The test contains {len(pairs)} pairs built from {len(objects)} unique objects. {sum(value > 1 for value in appearances.values())} objects appear in more than one pair, so pair errors are correlated. These overlapping slices are descriptive comparisons; they are not independent statistical tests or evidence for visual/material causes.",
               "", "## Largest object weight errors", "",
               "| Object ID | Product | Gold (lb) | Predicted (lb) | Absolute error (lb) | Signed error (lb) | Wrong pairs / appearances |",
               "|---|---|---:|---:|---:|---:|---:|"])
    for object_id, obj in sorted(objects.items(), key=lambda item: abs(item[1]["predicted"] - item[1]["gold"]), reverse=True)[:8]:
        signed = obj["predicted"] - obj["gold"]
        md.append(f"| {object_id} | {obj['product']} | {obj['gold']:.5g} | {obj['predicted']:.5g} | {abs(signed):.5g} | {signed:+.5g} | {error_appearances[object_id]} / {appearances[object_id]} |")
    mean_signed_error = statistics.mean(obj["predicted"] - obj["gold"] for obj in objects.values())
    md.extend(["", f"Mean signed object error is {mean_signed_error:+.5g} lb; a positive value means overestimation. Object metrics count each unique object once, while the pair table measures ordering accuracy."])
    severe = max(objects.items(), key=lambda item: abs(item[1]["predicted"] - item[1]["gold"]))
    severe_id, severe_obj = severe
    md.extend(["", f"Strong ordering accuracy does not guarantee accurate absolute mass. For example, object {severe_id} has gold weight {severe_obj['gold']:.4g} lb and predicted weight {severe_obj['predicted']:.4g} lb, yet only {error_appearances[severe_id]} of its {appearances[severe_id]} comparisons are wrong. Both regression error and pair accuracy should therefore be retained in future evaluations."])
    prior_path = out.parent / "pairwise-results" / "test" / "preds.jsonl"
    if prior_path.is_file():
        prior = indexed(prior_path)
        if set(prior) == set(pairs):
            md.extend(["", "## Historical pairwise fine-tuning comparison", "",
                       "| Slice | Pairs | Prior pairwise accuracy | Pointwise accuracy | Pointwise minus pairwise |",
                       "|---|---:|---:|---:|---:|"])
            for title, ids in [("Full test", set(pairs)), ("Volume-rule errors", hard)]:
                if ids:
                    previous, current = scores(ids, prior), scores(ids, predictions)
                    md.append(f"| {title} | {len(ids)} | {previous['accuracy']:.1%} | {current['accuracy']:.1%} | {(current['accuracy'] - previous['accuracy']) * 100:+.1f} pp |")
            md.extend(["", "The prior pairwise predictions are matched on the same test pair IDs. This historical comparison is added after pointwise selection and is not used for tuning or checkpoint selection."])
    md.extend(["", "## Matched cases", "",
               "| Change | Pair | Gold | Baseline → tuned | Object A (gold → predicted lb) | Object B (gold → predicted lb) | Hard subset |",
               "|---|---|---|---|---|---|---|"])
    for change in ("corrected", "regressed", "still_wrong"):
        examples = sorted((row for row in cases if row["change"] == change),
                          key=lambda row: (not row["hard_subset"], row["pair_id"]))[:2]
        for row in examples:
            md.append(f"| {change} | {row['pair_id']} | {row['gold']} | {row['baseline']} → {row['tuned']} | {row['product_a']} ({row['gold_weight_a_lb']:.4g} → {row['predicted_weight_a_lb']:.4g}) | {row['product_b']} ({row['gold_weight_b_lb']:.4g} → {row['predicted_weight_b_lb']:.4g}) | {row['hard_subset']} |")
    md.extend(["", "All matched cases and predicted/gold object weights are in [test_cases.csv](test_cases.csv). Full per-object predictions are in [test/weights.jsonl](test/weights.jsonl).", "", "## Elapsed time", ""])
    interrupted_pilot = optional(out / "pilot-interrupt.time.json")
    resumed_pilot = optional(out / "pilot.time.json")
    if interrupted_pilot and resumed_pilot:
        total_pilot_seconds = interrupted_pilot.get("seconds", 0) + resumed_pilot.get("seconds", 0)
        md.append(f"- Pilot total, including the intentional interruption and resumed segment: {elapsed(total_pilot_seconds)}. This sums active stage times and excludes the gap between processes.")
    for name in ["pilot-interrupt", "pilot", "reload", *names, "test"]:
        timing = optional(out / f"{name}.time.json")
        if timing and "seconds" in timing:
            label = "pilot resumed segment" if name == "pilot" and interrupted_pilot else name
            md.append(f"- {label}: {elapsed(timing['seconds'])}; exit code {timing.get('exit_code', 'unknown')}.")
    runtime = optional(out / "runtime_summary.json")
    if runtime:
        md.extend(["", f"- Full-training wall time from first launch to completion, including the repair/approval wait: {elapsed(runtime['full_wall_seconds'])}.",
                   f"- End-to-end pilot, recovery, full training, test and initial report generation: {elapsed(runtime['pipeline_wall_seconds'])}.",
                   f"- The pilot projected {elapsed(runtime['pilot_projected_full_seconds'])}; active full-stage intervals totaled {elapsed(runtime['active_full_seconds'])}, {runtime['projection_error_percent']:.1f}% above that estimate.",
                   "", "The 1h 55m full-stage total includes model loading, checkpoint saves, three dev evaluations and the failed initial attempt. The longer calendar times include the idle repair/approval interval. No additional hyperparameter configuration or seed was run, and test results were not used for selection."])
    md.extend(["", "## Running and recovering on this Windows machine", "",
               f"The source checkout used for this experiment is `{repo}`. Run the following in PowerShell to launch or recover the workflow:", "", "```powershell",
               f"& '{out.parent / '.venv/Scripts/python.exe'}' '{out / 'start.py'}'", "```", "",
               "The supervisor and model workers use pythonw.exe without console windows. The supervisor prevents automatic sleep during active work, holds a single-instance lock, updates supervisor_status.json every 30 seconds and retries failed workflows up to three times. Training saves every 25 updates; latest_checkpoint.json identifies the complete resumable state. Completed stages are skipped on restart.", "",
               "Locking the screen or closing a separate log-viewing window can leave training running. Keep the machine powered; manual sleep, logout, shutdown or reboot can stop the job. After a reboot, run start.py again to resume the last durable checkpoint. best/ and last/ are inference adapters; use the full run directory with its latest_checkpoint.json for training recovery.", "", "To inspect progress:", "", "```powershell",
               f"& '{out.parent / '.venv/Scripts/python.exe'}' '{out / 'status.py'}'",
               f"Get-Content '{out / 'full.stdout.log'}' -Tail 5 -Wait", "```", "",
               "Checkpoint recovery restores the adapter, mass head, optimizer, scheduler, RNG and example cursor. Unsaved updates are replayed; absolute uninterrupted execution cannot be guaranteed across power loss or operating-system termination."])
    atomic_text(out / "analysis.md", "\n".join(md) + "\n")
    print(f"Wrote {out / 'analysis.md'} and {csv_path}", flush=True)


if __name__ == "__main__":
    report()
