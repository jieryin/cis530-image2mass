"""Run resumable, strictly scored Qwen3-VL zero-shot baselines on the saved pairs.

Run from the repository root. --dry-run validates all inputs without importing
the model, using CUDA, or writing files. Pilot limits count only new inferences;
final pair predictions and scores are published only at complete coverage.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASELINE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai_strong_baseline.api import BaselineError
from openai_strong_baseline.runner import (
    atomic_write as _atomic_write, collect_objects, load_input, read_pairs, resolve_split,
    sha256,
)
from qwen_strong_baseline.pairwise import predict as pairwise
from qwen_strong_baseline.pointwise import predict as pointwise
from qwen_strong_baseline.numeric_format import NUMERIC_FORMAT_POLICY
from prompts.pairwise import PROMPT as PAIRWISE_PROMPT, build_messages as build_pairwise_messages
from prompts.pointwise import PROMPT as POINTWISE_PROMPT, build_messages as build_pointwise_messages
from score import ScoringError, load_labels, print_report, score

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
SCHEMA_VERSION = 1


def atomic_write(path, text):
    # Windows readers and antivirus can briefly prevent replacing an existing
    # file. Retry only that filesystem operation, never model generation.
    for attempt in range(40):
        try:
            return _atomic_write(path, text)
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    atomic_write(path, "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def now():
    return datetime.now(timezone.utc).isoformat()


def dependency_versions():
    versions = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "transformers", "accelerate", "huggingface-hub",
                 "numpy", "Pillow", "lz4", "safetensors", "tokenizers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def configuration(args, mode):
    # Pin source and dependency versions as well as settings. A changed inference
    # implementation must never silently borrow predictions from an old run.
    sources = [BASELINE_DIR / "model.py", BASELINE_DIR / mode / "predict.py",
               Path(__file__), REPO_ROOT / "load_record.py",
               REPO_ROOT / "openai_strong_baseline" / "runner.py",
               REPO_ROOT / "openai_strong_baseline" / "api.py"]
    sources.extend(sorted((REPO_ROOT / "prompts").glob("*.py")))
    if mode == "pointwise":
        sources.append(BASELINE_DIR / "numeric_format.py")
    return {
        "mode": mode, "model": args.model, "revision": args.revision,
        "prompt": PAIRWISE_PROMPT if mode == "pairwise" else POINTWISE_PROMPT,
        "dtype": "bfloat16", "device": "cuda:0", "batch_size": 1,
        "do_sample": False, "max_new_tokens": args.max_new_tokens,
        "format_constraint": NUMERIC_FORMAT_POLICY if mode == "pointwise" else None,
        "attention_implementation": "sdpa", "dimension_unit": "inches",
        "weight_unit": "pounds", "tie_label": "B",
        "processor": "AutoProcessor standard settings from the pinned revision",
        "invalid_output_policy": "save raw output and abort; no retries or fallback labels",
        "dependencies": dependency_versions(),
        "source_sha256": {str(p.relative_to(REPO_ROOT)): sha256(p) for p in sources},
    }


def validate_inputs(args, record_loader):
    pairs = read_pairs(args.pairs)
    objects = collect_objects(pairs)
    args.output = None  # Compatibility with the existing split validator.
    args.split = resolve_split(args, pairs)
    root = Path(args.root).resolve()
    paths, fingerprints = {}, {}
    for key, obj in objects.items():
        path = (root / obj["path"]).resolve()
        if not path.is_relative_to(root):
            raise BaselineError(f"Object {key}: path escapes --root")
        if not path.is_file():
            raise BaselineError(f"Missing object {key}: {path}")
        image, dims = load_input(record_loader, path, obj)
        try:
            if getattr(image, "mode", None) != "RGB":
                raise BaselineError(f"Object {key}: loader must return an RGB image")
            if hasattr(image, "load"):
                image.load()
        finally:
            if hasattr(image, "close"):
                image.close()
        paths[key] = path
        fingerprints[key] = {"id": obj["id"], "path": obj["path"],
                             "dims": dims, "sha256": sha256(path)}
    # Validate the gold contract, but never use its values to form model input.
    load_labels(Path(args.pairs), "label")
    inputs = {"pairs_sha256": sha256(args.pairs), "objects": fingerprints}
    return pairs, objects, paths, inputs


def parse_result(mode, response):
    if not isinstance(response, dict) or not isinstance(response.get("raw_output"), str):
        raise BaselineError("Predictor must return raw_output text")
    if response.get("truncated", False):
        raise BaselineError("Model output reached the token limit; inspect dev and use a new output directory")
    if response.get("parameter_versions_unchanged", True) is not True:
        raise BaselineError("Model parameters changed during inference")
    return (pairwise if mode == "pairwise" else pointwise).parse_output(response["raw_output"])


def prepare_mode(args, mode, pairs, objects, inputs):
    folder = Path(args.output_dir) / mode / "output" / args.split
    if folder.resolve().is_relative_to(Path(args.root).resolve()):
        raise BaselineError("Output directory must be outside the input dataset root")
    jobs = ([p["pair_id"] for p in pairs] if mode == "pairwise" else list(objects))
    config = configuration(args, mode)
    cache = {"version": SCHEMA_VERSION, "config": config, "inputs": inputs,
             "predictions": {}, "errors": {}, "created_at": now()}
    cache_path = folder / "cache.json"
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise BaselineError(f"Cannot read cache {cache_path}: {exc}") from exc
        if (not isinstance(cache, dict) or cache.get("version") != SCHEMA_VERSION
                or cache.get("config") != config or cache.get("inputs") != inputs
                or not isinstance(cache.get("predictions"), dict)
                or not isinstance(cache.get("errors"), dict)):
            raise BaselineError(f"Cache configuration/input mismatch in {cache_path}; use a new output directory")
        if set(cache["predictions"]) - set(jobs) or set(cache["errors"]) - set(jobs):
            raise BaselineError("Cache contains unknown prediction IDs")
        for key, row in cache["predictions"].items():
            if not isinstance(row, dict):
                raise BaselineError(f"Invalid cached prediction for {key}")
            parsed = parse_result(mode, row)
            if type(row.get("value")) is not type(parsed) or row["value"] != parsed:
                raise BaselineError(f"Invalid cached prediction for {key}")
        if cache["errors"]:
            raise BaselineError(f"Cache contains {len(cache['errors'])} failed output(s); inspect errors and use a new directory after fixing the dev configuration")
    return {"mode": mode, "folder": folder, "cache": cache, "jobs": jobs,
            "limit": args.max_new_pairs if mode == "pairwise" else args.max_new_objects}


def summarize(state):
    rows = list(state["cache"]["predictions"].values())
    seconds = sum(row.get("elapsed_seconds", 0.0) for row in rows)
    return {
        "required_inferences": len(state["jobs"]), "completed_inferences": len(rows),
        "remaining_inferences": len(state["jobs"]) - len(rows),
        "invalid_outputs": len(state["cache"]["errors"]),
        "inference_seconds": seconds,
        "mean_inference_seconds": seconds / len(rows) if rows else None,
        "peak_memory_allocated_bytes": max((r.get("peak_memory_allocated_bytes", 0) for r in rows), default=0),
        "timing_policy": "CUDA synchronized; excludes model download/load, input validation, cache writes and scoring; includes processing and generation",
    }


def persist(state, args, *, status="incomplete", error=None):
    cache = state["cache"]
    cache["updated_at"] = now()
    write_json(state["folder"] / "cache.json", cache)
    report = {
        "status": status, "mode": state["mode"], "split": args.split,
        "pairs_path": str(Path(args.pairs).resolve()), "root": str(Path(args.root).resolve()),
        "pairs_sha256": cache["inputs"]["pairs_sha256"], "pair_count": state["pair_count"],
        "unique_objects": len(cache["inputs"]["objects"]), "config": cache["config"],
        "model_metadata": cache.get("model_metadata"), "created_at": cache["created_at"],
        "updated_at": cache["updated_at"], **summarize(state),
    }
    if error:
        report["error"] = error
    if "tie_pairs" in state:
        report["tie_pairs"] = state["tie_pairs"]
    write_json(state["folder"] / "run.json", report)
    if state["mode"] == "pointwise":
        write_jsonl(state["folder"] / "weight_preds.jsonl", (
            {"id": int(key), "weight_lb": cache["predictions"][key]["value"]}
            for key in state["jobs"] if key in cache["predictions"]))
    return report


def save_complete(state, args, pairs):
    accepted = state["cache"]["predictions"]
    if len(accepted) != len(state["jobs"]) or state["cache"]["errors"]:
        return persist(state, args)
    if state["mode"] == "pairwise":
        rows = [{"pair_id": pair["pair_id"], "pred": accepted[pair["pair_id"]]["value"]} for pair in pairs]
    else:
        weights = {key: item["value"] for key, item in accepted.items()}
        rows = [{"pair_id": pair["pair_id"], "pred": pointwise.compare(
            weights[str(pair["id_a"])], weights[str(pair["id_b"])])} for pair in pairs]
        state["tie_pairs"] = sum(weights[str(p["id_a"])] == weights[str(p["id_b"])] for p in pairs)
    result = score(load_labels(Path(args.pairs), "label"), {row["pair_id"]: row["pred"] for row in rows})
    write_jsonl(state["folder"] / "preds.jsonl", rows)
    # Re-read the actual exported file through the original strict scorer.
    result = score(load_labels(Path(args.pairs), "label"), load_labels(state["folder"] / "preds.jsonl", "pred"))
    write_json(state["folder"] / "scores.json", result)
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        print_report(result)
    atomic_write(state["folder"] / "scores.txt", text.getvalue())
    report = persist(state, args, status="complete")
    print(f"{state['mode']} {args.split}: complete", flush=True)
    print_report(result)
    return report


def run(args, *, predictor=None, record_loader=None):
    if args.max_new_tokens < 1:
        raise BaselineError("--max-new-tokens must be positive")
    if any(n is not None and n < 1 for n in (args.max_new_pairs, args.max_new_objects)):
        raise BaselineError("Pilot limits must be positive")
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        raise BaselineError("--revision must be an immutable 40-character lowercase commit SHA")
    if record_loader is None:
        from load_record import load_record
        record_loader = load_record
    pairs, objects, paths, inputs = validate_inputs(args, record_loader)
    modes = ("pairwise", "pointwise") if args.mode == "both" else (args.mode,)
    states = [prepare_mode(args, mode, pairs, objects, inputs) for mode in modes]
    for state in states:
        state["pair_count"] = len(pairs)
        print(f"{state['mode']} {args.split}: {len(pairs)} pairs, {len(objects)} unique objects; "
              f"{summarize(state)['completed_inferences']}/{len(state['jobs'])} cached", flush=True)
    if args.dry_run:
        return {state["mode"]: summarize(state) for state in states}
    missing = any(len(s["cache"]["predictions"]) < len(s["jobs"]) for s in states)
    if missing and predictor is None:
        from qwen_strong_baseline.model import QwenPredictor
        predictor = QwenPredictor(args.model, revision=args.revision, max_new_tokens=args.max_new_tokens)
    pair_by_id = {p["pair_id"]: p for p in pairs}
    reports = {}
    for state in states:
        if predictor is not None:
            metadata = json.loads(json.dumps(predictor.metadata, allow_nan=False))
            previous = state["cache"].get("model_metadata")
            if previous is not None and previous != metadata:
                raise BaselineError("Model runtime metadata changed; use a new output directory")
            state["cache"]["model_metadata"] = metadata
        new_count = 0
        persist(state, args)
        for key in state["jobs"]:
            if key in state["cache"]["predictions"]:
                continue
            if state["limit"] is not None and new_count >= state["limit"]:
                break
            images = []
            response = None
            try:
                if state["mode"] == "pairwise":
                    pair = pair_by_id[key]
                    values = []
                    for side in ("a", "b"):
                        object_key = str(pair[f"id_{side}"])
                        image, dims = load_input(record_loader, paths[object_key], objects[object_key])
                        images.append(image)
                        values.extend((image, dims))
                    messages = build_pairwise_messages(*values)
                else:
                    image, dims = load_input(record_loader, paths[key], objects[key])
                    images.append(image)
                    messages = build_pointwise_messages(image, dims)
                response = predictor(messages)
                value = parse_result(state["mode"], response)
                elapsed = response.get("elapsed_seconds", 0.0)
                if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                    raise BaselineError("Invalid inference timing metadata")
            except Exception as exc:
                state["cache"]["errors"][key] = {
                    "error": str(exc), "response": response, "failed_at": now(),
                }
                persist(state, args, status="failed", error=str(exc))
                raise BaselineError(f"{state['mode']} item {key}: {exc}. Raw output/progress saved; no final scores for incomplete coverage") from exc
            finally:
                for image in images:
                    if hasattr(image, "close"):
                        image.close()
            state["cache"]["predictions"][key] = {**response, "value": value}
            new_count += 1
            # Filesystem failures must not relabel a valid response as an
            # invalid deterministic model output. Prior atomic saves survive.
            persist(state, args)
            done = len(state["cache"]["predictions"])
            if done % 20 == 0 or done == len(state["jobs"]):
                print(f"{state['mode']} {args.split}: {done}/{len(state['jobs'])}; "
                      f"mean {summarize(state)['mean_inference_seconds']:.3f} s", flush=True)
        reports[state["mode"]] = save_complete(state, args, pairs)
        print(json.dumps({"mode": state["mode"], **summarize(state)}, indent=2), flush=True)
    return reports


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pairwise", "pointwise", "both"), default="pairwise")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "amazon_data_10k")
    parser.add_argument("--output-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--split", choices=("train", "dev", "test"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-new-pairs", type=int)
    parser.add_argument("--max-new-objects", type=int)
    return parser


def main():
    args = build_parser().parse_args()
    try:
        run(args)
    except (BaselineError, ScoringError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Completed predictions were saved; rerun to resume.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
