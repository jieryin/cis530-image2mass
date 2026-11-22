r"""Run and score OpenAI pairwise and pointwise mass-comparison baselines.

From the repository root (Python 3.11):
    python -m pip install -r requirements.txt
    # Put the API key in openai_strong_baseline/.env.
    python openai_strong_baseline/runner.py \
        --mode both --pairs pairs/dev_pairs.jsonl --dry-run
    python openai_strong_baseline/runner.py \
        --mode both --pairs pairs/dev_pairs.jsonl \
        --max-new-pairs 20 --max-new-objects 20
    # Remove both limits to resume and finish. Results are scored automatically.
    # Run all three saved pair sets, each with separate outputs:
    for split in train dev test; do
        python openai_strong_baseline/runner.py \
            --mode both --pairs "pairs/${split}_pairs.jsonl" || break
    done

Default outputs:
    openai_strong_baseline/pairwise/output/SPLIT/preds.jsonl
    openai_strong_baseline/pointwise/output/SPLIT/weight_preds.jsonl
    openai_strong_baseline/pointwise/output/SPLIT/preds.jsonl
SPLIT is train, dev, or test, inferred from the records in the pair file.
Use --split for custom inputs whose split cannot be inferred. Each split has
independent predictions, caches, and scores. A smoke subset and its full split
reuse matching cached predictions; completing the full split replaces its
prediction and score files. Explicit --output paths are used exactly as given.
Each mode also saves cache.json, run.json, scores.json, and scores.txt.
Both preds.jsonl files work directly with the unchanged score.py. Numeric
weight_preds.jsonl contains one {id, weight_lb} row per predicted object.
Pilots save cache/progress (and available numeric weights), but final A/B
predictions and scores are written only after that mode finishes its split.

Pairwise: two labeled images and dimensions -> only A or B.
Pointwise: one image and dimensions -> a numeric weight in pounds; compare
estimated weights for each pair, using B on exact ties. No targets reach the API.

Defaults: high reasoning, 4,096 output tokens, and 16 concurrent requests.
Both modes share one concurrency limit. Jobs are processed asynchronously;
final prediction files preserve the original pair/object order. On failure or
Ctrl+C, stop starting new requests and save responses already in flight before
exiting. Raw outputs, usage, configuration and input checksums are cached.
Use a new output directory/cache when changing prediction settings. The output
limit includes reasoning and is not an account dollar limit. Call async_run()
from an existing event loop, or run() from synchronous Python code.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import math
import os
import shlex
import sys
import tempfile
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASELINE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai_strong_baseline.api import (
    BaselineError, DEFAULT_REASONING_EFFORT, DEFAULT_MAX_OUTPUT_TOKENS, dimensions, positive_number,
)
from openai_strong_baseline.pairwise.predict import predict as predict_pair
from openai_strong_baseline.pointwise.predict import predict as predict_weight
from prompts.pairwise import PROMPT as PAIRWISE_PROMPT
from prompts.pointwise import PROMPT as POINTWISE_PROMPT
from score import ScoringError, load_labels, print_report, score

DEFAULT_MODEL = "gpt-6.1-sol"


def read_env(path):
    """Read only API-key/model settings; quoted values and comments are supported."""
    settings = {}
    if not Path(path).exists():
        return settings
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        if name.strip() not in {"OPENAI_API_KEY", "OPENAI_MODEL"}:
            continue
        try:
            parts = shlex.split(value, comments=True)
        except ValueError as exc:
            raise BaselineError(f"{path}:{number}: invalid quoting") from exc
        if not separator or len(parts) > 1:
            raise BaselineError(f"{path}:{number}: expected one value after '='")
        settings[name.strip()] = parts[0] if parts else ""
    return settings


def read_pairs(path):
    """Read pair metadata, without requiring or consuming evaluation targets."""
    pairs, seen = [], set()
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                pair = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BaselineError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(pair, dict):
                raise BaselineError(f"{path}:{number}: expected a JSON object")
            pair_id = pair.get("pair_id")
            if not isinstance(pair_id, str) or not pair_id:
                raise BaselineError(f"{path}:{number}: pair_id must be a nonempty string")
            if pair_id in seen:
                raise BaselineError(f"{path}:{number}: duplicate pair_id {pair_id!r}")
            seen.add(pair_id)
            pairs.append(pair)
    if not pairs:
        raise BaselineError(f"{path}: no pairs found")
    return pairs


def collect_objects(pairs):
    """Deduplicate objects and reject inconsistent identity or input metadata."""
    objects, path_ids = {}, {}
    for pair in pairs:
        for side in ("a", "b"):
            object_id = pair.get(f"id_{side}")
            if type(object_id) is not int or object_id < 0:
                raise BaselineError(f"{pair['pair_id']}: id_{side} must be a nonnegative integer")
            path = pair.get(f"path_{side}")
            if not isinstance(path, str) or not path or Path(path).is_absolute():
                raise BaselineError(f"{pair['pair_id']}: path_{side} must be relative to --root")
            obj = {
                "id": object_id,
                "path": path,
                "dims": dimensions(pair.get(f"dims_{side}"), f"{pair['pair_id']}: dims_{side}"),
            }
            key = str(object_id)
            if key in objects and objects[key] != obj:
                raise BaselineError(f"Object {object_id} has conflicting paths or dimensions")
            if path in path_ids and path_ids[path] != object_id:
                raise BaselineError(f"Record path {path!r} has conflicting IDs")
            objects[key] = obj
            path_ids[path] = object_id
        if pair["id_a"] == pair["id_b"]:
            raise BaselineError(f"{pair['pair_id']}: cannot compare an object with itself")
    return objects


def atomic_write(path, text):
    """Keep the last complete file if a write or prediction run is interrupted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    atomic_write(path, "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def resolve_split(args, pairs):
    """Keep dataset splits separate, including subsets with custom filenames."""
    split_names = {"train", "dev", "test"}
    root = Path(args.root).resolve()
    prefixes = set()
    for pair in pairs:
        for side in ("a", "b"):
            path = (root / pair[f"path_{side}"]).resolve()
            if not path.is_relative_to(root):
                raise BaselineError(f"Object {pair[f'id_{side}']}: path escapes --root")
            parts = path.relative_to(root).parts
            prefixes.add(parts[0] if parts else "")
    known = prefixes & split_names
    if len(known) > 1:
        raise BaselineError("Pair file mixes train/dev/test records; run each split separately")
    if args.split:
        if known and args.split not in known:
            raise BaselineError(f"--split {args.split} conflicts with the records' split {next(iter(known))}")
        return args.split
    if known:
        if prefixes != known:
            raise BaselineError("Some record paths have no split prefix; specify --split explicitly")
        return next(iter(known))
    by_filename = {f"{split}_pairs": split for split in split_names}
    inferred = by_filename.get(Path(args.pairs).stem)
    if inferred:
        return inferred
    if args.output:
        # Explicit prediction paths already identify the destination for custom data.
        return None
    raise BaselineError("Cannot infer train/dev/test from these inputs; specify --split")


def output_paths(args, mode):
    if args.output:
        predictions = Path(args.output)
        paths = {
            "predictions": predictions,
            "cache": predictions.with_suffix(".cache.json"),
            "scores_json": predictions.with_suffix(".scores.json"),
            "scores_text": predictions.with_suffix(".scores.txt"),
            "run": predictions.with_suffix(".run.json"),
        }
        if mode == "pointwise":
            paths["weights"] = predictions.with_suffix(".weight_preds.jsonl")
    else:
        directory = Path(args.output_dir or BASELINE_DIR) / mode / "output" / args.split
        predictions = directory / "preds.jsonl"
        paths = {
            "predictions": predictions,
            "cache": directory / "cache.json",
            "scores_json": directory / "scores.json",
            "scores_text": directory / "scores.txt",
            "run": directory / "run.json",
        }
        if mode == "pointwise":
            paths["weights"] = directory / "weight_preds.jsonl"
    if args.cache:
        paths["cache"] = Path(args.cache)
    resolved = [path.resolve() for path in paths.values()]
    if len(resolved) != len(set(resolved)):
        raise BaselineError("Prediction, cache, weight, and score paths must be different files")
    return paths


def load_input(record_loader, path, obj):
    try:
        record = record_loader(path)
    except Exception as exc:
        raise BaselineError(f"Cannot load object {obj['id']} from {path}: {exc}") from exc
    if record["id"] != obj["id"]:
        raise BaselineError(f"Record ID mismatch for object {obj['id']}")
    actual_dims = dimensions(record["dims"], f"Record dimensions for object {obj['id']}")
    if not all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
               for a, b in zip(actual_dims, obj["dims"])):
        raise BaselineError(f"Record dimension mismatch for object {obj['id']}")
    # Keep gold weight and tracking metadata outside model-facing arguments.
    return record["image"], actual_dims


def prepare_mode(args, mode, pairs, objects, record_paths, fingerprints):
    """Validate every mode's files/cache before any API request is scheduled."""
    paths = output_paths(args, mode)
    protected = {Path(args.pairs).resolve(), Path(args.env_file).resolve(), *record_paths.values()}
    if any(path.resolve() in protected for path in paths.values()):
        raise BaselineError("Output files must not overwrite dataset inputs or local settings")
    config = {
        "mode": mode, "model": args.model,
        "prompt": PAIRWISE_PROMPT if mode == "pairwise" else POINTWISE_PROMPT,
        "response_format": "text", "image_format": "PNG", "detail": args.detail,
        "dimension_unit": "inches", "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens, "service_tier": "default",
    }
    if mode == "pointwise":
        config.update(weight_unit="pounds", tie_label="B")
        jobs = {key: {"object": fingerprints[key]} for key in objects}
        limit = args.max_new_objects
    else:
        jobs = {
            pair["pair_id"]: {side: fingerprints[str(pair[f"id_{side}"])] for side in ("a", "b")}
            for pair in pairs
        }
        limit = args.max_new_pairs
    cache = {"version": 3, "config": config, "predictions": {}}
    if paths["cache"].exists():
        try:
            cache = json.loads(paths["cache"].read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BaselineError(f"Invalid cache JSON: {paths['cache']}") from exc
        if (not isinstance(cache, dict) or cache.get("version") != 3
                or cache.get("config") != config or not isinstance(cache.get("predictions"), dict)):
            raise BaselineError("Cache configuration mismatch; choose a new output directory or --cache")
    accepted = cache["predictions"]
    for key, inputs in jobs.items():
        if key not in accepted:
            continue
        saved = accepted[key]
        if not isinstance(saved, dict) or saved.get("inputs") != inputs:
            raise BaselineError(f"Cached input mismatch for {mode} item {key}; use a new cache")
        if mode == "pairwise":
            if saved.get("pred") not in ("A", "B"):
                raise BaselineError(f"Cached prediction {key} must be A or B")
        else:
            positive_number(saved.get("weight_lb"), f"Cached weight for object {key}")
    pending = [key for key in jobs if key not in accepted]
    selected = pending[:limit] if limit is not None else pending
    print(f"{mode}: {len(jobs)} requests total; {len(jobs) - len(pending)} cached; "
          f"{len(pending)} pending; {len(selected)} planned now", flush=True)
    print(f"Predictions: {paths['predictions']}", flush=True)
    metadata = {
        "mode": mode, "split": args.split, "concurrency": args.concurrency,
        "job_order": ("existing pair file order" if mode == "pairwise"
                      else "object first appearance in pair file"),
        "completion_order": "asynchronous; final files preserve input order",
        "config": config, "pair_file": str(Path(args.pairs).resolve()),
        "pair_file_sha256": sha256(args.pairs), "pairs": len(pairs), "objects": len(objects),
        "total_requests": len(jobs), "cached_before_run": len(jobs) - len(pending),
        "files": {name: str(path.resolve()) for name, path in paths.items()},
    }
    return {
        "mode": mode, "paths": paths, "cache": cache, "accepted": accepted,
        "jobs": jobs, "selected": selected, "metadata": metadata,
        "objects": objects, "failures": [],
    }


def save_progress(state, status):
    jobs, accepted = state["jobs"], state["accepted"]
    done = [key for key in jobs if key in accepted]
    usage = [accepted[key].get("usage") for key in done if accepted[key].get("usage") is not None]
    metadata, paths = state["metadata"], state["paths"]
    metadata.update(
        status=status, completed_requests=len(done), remaining_requests=len(jobs) - len(done),
        new_predictions_this_run=len(done) - metadata["cached_before_run"],
        usage={
            "requests_with_usage": len(usage),
            "input_tokens": sum(row.get("input_tokens", 0) for row in usage),
            "output_tokens": sum(row.get("output_tokens", 0) for row in usage),
        },
    )
    write_json(paths["run"], metadata)
    if state["mode"] == "pointwise":
        write_jsonl(paths["weights"], (
            {"id": state["objects"][key]["id"], "weight_lb": accepted[key]["weight_lb"]} for key in done
        ))


def save_results(state, pairs, gold):
    mode, accepted, paths = state["mode"], state["accepted"], state["paths"]
    if mode == "pairwise":
        rows = [{"pair_id": pair["pair_id"], "pred": accepted[pair["pair_id"]]["pred"]} for pair in pairs]
    else:
        rows = []
        for pair in pairs:
            wa = accepted[str(pair["id_a"])]["weight_lb"]
            wb = accepted[str(pair["id_b"])]["weight_lb"]
            if wa > wb:
                prediction = "A"
            elif wb > wa:
                prediction = "B"
            else:
                prediction = "B"
            rows.append({"pair_id": pair["pair_id"], "pred": prediction})
    metrics = score(gold, {row["pair_id"]: row["pred"] for row in rows})
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        print_report(metrics)
    write_jsonl(paths["predictions"], rows)
    write_json(paths["scores_json"], metrics)
    atomic_write(paths["scores_text"], report.getvalue())
    save_progress(state, "complete")
    print(f"{mode}: wrote {len(rows)} scorer-ready predictions to {paths['predictions']}")
    print(report.getvalue(), end="")
    print(f"Scores: {paths['scores_json']} and {paths['scores_text']}")


async def run_mode(args, state, pairs, record_paths, gold, get_client, record_loader,
                   semaphore, stop, failures):
    """Bound workers, drain in-flight requests, and serialize all cache writes."""
    mode, paths = state["mode"], state["paths"]
    jobs, accepted = state["jobs"], state["accepted"]
    work = iter(state["selected"])
    completed = 0
    options = {"reasoning_effort": args.reasoning_effort, "max_output_tokens": args.max_output_tokens}

    def fail(key, exc):
        stop.set()
        message = f"{mode} item {key}: {exc}"
        state["failures"].append(message)
        failures.append(message)

    def load_job(key):
        if mode == "pointwise":
            return load_input(record_loader, record_paths[key], state["objects"][key])
        a, b = jobs[key]["a"], jobs[key]["b"]
        image_a, dims_a = load_input(record_loader, record_paths[str(a["id"])], a)
        image_b, dims_b = load_input(record_loader, record_paths[str(b["id"])], b)
        return image_a, dims_a, image_b, dims_b

    async def worker():
        nonlocal completed
        while not stop.is_set():
            # No await between claiming jobs: each selected item is claimed once.
            try:
                key = next(work)
            except StopIteration:
                return
            async with semaphore:
                if stop.is_set():
                    return
                try:
                    inputs = await asyncio.to_thread(load_job, key)
                    if stop.is_set():
                        return
                    predict = predict_pair if mode == "pairwise" else predict_weight
                    prediction = await predict(get_client(), args.model, *inputs, args.detail, **options)
                    # Deliberately no await during commit: concurrent completions cannot
                    # interleave cache mutation, atomic writes, or progress snapshots.
                    accepted[key] = {"inputs": jobs[key], **prediction}
                    write_json(paths["cache"], state["cache"])
                    save_progress(state, "running")
                    completed += 1
                    print(f"{mode}: saved {completed}/{len(state['selected'])} new predictions (id={key})", flush=True)
                except asyncio.CancelledError:
                    stop.set()
                    raise
                except Exception as exc:
                    fail(key, exc)
                    return

    workers = []
    try:
        save_progress(state, "running")
        workers = [asyncio.create_task(worker()) for _ in range(min(args.concurrency, len(state["selected"])))]
        outcomes = await asyncio.gather(*workers, return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                fail("worker", outcome)
        if state["failures"]:
            save_progress(state, "failed")
        elif any(key not in accepted for key in jobs):
            save_progress(state, "partial")
            remaining = sum(key not in accepted for key in jobs)
            print(f"{mode}: {remaining} requests remain. Saved progress; rerun to resume.")
        else:
            save_results(state, pairs, gold)
    except asyncio.CancelledError:
        stop.set()
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        save_progress(state, "failed")
        raise
    except Exception as exc:
        fail("saving results", exc)
        save_progress(state, "failed")


async def async_run(args, *, client=None, record_loader=None):
    """Run both modes with one shared concurrency limit and AsyncOpenAI client."""
    args = argparse.Namespace(**vars(args))
    if args.concurrency < 1:
        raise BaselineError("--concurrency must be positive")
    if args.mode == "both" and (args.output or args.cache):
        raise BaselineError("Use --output-dir with --mode both; --output/--cache select a single mode's files")
    if args.mode == "pairwise" and args.max_new_objects is not None:
        raise BaselineError("--max-new-objects applies to pointwise mode; use --max-new-pairs")
    if args.mode == "pointwise" and args.max_new_pairs is not None:
        raise BaselineError("--max-new-pairs applies to pairwise mode; use --max-new-objects")
    env = read_env(args.env_file)
    args.model = args.model or os.environ.get("OPENAI_MODEL") or env.get("OPENAI_MODEL") or DEFAULT_MODEL
    if not args.model.strip():
        raise BaselineError("Model cannot be empty")
    pairs = read_pairs(args.pairs)
    gold = load_labels(Path(args.pairs), "label")
    objects = collect_objects(pairs)
    args.split = resolve_split(args, pairs)
    print(f"Model: {args.model}; reasoning: {args.reasoning_effort}; "
          f"output limit: {args.max_output_tokens}; concurrency: {args.concurrency}", flush=True)
    print(f"Split: {args.split or 'custom'}; pairs: {len(pairs)}; unique objects: {len(objects)}", flush=True)
    root = Path(args.root).resolve()
    record_paths, missing = {}, []
    for key, obj in objects.items():
        path = (root / obj["path"]).resolve()
        if not path.is_relative_to(root):
            raise BaselineError(f"Object {obj['id']}: path escapes --root")
        record_paths[key] = path
        if not path.is_file():
            missing.append(obj["path"])
    if missing:
        raise BaselineError(
            f"Missing {len(missing)} object records under {root}; first: {missing[0]}. "
            "Download the archive linked in README.md. No API calls made."
        )
    fingerprints = {key: {**objects[key], "source_sha256": sha256(path)} for key, path in record_paths.items()}
    modes = ("pairwise", "pointwise") if args.mode == "both" else (args.mode,)
    # Preflight all modes before constructing a client or allowing any paid request.
    states = [prepare_mode(args, mode, pairs, objects, record_paths, fingerprints) for mode in modes]
    if args.dry_run:
        print("Dry run: no API calls or files written.")
        return
    if record_loader is None and any(state["selected"] for state in states):
        try:
            from load_record import load_record
        except ImportError as exc:
            raise BaselineError("Install record dependencies: python -m pip install -r requirements.txt") from exc
        record_loader = load_record

    owned_client = None

    def get_client():
        nonlocal owned_client
        if client is not None:
            return client
        if owned_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise BaselineError("Install dependencies: python -m pip install -r requirements.txt") from exc
            api_key = os.environ.get("OPENAI_API_KEY") or env.get("OPENAI_API_KEY")
            if not api_key:
                raise BaselineError(f"Add OPENAI_API_KEY to {args.env_file} before running predictions")
            # Construction has no await, so simultaneous workers share one client.
            owned_client = AsyncOpenAI(api_key=api_key, timeout=args.timeout, max_retries=args.max_retries)
        return owned_client

    semaphore, stop, failures = asyncio.Semaphore(args.concurrency), asyncio.Event(), []
    tasks = [asyncio.create_task(run_mode(
        args, state, pairs, record_paths, gold, get_client, record_loader, semaphore, stop, failures,
    )) for state in states]
    draining = asyncio.gather(*tasks, return_exceptions=True)
    try:
        try:
            outcomes = await asyncio.shield(draining)
        except asyncio.CancelledError:
            stop.set()
            print("Stopping new requests; waiting for in-flight responses to be saved...", flush=True)
            await asyncio.shield(draining)
            raise
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                failures.append(str(outcome))
        if failures:
            raise BaselineError(
                f"{failures[0]}. Successfully cached responses are preserved; "
                "rerun the same command to resume."
            )
    finally:
        if owned_client is not None:
            await owned_client.close()


def run(args, *, client=None, record_loader=None):
    """Synchronous CLI/test entry point; use async_run from an existing event loop."""
    return asyncio.run(async_run(args, client=client, record_loader=record_loader))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("pairwise", "pointwise", "both"), default="pairwise")
    parser.add_argument("--pairs", type=Path, required=True, help="Existing train/dev/test pair JSONL with gold labels")
    parser.add_argument("--split", choices=("train", "dev", "test"), help="Output split (inferred from record paths or standard pair filename)")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "amazon_data_10k", help="Root of .pklz records")
    parser.add_argument("--model", help="Overrides OPENAI_MODEL in environment/.env (default: gpt-6.1-sol)")
    parser.add_argument("--env-file", type=Path, default=BASELINE_DIR / ".env", help="Shared local API key/model settings")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--output-dir", type=Path, help="Parent for MODE/output/SPLIT result folders (default: baseline folder)")
    outputs.add_argument("--output", type=Path, help="Override A/B prediction path for a single mode")
    parser.add_argument("--cache", type=Path, help="Override cache path for a single mode")
    parser.add_argument("--detail", choices=("low", "high", "auto"), default="high")
    parser.add_argument("--reasoning-effort", choices=("default", "none", "low", "medium", "high", "xhigh", "max"), default=DEFAULT_REASONING_EFFORT, help="Reasoning level (default: high)")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="Per-request cap including reasoning tokens (default: 4096)")
    parser.add_argument("--concurrency", type=int, default=16, help="Maximum in-flight requests across both modes (default: 16)")
    parser.add_argument("--max-new-pairs", type=int, help="Pairwise pilot: at most N new pair predictions")
    parser.add_argument("--max-new-objects", type=int, help="Pointwise pilot: at most N new object predictions")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Check inputs/cache without API calls or writes")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.max_retries < 0:
        parser.error("--timeout must be finite and positive; --max-retries must be nonnegative")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.max_output_tokens < 16:
        parser.error("--max-output-tokens must be at least 16")
    if any(value is not None and value < 1 for value in (args.max_new_pairs, args.max_new_objects)):
        parser.error("Pilot limits must be positive")
    try:
        run(args)
    except (BaselineError, ScoringError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to resume saved predictions.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
