"""Validate the existing project records and pair files without model inference.

This metadata audit may inspect gold weights. Nothing from this utility is a
model message, and it never changes the data or constructs new pairs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from load_record import load_record


EXPECTED_COUNTS = {"train": 8000, "dev": 1000, "test": 1000}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def valid_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def validate_data(root, pairs_dir):
    """Return an audit report; decode and close each image before the next one."""
    root, pairs_dir = Path(root).resolve(), Path(pairs_dir).resolve()
    errors = []
    report = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "pairs_dir": str(pairs_dir),
        "model_loaded": False,
        "splits": {},
        "errors": errors,
    }
    manifest_path = root / "manifest.csv"
    if not manifest_path.is_file():
        errors.append("Missing manifest.csv")
        report["passed"] = False
        return report
    report["manifest_sha256"] = sha256(manifest_path)
    manifest = {}
    manifest_ids = set()
    try:
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                relative = row["subset_path"]
                if relative in manifest:
                    errors.append(f"Duplicate manifest path: {relative}")
                manifest[relative] = row
                object_id = int(row["id"])
                if object_id in manifest_ids:
                    errors.append(f"Duplicate manifest id: {object_id}")
                manifest_ids.add(object_id)
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        errors.append(f"Cannot read manifest: {exc}")
        report["passed"] = False
        return report
    report["manifest_records"] = len(manifest)
    sampling_path = root / "sampling.json"
    if sampling_path.is_file():
        try:
            expected_hash = json.loads(sampling_path.read_text(encoding="utf-8")).get("manifest_sha256")
            if expected_hash and expected_hash != report["manifest_sha256"]:
                errors.append("Manifest hash differs from sampling.json")
        except (OSError, ValueError) as exc:
            errors.append(f"Cannot read sampling.json: {exc}")

    metadata = {}
    all_record_ids = {}
    record_ids_by_split = {}
    found_paths = set()
    for split, expected_count in EXPECTED_COUNTS.items():
        files = sorted((root / split).glob("*.pklz"))
        split_ids = set()
        decoded = 0
        byte_count = 0
        split_report = {"record_files": len(files), "expected_records": expected_count}
        report["splits"][split] = split_report
        if len(files) != expected_count:
            errors.append(f"{split}: expected {expected_count} record files, found {len(files)}")
        for path in files:
            relative = path.relative_to(root).as_posix()
            found_paths.add(relative)
            row = manifest.get(relative)
            if row is None:
                errors.append(f"Record absent from manifest: {relative}")
            else:
                if row["project_split"] != split:
                    errors.append(f"Manifest split mismatch: {relative}")
                if sha256(path) != row.get("file_sha256"):
                    errors.append(f"Record checksum mismatch: {relative}")
            byte_count += path.stat().st_size
            image = None
            try:
                record = load_record(path)
                image = record["image"]
                image.load()
                if image.mode != "RGB" or min(image.size) <= 0:
                    raise ValueError("image must decode to nonempty RGB")
                decoded += 1
                object_id = record["id"]
                dims = tuple(record["dims"])
                weight = record["weight"]
                if len(dims) != 3 or not all(valid_number(d) for d in dims):
                    errors.append(f"Invalid dimensions: {relative}")
                if not valid_number(weight):
                    errors.append(f"Invalid weight: {relative}")
                if int(path.name.split("_", 1)[0]) != object_id:
                    errors.append(f"Filename id differs from record: {relative}")
                if object_id in all_record_ids:
                    errors.append(f"Record id reused across files: {object_id}")
                all_record_ids[object_id] = relative
                split_ids.add(object_id)
                metadata[relative] = {"id": object_id, "dims": dims, "weight": weight}
                if row is not None:
                    if int(row["id"]) != object_id:
                        errors.append(f"Manifest id mismatch: {relative}")
                    if "weight_lb" in row and float(row["weight_lb"]) != weight:
                        errors.append(f"Manifest weight mismatch: {relative}")
                    if "dimension_1_in" in row:
                        expected_dims = tuple(float(row[f"dimension_{i}_in"]) for i in (1, 2, 3))
                        if expected_dims != dims:
                            errors.append(f"Manifest dimensions mismatch: {relative}")
                    if "file_bytes" in row and int(row["file_bytes"]) != path.stat().st_size:
                        errors.append(f"Manifest byte count mismatch: {relative}")
            except Exception as exc:
                errors.append(f"Cannot validate record {relative}: {type(exc).__name__}: {exc}")
            finally:
                if image is not None:
                    image.close()
        record_ids_by_split[split] = split_ids
        split_report.update({"unique_record_ids": len(split_ids), "decoded_rgb_images": decoded,
                             "record_bytes": byte_count})
    for relative in sorted(set(manifest) - found_paths):
        errors.append(f"Manifest record missing on disk: {relative}")

    referenced_ids_by_split = {}
    for split, expected_count in EXPECTED_COUNTS.items():
        path = pairs_dir / f"{split}_pairs.jsonl"
        split_report = report["splits"][split]
        if not path.is_file():
            errors.append(f"Missing pairs file: {path}")
            referenced_ids_by_split[split] = set()
            continue
        split_report["pairs_sha256"] = sha256(path)
        pair_ids, object_ids, labels = set(), set(), {"A": 0, "B": 0}
        pairs_count = 0
        object_inputs = {}
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                pairs_count += 1
                context = f"{path.name}:{line_number}"
                try:
                    pair = json.loads(line)
                    pair_id = pair["pair_id"]
                    if not isinstance(pair_id, str) or not pair_id.startswith(f"{split}-"):
                        raise ValueError("pair_id must have the matching split prefix")
                    if pair_id in pair_ids:
                        raise ValueError(f"duplicate pair_id {pair_id}")
                    pair_ids.add(pair_id)
                    label = pair["label"]
                    if label not in labels:
                        raise ValueError("label must be A or B")
                    labels[label] += 1
                    for side in ("a", "b"):
                        relative = pair[f"path_{side}"]
                        record_path = (root / relative).resolve()
                        if root not in record_path.parents or not relative.startswith(f"{split}/"):
                            raise ValueError(f"path_{side} is outside its split")
                        saved = metadata.get(relative)
                        if saved is None:
                            raise ValueError(f"path_{side} has no successfully validated record")
                        object_id = pair[f"id_{side}"]
                        if isinstance(object_id, bool) or not isinstance(object_id, int):
                            raise ValueError(f"id_{side} is not an integer")
                        dims = tuple(pair[f"dims_{side}"])
                        if object_id != saved["id"] or dims != saved["dims"] or pair[f"weight_{side}"] != saved["weight"]:
                            raise ValueError(f"id/dimensions/weight mismatch for side {side}")
                        signature = (relative, dims)
                        if object_id in object_inputs and object_inputs[object_id] != signature:
                            raise ValueError(f"inconsistent object input for id {object_id}")
                        object_inputs[object_id] = signature
                        object_ids.add(object_id)
                    if pair["id_a"] == pair["id_b"]:
                        raise ValueError("pair contains the same object twice")
                    if label != ("A" if pair["weight_a"] > pair["weight_b"] else "B"):
                        raise ValueError("gold label differs from saved record weights")
                    if max(pair["weight_a"], pair["weight_b"]) / min(pair["weight_a"], pair["weight_b"]) < 1.25:
                        raise ValueError("pair violates minimum mass ratio 1.25")
                except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
                    errors.append(f"{context}: {exc}")
        if pairs_count != expected_count:
            errors.append(f"{split}: expected {expected_count} pairs, found {pairs_count}")
        referenced_ids_by_split[split] = object_ids
        split_report.update({"pairs": pairs_count, "unique_pair_ids": len(pair_ids),
                             "unique_referenced_objects": len(object_ids), "labels": labels})
    report["cross_split_record_id_overlap"] = {}
    report["cross_split_referenced_id_overlap"] = {}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        records_overlap = len(record_ids_by_split[a] & record_ids_by_split[b])
        referenced_overlap = len(referenced_ids_by_split[a] & referenced_ids_by_split[b])
        report["cross_split_record_id_overlap"][f"{a}_{b}"] = records_overlap
        report["cross_split_referenced_id_overlap"][f"{a}_{b}"] = referenced_overlap
        if records_overlap or referenced_overlap:
            errors.append(f"Object leakage between {a} and {b}")
    report["passed"] = not errors
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("amazon_data_10k"))
    parser.add_argument("--pairs-dir", type=Path, default=Path("pairs"))
    parser.add_argument("--json-output", type=Path, help="Optionally save the audit report")
    args = parser.parse_args(argv)
    report = validate_data(args.root, args.pairs_dir)
    text = json.dumps(report, indent=2) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
