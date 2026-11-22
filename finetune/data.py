"""Shared pair-file validation, record loading, and unique-object extraction.

Validation uses only the standard library until records are actually opened.
An optional ``answer`` may start with A/B followed by an explanation. Only that
leading choice is retained: the training objective scores the next answer token
from the prompt, so trailing explanation tokens never participate in the loss.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

from prompts._shared import dimensions, positive_number


SPLITS = {"train", "dev", "test"}
_LEADING_CHOICE = re.compile(r"\A([AB])(?=$|\s|[.,:;!?])")


def normalize_answer(answer, label):
    """Return a leading standalone A/B choice, rejecting rationale-first text."""
    if not isinstance(answer, str):
        raise ValueError("answer must be a string starting with A or B")
    match = _LEADING_CHOICE.match(answer.strip())
    if match is None:
        raise ValueError("answer must start with standalone A or B; rationale-first answers are unsupported")
    choice = match.group(1)
    if choice != label:
        raise ValueError("answer choice disagrees with the gold label")
    return choice


def _relative_path(value, context):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a relative .pklz path")
    path = PurePosixPath(value)
    if (path.is_absolute() or PureWindowsPath(value).drive or "\\" in value
            or any(part in ("", ".", "..") for part in value.split("/"))
            or path.suffix != ".pklz" or "\x00" in value):
        raise ValueError(f"{context} must be a canonical relative .pklz path without traversal")
    if len(path.parts) < 2 or path.parts[0] not in SPLITS:
        raise ValueError(f"{context} must have a train/dev/test split prefix")
    return value


def _resolved_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Record path escapes the data root: {relative!r}")
    parts = path.relative_to(root).parts
    if not parts or parts[0] != PurePosixPath(relative).parts[0]:
        raise ValueError(f"Resolved record path changes the data split: {relative!r}")
    return path


def _validate_pair(pair):
    """Validate and normalize one JSON row at the dataset boundary."""
    if not isinstance(pair, dict):
        raise ValueError("expected a JSON object")
    pair_id = pair.get("pair_id")
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("pair_id must be a nonempty string")
    split, separator, suffix = pair_id.partition("-")
    if split not in SPLITS or not separator or not suffix:
        raise ValueError("pair_id must start with train-, dev-, or test-")
    for side in ("a", "b"):
        relative = _relative_path(pair.get(f"path_{side}"), f"{pair_id}: path_{side}")
        if PurePosixPath(relative).parts[0] != split:
            raise ValueError(f"{pair_id}: record path and pair_id split mismatch")
        object_id = pair.get(f"id_{side}")
        if type(object_id) is not int or object_id < 0:
            raise ValueError(f"{pair_id}: id_{side} must be a nonnegative integer")
        pair[f"dims_{side}"] = dimensions(pair.get(f"dims_{side}"), f"{pair_id}: dims_{side}")
        pair[f"weight_{side}"] = positive_number(pair.get(f"weight_{side}"), f"{pair_id}: weight_{side}")
    if pair["id_a"] == pair["id_b"] or pair["path_a"] == pair["path_b"]:
        raise ValueError(f"{pair_id}: cannot compare an object with itself")
    label = pair.get("label")
    expected = "A" if pair["weight_a"] > pair["weight_b"] else "B"
    if label != expected:
        raise ValueError(f"{pair_id}: gold label disagrees with object weights")
    pair["answer"] = normalize_answer(pair.get("answer", label), label)
    return pair


def load_object_record(obj, root):
    """Load one checked RGB record for either task; caller owns its image."""
    # The existing restricted unpickler blocks arbitrary imported globals.
    from load_record import load_record

    record = load_record(Path(root) / obj["path"])
    try:
        if (record["id"] != obj["id"]
                or tuple(record["dims"]) != tuple(obj["dims"])
                or record["weight"] != obj["weight"]):
            raise ValueError(f"Object {obj['id']}: record id/dimensions/weight mismatch")
        return record
    except Exception:
        record["image"].close()
        raise


def pair_object(pair, side):
    """Extract one object's metadata from a validated pair."""
    obj = {field: pair[f"{field}_{side}"] for field in ("id", "path", "dims", "weight")}
    obj["split"] = pair["pair_id"].partition("-")[0]
    return obj


def read_pairs(path, root=None, check_images=False):
    """Read one split, validate targets/identities, and optionally decode records.

    ``root`` enables resolved-path containment checks even without decoding.
    ``check_images=True`` requires it and validates each unique record once.
    """
    if check_images and root is None:
        raise ValueError("root is required when check_images=True")
    pairs, pair_ids, unordered_pairs = [], set(), set()
    objects, path_ids, splits = {}, {}, set()
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                pair = _validate_pair(json.loads(line))
                pair_id = pair["pair_id"]
                if pair_id in pair_ids:
                    raise ValueError(f"duplicate pair_id {pair_id!r}")
                pair_ids.add(pair_id)
                key = frozenset((pair["id_a"], pair["id_b"]))
                if key in unordered_pairs:
                    raise ValueError("duplicate unordered object pair")
                unordered_pairs.add(key)
                splits.add(pair_id.partition("-")[0])
                if len(splits) != 1:
                    raise ValueError("pair file mixes train/dev/test splits")
                for side in ("a", "b"):
                    object_id, relative = pair[f"id_{side}"], pair[f"path_{side}"]
                    signature = (relative, tuple(pair[f"dims_{side}"]), pair[f"weight_{side}"])
                    if object_id in objects and objects[object_id] != signature:
                        raise ValueError(f"Object {object_id} has conflicting paths, dimensions, or weights")
                    if relative in path_ids and path_ids[relative] != object_id:
                        raise ValueError(f"Record path {relative!r} has conflicting IDs")
                    if object_id not in objects:
                        if root is not None:
                            _resolved_path(root, relative)
                        if check_images:
                            record = load_object_record(pair_object(pair, side), root)
                            record["image"].close()
                    objects[object_id] = signature
                    path_ids[relative] = object_id
                pairs.append(pair)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    if not pairs:
        raise ValueError(f"{path}: no pairs found")
    return pairs


def validate_disjoint(train, dev):
    """Check train/dev roles and overlap for rows validated by ``read_pairs``."""
    ids_by_split = {}
    for expected, pairs in (("train", train), ("dev", dev)):
        if pairs[0]["pair_id"].partition("-")[0] != expected:
            raise ValueError(f"Expected {expected} data; found a different split")
        ids_by_split[expected] = {pair[f"id_{side}"] for pair in pairs for side in ("a", "b")}
    if ids_by_split["train"] & ids_by_split["dev"]:
        raise ValueError("Object ID leakage between train and dev")


def unique_objects(pairs):
    """Extract objects from validated pairs in stable first-appearance order.

    Shared objects are trained/evaluated once in pointwise mode even when they
    occur in multiple pairs. ``read_pairs`` owns metadata and split validation.
    """
    objects = {}
    for pair in pairs:
        for side in ("a", "b"):
            object_id = pair[f"id_{side}"]
            if object_id not in objects:
                objects[object_id] = pair_object(pair, side)
    return list(objects.values())


def validate_subset(rows, subset):
    """Require an evaluation slice to preserve every input and target field."""
    full = {row["pair_id"]: row for row in rows}
    fields = ("id_a", "id_b", "path_a", "path_b", "weight_a", "weight_b", "dims_a", "dims_b", "label")
    for row in subset:
        parent = full.get(row["pair_id"])
        if parent is None or any(parent[field] != row[field] for field in fields):
            raise ValueError(f"Subset is not an exact slice of full pairs: {row['pair_id']}")
