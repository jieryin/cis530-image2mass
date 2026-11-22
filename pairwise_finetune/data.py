"""Validated pairwise supervision and target-free Qwen input messages.

Validation uses only the standard library until records are actually opened.
An optional ``answer`` may start with A/B followed by an explanation. Only that
leading choice is retained: the training objective scores the next answer token
from the prompt, so trailing explanation tokens never participate in the loss.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path, PurePosixPath, PureWindowsPath


SPLITS = {"train", "dev", "test"}
_LEADING_CHOICE = re.compile(r"\A([AB])(?=$|\s|[.,:;!?])")


def normalize_answer(answer, label):
    """Return a leading standalone A/B choice, rejecting rationale-first text."""
    if label not in ("A", "B"):
        raise ValueError("label must be A or B")
    if not isinstance(answer, str):
        raise ValueError("answer must be a string starting with A or B")
    match = _LEADING_CHOICE.match(answer.strip())
    if match is None:
        raise ValueError("answer must start with standalone A or B; rationale-first answers are unsupported")
    choice = match.group(1)
    if choice != label:
        raise ValueError("answer choice disagrees with the gold label")
    return choice


def _positive(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a positive finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{context} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{context} must be a positive finite number")
    return result


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


def pair_split(pair):
    """Infer one split and require pair IDs and both object paths to agree."""
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
    return split


def _validate_pair(row):
    if not isinstance(row, dict):
        raise ValueError("expected a JSON object")
    pair = dict(row)
    pair_split(pair)
    context = pair["pair_id"]
    for side in ("a", "b"):
        object_id = pair.get(f"id_{side}")
        if type(object_id) is not int or object_id < 0:
            raise ValueError(f"{context}: id_{side} must be a nonnegative integer")
        dims = pair.get(f"dims_{side}")
        if not isinstance(dims, (list, tuple)) or len(dims) != 3:
            raise ValueError(f"{context}: dims_{side} must contain exactly three dimensions")
        pair[f"dims_{side}"] = [_positive(d, f"{context}: dims_{side}") for d in dims]
        pair[f"weight_{side}"] = _positive(pair.get(f"weight_{side}"), f"{context}: weight_{side}")
    if pair["id_a"] == pair["id_b"] or pair["path_a"] == pair["path_b"]:
        raise ValueError(f"{context}: cannot compare an object with itself")
    label = pair.get("label")
    if label not in ("A", "B"):
        raise ValueError(f"{context}: label must be A or B")
    expected = "A" if pair["weight_a"] > pair["weight_b"] else "B"
    if label != expected:
        raise ValueError(f"{context}: gold label disagrees with object weights")
    pair["answer"] = normalize_answer(pair.get("answer", label), label)
    return pair


def _load_validated_record(pair, side, root):
    # The existing restricted unpickler blocks arbitrary imported globals.
    from load_record import load_record

    path = _resolved_path(root, pair[f"path_{side}"])
    record = load_record(path)
    try:
        if (record["id"] != pair[f"id_{side}"]
                or tuple(record["dims"]) != tuple(pair[f"dims_{side}"])
                or record["weight"] != pair[f"weight_{side}"]):
            raise ValueError(f"{pair['pair_id']}: record id/dimensions/weight mismatch for side {side}")
        image = record["image"]
        image.load()
        if image.mode != "RGB" or min(image.size) <= 0:
            raise ValueError(f"{pair['pair_id']}: record image must be nonempty RGB")
        return record
    except Exception:
        image = record.get("image")
        if image is not None:
            image.close()
        raise


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
                splits.add(pair_split(pair))
                if len(splits) != 1:
                    raise ValueError("pair file mixes train/dev/test splits")
                for side in ("a", "b"):
                    object_id, relative = pair[f"id_{side}"], pair[f"path_{side}"]
                    signature = (relative, tuple(pair[f"dims_{side}"]), pair[f"weight_{side}"])
                    if object_id in objects and objects[object_id] != signature:
                        raise ValueError(f"Object {object_id} has conflicting paths, dimensions, or weights")
                    if relative in path_ids and path_ids[relative] != object_id:
                        raise ValueError(f"Record path {relative!r} has conflicting IDs")
                    if root is not None:
                        _resolved_path(root, relative)
                    if check_images and object_id not in objects:
                        record = _load_validated_record(pair, side, root)
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
    """Require the intended train/dev roles and reject shared objects or paths."""
    ids_by_split, paths_by_split = {}, {}
    for expected, pairs in (("train", train), ("dev", dev)):
        if not pairs:
            raise ValueError(f"{expected}: no pairs found")
        ids, paths = set(), set()
        for pair in pairs:
            pair = _validate_pair(pair)
            if pair_split(pair) != expected:
                raise ValueError(f"Expected {expected} data; found a different split")
            for side in ("a", "b"):
                ids.add(pair[f"id_{side}"])
                paths.add(pair[f"path_{side}"])
        ids_by_split[expected], paths_by_split[expected] = ids, paths
    if ids_by_split["train"] & ids_by_split["dev"]:
        raise ValueError("Object ID leakage between train and dev")
    if paths_by_split["train"] & paths_by_split["dev"]:
        raise ValueError("Record path leakage between train and dev")


def build_messages_for_pair(pair, root):
    """Load checked images, returning prompt-only messages without supervision.

    The caller owns the returned image objects. Neither answer text nor gold
    weights, IDs, paths, or thickness masks enter the existing Qwen prompt.
    """
    from qwen_strong_baseline.pairwise.predict import build_messages

    pair = _validate_pair(pair)
    records = []
    try:
        for side in ("a", "b"):
            records.append(_load_validated_record(pair, side, root))
        return build_messages(records[0]["image"], pair["dims_a"],
                              records[1]["image"], pair["dims_b"])
    except Exception:
        for record in records:
            record["image"].close()
        raise
