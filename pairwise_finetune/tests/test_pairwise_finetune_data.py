"""Supervision, leakage, and record checks without model/image dependencies."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from pairwise_finetune.data import (
    build_messages_for_pair, normalize_answer, read_pairs, validate_disjoint,
)


def example(split="train", first=10001, second=10002, number=0):
    return {
        "pair_id": f"{split}-{number:05d}",
        "id_a": first, "id_b": second,
        "path_a": f"{split}/{first}_secret_name_a.pklz",
        "path_b": f"{split}/{second}_secret_name_b.pklz",
        "dims_a": [1, 2, 3], "dims_b": [2, 3, 4],
        "weight_a": 9876, "weight_b": 5432, "label": "A",
    }


class FakeImage:
    mode = "RGB"
    size = (8, 8)

    def __init__(self):
        self.loaded = False
        self.closed = False

    def load(self):
        self.loaded = True

    def close(self):
        self.closed = True


def records_for(pair):
    return {
        pair[f"path_{side}"]: {
            "id": pair[f"id_{side}"], "dims": tuple(pair[f"dims_{side}"]),
            "weight": pair[f"weight_{side}"], "image": FakeImage(),
        }
        for side in ("a", "b")
    }


class PairDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.pairs_file = self.root / "pairs.jsonl"

    def read(self, rows, **kwargs):
        self.pairs_file.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        return read_pairs(self.pairs_file, **kwargs)

    def test_plain_and_long_answers_have_identical_supervision(self):
        for answer in ("A", "  A  ", "A. " + "long rationale mentioning B " * 1000, "A\nexplanation"):
            with self.subTest(answer=answer[:25]):
                pair = example()
                pair["answer"] = answer
                self.assertEqual(self.read([pair])[0]["answer"], "A")
        self.assertEqual(self.read([example()])[0]["answer"], "A")
        self.assertEqual(normalize_answer("B\nA is lighter.", "B"), "B")

    def test_answer_must_lead_with_matching_standalone_choice(self):
        for answer in ("", "AB", "Apple", "A/B", "Answer: A", "Because B is lighter, A", "B. explanation", None):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                normalize_answer(answer, "A")

    def test_incorrect_labels_and_invalid_targets_are_rejected(self):
        for key, value in (("label", "B"), ("label", 1), ("weight_a", 0),
                           ("weight_a", -1), ("weight_a", True), ("weight_a", "9876"),
                           ("weight_a", float("nan")), ("weight_b", float("inf"))):
            pair = example()
            pair[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.read([pair])

    def test_tie_uses_existing_b_label_convention(self):
        pair = example()
        pair["weight_b"] = pair["weight_a"]
        pair["label"] = "B"
        self.assertEqual(self.read([pair])[0]["answer"], "B")

    def test_dimensions_and_identity_are_validated(self):
        for key, value in (("dims_a", [1, 2]), ("dims_a", [1, 2, 0]),
                           ("dims_a", [1, 2, True]), ("dims_b", [1, 2, float("nan")]),
                           ("id_a", True), ("id_a", -1), ("id_a", 1.5), ("id_a", 10002)):
            pair = example()
            pair[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.read([pair])

    def test_paths_cannot_escape_or_change_splits(self):
        for path in ("/train/a.pklz", "../a.pklz", "train/../dev/a.pklz", "C:\\train\\a.pklz",
                     "train//a.pklz", "train/./a.pklz", "dev/a.pklz", "a.pklz", "train/a.txt"):
            pair = example()
            pair["path_a"] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.read([pair])

    def test_symlink_escape_and_split_alias_are_rejected(self):
        (self.root / "train").mkdir()
        pair = example()
        pair["path_a"] = "train/alias.pklz"
        alias = self.root / "train" / "alias.pklz"
        alias.symlink_to(self.root.parent / "outside-record.pklz")
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.read([pair], root=self.root)
        alias.unlink()
        alias.symlink_to(self.root / "dev" / "record.pklz")
        with self.assertRaisesRegex(ValueError, "changes the data split"):
            self.read([pair], root=self.root)

    def test_duplicate_pair_ids_and_reversed_pairs_are_rejected(self):
        first = example()
        second = example(first=10003, second=10004)
        with self.assertRaisesRegex(ValueError, "duplicate pair_id"):
            self.read([first, second])
        reverse = copy.deepcopy(first)
        reverse["pair_id"] = "train-00001"
        for field in ("id", "path", "dims", "weight"):
            reverse[f"{field}_a"], reverse[f"{field}_b"] = reverse[f"{field}_b"], reverse[f"{field}_a"]
        reverse["label"] = "B"
        with self.assertRaisesRegex(ValueError, "duplicate unordered"):
            self.read([first, reverse])

    def test_reused_objects_require_consistent_metadata(self):
        first = example()
        for field, value in (("weight_a", 10000), ("dims_a", [8, 9, 10]), ("path_a", "train/alias.pklz")):
            second = example(second=10003, number=1)
            second[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "conflicting"):
                self.read([first, second])

    def test_same_path_cannot_claim_multiple_ids(self):
        first = example()
        second = example(first=10003, second=10004, number=1)
        second["path_a"] = first["path_a"]
        with self.assertRaisesRegex(ValueError, "conflicting IDs"):
            self.read([first, second])

    def test_files_and_train_dev_arguments_keep_split_roles(self):
        train, dev = example(), example("dev", first=20001, second=20002)
        with self.assertRaisesRegex(ValueError, "mixes"):
            self.read([train, dev])
        validate_disjoint([train], [dev])
        with self.assertRaisesRegex(ValueError, "Expected train"):
            validate_disjoint([dev], [train])
        with self.assertRaises(ValueError):
            validate_disjoint([train], [example("test", first=20001, second=20002)])

    def test_object_ids_must_be_disjoint_even_when_paths_differ(self):
        with self.assertRaisesRegex(ValueError, "Object ID leakage"):
            validate_disjoint([example()], [example("dev", first=10001, second=20002)])

    def test_empty_and_invalid_json_fail_with_location(self):
        with self.assertRaisesRegex(ValueError, "no pairs"):
            self.read([])
        self.pairs_file.write_text("\n{broken", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, r"pairs.jsonl:2:"):
            read_pairs(self.pairs_file)
        with self.assertRaisesRegex(ValueError, "expected a JSON object"):
            self.read([["A"]])

    def test_image_check_requires_root(self):
        with self.assertRaisesRegex(ValueError, "root is required"):
            self.read([example()], check_images=True)

    def test_image_check_decodes_each_unique_record_and_closes_it(self):
        first, second = example(), example(second=10003, number=1)
        records = {**records_for(first), **records_for(second)}
        loader = Mock(side_effect=lambda path: records[path.relative_to(self.root).as_posix()])
        with patch.dict(sys.modules, {"load_record": types.SimpleNamespace(load_record=loader)}):
            rows = self.read([first, second], root=self.root, check_images=True)
        self.assertEqual(len(rows), 2)
        self.assertEqual(loader.call_count, 3)
        self.assertTrue(all(record["image"].loaded and record["image"].closed for record in records.values()))

    def test_prompt_contains_only_images_and_dimensions(self):
        pair = example()
        pair["answer"] = "A. SECRET_RATIONALE with gold weight 9876"
        records = records_for(pair)
        loader = Mock(side_effect=lambda path: records[path.relative_to(self.root).as_posix()])
        with patch.dict(sys.modules, {"load_record": types.SimpleNamespace(load_record=loader)}):
            messages = build_messages_for_pair(pair, self.root)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        content = messages[0]["content"]
        self.assertEqual(sum(part["type"] == "image" for part in content), 2)
        text = " ".join(part["text"] for part in content if part["type"] == "text")
        for forbidden in ("SECRET_RATIONALE", "9876", "5432", "secret_name", "10001", "10002", "train-"):
            self.assertNotIn(forbidden, text)
        self.assertIn("[1.0, 2.0, 3.0]", text)
        self.assertTrue(all(record["image"].loaded and not record["image"].closed for record in records.values()))

    def test_record_mismatch_closes_images_before_raising(self):
        for field, wrong in (("id", 123), ("dims", (9, 9, 9)), ("weight", 1)):
            pair = example()
            records = records_for(pair)
            records[pair["path_b"]][field] = wrong
            loader = Mock(side_effect=lambda path: records[path.relative_to(self.root).as_posix()])
            with self.subTest(field=field), patch.dict(sys.modules, {"load_record": types.SimpleNamespace(load_record=loader)}):
                with self.assertRaisesRegex(ValueError, "record id/dimensions/weight mismatch"):
                    build_messages_for_pair(pair, self.root)
            self.assertTrue(all(record["image"].closed for record in records.values()))


if __name__ == "__main__":
    unittest.main()
