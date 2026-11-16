"""Regression tests for the strict JSONL scoring contract."""

import json
import tempfile
import unittest
from pathlib import Path

from score import ScoringError, load_labels, score


class ScoreTests(unittest.TestCase):
    def write_jsonl(self, directory, name, rows):
        """Create a small temporary JSONL fixture for loader validation tests."""
        path = Path(directory) / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_scores_by_pair_id_instead_of_row_order(self):
        # Prediction order is intentionally different from gold order.
        gold = {"p1": "A", "p2": "B", "p3": "A"}
        predictions = {"p3": "A", "p1": "B", "p2": "B"}
        result = score(gold, predictions)
        self.assertEqual(result["correct"], 2)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["confusion_matrix"]["A"]["B"], 1)

    def test_rejects_missing_and_extra_predictions(self):
        with self.assertRaisesRegex(ScoringError, "missing 1 prediction"):
            score({"p1": "A", "p2": "B"}, {"p1": "A"})
        with self.assertRaisesRegex(ScoringError, "unknown pair_id"):
            score({"p1": "A"}, {"p1": "A", "p2": "B"})

    def test_loader_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(
                directory,
                "pred.jsonl",
                [{"pair_id": "p1", "pred": "A"}, {"pair_id": "p1", "pred": "B"}],
            )
            with self.assertRaisesRegex(ScoringError, "duplicate pair_id"):
                load_labels(path, "pred")

    def test_loader_rejects_invalid_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, "pred.jsonl", [{"pair_id": "p1", "pred": "C"}])
            with self.assertRaisesRegex(ScoringError, "must be A or B"):
                load_labels(path, "pred")


if __name__ == "__main__":
    unittest.main()
