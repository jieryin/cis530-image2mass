"""Subset reporting and a small offline training-loop integration test."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from finetune.pairwise_finetune.task import subset_metrics
from finetune.data import validate_subset


def pair(split, index):
    offset = 100 if split == "train" else 200
    first, second = offset + 2 * index, offset + 2 * index + 1
    return {
        "pair_id": f"{split}-{index:05d}",
        "id_a": first, "id_b": second,
        "path_a": f"{split}/{first}.pklz", "path_b": f"{split}/{second}.pklz",
        "dims_a": [1.0, 2.0, 3.0], "dims_b": [2.0, 3.0, 4.0],
        "weight_a": 2.0, "weight_b": 1.0, "label": "A",
    }


class SubsetMetricsTests(unittest.TestCase):
    def test_subset_uses_full_run_predictions_and_its_own_denominator(self):
        rows = [pair("test", i) for i in range(3)]
        rows[1]["label"] = "B"
        rows[1]["weight_a"], rows[1]["weight_b"] = 1.0, 2.0
        predictions = {rows[0]["pair_id"]: "A", rows[1]["pair_id"]: "A", rows[2]["pair_id"]: "B"}
        probabilities = {rows[0]["pair_id"]: 0.8, rows[1]["pair_id"]: 0.6, rows[2]["pair_id"]: 0.4}
        metrics = subset_metrics(copy.deepcopy(rows[:2]), predictions, probabilities)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["mse"], (0.2 ** 2 + 0.6 ** 2) / 2)

    def test_unknown_pair_and_any_changed_input_or_target_are_rejected(self):
        original = pair("test", 0)
        mutations = {
            "pair_id": "test-99999", "id_a": 999, "id_b": 998,
            "path_a": "test/other_a.pklz", "path_b": "test/other_b.pklz",
            "dims_a": [9, 9, 9], "dims_b": [8, 8, 8],
            "weight_a": 3, "weight_b": 0.5, "label": "B",
        }
        for field, value in mutations.items():
            modified = copy.deepcopy(original)
            modified[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "not an exact slice"):
                validate_subset([original], [modified])


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Install PyTorch for training-loop checks")
class TrainingWorkflowTests(unittest.TestCase):
    def run_toy_training(self, max_steps):
        import torch
        from finetune import train
        from finetune.pairwise_finetune import task as pairwise

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, input_ids, logits_to_keep=1, use_cache=False):
                assert logits_to_keep == 1 and not use_cache
                logits = torch.stack((self.weight, self.weight * 0)).reshape(1, 1, 2)
                return type("Output", (), {"logits": logits})()

            def save_pretrained(self, directory):
                directory = Path(directory)
                directory.mkdir(parents=True, exist_ok=True)
                torch.save(self.state_dict(), directory / "toy_adapter.pt")

        class TinyProcessor:
            chat_template = "<assistant>\n"

            @property
            def image_processor(self):
                return self

            def to_dict(self):
                return {"size": {"shortest_edge": 1024, "longest_edge": 1024}}

            def save_pretrained(self, directory):
                (Path(directory) / "toy_processor.json").write_text("{}", encoding="utf-8")

        gradients = []

        class RecordingOptimizer(torch.optim.SGD):
            def step(self, closure=None):
                gradients.append([parameter.grad.detach().clone()
                                  for group in self.param_groups for parameter in group["params"]])
                return super().step(closure)

        # A zero update keeps each per-pair analytical gradient at -0.25.
        # Thus a mistakenly fixed /grad_accum denominator on the final singleton
        # group is observable as -0.125 rather than masked by Adam normalization.
        def optimizer_factory(parameters, **kwargs):
            return RecordingOptimizer(parameters, lr=0.0)

        def prepare_inputs(processor, row, root, device):
            return {"input_ids": torch.tensor([[1, 2, 3]], device=device)}

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, output = directory / "records", directory / "run"
            training, development = [pair("train", i) for i in range(5)], [pair("dev", 0)]
            for row in training + development:
                for side in ("a", "b"):
                    record_path = root / row[f"path_{side}"]
                    record_path.parent.mkdir(parents=True, exist_ok=True)
                    record_path.touch()
            train_path, dev_path = directory / "train.jsonl", directory / "dev.jsonl"
            train_path.write_text("\n".join(json.dumps(row) for row in training), encoding="utf-8")
            dev_path.write_text("\n".join(json.dumps(row) for row in development), encoding="utf-8")
            model = TinyModel()
            with (
                patch.object(train, "load_model", return_value=(model, TinyProcessor())),
                patch.object(pairwise, "setup", return_value=(0, 1)),
                patch.object(pairwise, "prepare_inputs", side_effect=prepare_inputs),
                patch.object(torch.optim, "AdamW", side_effect=optimizer_factory),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                train.main([
                    "--train", str(train_path), "--dev", str(dev_path),
                    "--root", str(root), "--output", str(output),
                    "--device", "cpu", "--epochs", "1", "--grad-accum", "2",
                    "--max-steps", str(max_steps),
                ])
            run = json.loads((output / "run.json").read_text(encoding="utf-8"))
            history = [json.loads(line) for line in (output / "history.jsonl").read_text(encoding="utf-8").splitlines()]
            snapshots = {}
            for name in ("best", "last"):
                checkpoint = output / name
                self.assertTrue((checkpoint / "toy_adapter.pt").is_file())
                self.assertTrue((checkpoint / "toy_processor.json").is_file())
                snapshots[name] = json.loads((checkpoint / "training.json").read_text(encoding="utf-8"))
            predictions = [json.loads(line) for line in (output / "dev_epoch_1" / "preds.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(predictions[0]["pred"], "B")  # Explicit tie convention.
            self.assertAlmostEqual(predictions[0]["p_a"], 0.5)
        return run, history, snapshots, gradients

    def test_partial_accumulation_group_uses_actual_size_and_saves_checkpoints(self):
        run, history, snapshots, gradients = self.run_toy_training(max_steps=3)
        self.assertEqual(run["status"], "pilot_complete")
        self.assertEqual(run["optimizer_steps"], 3)
        self.assertEqual(run["examples_seen"], 5)
        self.assertEqual([event["examples"] for event in history], [2, 2, 1])
        self.assertEqual(len(gradients), 3)
        for group in gradients:
            self.assertAlmostEqual(group[0].item(), -0.25)
        for snapshot in snapshots.values():
            self.assertTrue(snapshot["epoch_complete"])
            self.assertFalse(snapshot["optimizer_state_saved"])
            self.assertAlmostEqual(snapshot["train_loss"], 0.25)
            self.assertEqual(snapshot["config"]["probability_space"], "binary")

    def test_pilot_stops_after_optimizer_step_and_marks_partial_epoch(self):
        run, history, snapshots, gradients = self.run_toy_training(max_steps=1)
        self.assertEqual(run["status"], "pilot_complete")
        self.assertEqual(run["optimizer_steps"], 1)
        self.assertEqual(run["examples_seen"], 2)
        self.assertEqual(len(history), 1)
        self.assertEqual(len(gradients), 1)
        self.assertFalse(snapshots["last"]["epoch_complete"])


if __name__ == "__main__":
    unittest.main()
