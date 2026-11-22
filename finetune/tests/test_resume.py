"""Offline recovery tests with real AdamW updates and stochastic forwards."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch


def make_pair(split, index):
    offset = 100 if split == "train" else 200
    a, b = offset + 2 * index, offset + 2 * index + 1
    label = "A" if index % 2 == 0 else "B"
    return {
        "pair_id": f"{split}-{index:05d}", "id_a": a, "id_b": b,
        "path_a": f"{split}/{a}.pklz", "path_b": f"{split}/{b}.pklz",
        "dims_a": [1, 2, 3], "dims_b": [2, 3, 4],
        "weight_a": 2 if label == "A" else 1,
        "weight_b": 1 if label == "A" else 2, "label": label,
    }


@unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                     "Install PyTorch for training recovery checks")
class ResumeWorkflowTests(unittest.TestCase):
    def setUp(self):
        import torch
        from finetune import train
        from finetune.pairwise_finetune import task as pairwise

        self.torch, self.train, self.pairwise = torch, train, pairwise
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "records"
        self.train_path = self.directory / "train.jsonl"
        self.dev_path = self.directory / "dev.jsonl"
        for split, count, path in (("train", 5, self.train_path), ("dev", 2, self.dev_path)):
            rows = [make_pair(split, index) for index in range(count)]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            for row in rows:
                for side in ("a", "b"):
                    record = self.root / row[f"path_{side}"]
                    record.parent.mkdir(parents=True, exist_ok=True)
                    record.touch()

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4) * 0.1)
                self.dropout = torch.nn.Dropout(0.4)

            def forward(self, input_ids, logits_to_keep=1, use_cache=False):
                assert logits_to_keep == 1 and not use_cache
                value = (self.dropout(self.weight) * input_ids.float()).sum()
                logits = torch.stack((value, -value)).reshape(1, 1, 2)
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

        self.TinyModel, self.TinyProcessor = TinyModel, TinyProcessor

    def arguments(self, output):
        return [
            "--train", str(self.train_path), "--dev", str(self.dev_path),
            "--root", str(self.root), "--output", str(output), "--device", "cpu",
            "--epochs", "2", "--grad-accum", "2", "--save-every", "2",
            "--lr", "0.03", "--seed", "530",
        ]

    def run_training(self, output, *, resume=False, interrupt_pair=None, interrupt_dev=False,
                     extra_args=()):
        torch, train = self.torch, self.train
        train_order = []
        real_evaluate = self.pairwise.evaluate_rows

        def load_model(device, dtype, **kwargs):
            model = self.TinyModel().to(device)
            adapter = kwargs.get("adapter")
            if adapter is not None:
                self.assertTrue(kwargs.get("trainable_adapter"))
                model.load_state_dict(torch.load(Path(adapter) / "toy_adapter.pt",
                                                 map_location=device, weights_only=True))
            return model, self.TinyProcessor()

        def prepare_inputs(processor, row, root, device):
            if row["pair_id"].startswith("train-"):
                if interrupt_pair is not None and len(train_order) == interrupt_pair:
                    raise RuntimeError("simulated interruption during training")
                train_order.append(row["pair_id"])
            # Exercise Python RNG as well as dropout's torch RNG. These values
            # must restart at the saved cursor, including a repeated dev pass.
            feature = (row["id_a"] % 10 + 1) / 10 + random.random() * 0.05
            return {"input_ids": torch.tensor([[feature, 0.3, -0.2, 0.8]], device=device)}

        def evaluate_rows(*args, **kwargs):
            if interrupt_dev:
                partial_args = list(args)
                partial_args[3] = args[3][:1]
                real_evaluate(*partial_args, **kwargs)
                raise RuntimeError("simulated interruption during dev")
            return real_evaluate(*args, **kwargs)

        argv = (["--resume", str(output)] if resume else self.arguments(output)) + list(extra_args)
        with (
            patch.object(train, "load_model", side_effect=load_model),
            patch.object(self.pairwise, "setup", return_value=(0, 1)),
            patch.object(self.pairwise, "prepare_inputs", side_effect=prepare_inputs),
            patch.object(self.pairwise, "evaluate_rows", side_effect=evaluate_rows),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            if interrupt_pair is not None or interrupt_dev:
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    train.main(argv)
            else:
                train.main(argv)
        return train_order

    def history(self, output):
        return [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]

    def assert_state_equal(self, actual, expected):
        if self.torch.is_tensor(expected):
            self.torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_state_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(actual, expected)

    def assert_runs_equal(self, actual, expected, group_sizes=(2, 2, 1, 2, 2, 1)):
        from finetune.checkpoint import load_checkpoint

        for name in ("last", "best"):
            self.assert_state_equal(
                self.torch.load(actual / name / "toy_adapter.pt", weights_only=True),
                self.torch.load(expected / name / "toy_adapter.pt", weights_only=True),
            )
        _, actual_metadata, actual_state = load_checkpoint(actual)
        _, expected_metadata, expected_state = load_checkpoint(expected)
        for key in ("optimizer", "scheduler", "rng"):
            self.assert_state_equal(actual_state[key], expected_state[key])
        for key in ("epoch", "optimizer_steps", "examples_seen"):
            self.assertEqual(actual_metadata["progress"][key], expected_metadata["progress"][key])
        actual_history, expected_history = self.history(actual), self.history(expected)
        self.assertEqual([row["optimizer_step"] for row in actual_history],
                         list(range(1, len(group_sizes) + 1)))
        for a, b in zip(actual_history, expected_history):
            for key in ("epoch", "optimizer_step", "examples", "loss", "lr"):
                self.assertEqual(a[key], b[key], key)
        self.assertEqual([row["examples"] for row in actual_history], list(group_sizes))
        # The logged LR is the rate used for the update, not the next rate.
        self.assertGreater(actual_history[-1]["lr"], 0)

    def test_mid_epoch_resume_replays_only_unsaved_pairs_and_restores_adam_rng(self):
        expected, actual = self.directory / "expected", self.directory / "resumed"
        uninterrupted_order = self.run_training(expected)
        interrupted_order = self.run_training(actual, interrupt_pair=9)
        self.assertEqual(interrupted_order, uninterrupted_order[:9])
        self.assertEqual(self.history(actual)[-1]["optimizer_step"], 5)
        resumed_order = self.run_training(actual, resume=True)
        # Step 4 is durable: epoch one (5 pairs), then 2 pairs of epoch two.
        self.assertEqual(resumed_order, uninterrupted_order[7:])
        self.assert_runs_equal(actual, expected)

    def test_interrupted_dev_restarts_evaluation_without_repeating_epoch_training(self):
        expected, actual = self.directory / "expected", self.directory / "resumed"
        uninterrupted_order = self.run_training(expected)
        interrupted_order = self.run_training(actual, interrupt_dev=True)
        self.assertEqual(interrupted_order, uninterrupted_order[:5])
        resumed_order = self.run_training(actual, resume=True)
        self.assertEqual(resumed_order, uninterrupted_order[5:])
        self.assert_runs_equal(actual, expected)

    def test_interrupted_pilot_dev_does_not_train_past_saved_max_steps(self):
        expected, actual = self.directory / "expected", self.directory / "resumed"
        limits = ("--max-steps", "2")
        expected_order = self.run_training(expected, extra_args=limits)
        self.assertEqual(len(expected_order), 4)
        self.assertEqual(self.run_training(actual, interrupt_dev=True, extra_args=limits), expected_order)
        self.assertEqual(self.run_training(actual, resume=True), [])
        self.assert_runs_equal(actual, expected, group_sizes=(2, 2))
        run = json.loads((actual / "run.json").read_text())
        self.assertEqual(run["status"], "pilot_complete")
        self.assertEqual(run["examples_seen"], 4)

    def test_changed_training_options_and_data_are_rejected_before_model_load(self):
        output = self.directory / "resumed"
        self.run_training(output, interrupt_dev=True)
        with patch.object(self.train, "load_model") as load_model, contextlib.redirect_stderr(io.StringIO()):
            for option, value in (("--lr", "0.5"), ("--grad-accum", "3"), ("--epochs", "4")):
                with self.subTest(option=option), self.assertRaises((ValueError, SystemExit)):
                    self.train.main(["--resume", str(output), option, value])
            load_model.assert_not_called()
        # Even a valid JSONL formatting change must invalidate the saved hash.
        with self.train_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with patch.object(self.train, "load_model") as load_model:
            with self.assertRaises(ValueError):
                self.train.main(["--resume", str(output)])
            load_model.assert_not_called()

    def test_resuming_completed_run_does_not_add_updates_or_history(self):
        output = self.directory / "complete"
        self.run_training(output)
        before = (output / "history.jsonl").read_bytes()
        self.assertEqual(self.run_training(output, resume=True), [])
        self.assertEqual((output / "history.jsonl").read_bytes(), before)
        run = json.loads((output / "run.json").read_text())
        self.assertEqual(run["optimizer_steps"], 6)
        self.assertEqual(run["examples_seen"], 10)

    def test_moved_run_resumes_in_selected_directory_with_complete_history(self):
        expected, original = self.directory / "expected", self.directory / "original"
        moved = self.directory / "moved"
        expected_order = self.run_training(expected)
        self.run_training(original, interrupt_dev=True)
        original.rename(moved)
        self.assertEqual(self.run_training(moved, resume=True), expected_order[5:])
        self.assertFalse(original.exists())
        self.assert_runs_equal(moved, expected)
        run = json.loads((moved / "run.json").read_text())
        self.assertEqual(run["config"]["output"], str(moved.resolve()))

    def test_copied_completed_run_keeps_original_untouched_and_rejects_output_redirect(self):
        from finetune.checkpoint import load_checkpoint

        original, copied = self.directory / "original", self.directory / "copied"
        self.run_training(original)
        shutil.copytree(original, copied)
        original_run = (original / "run.json").read_bytes()
        copied_history = (copied / "history.jsonl").read_bytes()
        saved_config = load_checkpoint(copied)[1]["config"]
        with self.assertRaisesRegex(ValueError, "--output"):
            self.run_training(copied, resume=True, extra_args=("--output", str(original)))
        self.assertEqual(self.run_training(copied, resume=True, extra_args=("--output", str(copied))), [])
        self.assertEqual((original / "run.json").read_bytes(), original_run)
        self.assertEqual((copied / "history.jsonl").read_bytes(), copied_history)
        self.assertEqual(load_checkpoint(copied)[1]["config"], saved_config)
        run = json.loads((copied / "run.json").read_text())
        self.assertEqual(run["config"]["output"], str(copied.resolve()))


if __name__ == "__main__":
    unittest.main()
