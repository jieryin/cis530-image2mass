"""Checkpoint durability and random-state checks without model downloads."""

import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from finetune import checkpoint


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Install PyTorch for checkpoint checks")
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name).resolve()

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.25))

            def save_pretrained(self, path):
                torch.save(self.state_dict(), Path(path) / "adapter.pt")

        class Processor:
            def save_pretrained(self, path):
                (Path(path) / "processor.json").write_text("{}", encoding="utf-8")

        self.model, self.processor = Model(), Processor()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1 / (step + 1))
        (self.model.weight ** 2).backward()
        self.optimizer.step()
        self.scheduler.step()
        self.config = {"model": "offline-test", "seed": 530}

    def save(self, step=1, phase="train"):
        return checkpoint.save_checkpoint(
            self.output, self.model, self.processor, self.optimizer, self.scheduler,
            config=self.config,
            progress={"optimizer_steps": step, "phase": phase, "epoch": 1, "epoch_seen": step * 8},
            device=self.torch.device("cpu"),
        )

    def test_optimizer_scheduler_metadata_and_rng_roundtrip(self):
        torch = self.torch
        random.seed(530)
        torch.manual_seed(530)
        saved = self.save()
        expected_python, expected_tensor = random.random(), torch.rand(4)
        random.seed(99)
        torch.manual_seed(99)
        path, metadata, state = checkpoint.load_checkpoint(self.output)
        self.assertEqual(path, saved)
        self.assertEqual(metadata["config"], self.config)
        self.assertEqual(metadata["progress"]["epoch_seen"], 8)
        self.assertTrue(metadata["optimizer_state_saved"])
        self.assertEqual(state["scheduler"], self.scheduler.state_dict())
        expected_state = self.optimizer.state_dict()["state"][0]
        for key, value in expected_state.items():
            torch.testing.assert_close(state["optimizer"]["state"][0][key], value)
        checkpoint.restore_rng_state(state["rng"], torch.device("cpu"))
        self.assertEqual(random.random(), expected_python)
        torch.testing.assert_close(torch.rand(4), expected_tensor, rtol=0, atol=0)

    def test_failed_serialization_preserves_previous_checkpoint(self):
        saved = self.save()
        previous_pointer = (self.output / "latest_checkpoint.json").read_bytes()
        with patch.object(self.model, "save_pretrained", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.save(step=2)
        self.assertEqual((self.output / "latest_checkpoint.json").read_bytes(), previous_pointer)
        self.assertEqual(checkpoint.load_checkpoint(self.output)[0], saved)
        self.assertEqual(list((self.output / "checkpoints").iterdir()), [saved])

    def test_failed_pointer_publication_keeps_previous_resume_target(self):
        saved = self.save()
        original_write = checkpoint.write_json

        def fail_pointer(path, value):
            if Path(path).name == "latest_checkpoint.json":
                raise OSError("pointer write failed")
            return original_write(path, value)

        with patch.object(checkpoint, "write_json", side_effect=fail_pointer):
            with self.assertRaisesRegex(OSError, "pointer write failed"):
                self.save(step=2)
        self.assertEqual(checkpoint.load_checkpoint(self.output)[0], saved)
        self.assertTrue(saved.exists())

    def test_retains_two_completed_checkpoints_and_leaves_unrelated_files(self):
        old = self.save()
        unrelated = self.output / "checkpoints" / "keep-me"
        unrelated.mkdir()
        (unrelated / "notes.txt").write_text("user notes", encoding="utf-8")
        second = self.save(step=2)
        newest = self.save(step=2, phase="epoch_complete")
        self.assertFalse(old.exists())
        self.assertTrue(second.exists())
        self.assertTrue(newest.exists())
        self.assertEqual((unrelated / "notes.txt").read_text(encoding="utf-8"), "user notes")
        self.assertEqual(checkpoint.load_checkpoint(self.output)[0], newest)
        self.assertEqual(len(list((self.output / "checkpoints").glob("checkpoint-*"))), 2)

    def test_rejects_inference_only_checkpoint_and_escaping_pointer(self):
        inference = self.output / "best"
        checkpoint.save_adapter(inference, self.model, self.processor, {"config": self.config})
        self.assertTrue((inference / "adapter.pt").is_file())
        self.assertTrue((inference / "processor.json").is_file())
        with self.assertRaises(FileNotFoundError):
            checkpoint.load_checkpoint(inference)
        saved = self.save()
        (saved / "training.json").write_text(json.dumps({"config": self.config}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Not a resumable"):
            checkpoint.load_checkpoint(self.output)
        (self.output / "latest_checkpoint.json").write_text(
            json.dumps({"checkpoint": "../elsewhere"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid latest checkpoint pointer"):
            checkpoint.load_checkpoint(self.output)

    def test_rejects_rng_device_change_before_altering_cpu_generators(self):
        self.save()
        _, _, state = checkpoint.load_checkpoint(self.output)
        state["rng"]["device_type"] = "cuda"
        python_before, torch_before = random.getstate(), self.torch.get_rng_state()
        with self.assertRaisesRegex(ValueError, "same device type"):
            checkpoint.restore_rng_state(state["rng"], self.torch.device("cpu"))
        self.assertEqual(random.getstate(), python_before)
        self.torch.testing.assert_close(self.torch.get_rng_state(), torch_before)


if __name__ == "__main__":
    unittest.main()
