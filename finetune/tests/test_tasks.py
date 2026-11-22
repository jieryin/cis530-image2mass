"""Task boundaries and shared orchestration, without pretrained-model downloads."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from finetune import train
from finetune.data import unique_objects
from finetune import get_task
from finetune.pairwise_finetune import task as pairwise
from finetune.pointwise_finetune import task as pointwise
from prompts.pairwise import PROMPT as PAIRWISE_PROMPT
from prompts.pointwise import PROMPT as POINTWISE_PROMPT


HAS_TORCH = importlib.util.find_spec("torch") is not None


def pairs(split):
    offset = 100 if split == "train" else 200
    objects = [(offset, [1, 2, 3], 3), (offset + 1, [2, 3, 4], 1), (offset + 2, [3, 4, 5], 4)]
    rows = []
    for index, other in enumerate(objects[1:]):
        first = objects[0]
        rows.append({
            "pair_id": f"{split}-{index}", "id_a": first[0], "id_b": other[0],
            "path_a": f"{split}/{first[0]}.pklz", "path_b": f"{split}/{other[0]}.pklz",
            "dims_a": first[1], "dims_b": other[1],
            "weight_a": first[2], "weight_b": other[2],
            "label": "A" if first[2] > other[2] else "B",
        })
    return rows


class _TaskFixture:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "records"
        self.training, self.dev = pairs("train"), pairs("dev")
        self.train_path, self.dev_path = self.directory / "train.jsonl", self.directory / "dev.jsonl"
        for path, rows in ((self.train_path, self.training), (self.dev_path, self.dev)):
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            for obj in unique_objects(rows):
                record = self.root / obj["path"]
                record.parent.mkdir(parents=True, exist_ok=True)
                record.touch()
        self.output = self.directory / "output"

    def arguments(self, task="pointwise"):
        return ["--task", task, "--train", str(self.train_path), "--dev", str(self.dev_path),
                "--root", str(self.root), "--output", str(self.output), "--device", "cpu"]


class TaskDataTests(_TaskFixture, unittest.TestCase):
    def test_task_selection_preserves_prompts_and_deduplicates_only_pointwise(self):
        self.assertIs(get_task("pairwise"), pairwise)
        self.assertIs(get_task("pointwise"), pointwise)
        self.assertEqual(pairwise.PROMPT, PAIRWISE_PROMPT)
        self.assertEqual(pointwise.PROMPT, POINTWISE_PROMPT)
        self.assertEqual(pairwise.training_rows(self.training), self.training)
        objects = pointwise.training_rows(self.training)
        self.assertEqual([obj["id"] for obj in objects], [100, 101, 102])
        self.assertEqual([obj["weight"] for obj in objects], [3, 1, 4])
        self.assertTrue(all(set(obj) == {"id", "path", "dims", "weight", "split"} for obj in objects))
        self.assertEqual(pairwise.metadata((3, 7))["answer_policy"], "leading_choice_only_v1")
        self.assertEqual(pairwise.metadata((3, 7))["answer_token_ids"], [3, 7])

    def test_pointwise_dry_run_counts_unique_objects_and_creates_no_run(self):
        stream = io.StringIO()
        with patch.object(train, "load_model") as loader, contextlib.redirect_stdout(stream):
            train.main(self.arguments() + ["--dry-run"])
        result = json.loads(stream.getvalue())
        self.assertEqual((result["task"], result["count_unit"]), ("pointwise", "objects"))
        self.assertEqual(result["train_examples"], 3)
        self.assertEqual(result["dev_pairs"], 2)
        self.assertFalse(result["model_loaded"])
        loader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_pointwise_training_loss_is_an_explicit_placeholder(self):
        with self.assertRaisesRegex(NotImplementedError, "training_loss"):
            pointwise.training_loss(None, None, None, {"weight": 3}, None, None)
        with self.assertRaises(NotImplementedError):
            pointwise.selection_key({"accuracy": 1.0})
        self.assertEqual(pointwise.metadata()["loss"], "not_implemented")
        self.assertEqual(pointwise.metadata()["selection"], "not_implemented")

    def test_resume_rejects_changing_task_before_model_or_checkpoint_use(self):
        args = train.parse_args(self.arguments("pointwise"))
        # A real resume restores all recorded options; a task conflict must be
        # rejected before any task-specific state is interpreted.
        saved = {name: str(value) if isinstance(value, Path) else value
                 for name, value in vars(args).items() if not name.startswith("_") and name != "resume"}
        saved["task"] = "pairwise"
        with self.assertRaisesRegex(ValueError, "--task"):
            train.resume_arguments(args, saved)

    def test_evaluation_rejects_explicit_task_mismatch_before_model_load(self):
        from finetune import evaluate

        checkpoint = self.directory / "adapter"
        checkpoint.mkdir()
        (checkpoint / "training.json").write_text(json.dumps({"config": {
            "task": "pairwise", "model": train.MODEL_ID, "revision": train.REVISION,
            **pairwise.metadata((3, 7)), "dtype": "float32", "max_pixels": 262144,
        }}), encoding="utf-8")
        with patch.object(evaluate, "load_model") as loader:
            with self.assertRaisesRegex(ValueError, "task"):
                evaluate.main(["--task", "pointwise", "--checkpoint", str(checkpoint),
                               "--pairs", str(self.dev_path), "--root", str(self.root),
                               "--output", str(self.output)])
        loader.assert_not_called()
        self.assertFalse(self.output.exists())


@unittest.skipUnless(HAS_TORCH, "Install PyTorch for task tensor/workflow checks")
class TaskTensorTests(_TaskFixture, unittest.TestCase):
    def test_evaluation_infers_task_without_enforcing_training_only_settings(self):
        from finetune import evaluate

        checkpoint = self.directory / "adapter"
        checkpoint.mkdir()
        config = {"task": "pairwise", "model": train.MODEL_ID, "revision": train.REVISION,
                  **pairwise.metadata((1, 3)), "dtype": "float32", "max_pixels": 262144,
                  "task_prompt": pairwise.PROMPT, "loss": "training-only decision",
                  "selection": "training-only decision"}
        (checkpoint / "training.json").write_text(json.dumps({"config": config}), encoding="utf-8")
        predictions = {row["pair_id"]: row["label"] for row in self.dev}
        probabilities = {row["pair_id"]: float(row["label"] == "A") for row in self.dev}
        metrics = pairwise.subset_metrics(self.dev, predictions, probabilities)
        with patch.object(evaluate, "load_model", return_value=(None, None)), \
                patch.object(pairwise, "setup", return_value=(1, 3)), \
                patch.object(pairwise, "evaluate_rows", return_value=(predictions, probabilities, metrics)), \
                contextlib.redirect_stdout(io.StringIO()):
            evaluate.main(["--checkpoint", str(checkpoint), "--pairs", str(self.dev_path),
                           "--root", str(self.root), "--output", str(self.output), "--device", "cpu"])
        self.assertEqual(json.loads((self.output / "scores.json").read_text())["accuracy"], 1)
        self.assertEqual(json.loads((self.output / "run.json").read_text())["task"], "pairwise")

    def test_pairwise_loss_and_prediction_keep_original_probability_rule(self):
        import torch

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.zeros(1, 1, 5))

            def forward(self, **kwargs):
                self.request = kwargs
                return SimpleNamespace(logits=self.logits)

        model = TinyModel()
        with patch.object(pairwise, "prepare_inputs", return_value={"input_ids": torch.tensor([[1]])}):
            loss = pairwise.training_loss(model, None, (1, 3), self.training[0], None, torch.device("cpu"))
        self.assertEqual(loss.item(), 0.25)
        loss.backward()
        self.assertEqual(model.logits.grad[0, 0, 1].item(), -0.25)
        self.assertEqual(model.logits.grad[0, 0, 3].item(), 0.25)
        self.assertEqual(model.request["logits_to_keep"], 1)
        self.assertNotIn("labels", model.request)

    def test_pointwise_evaluation_predicts_each_object_once_and_scores_same_slice(self):
        import torch

        seen = []

        def prepare(processor, obj, root, device):
            seen.append(obj["id"])
            return {"object_id": obj["id"]}

        with patch.object(pointwise, "prepare_inputs", side_effect=prepare), \
                patch.object(pointwise, "predict_weight", side_effect=[2.5, 1.2, 4.2]):
            predictions, weights, metrics = pointwise.evaluate_rows(
                SimpleNamespace(eval=lambda: None), None, None, self.dev, None, torch.device("cpu"))
        self.assertEqual(seen, [200, 201, 202])
        self.assertEqual(predictions, {"dev-0": "A", "dev-1": "B"})
        self.assertEqual(metrics["accuracy"], 1)
        self.assertEqual(metrics["objects"], 3)
        subset_metrics = pointwise.subset_metrics(self.dev[:1], predictions, weights)
        self.assertEqual(subset_metrics["objects"], 2)
        self.assertEqual(subset_metrics["accuracy"], 1)
        pointwise.save_predictions(self.output, predictions, weights, metrics)
        saved = [json.loads(line) for line in (self.output / "weights.jsonl").read_text().splitlines()]
        self.assertEqual(saved, [{"id": 200, "weight_lb": 2.5}, {"id": 201, "weight_lb": 1.2},
                                 {"id": 202, "weight_lb": 4.2}])

    def test_pointwise_generation_keeps_numeric_grammar_and_rejects_truncation(self):
        import torch

        callback = lambda batch, tokens: [3, 9]
        constraint = SimpleNamespace(eos_token_ids=(9,), for_prompt=lambda length: callback)
        processor = SimpleNamespace(batch_decode=lambda *args, **kwargs: ["2.5"])
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return torch.tensor([[1, 2, 3, 9]])

        model = SimpleNamespace(generate=generate)
        inputs = {"input_ids": torch.tensor([[1, 2]])}
        self.assertEqual(pointwise.predict_weight(model, processor, inputs, constraint), 2.5)
        self.assertIs(calls[0]["prefix_allowed_tokens_fn"], callback)
        self.assertFalse(calls[0]["do_sample"])
        self.assertEqual(calls[0]["max_new_tokens"], 32)
        processor.batch_decode = lambda *args, **kwargs: ["The weight is 2.5"]
        with self.assertRaises(ValueError):
            pointwise.predict_weight(model, processor, inputs, constraint)
        model.generate = lambda **kwargs: torch.tensor([[1, 2] + [3] * 32])
        with self.assertRaisesRegex(ValueError, "token limit"):
            pointwise.predict_weight(model, processor, inputs, constraint)

    def test_common_loop_uses_pointwise_example_and_loss_hooks(self):
        import torch

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))

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

        model, seen = TinyModel(), []

        def toy_loss(model, processor, context, row, root, device):
            # This test-only scalar objective proves that common orchestration
            # accepts object examples. It does not implement numeric text loss.
            seen.append(row["id"])
            self.assertNotIn("label", row)
            return (model.weight - row["weight"]) ** 2

        expected_predictions = {row["pair_id"]: row["label"] for row in self.dev}
        expected_weights = {obj["id"]: obj["weight"] for obj in unique_objects(self.dev)}
        evaluation = (expected_predictions, expected_weights, {"accuracy": 1.0})
        with patch.object(train, "load_model", return_value=(model, TinyProcessor())), \
                patch.object(pointwise, "setup", return_value=None), \
                patch.object(pointwise, "training_loss", side_effect=toy_loss), \
                patch.object(pointwise, "selection_key", side_effect=lambda metrics: metrics["accuracy"]), \
                patch.object(pointwise, "evaluate_rows", return_value=evaluation), \
                contextlib.redirect_stdout(io.StringIO()):
            train.main(self.arguments() + ["--epochs", "1", "--grad-accum", "2", "--save-every", "1"])
        self.assertEqual(sorted(seen), [100, 101, 102])
        self.assertNotEqual(model.weight.item(), 0)
        run = json.loads((self.output / "run.json").read_text())
        self.assertEqual((run["config"]["task"], run["examples_seen"], run["optimizer_steps"]),
                         ("pointwise", 3, 2))
        history = [json.loads(line) for line in (self.output / "history.jsonl").read_text().splitlines()]
        self.assertEqual([row["examples"] for row in history], [2, 1])
        self.assertTrue((self.output / "latest_checkpoint.json").is_file())


if __name__ == "__main__":
    unittest.main()
