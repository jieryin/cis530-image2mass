"""Offline checks for the API baseline's input, cache, and scoring contracts."""

import asyncio
import base64
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from openai_strong_baseline import runner as baseline
from score import load_labels, print_report, score


class FakeImage:
    def __init__(self, data=b"fake image bytes for offline tests"):
        self.data = data

    def save(self, buffer, format):
        if format != "PNG":
            raise AssertionError("The baseline should send a supported PNG image")
        buffer.write(self.data)


class FakeResponses:
    def __init__(self, results):
        self.results = results if callable(results) else iter(results)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results(kwargs) if callable(self.results) else next(self.results)
        return self.make_response(result, kwargs)

    def make_response(self, result, kwargs):
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(
            status="completed",
            output=[],
            output_text=str(result),
            id=f"response-{len(self.calls)}",
            model=kwargs["model"],
            usage=None,
        )


class ConcurrentResponses(FakeResponses):
    """Track real coroutine overlap and derive responses from fixture image pixels."""

    def __init__(self, *, delays=None, fail_key=None, overlap_target=2):
        super().__init__([])
        self.delays = delays or {}
        self.fail_key = fail_key
        self.started = []
        self.completed = []
        self.active = 0
        self.max_active = 0
        self.overlap_target = overlap_target
        self.overlap_ready = asyncio.Event()

    async def create(self, **kwargs):
        key = tuple(
            int(base64.b64decode(part["image_url"].split(",", 1)[1]).rsplit(b" ", 1)[1])
            for message in kwargs["input"] for part in message["content"]
            if part["type"] == "input_image"
        )
        self.calls.append(kwargs)
        self.started.append(key)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if len(self.started) >= self.overlap_target:
            self.overlap_ready.set()
        try:
            await asyncio.wait_for(self.overlap_ready.wait(), timeout=2)
            if key == self.fail_key:
                raise RuntimeError("offline concurrent API failure")
            await asyncio.sleep(self.delays.get(key, 0.01))
            weights = {0: 8.0, 1: 3.0, 2: 12.0}
            result = weights[key[0]] if len(key) == 1 else ("A" if weights[key[0]] > weights[key[1]] else "B")
            self.completed.append(key)
            return self.make_response(result, kwargs)
        finally:
            self.active -= 1


class OpenAIStrongBaselineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.root = self.directory / "records"
        self.root.mkdir()
        self.pairs_path = self.directory / "test_pairs.jsonl"
        self.output = self.directory / "predictions.jsonl"
        self.cache = self.directory / "cache.json"
        self.env_file = self.directory / "test.env"
        self.records = {}
        for index, object_id in enumerate((71001, 71002, 71003)):
            path = self.root / f"private-record-{object_id}.pklz"
            path.write_bytes(f"source bytes for object {object_id}".encode())
            self.records[path] = {
                "id": object_id,
                "image": FakeImage(f"image pixels {index}".encode()),
                "dims": (2.25 + index, 3.5 + index, 4.75 + index),
                "weight": 987654.321 + object_id,
            }
        self.pairs = [
            self.make_pair("private-pair-one", 71001, 71002, "A"),
            self.make_pair("private-pair-two", 71001, 71003, "B"),
            self.make_pair("private-pair-three", 71002, 71003, "B"),
        ]
        self.write_pairs()

    def write_pairs(self):
        self.pairs_path.write_text(
            "".join(json.dumps(pair) + "\n" for pair in self.pairs),
            encoding="utf-8",
        )

    def prefix_record_paths(self, split):
        for pair in self.pairs:
            for side in ("a", "b"):
                key = f"path_{side}"
                original = self.root / pair[key]
                destination = self.root / split / original.name
                destination.parent.mkdir(exist_ok=True)
                if destination != original:
                    destination.write_bytes(original.read_bytes())
                    self.records[destination] = self.records[original]
                pair[key] = f"{split}/{original.name}"

    def make_pair(self, pair_id, object_a, object_b, label):
        pair = {"pair_id": pair_id, "label": label}
        for side, object_id in (("a", object_a), ("b", object_b)):
            pair[f"id_{side}"] = object_id
            pair[f"path_{side}"] = f"private-record-{object_id}.pklz"
            record = self.records[self.root / pair[f"path_{side}"]]
            pair[f"dims_{side}"] = list(record["dims"])
            pair[f"weight_{side}"] = record["weight"]
        return pair

    def arguments(self, *extra, model="offline-test-model", single_output=True):
        options = [
            "--pairs", str(self.pairs_path),
            "--root", str(self.root),
            "--env-file", str(self.env_file),
            "--concurrency", "1",
        ]
        if single_output:
            options += ["--output", str(self.output), "--cache", str(self.cache)]
        if model is not None:
            options += ["--model", model]
        return baseline.build_parser().parse_args([*options, *extra])

    def run_baseline(self, results, *extra, model="offline-test-model", single_output=True):
        client = SimpleNamespace(responses=FakeResponses(results))
        with contextlib.redirect_stdout(io.StringIO()):
            baseline.run(
                self.arguments(*extra, model=model, single_output=single_output),
                client=client,
                record_loader=lambda path: self.records[Path(path)],
            )
        return client.responses.calls

    def run_client(self, client, *extra, single_output=True, async_entry=False):
        with contextlib.redirect_stdout(io.StringIO()):
            args = self.arguments(*extra, single_output=single_output)
            loader = lambda path: self.records[Path(path)]
            if async_entry:
                asyncio.run(baseline.async_run(args, client=client, record_loader=loader))
            else:
                baseline.run(args, client=client, record_loader=loader)

    def test_api_inputs_include_images_and_dimensions_without_gold_metadata(self):
        calls = self.run_baseline(["A", "B", "B"])
        for pair, call in zip(self.pairs, calls):
            payload = json.dumps(call["input"])
            self.assertIn("input_image", payload)
            self.assertIn("input_text", payload)
            self.assertIn("data:image/png;base64,", payload)
            content = [item for message in call["input"] for item in message["content"]]
            image_indices = [index for index, item in enumerate(content) if item["type"] == "input_image"]
            self.assertEqual(len(image_indices), 2)
            for side, index in zip(("a", "b"), image_indices):
                record = self.records[self.root / pair[f"path_{side}"]]
                self.assertEqual(
                    base64.b64decode(content[index]["image_url"].split(",", 1)[1]),
                    record["image"].data,
                )
                description = content[index - 1]
                self.assertEqual(description["type"], "input_text")
                self.assertRegex(description["text"], rf"\b{side.upper()}\b")
                for dimension in record["dims"]:
                    self.assertIn(str(dimension), description["text"])
            for gold_pair in self.pairs:
                self.assertNotIn(gold_pair["pair_id"], payload)
                for side in ("a", "b"):
                    for key in ("id", "path", "weight"):
                        self.assertNotIn(str(gold_pair[f"{key}_{side}"]), payload)
            self.assertNotIn("json_schema", json.dumps(call))
            self.assertNotIn("do not return numerical weight", payload.lower())
            self.assertNotIn("seed", call)

    def test_each_pair_gets_one_call_even_with_shared_objects_and_output_scores(self):
        self.pairs.append(self.make_pair("private-pair-four", 71003, 71002, "A"))
        self.write_pairs()
        calls = self.run_baseline(["A", "B", "B", "A"])
        self.assertEqual(len(calls), 4)
        gold = load_labels(self.pairs_path, "label")
        predictions = load_labels(self.output, "pred")
        result = score(gold, predictions)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["coverage"], 1.0)
        self.assert_score_files(self.output.parent, self.output)

    def assert_score_files(self, directory, predictions_path, *, canonical=False):
        expected = score(load_labels(self.pairs_path, "label"), load_labels(predictions_path, "pred"))
        json_path = directory / "scores.json" if canonical else predictions_path.with_suffix(".scores.json")
        text_path = directory / "scores.txt" if canonical else predictions_path.with_suffix(".scores.txt")
        self.assertEqual(json.loads(json_path.read_text()), expected)
        report = io.StringIO()
        with contextlib.redirect_stdout(report):
            print_report(expected)
        self.assertEqual(text_path.read_text(), report.getvalue())

    def test_resume_uses_cached_predictions_without_api_calls(self):
        self.run_baseline(["A", "B", "B"])
        expected = self.output.read_bytes()
        self.output.unlink()
        calls = self.run_baseline([])
        self.assertEqual(calls, [])
        self.assertEqual(self.output.read_bytes(), expected)

    def test_different_model_rejects_existing_cache(self):
        self.run_baseline(["A", "B", "B"])
        with self.assertRaises(baseline.BaselineError):
            self.run_baseline([], "--model", "different-offline-model")

    def test_changed_source_record_rejects_existing_cache(self):
        self.run_baseline(["A", "B", "B"])
        source = self.root / "private-record-71001.pklz"
        source.write_bytes(b"different source record bytes")
        with self.assertRaises(baseline.BaselineError):
            self.run_baseline([])

    def test_reversing_pair_orientation_rejects_cached_prediction(self):
        self.run_baseline(["A", "B", "B"])
        self.pairs[0] = self.make_pair("private-pair-one", 71002, 71001, "B")
        self.write_pairs()
        with self.assertRaises(baseline.BaselineError):
            self.run_baseline([])

    def test_legacy_weight_cache_cannot_be_reused(self):
        self.cache.write_text(json.dumps({
            "version": 1,
            "config": {"model": "offline-test-model", "task": "weight"},
            "predictions": {"71001": {"weight_lb": 8.0}},
        }), encoding="utf-8")
        with self.assertRaises(baseline.BaselineError):
            self.run_baseline([])

    def test_api_failure_preserves_progress_without_partial_prediction_file(self):
        with self.assertRaises((RuntimeError, baseline.BaselineError)):
            self.run_baseline(["A", RuntimeError("offline simulated API failure")])
        self.assertFalse(self.output.exists())
        self.assertTrue(self.cache.exists())
        calls = self.run_baseline(["B", "B"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(load_labels(self.output, "pred")), 3)

    def test_dry_run_makes_no_api_calls_or_prediction_file(self):
        calls = self.run_baseline([], "--dry-run")
        self.assertEqual(calls, [])
        self.assertFalse(self.output.exists())

    def test_missing_or_invalid_labels_cannot_produce_predictions(self):
        for result in ("", '{"pred": "A"}', "a", "C", "A is heavier", None, 1):
            with self.subTest(result=result):
                with self.assertRaises(baseline.BaselineError):
                    self.run_baseline([result])
                self.assertFalse(self.output.exists())

    def test_env_parser_handles_quoted_values_comments_and_missing_files(self):
        self.assertEqual(baseline.read_env(self.env_file), {})
        self.env_file.write_text(
            '# Local test settings\n'
            'export OPENAI_API_KEY="fake-key#literal" # trailing comment\n'
            "OPENAI_MODEL='file-model'\n"
            'UNRELATED_SETTING=ignored\n',
            encoding="utf-8",
        )
        self.assertEqual(baseline.read_env(self.env_file), {
            "OPENAI_API_KEY": "fake-key#literal", "OPENAI_MODEL": "file-model",
        })
        self.env_file.write_text('OPENAI_API_KEY="unfinished\n', encoding="utf-8")
        with self.assertRaises(baseline.BaselineError):
            baseline.read_env(self.env_file)

    def test_model_selection_prefers_cli_then_process_then_file_then_default(self):
        cases = [
            ("cli-model", {"OPENAI_MODEL": "process-model"}, "file-model", "cli-model"),
            (None, {"OPENAI_MODEL": "process-model"}, "file-model", "process-model"),
            (None, {}, "file-model", "file-model"),
            (None, {}, None, "gpt-6.1-sol"),
        ]
        for index, (cli_model, environment, file_model, expected) in enumerate(cases):
            with self.subTest(expected=expected), patch.dict(os.environ, environment, clear=True):
                self.env_file.write_text(
                    f"OPENAI_MODEL={file_model}\n" if file_model else "",
                    encoding="utf-8",
                )
                calls = self.run_baseline(
                    ["A", "B", "B"],
                    "--cache", str(self.directory / f"model-{index}.json"),
                    model=cli_model,
                )
                self.assertEqual({call["model"] for call in calls}, {expected})

    def test_api_key_selection_prefers_process_environment_to_env_file(self):
        self.env_file.write_text("OPENAI_API_KEY='fake-file-key'\n", encoding="utf-8")
        cases = [({}, "fake-file-key"), ({"OPENAI_API_KEY": "fake-process-key"}, "fake-process-key")]
        for index, (environment, expected) in enumerate(cases):
            with self.subTest(source=index):
                client = SimpleNamespace(responses=FakeResponses(["A", "B", "B"]), close=AsyncMock())
                constructor = Mock(return_value=client)
                with (patch.dict(os.environ, environment, clear=True),
                      patch.dict(sys.modules, {"openai": SimpleNamespace(AsyncOpenAI=constructor)}),
                      contextlib.redirect_stdout(io.StringIO())):
                    baseline.run(
                        self.arguments("--cache", str(self.directory / f"key-{index}.json")),
                        record_loader=lambda path: self.records[Path(path)],
                    )
                self.assertEqual(constructor.call_args.kwargs["api_key"], expected)
                client.close.assert_awaited_once()

    def test_api_requests_use_high_reasoning_and_4096_output_tokens(self):
        defaults = baseline.build_parser().parse_args(["--pairs", str(self.pairs_path)])
        self.assertEqual(defaults.concurrency, 16)
        calls = self.run_baseline(["A", "B", "B"])
        for call in calls:
            self.assertEqual(call["reasoning"], {"effort": "high"})
            self.assertEqual(call["max_output_tokens"], 4096)
        with self.assertRaises(baseline.BaselineError):
            self.run_baseline([], "--max-output-tokens", "2048")

    def test_custom_token_cap_and_default_reasoning_reach_api(self):
        calls = self.run_baseline(
            ["A", "B", "B"], "--max-output-tokens", "256", "--reasoning-effort", "default",
        )
        for call in calls:
            self.assertEqual(call["max_output_tokens"], 256)
            self.assertNotIn("reasoning", call)

    def test_pilot_limits_new_pairs_and_resumes_to_complete_predictions(self):
        calls = self.run_baseline(["A"], "--max-new-pairs", "1")
        self.assertEqual(len(calls), 1)
        self.assertTrue(self.cache.exists())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix(".scores.json").exists())
        self.assertEqual(json.loads(self.output.with_suffix(".run.json").read_text())["status"], "partial")
        calls = self.run_baseline(["B", "B"], "--max-new-pairs", "2")
        self.assertEqual(len(calls), 2)
        result = score(load_labels(self.pairs_path, "label"), load_labels(self.output, "pred"))
        self.assertEqual(result["accuracy"], 1.0)

    def test_pointwise_predicts_shared_objects_once_and_saves_weights_and_scores(self):
        self.pairs = [self.pairs[2], self.pairs[0], self.pairs[1],
                      self.make_pair("private-pair-four", 71003, 71002, "A")]
        self.write_pairs()
        calls = self.run_baseline([3.0, 12.0, 8.0], "--mode", "pointwise")
        self.assertEqual(len(calls), 3)
        weight_rows = [json.loads(line) for line in self.output.with_suffix(".weight_preds.jsonl").read_text().splitlines()]
        self.assertEqual(weight_rows, [
            {"id": 71002, "weight_lb": 3.0},
            {"id": 71003, "weight_lb": 12.0},
            {"id": 71001, "weight_lb": 8.0},
        ])
        self.assertEqual(score(load_labels(self.pairs_path, "label"), load_labels(self.output, "pred"))["accuracy"], 1.0)
        self.assert_score_files(self.directory, self.output)
        ordered_records = [self.records[self.root / f"private-record-{object_id}.pklz"]
                           for object_id in (71002, 71003, 71001)]
        for call, record in zip(calls, ordered_records):
            content = [part for message in call["input"] for part in message["content"]]
            images = [part for part in content if part["type"] == "input_image"]
            self.assertEqual(len(images), 1)
            self.assertEqual(base64.b64decode(images[0]["image_url"].split(",", 1)[1]), record["image"].data)
            payload = json.dumps(call["input"])
            for dimension in record["dims"]:
                self.assertIn(str(dimension), payload)
            for pair in self.pairs:
                self.assertNotIn(pair["pair_id"], payload)
                for side in ("a", "b"):
                    for key in ("id", "path", "weight"):
                        self.assertNotIn(str(pair[f"{key}_{side}"]), payload)
            self.assertNotIn("json_schema", json.dumps(call))
            self.assertNotIn("seed", call)
        run = json.loads(self.output.with_suffix(".run.json").read_text())
        self.assertEqual(run["job_order"], "object first appearance in pair file")

    def test_pointwise_equal_weights_use_b(self):
        self.run_baseline([5.0, 5.0, 5.0], "--mode", "pointwise")
        self.assertEqual(set(load_labels(self.output, "pred").values()), {"B"})

    def test_pointwise_invalid_weights_cannot_produce_predictions(self):
        for result in ("", "0", "-2", "nan", "inf", "0.75 lb", '{"weight_lb": 2}'):
            with self.subTest(result=result):
                with self.assertRaises(baseline.BaselineError):
                    self.run_baseline([result], "--mode", "pointwise")
                self.assertFalse(self.output.exists())
                self.assertFalse(self.output.with_suffix(".scores.json").exists())

    def test_pointwise_pilot_writes_available_weights_then_resumes(self):
        calls = self.run_baseline([8.0], "--mode", "pointwise", "--max-new-objects", "1")
        self.assertEqual(len(calls), 1)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix(".scores.json").exists())
        rows = [json.loads(line) for line in self.output.with_suffix(".weight_preds.jsonl").read_text().splitlines()]
        self.assertEqual(rows, [{"id": 71001, "weight_lb": 8.0}])
        self.assertEqual(json.loads(self.output.with_suffix(".run.json").read_text())["status"], "partial")
        calls = self.run_baseline([3.0, 12.0], "--mode", "pointwise")
        self.assertEqual(len(calls), 2)
        self.assert_score_files(self.directory, self.output)
        self.assertEqual(json.loads(self.output.with_suffix(".run.json").read_text())["status"], "complete")

    def test_both_modes_have_separate_caches_predictions_and_scores(self):
        output_root = self.directory / "experiments"
        outputs_by_image_count = {2: iter(["A", "B", "B"]), 1: iter([8.0, 3.0, 12.0])}

        def response_for_request(request):
            image_count = sum(part["type"] == "input_image"
                              for message in request["input"] for part in message["content"])
            return next(outputs_by_image_count[image_count])

        calls = self.run_baseline(response_for_request, "--mode", "both", "--output-dir", str(output_root), single_output=False)
        self.assertEqual(len(calls), 6)
        for mode in ("pairwise", "pointwise"):
            directory = output_root / mode / "output" / "test"
            self.assert_score_files(directory, directory / "preds.jsonl", canonical=True)
            cache = json.loads((directory / "cache.json").read_text())
            self.assertEqual(cache["version"], 3)
            self.assertEqual(cache["config"]["mode"], mode)
            run = json.loads((directory / "run.json").read_text())
            self.assertEqual(run["mode"], mode)
            self.assertEqual(run["status"], "complete")
        self.assertFalse((output_root / "pairwise" / "output" / "test" / "weight_preds.jsonl").exists())
        self.assertTrue((output_root / "pointwise" / "output" / "test" / "weight_preds.jsonl").exists())
        self.assertEqual(self.run_baseline([], "--mode", "both", "--output-dir", str(output_root), single_output=False), [])

    def test_train_dev_test_keep_independent_predictions_caches_and_reports(self):
        output_root = self.directory / "split-experiments"
        previous_files = {}
        for split in ("dev", "test", "train"):
            with self.subTest(split=split):
                self.pairs_path = self.directory / f"{split}_pairs.jsonl"
                self.write_pairs()
                outputs = {2: iter(["A", "B", "B"]), 1: iter([8.0, 3.0, 12.0])}

                def response(request):
                    count = sum(part["type"] == "input_image"
                                for message in request["input"] for part in message["content"])
                    return next(outputs[count])

                calls = self.run_baseline(response, "--mode", "both", "--output-dir", str(output_root), single_output=False)
                self.assertEqual(len(calls), 6)
                self.assertEqual({path: path.read_bytes() for path in previous_files}, previous_files)
                for mode in ("pairwise", "pointwise"):
                    directory = output_root / mode / "output" / split
                    self.assert_score_files(directory, directory / "preds.jsonl", canonical=True)
                    self.assertEqual(json.loads((directory / "run.json").read_text())["split"], split)
                    self.assertEqual(len(json.loads((directory / "cache.json").read_text())["predictions"]), 3)
                    previous_files.update({path: path.read_bytes() for path in directory.iterdir() if path.is_file()})

    def test_smoke_prefix_reuses_dev_cache_and_expands_complete_scores(self):
        output_root = self.directory / "smoke-experiments"
        self.prefix_record_paths("dev")
        full_pairs = list(self.pairs)
        self.pairs = self.pairs[:1]
        self.pairs_path = self.directory / "smoke_pairs.jsonl"
        self.write_pairs()
        outputs = {2: iter(["A"]), 1: iter([8.0, 3.0])}

        def response(request):
            count = sum(part["type"] == "input_image"
                        for message in request["input"] for part in message["content"])
            return next(outputs[count])

        calls = self.run_baseline(response, "--mode", "both", "--output-dir", str(output_root), single_output=False)
        self.assertEqual(len(calls), 3)
        saved = {}
        for mode in ("pairwise", "pointwise"):
            directory = output_root / mode / "output" / "dev"
            saved[mode] = json.loads((directory / "cache.json").read_text())["predictions"]
            self.assertEqual(json.loads((directory / "scores.json").read_text())["examples"], 1)
            self.assertEqual(json.loads((directory / "run.json").read_text())["split"], "dev")
        self.pairs = full_pairs
        self.pairs_path = self.directory / "dev_pairs.jsonl"
        self.write_pairs()
        outputs = {2: iter(["B", "B"]), 1: iter([12.0])}
        calls = self.run_baseline(response, "--mode", "both", "--output-dir", str(output_root), single_output=False)
        self.assertEqual(len(calls), 3)
        for mode in ("pairwise", "pointwise"):
            directory = output_root / mode / "output" / "dev"
            self.assert_score_files(directory, directory / "preds.jsonl", canonical=True)
            cache = json.loads((directory / "cache.json").read_text())["predictions"]
            self.assertEqual({key: cache[key] for key in saved[mode]}, saved[mode])
            self.assertEqual(len(cache), 3)
            run = json.loads((directory / "run.json").read_text())
            self.assertEqual(run["cached_before_run"], len(saved[mode]))
            self.assertEqual(run["new_predictions_this_run"], 3 - len(saved[mode]))

    def test_conflicting_or_mixed_splits_fail_before_requests_or_output_writes(self):
        for mixed in (False, True):
            with self.subTest(mixed=mixed):
                self.prefix_record_paths("dev")
                if mixed:
                    original = self.root / self.pairs[0]["path_b"]
                    destination = self.root / "test" / original.name
                    destination.parent.mkdir(exist_ok=True)
                    destination.write_bytes(original.read_bytes())
                    self.records[destination] = self.records[original]
                    self.pairs[0]["path_b"] = f"test/{original.name}"
                self.write_pairs()
                output_root = self.directory / "invalid-split-output"
                client = SimpleNamespace(responses=FakeResponses([]))
                extra = [] if mixed else ["--split", "test"]
                with self.assertRaises(baseline.BaselineError):
                    self.run_client(client, "--output-dir", str(output_root), *extra, single_output=False)
                self.assertEqual(client.responses.calls, [])
                self.assertFalse(output_root.exists())

    def test_split_inference_normalizes_parent_components_before_validating_override(self):
        self.prefix_record_paths("test")
        for pair in self.pairs:
            for side in ("a", "b"):
                pair[f"path_{side}"] = f"dev/../{pair[f'path_{side}']}"
        self.pairs_path = self.directory / "dev_pairs.jsonl"
        self.write_pairs()
        output_root = self.directory / "normalized-output"
        self.run_baseline(["A", "B", "B"], "--output-dir", str(output_root), single_output=False)
        directory = output_root / "pairwise" / "output" / "test"
        self.assertEqual(json.loads((directory / "run.json").read_text())["split"], "test")
        self.assert_score_files(directory, directory / "preds.jsonl", canonical=True)
        self.assertFalse((output_root / "pairwise" / "output" / "dev").exists())
        rejected_output = self.directory / "rejected-normalized-output"
        client = SimpleNamespace(responses=FakeResponses([]))
        with self.assertRaises(baseline.BaselineError):
            self.run_client(client, "--split", "dev", "--output-dir", str(rejected_output), single_output=False)
        self.assertEqual(client.responses.calls, [])
        self.assertFalse(rejected_output.exists())

    def test_unknown_split_requires_default_output_hint_but_accepts_explicit_output(self):
        self.pairs_path = self.directory / "custom_examples.jsonl"
        self.write_pairs()
        output_root = self.directory / "unknown-split-output"
        client = SimpleNamespace(responses=FakeResponses([]))
        with self.assertRaises(baseline.BaselineError):
            self.run_client(client, "--output-dir", str(output_root), single_output=False)
        self.assertEqual(client.responses.calls, [])
        self.assertFalse(output_root.exists())
        calls = self.run_baseline(["A", "B", "B"])
        self.assertEqual(len(calls), 3)
        self.assertIsNone(json.loads(self.output.with_suffix(".run.json").read_text())["split"])
        self.assert_score_files(self.directory, self.output)

    def test_changing_mode_rejects_existing_cache(self):
        self.run_baseline(["A", "B", "B"])
        with self.assertRaises(baseline.BaselineError):
            self.run_baseline([], "--mode", "pointwise")

    def test_distinct_output_names_in_one_directory_preserve_both_modes_artifacts(self):
        pairwise_output = self.directory / "pairwise_predictions.jsonl"
        pointwise_output = self.directory / "pointwise_predictions.jsonl"
        self.run_baseline(["A", "B", "B"], "--output", str(pairwise_output), single_output=False)
        pairwise_files = [pairwise_output, *[pairwise_output.with_suffix(suffix)
                          for suffix in (".cache.json", ".scores.json", ".scores.txt", ".run.json")]]
        before = {path: path.read_bytes() for path in pairwise_files}
        self.run_baseline([8.0, 3.0, 12.0], "--mode", "pointwise",
                          "--output", str(pointwise_output), single_output=False)
        self.assertEqual({path: path.read_bytes() for path in pairwise_files}, before)
        for mode, output in (("pairwise", pairwise_output), ("pointwise", pointwise_output)):
            self.assert_score_files(self.directory, output)
            self.assertEqual(json.loads(output.with_suffix(".cache.json").read_text())["config"]["mode"], mode)
            run = json.loads(output.with_suffix(".run.json").read_text())
            self.assertEqual(run["mode"], mode)
            self.assertEqual(run["status"], "complete")
        self.assertFalse(pairwise_output.with_suffix(".weight_preds.jsonl").exists())
        self.assertTrue(pointwise_output.with_suffix(".weight_preds.jsonl").exists())

    def test_both_modes_reject_single_mode_output_overrides_before_api_calls(self):
        client = SimpleNamespace(responses=FakeResponses([]))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(baseline.BaselineError):
            baseline.run(self.arguments("--mode", "both"), client=client,
                         record_loader=lambda path: self.records[Path(path)])
        self.assertEqual(client.responses.calls, [])

    def test_pair_order_is_preserved_without_seed_settings(self):
        (self.directory / "pairing.json").write_text('{"seed": 7}', encoding="utf-8")
        self.pairs.reverse()
        self.write_pairs()
        calls = self.run_baseline(["B", "B", "A"])
        self.assertEqual(len(calls), 3)
        run = json.loads(self.output.with_suffix(".run.json").read_text())
        self.assertNotIn("seed", json.dumps(run))
        self.assertNotIn("seed", json.dumps(json.loads(self.cache.read_text())["config"]))
        self.assertEqual(run["job_order"], "existing pair file order")
        self.assertEqual(run["status"], "complete")
        self.assertEqual(run["pair_file_sha256"], hashlib.sha256(self.pairs_path.read_bytes()).hexdigest())
        output_ids = [json.loads(line)["pair_id"] for line in self.output.read_text().splitlines()]
        self.assertEqual(output_ids, [pair["pair_id"] for pair in self.pairs])
        for call in calls:
            self.assertNotIn("seed", call)

    def test_seed_cli_option_is_not_supported(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.arguments("--seed", "42")

    def test_async_requests_overlap_without_exceeding_concurrency_limit(self):
        responses = ConcurrentResponses()
        self.run_client(SimpleNamespace(responses=responses), "--concurrency", "2", async_entry=True)
        self.assertEqual(len(responses.started), 3)
        self.assertEqual(responses.max_active, 2)
        self.assertEqual(responses.active, 0)
        self.assert_score_files(self.directory, self.output)

    def test_out_of_order_completions_keep_predictions_attached_to_original_pair_ids(self):
        responses = ConcurrentResponses(delays={(0, 1): 0.08, (0, 2): 0.04, (1, 2): 0.001}, overlap_target=3)
        self.run_client(SimpleNamespace(responses=responses), "--concurrency", "3")
        self.assertEqual(responses.max_active, 3)
        self.assertNotEqual(responses.completed, responses.started)
        self.assertEqual([json.loads(line) for line in self.output.read_text().splitlines()], [
            {"pair_id": pair["pair_id"], "pred": pair["label"]} for pair in self.pairs
        ])

    def test_concurrent_pointwise_predicts_each_shared_object_once(self):
        self.pairs.append(self.make_pair("private-pair-four", 71003, 71002, "A"))
        self.write_pairs()
        responses = ConcurrentResponses(overlap_target=3)
        self.run_client(SimpleNamespace(responses=responses), "--mode", "pointwise", "--concurrency", "3")
        self.assertEqual(sorted(responses.started), [(0,), (1,), (2,)])
        self.assertEqual(responses.max_active, 3)
        self.assert_score_files(self.directory, self.output)

    def test_both_mode_pilots_share_global_concurrency_limit_and_resume_only_missing(self):
        output_root = self.directory / "concurrent-experiments"
        responses = ConcurrentResponses()
        self.run_client(SimpleNamespace(responses=responses), "--mode", "both",
                        "--output-dir", str(output_root), "--concurrency", "2",
                        "--max-new-pairs", "2", "--max-new-objects", "2", single_output=False)
        self.assertEqual(len(responses.started), 4)
        self.assertEqual(responses.max_active, 2)
        for mode in ("pairwise", "pointwise"):
            directory = output_root / mode / "output" / "test"
            self.assertEqual(len(json.loads((directory / "cache.json").read_text())["predictions"]), 2)
            self.assertEqual(json.loads((directory / "run.json").read_text())["status"], "partial")
            self.assertFalse((directory / "preds.jsonl").exists())
            self.assertFalse((directory / "scores.json").exists())
        resumed = ConcurrentResponses()
        self.run_client(SimpleNamespace(responses=resumed), "--mode", "both",
                        "--output-dir", str(output_root), "--concurrency", "2", single_output=False)
        self.assertEqual(len(resumed.started), 2)
        self.assertEqual(resumed.max_active, 2)
        self.assertTrue(set(responses.started).isdisjoint(resumed.started))
        for mode in ("pairwise", "pointwise"):
            directory = output_root / mode / "output" / "test"
            self.assert_score_files(directory, directory / "preds.jsonl", canonical=True)

    def test_first_failure_stops_new_requests_but_saves_inflight_success_before_resume(self):
        responses = ConcurrentResponses(fail_key=(0, 1), delays={(0, 2): 0.04})
        with self.assertRaises(baseline.BaselineError):
            self.run_client(SimpleNamespace(responses=responses), "--concurrency", "2")
        self.assertEqual(set(responses.started), {(0, 1), (0, 2)})
        self.assertEqual(responses.completed, [(0, 2)])
        self.assertEqual(responses.active, 0)
        self.assertEqual(set(json.loads(self.cache.read_text())["predictions"]), {"private-pair-two"})
        self.assertEqual(json.loads(self.output.with_suffix(".run.json").read_text())["status"], "failed")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix(".scores.json").exists())
        calls = self.run_baseline(["A", "B"])
        self.assertEqual(len(calls), 2)
        self.assert_score_files(self.directory, self.output)

    def test_borrowed_client_stays_open_and_concurrency_change_reuses_cache(self):
        client = SimpleNamespace(responses=FakeResponses(["A", "B", "B"]), close=AsyncMock())
        self.run_client(client)
        client.close.assert_not_awaited()
        resumed = SimpleNamespace(responses=FakeResponses([]), close=AsyncMock())
        self.run_client(resumed, "--concurrency", "3")
        self.assertEqual(resumed.responses.calls, [])
        resumed.close.assert_not_awaited()
        self.assertEqual(json.loads(self.output.with_suffix(".run.json").read_text())["concurrency"], 3)
        self.assertNotIn("concurrency", json.loads(self.cache.read_text())["config"])

    def test_cancellation_drains_inflight_responses_before_raising_and_keeps_client_open(self):
        async def scenario():
            ready, release = asyncio.Event(), asyncio.Event()
            completed = []

            class GatedResponses(FakeResponses):
                async def create(self, **kwargs):
                    self.calls.append(kwargs)
                    key = tuple(
                        int(base64.b64decode(part["image_url"].split(",", 1)[1]).rsplit(b" ", 1)[1])
                        for message in kwargs["input"] for part in message["content"]
                        if part["type"] == "input_image"
                    )
                    if len(self.calls) == 2:
                        ready.set()
                    await release.wait()
                    completed.append(key)
                    return self.make_response("A" if key == (0, 1) else "B", kwargs)

            client = SimpleNamespace(responses=GatedResponses([]), close=AsyncMock())
            task = asyncio.create_task(baseline.async_run(
                self.arguments("--concurrency", "2"), client=client,
                record_loader=lambda path: self.records[Path(path)],
            ))
            try:
                await asyncio.wait_for(ready.wait(), timeout=2)
                task.cancel()
                await asyncio.sleep(0)  # Let async_run handle cancellation and begin draining.
                self.assertFalse(task.done())
                self.assertEqual(len(client.responses.calls), 2)
                self.assertEqual(completed, [])
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
            self.assertEqual(set(completed), {(0, 1), (0, 2)})
            self.assertEqual(len(client.responses.calls), 2)
            saved = json.loads(self.cache.read_text())["predictions"]
            self.assertEqual(set(saved), {"private-pair-one", "private-pair-two"})
            self.assertEqual(saved["private-pair-one"]["pred"], "A")
            self.assertEqual(saved["private-pair-two"]["pred"], "B")
            self.assertFalse(self.output.exists())
            client.close.assert_not_awaited()

        with contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(scenario())

    def test_owned_async_client_is_closed_after_failure(self):
        self.env_file.write_text("OPENAI_API_KEY=fake-file-key\n", encoding="utf-8")
        client = SimpleNamespace(responses=FakeResponses([RuntimeError("offline API failure")]), close=AsyncMock())
        constructor = Mock(return_value=client)
        with (patch.dict(os.environ, {}, clear=True),
              patch.dict(sys.modules, {"openai": SimpleNamespace(AsyncOpenAI=constructor)}),
              contextlib.redirect_stdout(io.StringIO()), self.assertRaises(baseline.BaselineError)):
            baseline.run(self.arguments(), record_loader=lambda path: self.records[Path(path)])
        client.close.assert_awaited_once()

    def test_incomplete_and_refused_responses_are_rejected(self):
        responses = [
            SimpleNamespace(status="incomplete", output=[], output_text="A"),
            SimpleNamespace(
                status="completed", output_text="A",
                output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal")])],
            ),
        ]
        for response in responses:
            with self.subTest(status=response.status):
                client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=response)))
                with self.assertRaises(baseline.BaselineError):
                    asyncio.run(baseline.predict_pair(
                        client, "offline-test-model", FakeImage(), (1, 2, 3), FakeImage(), (4, 5, 6),
                    ))
                with self.assertRaises(baseline.BaselineError):
                    asyncio.run(baseline.predict_weight(client, "offline-test-model", FakeImage(), (1, 2, 3)))


if __name__ == "__main__":
    unittest.main()
