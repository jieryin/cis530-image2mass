"""Offline Qwen contract checks; no model weights, API, or GPU are needed."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from openai_strong_baseline.api import BaselineError
from prompts.pairwise import PROMPT as PAIRWISE_PROMPT
from prompts.pointwise import PROMPT as POINTWISE_PROMPT
from qwen_strong_baseline import runner
from qwen_strong_baseline.numeric_format import PositiveNumberConstraint
from qwen_strong_baseline.pairwise import predict as pairwise
from qwen_strong_baseline.pointwise import predict as pointwise
from score import ScoringError, load_labels, score


REVISION = "0123456789abcdef0123456789abcdef01234567"


class FakeNumericTokenizer:
    """Numeric pieces mixed with tokens that must never enter a scalar output."""

    def __init__(self):
        pieces = [*"0123456789", ".", "e", "E", "+", "-", ",", " lb", "NaN",
                  "Infinity", "because", " ", "00", "05", "0.", "1.5", "1e-",
                  "2E+", "1e309", "1e-999", "123"]
        self.vocab = {piece: index for index, piece in enumerate(pieces)}
        self.vocab.update({"<eos>": 90, "<end>": 91, "<pad>": 92})
        self.pieces = {index: piece for piece, index in self.vocab.items()}
        # A vocabulary spelling alone must not be trusted if decoding differs.
        self.pieces[self.vocab["123"]] = "not actually numeric"

    def get_vocab(self):
        return dict(self.vocab)

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self.pieces[index] for index in ids
                       if not skip_special_tokens or index not in (90, 91, 92))


class PositiveNumberConstraintTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeNumericTokenizer()
        self.constraint = PositiveNumberConstraint(self.tokenizer, [90, 91])

    def allowed(self, prefix):
        return set(self.constraint.allowed_tokens(prefix))

    def token(self, piece):
        return self.tokenizer.vocab[piece]

    def test_initial_tokens_follow_positive_json_syntax(self):
        allowed = self.allowed("")
        for piece in "0123456789":
            self.assertIn(self.token(piece), allowed)
        for piece in (".", "e", "E", "+", "-", "00", "05"):
            self.assertNotIn(self.token(piece), allowed)
        self.assertTrue({90, 91, 92}.isdisjoint(allowed))

    def test_leading_zero_requires_fraction_before_nonzero_digits(self):
        allowed = self.allowed("0")
        self.assertIn(self.token("."), allowed)
        for piece in [*"0123456789", "00", "05", "e", "E"]:
            self.assertNotIn(self.token(piece), allowed)
        self.assertTrue({90, 91}.isdisjoint(allowed))
        with self.assertRaises(BaselineError):
            self.allowed("01")

    def test_fraction_digits_and_exponent_order_are_enforced(self):
        fraction_start = self.allowed("1.")
        self.assertTrue({self.token(d) for d in "0123456789"}.issubset(fraction_start))
        for piece in (".", "e", "E", "+", "-"):
            self.assertNotIn(self.token(piece), fraction_start)
        self.assertTrue({90, 91}.isdisjoint(fraction_start))
        completed = self.allowed("1.2")
        self.assertTrue({90, 91, self.token("e"), self.token("E")}.issubset(completed))
        self.assertNotIn(self.token("."), completed)

    def test_exponents_allow_sign_only_immediately_after_marker(self):
        after_marker = self.allowed("1e")
        self.assertTrue({self.token("+"), self.token("-"), self.token("0")}.issubset(after_marker))
        self.assertTrue({90, 91}.isdisjoint(after_marker))
        after_sign = self.allowed("1e-")
        self.assertIn(self.token("2"), after_sign)
        self.assertNotIn(self.token("+"), after_sign)
        self.assertNotIn(self.token("-"), after_sign)
        self.assertTrue({90, 91}.isdisjoint(after_sign))
        self.assertTrue({90, 91}.issubset(self.allowed("1e-2")))
        for invalid in ("1e--", "1ee", "1e2.", "1.2.3"):
            with self.subTest(prefix=invalid), self.assertRaises(BaselineError):
                self.allowed(invalid)

    def test_eos_requires_complete_finite_strictly_positive_number(self):
        for valid in ("1", "0.001", "123.45", "1e-2", "2E+3"):
            with self.subTest(prefix=valid):
                self.assertTrue({90, 91}.issubset(self.allowed(valid)))
        for incomplete_or_zero in ("", "0", "0.0", "0.000", "1.", "1e", "1e+"):
            with self.subTest(prefix=incomplete_or_zero):
                self.assertTrue({90, 91}.isdisjoint(self.allowed(incomplete_or_zero)))
        for impossible in ("NaN", "Infinity", "-1", "+1", "0e5", "1e309", "1e-999"):
            with self.subTest(prefix=impossible), self.assertRaises(BaselineError):
                self.allowed(impossible)

    def test_multi_character_candidates_must_decode_exactly_and_fit_grammar(self):
        initial = self.allowed("")
        for piece in ("0.", "1.5", "1e-", "2E+"):
            self.assertIn(self.token(piece), initial)
        for piece in ("00", "05", "1e309", "1e-999", "123"):
            self.assertNotIn(self.token(piece), initial)
        # Multi-digit tokens remain available in a legal integer continuation.
        self.assertIn(self.token("00"), self.allowed("1"))

    def test_units_lists_explanations_and_special_tokens_are_always_masked(self):
        forbidden = {self.token(piece) for piece in (",", " lb", "NaN", "Infinity", "because", " ")}
        forbidden.add(92)
        for prefix in ("", "0", "1", "1.", "1.2", "1e", "1e-", "1e-2"):
            with self.subTest(prefix=prefix):
                self.assertTrue(forbidden.isdisjoint(self.allowed(prefix)))

    def test_callback_excludes_prompt_tokens_for_lists_and_tensor_like_ids(self):
        callback = self.constraint.for_prompt(3)
        # Deliberately invalid numeric prompt content must have no influence.
        prompt = [self.token("because"), self.token(","), 92]
        generated = [self.token("1"), self.token("."), self.token("5")]
        self.assertEqual(callback(0, prompt + generated), self.constraint.allowed_tokens("1.5"))

        class TensorLike:
            def __init__(self, ids):
                self.ids = ids

            def __getitem__(self, item):
                return TensorLike(self.ids[item])

            def tolist(self):
                return list(self.ids)

        self.assertEqual(callback(7, TensorLike(prompt + generated)),
                         self.constraint.allowed_tokens("1.5"))
        self.assertEqual(callback(0, prompt), self.constraint.allowed_tokens(""))

    def test_callback_validates_prompt_length_and_returned_cache_is_not_mutable(self):
        for invalid in (-1, True, 1.5):
            with self.subTest(length=invalid), self.assertRaises(BaselineError):
                self.constraint.for_prompt(invalid)
        allowed = self.constraint.allowed_tokens("1")
        allowed.clear()
        self.assertTrue(self.constraint.allowed_tokens("1"))


class FakePredictor:
    """Snapshot model-facing content while images are still open."""

    def __init__(self, outputs=None):
        self.outputs = iter(outputs) if outputs is not None else None
        self.calls = []
        self.metadata = {
            "model": "offline-test-model",
            "model_id": "offline-test-model",
            "revision": REVISION,
            "model_revision": REVISION,
            "dtype": "bfloat16",
            "device": "cuda:0",
            "max_new_tokens": 32,
            "do_sample": False,
            "processor": {"type": "offline fake"},
            "versions": {"torch": "offline", "transformers": "offline"},
        }

    def __call__(self, messages):
        snapshot = []
        for message in messages:
            content = []
            for part in message["content"]:
                if part["type"] == "image":
                    image = part["image"]
                    content.append({"type": "image", "mode": image.mode,
                                    "pixel": image.getpixel((0, 0)), "size": image.size})
                else:
                    content.append(dict(part))
            snapshot.append({"role": message["role"], "content": content})
        self.calls.append(snapshot)
        if self.outputs is not None:
            output = next(self.outputs)
        else:
            colors = [part["pixel"][0] for part in snapshot[0]["content"]
                      if part["type"] == "image"]
            weights = {10: 8.0, 20: 3.0, 30: 12.0}
            output = (str(weights[colors[0]]) if len(colors) == 1 else
                      ("A" if weights[colors[0]] > weights[colors[1]] else "B"))
        if isinstance(output, Exception):
            raise output
        if isinstance(output, dict):
            return output
        return {"raw_output": output, "elapsed_seconds": 0.125,
                "peak_memory_allocated_bytes": 123456,
                "generated_tokens": 2, "truncated": False}


class QwenPromptTests(unittest.TestCase):
    def test_prompts_match_existing_baseline_exactly(self):
        self.assertEqual(pairwise.PROMPT, PAIRWISE_PROMPT)
        self.assertEqual(pointwise.PROMPT, POINTWISE_PROMPT)

    def test_pairwise_dimensions_stay_adjacent_to_correct_rgb_image(self):
        image_a = Image.new("RGB", (3, 2), (10, 1, 2))
        image_b = Image.new("RGB", (2, 4), (20, 3, 4))
        self.addCleanup(image_a.close)
        self.addCleanup(image_b.close)
        messages = pairwise.build_messages(image_a, (1, 2, 3), image_b, (4, 5, 6))
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        content = messages[0]["content"]
        self.assertEqual([part["type"] for part in content],
                         ["text", "text", "image", "text", "image"])
        self.assertEqual(content[0]["text"], PAIRWISE_PROMPT)
        self.assertEqual(content[1]["text"],
                         'Object A: dimensions in inches = [1.0, 2.0, 3.0]. Image follows.')
        self.assertEqual(content[3]["text"],
                         'Object B: dimensions in inches = [4.0, 5.0, 6.0]. Image follows.')
        self.assertEqual(content[2]["image"].getpixel((0, 0)), (10, 1, 2))
        self.assertEqual(content[4]["image"].getpixel((0, 0)), (20, 3, 4))

    def test_pointwise_message_has_only_prompt_dimensions_and_image(self):
        image = Image.new("RGB", (2, 2), (10, 1, 2))
        self.addCleanup(image.close)
        messages = pointwise.build_messages(image, (1, 2, 3))
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        content = messages[0]["content"]
        self.assertEqual([part["type"] for part in content], ["text", "text", "image"])
        self.assertEqual(content[0]["text"], POINTWISE_PROMPT)
        self.assertEqual(content[1]["text"],
                         'Physical dimensions in inches = [1.0, 2.0, 3.0]. Image follows.')
        self.assertEqual(content[2]["image"].mode, "RGB")

    def test_pairwise_strips_whitespace_but_requires_exact_label(self):
        self.assertEqual(pairwise.parse_output(" \nA\t"), "A")
        self.assertEqual(pairwise.parse_output("B"), "B")
        for invalid in ("a", "A.", "A or B", "Answer: A", "", " B because heavier"):
            with self.subTest(invalid=invalid), self.assertRaises(BaselineError):
                pairwise.parse_output(invalid)

    def test_pointwise_rejects_nonpositive_nonfinite_and_extra_text(self):
        self.assertEqual(pointwise.parse_output(" 0.35\n"), 0.35)
        for invalid in ("0", "-1", "NaN", "Infinity", "1e309", "true", "null",
                        '"0.35"', "[0.35]", "0.35 lb", "about 0.35", ""):
            with self.subTest(invalid=invalid), self.assertRaises(BaselineError):
                pointwise.parse_output(invalid)

    def test_exact_ties_choose_b(self):
        self.assertEqual(pointwise.compare(2.0, 2.0), "B")
        self.assertEqual(pointwise.compare(2.01, 2.0), "A")
        self.assertEqual(pointwise.compare(1.99, 2.0), "B")

    def test_invalid_dimensions_cannot_reach_model_messages(self):
        image = Image.new("RGB", (2, 2))
        self.addCleanup(image.close)
        for invalid in ((1, 2), (1, 2, 3, 4), (1, 0, 3), (1, -2, 3),
                        (1, float("nan"), 3), (1, True, 3), (1, "2", 3)):
            with self.subTest(invalid=invalid), self.assertRaises(BaselineError):
                pointwise.build_messages(image, invalid)


class QwenRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.root = self.directory / "records"
        (self.root / "test").mkdir(parents=True)
        self.output = self.directory / "output"
        self.pairs_path = self.directory / "test_pairs.jsonl"
        self.objects = {}
        for index, object_id in enumerate((71001, 71002, 71003), 1):
            relative = f"test/private-record-{object_id}.pklz"
            path = self.root / relative
            path.write_bytes(f"source bytes for {object_id}".encode())
            self.objects[object_id] = {"path": relative, "dims": (index + 0.25, 3.5, 4.75),
                                       "weight": 987654.321 + object_id, "color": index * 10}
        self.pairs = [self.make_pair("private-pair-one", 71001, 71002, "A"),
                      self.make_pair("private-pair-two", 71001, 71003, "B"),
                      self.make_pair("private-pair-three", 71002, 71003, "B")]
        self.write_pairs()

    def make_pair(self, pair_id, object_a, object_b, label):
        pair = {"pair_id": pair_id, "label": label}
        for side, object_id in (("a", object_a), ("b", object_b)):
            obj = self.objects[object_id]
            pair.update({f"id_{side}": object_id, f"path_{side}": obj["path"],
                         f"dims_{side}": list(obj["dims"]), f"weight_{side}": obj["weight"]})
        return pair

    def write_pairs(self):
        self.pairs_path.write_text("".join(json.dumps(pair) + "\n" for pair in self.pairs),
                                   encoding="utf-8")

    def record_loader(self, path):
        path = Path(path).resolve()
        for object_id, obj in self.objects.items():
            if path == (self.root / obj["path"]).resolve():
                return {"id": object_id, "dims": obj["dims"], "weight": obj["weight"],
                        "image": Image.new("RGB", (2, 3), (obj["color"], 1, 2)),
                        "thickness_mask": "SECRET MASK"}
        raise AssertionError(f"Unexpected record: {path}")

    def arguments(self, *extra):
        return runner.build_parser().parse_args([
            "--mode", "both", "--pairs", str(self.pairs_path), "--root", str(self.root),
            "--output-dir", str(self.output), "--split", "test", "--model", "offline-test-model",
            "--revision", REVISION, *extra])

    def run_fake(self, *extra, predictor=None):
        predictor = predictor if predictor is not None else FakePredictor()
        with contextlib.redirect_stdout(io.StringIO()):
            runner.run(self.arguments(*extra), predictor=predictor, record_loader=self.record_loader)
        return predictor

    def mode_dir(self, mode):
        return self.output / mode / "output" / "test"

    def test_atomic_write_retries_transient_permission_error(self):
        path = self.directory / "atomic-write.txt"
        path.write_text("previous valid contents", encoding="utf-8")
        original_write = runner._atomic_write
        attempts = []

        def transient_write(target, text):
            attempts.append((target, text))
            if len(attempts) == 1:
                raise PermissionError("simulated temporary Windows sharing violation")
            return original_write(target, text)

        with patch.object(runner, "_atomic_write", side_effect=transient_write):
            with patch.object(runner.time, "sleep") as sleep:
                runner.atomic_write(path, "replacement contents")
        self.assertEqual(len(attempts), 2)
        sleep.assert_called_once_with(0.05)
        self.assertEqual(path.read_text(encoding="utf-8"), "replacement contents")

    def test_permanent_storage_error_has_bounded_retries_and_no_model_calls(self):
        predictor = FakePredictor([])
        with patch.object(runner, "_atomic_write", side_effect=PermissionError("permanent sharing violation")) as write:
            with patch.object(runner.time, "sleep") as sleep:
                with self.assertRaises(PermissionError):
                    self.run_fake("--mode", "pairwise", predictor=predictor)
        self.assertEqual(write.call_count, 40)
        self.assertEqual(sleep.call_count, 39)
        self.assertTrue(all(call.args == (0.05,) for call in sleep.call_args_list))
        self.assertEqual(predictor.calls, [])

    def test_storage_failure_after_valid_inference_is_not_an_invalid_response(self):
        original_persist = runner.persist
        saved_states = []

        def fail_second_persist(state, args, **kwargs):
            saved_states.append(state)
            if len(saved_states) == 1:
                return original_persist(state, args, **kwargs)
            raise PermissionError("simulated disk failure after a valid response")

        predictor = FakePredictor(["A"])
        with patch.object(runner, "persist", side_effect=fail_second_persist):
            with self.assertRaises(PermissionError):
                self.run_fake("--mode", "pairwise", predictor=predictor)
        self.assertEqual(len(predictor.calls), 1)
        self.assertEqual(len(saved_states), 2)
        self.assertEqual(saved_states[-1]["cache"]["errors"], {})
        self.assertEqual(saved_states[-1]["cache"]["predictions"][self.pairs[0]["pair_id"]]["value"], "A")
        directory = self.mode_dir("pairwise")
        previous_cache = json.loads((directory / "cache.json").read_text(encoding="utf-8"))
        self.assertEqual(previous_cache["errors"], {})
        self.assertEqual(previous_cache["predictions"], {})
        self.assertFalse((directory / "preds.jsonl").exists())
        self.assertFalse((directory / "scores.json").exists())

    def test_both_modes_produce_strict_complete_scores(self):
        predictor = self.run_fake()
        self.assertEqual(len(predictor.calls), 6)  # 3 pairs + 3 unique objects.
        for mode in ("pairwise", "pointwise"):
            directory = self.mode_dir(mode)
            metrics = score(load_labels(self.pairs_path, "label"),
                            load_labels(directory / "preds.jsonl", "pred"))
            self.assertEqual(metrics["accuracy"], 1.0)
            self.assertEqual(metrics["coverage"], 1.0)
            self.assertEqual(metrics, json.loads((directory / "scores.json").read_text()))
            self.assertTrue((directory / "scores.txt").exists())
            self.assertTrue((directory / "cache.json").exists())
            self.assertTrue((directory / "run.json").exists())
        weights = [json.loads(line) for line in
                   (self.mode_dir("pointwise") / "weight_preds.jsonl").read_text().splitlines()]
        self.assertEqual(len(weights), 3)
        self.assertEqual({row["id"] for row in weights}, set(self.objects))

    def test_model_inputs_exclude_all_tracking_and_gold_metadata(self):
        predictor = self.run_fake()
        for messages in predictor.calls:
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["role"], "user")
            text = json.dumps(messages)
            self.assertNotIn("SECRET MASK", text)
            self.assertNotIn("weight_a", text)
            self.assertNotIn("weight_b", text)
            for pair in self.pairs:
                self.assertNotIn(pair["pair_id"], text)
            for object_id, obj in self.objects.items():
                self.assertNotIn(str(object_id), text)
                self.assertNotIn(obj["path"], text)
                self.assertNotIn(str(obj["weight"]), text)
            for part in messages[0]["content"]:
                if part["type"] == "image":
                    self.assertEqual(part["mode"], "RGB")

    def test_pointwise_predicts_each_object_once_and_reuses_ties(self):
        predictor = self.run_fake("--mode", "pointwise", predictor=FakePredictor(["2", "2", "2"]))
        self.assertEqual(len(predictor.calls), 3)
        predictions = load_labels(self.mode_dir("pointwise") / "preds.jsonl", "pred")
        self.assertEqual(set(predictions.values()), {"B"})

    def test_compatible_completed_cache_requires_zero_new_inferences(self):
        self.run_fake()
        before = {mode: (self.mode_dir(mode) / "preds.jsonl").read_bytes()
                  for mode in ("pairwise", "pointwise")}
        resumed = self.run_fake(predictor=FakePredictor([]))
        self.assertEqual(resumed.calls, [])
        for mode, data in before.items():
            self.assertEqual((self.mode_dir(mode) / "preds.jsonl").read_bytes(), data)

    def test_pilot_limits_only_new_work_and_defers_full_scores(self):
        first = self.run_fake("--max-new-pairs", "1", "--max-new-objects", "1")
        self.assertEqual(len(first.calls), 2)
        for mode in ("pairwise", "pointwise"):
            self.assertFalse((self.mode_dir(mode) / "preds.jsonl").exists())
            self.assertFalse((self.mode_dir(mode) / "scores.json").exists())
        second = self.run_fake("--max-new-pairs", "1", "--max-new-objects", "1")
        self.assertEqual(len(second.calls), 2)
        third = self.run_fake()
        self.assertEqual(len(third.calls), 2)
        for mode in ("pairwise", "pointwise"):
            self.assertEqual(len(load_labels(self.mode_dir(mode) / "preds.jsonl", "pred")), 3)

    def test_dry_run_never_predicts_or_creates_output(self):
        predictor = self.run_fake("--dry-run", predictor=FakePredictor([]))
        self.assertEqual(predictor.calls, [])
        self.assertFalse(self.output.exists())

    def test_dry_run_does_not_construct_model(self):
        with patch("qwen_strong_baseline.model.QwenPredictor",
                   side_effect=AssertionError("dry-run must not load a model")):
            with contextlib.redirect_stdout(io.StringIO()):
                runner.run(self.arguments("--dry-run"), record_loader=self.record_loader)
        self.assertFalse(self.output.exists())

    def test_changed_generation_setting_rejects_cache_before_inference(self):
        self.run_fake()
        predictor = FakePredictor([])
        with self.assertRaises(BaselineError):
            self.run_fake("--max-new-tokens", "64", predictor=predictor)
        self.assertEqual(predictor.calls, [])

    def test_changed_prompt_rejects_cache_before_inference(self):
        self.run_fake("--mode", "pairwise")
        predictor = FakePredictor([])
        with patch.object(runner, "PAIRWISE_PROMPT", runner.PAIRWISE_PROMPT + " Changed prompt."):
            with self.assertRaises(BaselineError):
                self.run_fake("--mode", "pairwise", predictor=predictor)
        self.assertEqual(predictor.calls, [])

    def test_cached_parsed_value_must_match_its_raw_output(self):
        self.run_fake("--mode", "pairwise")
        cache_path = self.mode_dir("pairwise") / "cache.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        cache["predictions"][self.pairs[0]["pair_id"]]["value"] = "B"
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
        predictor = FakePredictor([])
        with self.assertRaises(BaselineError):
            self.run_fake("--mode", "pairwise", predictor=predictor)
        self.assertEqual(predictor.calls, [])

    def test_changed_model_revision_rejects_cache(self):
        self.run_fake()
        predictor = FakePredictor([])
        with self.assertRaises(BaselineError):
            self.run_fake("--revision", "a" * 40, predictor=predictor)
        self.assertEqual(predictor.calls, [])

    def test_changed_input_bytes_reject_cache(self):
        self.run_fake()
        (self.root / self.objects[71001]["path"]).write_bytes(b"new image record bytes")
        predictor = FakePredictor([])
        with self.assertRaises(BaselineError):
            self.run_fake(predictor=predictor)
        self.assertEqual(predictor.calls, [])

    def test_changed_pair_file_rejects_cache(self):
        self.run_fake()
        self.pairs[0]["label"] = "B"
        self.write_pairs()
        predictor = FakePredictor([])
        with self.assertRaises(BaselineError):
            self.run_fake(predictor=predictor)
        self.assertEqual(predictor.calls, [])

    def test_invalid_raw_response_is_preserved_and_never_scored(self):
        invalid = "Answer: A because it seems heavier"
        predictor = FakePredictor([invalid])
        with self.assertRaises(BaselineError):
            self.run_fake("--mode", "pairwise", predictor=predictor)
        self.assertEqual(len(predictor.calls), 1)
        directory = self.mode_dir("pairwise")
        retained = "\n".join(path.read_text(encoding="utf-8") for path in directory.glob("*.json"))
        self.assertIn(invalid, retained)
        self.assertFalse((directory / "preds.jsonl").exists())
        self.assertFalse((directory / "scores.json").exists())

    def test_failed_cache_refuses_repeating_deterministic_inference(self):
        with self.assertRaises(BaselineError):
            self.run_fake("--mode", "pointwise", predictor=FakePredictor(["not a weight"]))
        predictor = FakePredictor([])
        with self.assertRaisesRegex(BaselineError, "failed output"):
            self.run_fake("--mode", "pointwise", predictor=predictor)
        self.assertEqual(predictor.calls, [])

    def test_record_path_cannot_escape_data_root(self):
        outside = self.directory / "escaped-record.pklz"
        outside.write_bytes(b"must not be loaded")
        for pair in self.pairs:
            for side in ("a", "b"):
                if pair[f"id_{side}"] == 71001:
                    pair[f"path_{side}"] = "../escaped-record.pklz"
        self.write_pairs()
        predictor = FakePredictor([])
        with self.assertRaisesRegex(BaselineError, "escapes"):
            self.run_fake(predictor=predictor)
        self.assertEqual(predictor.calls, [])
        self.assertFalse(self.output.exists())

    def test_invalid_pair_dimensions_are_rejected_before_inference(self):
        for pair in self.pairs:
            for side in ("a", "b"):
                if pair[f"id_{side}"] == 71001:
                    pair[f"dims_{side}"] = [1, 0, 3]
        self.write_pairs()
        predictor = FakePredictor([])
        with self.assertRaises(BaselineError):
            self.run_fake(predictor=predictor)
        self.assertEqual(predictor.calls, [])
        self.assertFalse(self.output.exists())

    def test_truncated_response_is_rejected_even_if_label_looks_valid(self):
        predictor = FakePredictor([{"raw_output": "A", "elapsed_seconds": 0.1,
                                    "peak_memory_allocated_bytes": 1234,
                                    "generated_tokens": 32, "truncated": True}])
        with self.assertRaises(BaselineError):
            self.run_fake("--mode", "pairwise", predictor=predictor)
        self.assertFalse((self.mode_dir("pairwise") / "scores.json").exists())

    def test_missing_and_extra_predictions_remain_scoring_errors(self):
        gold = {pair["pair_id"]: pair["label"] for pair in self.pairs}
        with self.assertRaises(ScoringError):
            score(gold, {self.pairs[0]["pair_id"]: "A"})
        with self.assertRaises(ScoringError):
            score(gold, {**gold, "unknown-pair": "B"})


if __name__ == "__main__":
    unittest.main()
