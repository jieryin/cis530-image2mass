"""Answer-only gradients and offline Qwen3-VL/LoRA integration checks.

The integration model is tiny and randomly initialized. No pretrained weights,
tokenizer downloads, GPU, or real dataset images are needed.
"""

import copy
import importlib.util
import os
import tempfile
import unittest

from finetune.model import attach_lora, configure_image_pixels
from finetune.pairwise_finetune.task import (
    answer_mse, answer_probability, answer_token_ids, predict_probability,
)


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_MODEL_STACK = HAS_TORCH and all(
    importlib.util.find_spec(name) is not None for name in ("transformers", "peft")
)


class FakeProcessor:
    """Make continuation boundary behavior explicit without a real tokenizer."""

    prefix = "<assistant>\n"

    def __init__(self, a=(1, 2, 3), b=(1, 2, 7)):
        self.continuations = {self.prefix: [1, 2], self.prefix + "A": list(a), self.prefix + "B": list(b)}
        self.tokenizer = self

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize and add_generation_prompt
        return self.prefix

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        # Looking up bare "A" or "B" is deliberately unsupported: a token must
        # be validated after the actual assistant boundary, not in isolation.
        return self.continuations[text]


class ChoiceTokenTests(unittest.TestCase):
    def test_choices_are_verified_as_contextual_single_tokens(self):
        self.assertEqual(answer_token_ids(FakeProcessor()), (3, 7))

    def test_multi_token_answer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not one token"):
            answer_token_ids(FakeProcessor(a=(1, 2, 3, 4)))

    def test_boundary_retokenization_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not one token"):
            answer_token_ids(FakeProcessor(a=(1, 9, 3)))

    def test_choices_must_be_distinct(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            answer_token_ids(FakeProcessor(b=(1, 2, 3)))


@unittest.skipUnless(HAS_TORCH, "Install requirements.txt for tensor tests")
class AnswerObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_binary_probability_ignores_every_non_choice_logit(self):
        torch = self.torch
        logits = torch.tensor([[[1000.0, -50.0, 2.0, 900.0, -1.0]],
                               [[-900.0, 800.0, 0.0, 700.0, 0.0]]])
        probability = answer_probability(logits, (2, 4))
        expected = torch.tensor([torch.sigmoid(torch.tensor(3.0)), 0.5])
        torch.testing.assert_close(probability, expected)
        mutated = logits.clone()
        mutated[:, :, [0, 1, 3]] *= -100
        torch.testing.assert_close(answer_probability(mutated, (2, 4)), expected)

    def test_mse_reduction_and_only_ab_gradients(self):
        torch = self.torch
        logits = torch.zeros(2, 1, 9, requires_grad=True)
        targets = torch.tensor([1.0, 0.0])
        loss, probability = answer_mse(logits, targets, (3, 7))
        self.assertAlmostEqual(loss.item(), 0.25)
        torch.testing.assert_close(probability, torch.tensor([0.5, 0.5]))
        loss.backward()
        expected = torch.zeros_like(logits)
        # Mean over two examples: 2*(p-y)*p*(1-p)/2.
        expected[0, 0, 3], expected[0, 0, 7] = -0.125, 0.125
        expected[1, 0, 3], expected[1, 0, 7] = 0.125, -0.125
        torch.testing.assert_close(logits.grad, expected)

    def test_only_selected_answer_position_receives_direct_logit_gradient(self):
        torch = self.torch
        full_logits = torch.zeros(1, 11, 9, requires_grad=True)
        # The prompt occupies positions 0..3; position 4 predicts the answer;
        # later positions would predict a rationale and must remain unused.
        loss, _ = answer_mse(full_logits[:, 4:5, :], torch.tensor([1.0]), (3, 7))
        loss.backward()
        expected = torch.zeros_like(full_logits)
        expected[0, 4, 3], expected[0, 4, 7] = -0.25, 0.25
        torch.testing.assert_close(full_logits.grad, expected)

    def test_predictor_requests_only_last_prompt_logits_without_ce_labels(self):
        torch = self.torch

        class CapturingModel:
            def __init__(self):
                self.kwargs = None

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return type("Output", (), {"logits": torch.zeros(1, 1, 9)})()

        model = CapturingModel()
        inputs = {"input_ids": torch.tensor([[1, 2, 4, 8]])}
        probability = predict_probability(model, inputs, (3, 7))
        self.assertEqual(model.kwargs["logits_to_keep"], 1)
        self.assertFalse(model.kwargs["use_cache"])
        self.assertNotIn("labels", model.kwargs)
        torch.testing.assert_close(probability, torch.tensor([0.5]))

    def test_target_broadcasting_is_rejected(self):
        torch = self.torch
        logits = torch.zeros(2, 1, 9)
        for targets in (torch.tensor(1.0), torch.ones(1), torch.ones(2, 1), torch.ones(1, 2)):
            with self.subTest(shape=tuple(targets.shape)), self.assertRaisesRegex(ValueError, "broadcasting"):
                answer_mse(logits, targets, (3, 7))

    def test_full_sequence_and_missing_sequence_dimension_are_rejected(self):
        torch = self.torch
        for shape in ((1, 9), (1, 3, 9), (1, 0, 9)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "next-answer logits"):
                answer_probability(torch.zeros(shape), (3, 7))

    def test_half_precision_logits_are_scored_in_float32(self):
        torch = self.torch
        logits = torch.full((1, 1, 9), 10000.0, dtype=torch.float16, requires_grad=True)
        loss, probability = answer_mse(logits, torch.tensor([1.0]), (3, 7))
        self.assertEqual(probability.dtype, torch.float32)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(probability.item(), 0.5)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())


@unittest.skipUnless(HAS_MODEL_STACK, "Install requirements.txt for Qwen/LoRA integration")
class TinyQwenLoraTests(unittest.TestCase):
    def test_pixel_limit_is_applied_and_survives_processor_reload(self):
        from types import SimpleNamespace
        from PIL import Image
        from transformers import Qwen2VLImageProcessorFast

        image_processor = Qwen2VLImageProcessorFast(patch_size=16, merge_size=2)
        configure_image_pixels(SimpleNamespace(image_processor=image_processor), 16384)
        with tempfile.TemporaryDirectory() as directory:
            image_processor.save_pretrained(directory)
            restored = Qwen2VLImageProcessorFast.from_pretrained(directory)
            with Image.new("RGB", (512, 512)) as image:
                grid = restored(images=[image], return_tensors="pt")["image_grid_thw"][0]
            self.assertLessEqual(int(grid[1] * grid[2]) * 16 * 16, 16384)
            self.assertEqual(restored.size["longest_edge"], 16384)

    def test_two_image_checkpointed_backward_and_adapter_roundtrip(self):
        import torch
        from peft import PeftModel
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

        torch.manual_seed(530)
        device = torch.device(os.environ.get("PAIRWISE_TEST_DEVICE", "cpu"))
        config = Qwen3VLConfig(
            text_config={
                "vocab_size": 32, "hidden_size": 32, "intermediate_size": 64,
                "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 8,
                "max_position_embeddings": 128, "use_cache": False,
                "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
            },
            vision_config={
                "depth": 2, "hidden_size": 16, "intermediate_size": 32,
                "num_heads": 2, "patch_size": 2, "temporal_patch_size": 2,
                "spatial_merge_size": 2, "out_hidden_size": 32,
                "num_position_embeddings": 16, "deepstack_visual_indexes": [0],
            },
            image_token_id=12, video_token_id=13,
            vision_start_token_id=10, vision_end_token_id=11,
        )
        config._attn_implementation = "sdpa"
        base = Qwen3VLForConditionalGeneration(config).to(device)
        initial_base_state = {name: value.detach().clone() for name, value in base.state_dict().items()}
        model = attach_lora(base, rank=2, alpha=4, dropout=0)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        self.assertTrue(model.is_gradient_checkpointing)

        trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(any(".visual." in name for name in frozen))
        self.assertTrue(all("lora_" in name and ".language_model." in name for name in trainable))
        trainable_before = {name: parameter.detach().clone() for name, parameter in trainable.items()}

        # Two 2x2 patch grids merge to one image token each. Each flattened
        # patch has 3 channels * 2 temporal frames * 2x2 spatial pixels = 24.
        inputs = {
            "input_ids": torch.tensor([[1, 10, 12, 11, 4, 10, 12, 11, 5, 6]]),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
            "pixel_values": torch.randn(8, 24),
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
        }
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs, logits_to_keep=1, use_cache=False)
        self.assertEqual(tuple(outputs.logits.shape), (1, 1, 32))
        self.assertIsNone(outputs.loss)
        loss, probability = answer_mse(outputs.logits, torch.tensor([1.0], device=device), (3, 7))
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(0 < probability.item() < 1)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                            for parameter in trainable.values()))
        self.assertTrue(all(parameter.grad is None for name, parameter in model.named_parameters() if name in frozen))

        optimizer = torch.optim.AdamW(trainable.values(), lr=0.02)
        optimizer.step()
        self.assertTrue(any(not torch.equal(parameter, trainable_before[name])
                            for name, parameter in trainable.items()))
        for name, parameter in model.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(parameter, frozen[name]), f"Frozen parameter changed: {name}")

        model.eval()
        with torch.no_grad():
            expected = model(**inputs, logits_to_keep=1, use_cache=False).logits
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored_base = Qwen3VLForConditionalGeneration(copy.deepcopy(config)).to(device)
            restored_base.load_state_dict(initial_base_state)
            restored = PeftModel.from_pretrained(restored_base, directory, is_trainable=False)
            restored.eval()
            self.assertFalse(any(parameter.requires_grad for parameter in restored.parameters()))
            with torch.no_grad():
                actual = restored(**inputs, logits_to_keep=1, use_cache=False).logits
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
