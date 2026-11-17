"""GPU-only, deterministic Qwen3-VL inference shared by both baselines.

Heavy imports and model loading happen only when QwenPredictor is constructed,
so CLI validation and fake-predictor tests need neither CUDA nor model weights.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import time

from openai_strong_baseline.api import BaselineError
from qwen_strong_baseline.numeric_format import NUMERIC_FORMAT_POLICY, PositiveNumberConstraint


DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


def dependency_versions():
    """Record the local environment without importing GPU or network clients."""
    result = {"python": platform.python_version()}
    for package in ("torch", "torchvision", "transformers", "accelerate", "huggingface-hub",
                    "numpy", "Pillow", "lz4", "safetensors", "tokenizers"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


class QwenPredictor:
    """Load once on CUDA in BF16, and return raw responses before parsing.

    ``revision`` must be an immutable Hugging Face commit, resolved before the
    run/cache identity is created. No CPU or automatic device offload is used.
    """

    def __init__(self, model_id=DEFAULT_MODEL, *, revision, max_new_tokens=32):
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise BaselineError("Qwen revision must be an immutable 40-character commit SHA")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise BaselineError("max_new_tokens must be a positive integer")

        import torch
        from huggingface_hub import snapshot_download
        from huggingface_hub.constants import HF_HUB_OFFLINE
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise BaselineError("Qwen requires CUDA; CPU fallback is disabled")
        if not torch.cuda.is_bf16_supported():
            raise BaselineError("This GPU does not support the required BF16 configuration")
        self.torch = torch
        self.device = torch.device("cuda:0")
        self.max_new_tokens = max_new_tokens
        # Resolve the immutable snapshot first. Loading its tokenizer by local
        # path avoids a Transformers 4.57.3 Mistral-regex metadata probe that
        # otherwise tries the Hub even when all Qwen files are cached/offline.
        snapshot = snapshot_download(model_id, revision=revision,
                                     local_files_only=HF_HUB_OFFLINE)
        self.processor = AutoProcessor.from_pretrained(snapshot)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.model.requires_grad_(False)
        if any(parameter.device.type != "cuda" for parameter in self.model.parameters()):
            raise BaselineError("Qwen parameters are not all on CUDA; refusing mixed/CPU execution")
        self._parameter_versions = self._versions()
        actual_revision = getattr(self.model.config, "_commit_hash", None)
        if actual_revision != revision:
            raise BaselineError(f"Loaded model revision {actual_revision!r} differs from requested {revision!r}")
        self.numeric_constraint = PositiveNumberConstraint(
            self.processor.tokenizer, self.model.generation_config.eos_token_id,
        )

        self.identity = {
            "model": model_id,
            "model_revision": revision,
            "dtype": "bfloat16",
            "device": "cuda:0",
            "batch_size": 1,
            "attention_implementation": "sdpa",
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "pointwise_format_constraint": NUMERIC_FORMAT_POLICY,
            "pairwise_format_constraint": None,
            "preprocessing": "AutoProcessor defaults at model revision; no custom resize/crop",
        }
        chat_template = self.processor.chat_template
        if not isinstance(chat_template, str):
            chat_template = json.dumps(chat_template, sort_keys=True)
        self.metadata = {
            **self.identity,
            "versions": dependency_versions(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "image_processor": self.processor.image_processor.to_dict(),
            "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
            "generation": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": max_new_tokens,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "use_cache": True,
                "pointwise_prefix_allowed_tokens_fn": NUMERIC_FORMAT_POLICY,
                "pairwise_prefix_allowed_tokens_fn": None,
            },
            "training": False,
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "parameter_versions_unchanged": True,
            "timing": "CUDA synchronized; includes preprocessing, transfer, generation and decode; excludes download/loading",
            "peak_memory": "torch.cuda.max_memory_allocated on cuda:0, reset before each call",
        }

    def _versions(self):
        return tuple((name, parameter._version) for name, parameter in self.model.named_parameters())

    def __call__(self, messages):
        """Return the unmodified text and diagnostics; caller persists errors.

        A truncated response is returned with ``truncated=True`` so that the
        runner can save its raw text before refusing to parse or score it.
        """
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise BaselineError("Qwen baseline expects exactly one user message")
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise BaselineError("Qwen must remain frozen in evaluation mode")
        if self._versions() != self._parameter_versions:
            raise BaselineError("Qwen parameters changed after loading")
        content = messages[0].get("content", [])
        image_count = sum(part.get("type") == "image" for part in content)
        pointwise_input = (len(content) == 3 and image_count == 1
                           and [part.get("type") for part in content] == ["text", "text", "image"])
        generation_options = {}

        torch = self.torch
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            # Some Transformers versions include this auxiliary text field;
            # Qwen3-VL has no use for it (see the official model documentation).
            inputs.pop("token_type_ids", None)
            inputs = inputs.to(self.device)
            if pointwise_input:
                generation_options["prefix_allowed_tokens_fn"] = self.numeric_constraint.for_prompt(
                    inputs["input_ids"].shape[-1],
                )
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
                **generation_options,
            )
            generated = outputs[0, inputs["input_ids"].shape[-1]:]
            token_ids = generated.tolist()
            raw_output = self.processor.batch_decode(
                [generated], skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )[0]
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        peak_memory = torch.cuda.max_memory_allocated(self.device)
        if self._versions() != self._parameter_versions:
            self.metadata["parameter_versions_unchanged"] = False
            raise BaselineError("Qwen parameters changed during inference")

        eos_ids = self.model.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        eos_ids = set(eos_ids or [])
        ended = bool(token_ids and token_ids[-1] in eos_ids)
        return {
            "raw_output": raw_output,
            "elapsed_seconds": elapsed,
            "peak_memory_allocated_bytes": peak_memory,
            "generated_tokens": len(token_ids),
            "truncated": len(token_ids) >= self.max_new_tokens and not ended,
            "parameter_versions_unchanged": True,
            "format_constraint": NUMERIC_FORMAT_POLICY if pointwise_input else None,
        }
