"""Shared differentiable answer scoring, model loading, and evaluation.

The final prompt position predicts the FIRST answer token. No gold answer is
fed to the model. This is equivalent to masking every token except that first
answer in a causal-LM training example, without computing unused suffix logits.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_device(name="auto", dtype="auto"):
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; choose --device mps on Apple Silicon")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this Python/PyTorch/macOS environment")
    if dtype == "auto":
        dtype = "bfloat16" if name == "cuda" and torch.cuda.is_bf16_supported() else "float32"
    if name == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support BF16")
    torch_dtype = getattr(torch, dtype)
    # Fail before downloading weights if the selected device cannot use the dtype.
    torch.ones(1, device=name, dtype=torch_dtype)
    return torch.device(name), torch_dtype


def synchronize(device):
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_report(device):
    import torch

    if device.type == "cuda":
        return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if device.type == "mps":
        # These APIs report CURRENT allocation, not peak allocation.
        return {"current_allocated_bytes": torch.mps.current_allocated_memory(),
                "driver_allocated_bytes": torch.mps.driver_allocated_memory()}
    return {}


def answer_token_ids(processor):
    """Verify each choice is exactly one continuation token in the chat context."""
    messages = [{"role": "user", "content": [{"type": "text", "text": "Choose A or B."}]}]
    prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokenizer = processor.tokenizer
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    choices = []
    for answer in ("A", "B"):
        ids = tokenizer.encode(prefix + answer, add_special_tokens=False)
        if ids[:len(prefix_ids)] != prefix_ids or len(ids) != len(prefix_ids) + 1:
            raise ValueError(f"Answer {answer!r} is not one token after the assistant prefix")
        choices.append(ids[-1])
    if choices[0] == choices[1]:
        raise ValueError("A and B must map to distinct tokens")
    return tuple(choices)


def answer_probability(logits, choice_ids):
    """Use only last-position A/B logits; cast before softmax for stable MSE."""
    if logits.ndim != 3 or logits.shape[1] != 1:
        raise ValueError("Expected [batch, 1, vocabulary] next-answer logits")
    return logits[:, 0, list(choice_ids)].float().softmax(dim=-1)[:, 0]


def answer_mse(logits, targets, choice_ids):
    import torch.nn.functional as F

    probability = answer_probability(logits, choice_ids)
    if targets.shape != probability.shape:
        raise ValueError("Targets must have shape [batch]; broadcasting is not allowed")
    return F.mse_loss(probability, targets.float()), probability


def attach_lora(model, rank=8, alpha=16, dropout=0.05):
    from peft import LoraConfig, TaskType, get_peft_model

    # Exact full paths keep adapters out of the frozen vision encoder/merger.
    suffixes = ("q_proj", "k_proj", "v_proj", "o_proj")
    targets = [name for name, _ in model.named_modules()
               if ".language_model." in name and name.rsplit(".", 1)[-1] in suffixes]
    if not targets:
        raise ValueError("No Qwen language attention projections found for LoRA")
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
        lora_dropout=dropout, target_modules=targets, bias="none",
    ))
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name or ".language_model." not in name for name, _ in trainable):
        raise ValueError("Only language attention LoRA parameters may be trainable")
    return model


def configure_image_pixels(processor, max_pixels):
    """Set the fast processor itself; AutoProcessor can discard these kwargs."""
    image_processor = processor.image_processor
    image_processor.size = {"shortest_edge": min(65536, max_pixels), "longest_edge": max_pixels}
    image_processor.min_pixels = min(65536, max_pixels)
    image_processor.max_pixels = max_pixels


def load_model(device, dtype, *, max_pixels=262144, adapter=None, rank=8,
               alpha=16, dropout=0.05, checkpointing=True):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if adapter is None:
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
        configure_image_pixels(processor, max_pixels)
    else:
        processor = AutoProcessor.from_pretrained(adapter, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    model.config.use_cache = False
    if adapter is None:
        model = attach_lora(model, rank, alpha, dropout)
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    choices = answer_token_ids(processor)
    return model, processor, choices


def prepare_inputs(processor, pair, root, device):
    from .data import build_messages_for_pair

    messages = build_messages_for_pair(pair, root)
    try:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
    finally:
        for message in messages:
            for part in message["content"]:
                if part["type"] == "image":
                    part["image"].close()
    inputs.pop("token_type_ids", None)
    # Batch size is deliberately one: final prompt token cannot be padding.
    if inputs["input_ids"].shape[0] != 1 or not bool(inputs["attention_mask"][0, -1]):
        raise ValueError("Expected one unpadded prompt ending at the assistant answer boundary")
    return inputs.to(device)


def predict_probability(model, inputs, choices):
    # No labels= argument: HF's built-in cross-entropy is never computed.
    outputs = model(**inputs, logits_to_keep=1, use_cache=False)
    return answer_probability(outputs.logits, choices)


def evaluate_rows(model, processor, choices, rows, root, device):
    import torch
    from score import score

    model.eval()
    predictions, probabilities, gold = {}, {}, {}
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for pair in rows:
            inputs = prepare_inputs(processor, pair, root, device)
            probability = float(predict_probability(model, inputs, choices).item())
            if not 0 <= probability <= 1:
                raise ValueError(f"Non-finite probability for {pair['pair_id']}")
            pair_id = pair["pair_id"]
            probabilities[pair_id] = probability
            predictions[pair_id] = "A" if probability > 0.5 else "B"
            gold[pair_id] = pair["label"]
    synchronize(device)
    metrics = score(gold, predictions)
    metrics["mse"] = sum((probabilities[key] - (gold[key] == "A")) ** 2 for key in gold) / len(gold)
    metrics["elapsed_seconds"] = time.perf_counter() - started
    return predictions, probabilities, metrics


def save_predictions(directory, predictions, probabilities, metrics):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "preds.jsonl").open("w", encoding="utf-8") as handle:
        for pair_id, prediction in predictions.items():
            handle.write(json.dumps({"pair_id": pair_id, "pred": prediction,
                                     "p_a": probabilities[pair_id]}) + "\n")
    write_json(directory / "scores.json", metrics)


def environment_metadata():
    from importlib.metadata import version

    return {"python": platform.python_version(),
            **{package: version(package) for package in ("torch", "torchvision", "transformers", "peft", "accelerate")}}
