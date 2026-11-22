"""One-image numeric-text task; its training loss is intentionally a placeholder."""

from __future__ import annotations

import time
from pathlib import Path

from openai_strong_baseline.runner import write_json, write_jsonl
from prompts.pointwise import PROMPT, build_messages
from qwen_strong_baseline.numeric_format import NUMERIC_FORMAT_POLICY, PositiveNumberConstraint
from qwen_strong_baseline.pointwise.predict import compare, parse_output
from score import score
from ..data import load_object_record, unique_objects
from ..model import prepare_messages, synchronize


NAME = "pointwise"
COUNT_UNIT = "objects"
MAX_NEW_TOKENS = 32


def training_rows(pairs):
    return unique_objects(pairs)


def build_messages_for_object(obj, root):
    """Load a validated object's image; only image and dimensions enter the prompt.

    The caller owns the returned image.
    """
    record = load_object_record(obj, root)
    try:
        return build_messages(record["image"], obj["dims"])
    except Exception:
        record["image"].close()
        raise


def setup(model, processor):
    return PositiveNumberConstraint(processor.tokenizer, model.generation_config.eos_token_id)


def metadata(context=None):
    return {"answer_policy": NUMERIC_FORMAT_POLICY, "loss": "not_implemented",
            "tie_label": "B", "weight_unit": "pounds", "output_type": "numeric_text",
            "max_new_tokens": MAX_NEW_TOKENS,
            "selection": "not_implemented"}


def prepare_inputs(processor, obj, root, device):
    return prepare_messages(processor, build_messages_for_object(obj, root), device)


def training_loss(model, processor, context, row, root, device):
    """TODO: define the pointwise objective; numeric-text output is agreed."""
    raise NotImplementedError(
        "Pointwise text training loss is not implemented yet. "
        "Implement finetune/pointwise_finetune/task.py:training_loss before training; "
        "--dry-run can validate the shared data pipeline now."
    )


def predict_weight(model, processor, inputs, constraint):
    """Reuse the zero-shot baseline's numeric grammar and greedy text decoding."""
    prompt_length = inputs["input_ids"].shape[-1]
    outputs = model.generate(
        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
        temperature=None, top_p=None, top_k=None, use_cache=True,
        prefix_allowed_tokens_fn=constraint.for_prompt(prompt_length),
    )
    generated = outputs[0, prompt_length:]
    if len(generated) >= MAX_NEW_TOKENS and generated[-1].item() not in constraint.eos_token_ids:
        raise ValueError("Pointwise weight generation reached the token limit without EOS")
    text = processor.batch_decode(
        [generated], skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    return parse_output(text)


def evaluate_rows(model, processor, context, rows, root, device):
    import torch

    model.eval()
    weights = {}
    objects = unique_objects(rows)
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for obj in objects:
            inputs = prepare_inputs(processor, obj, root, device)
            weights[obj["id"]] = predict_weight(model, processor, inputs, context)
    predictions = {row["pair_id"]: compare(weights[row["id_a"]], weights[row["id_b"]]) for row in rows}
    gold = {row["pair_id"]: row["label"] for row in rows}
    synchronize(device)
    metrics = score(gold, predictions)
    metrics["objects"] = len(objects)
    metrics["elapsed_seconds"] = time.perf_counter() - started
    return predictions, weights, metrics


def save_predictions(directory, predictions, weights, metrics):
    directory = Path(directory)
    write_jsonl(directory / "preds.jsonl", (
        {"pair_id": pair_id, "pred": prediction} for pair_id, prediction in predictions.items()
    ))
    write_jsonl(directory / "weights.jsonl", (
        {"id": object_id, "weight_lb": weight} for object_id, weight in weights.items()
    ))
    write_json(directory / "scores.json", metrics)


def subset_metrics(subset, predictions, weights):
    selected = {row["pair_id"]: predictions[row["pair_id"]] for row in subset}
    gold = {row["pair_id"]: row["label"] for row in subset}
    metrics = score(gold, selected)
    metrics["objects"] = len(unique_objects(subset))
    return metrics


def selection_key(metrics):
    """TODO: choose how pointwise checkpoints should be ranked."""
    raise NotImplementedError("Pointwise checkpoint selection_key is not implemented yet.")
