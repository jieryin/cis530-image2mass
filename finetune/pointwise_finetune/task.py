"""One-image pointwise mass regression with answer-only MSE supervision."""

from __future__ import annotations

import math
import time
from pathlib import Path

from openai_strong_baseline.runner import write_json, write_jsonl
from prompts.pointwise import PROMPT, build_messages
from qwen_strong_baseline.pointwise.predict import compare
from score import score
from ..data import load_object_record, unique_objects
from ..model import get_mass_head, prepare_messages, synchronize


NAME = "pointwise"
COUNT_UNIT = "objects"
TARGET_TRANSFORM = "log1p_pounds"
MIN_WEIGHT_LB = 1e-8


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
    # PEFT saves/restores this module through ``modules_to_save``.
    return get_mass_head(model)


def metadata(context=None):
    return {"answer_policy": "scalar_regression_v1",
            "loss": "mse_on_log1p_weight_lb",
            "target_transform": TARGET_TRANSFORM,
            "tie_label": "B", "weight_unit": "pounds",
            "output_type": "scalar_regression",
            "selection": "dev accuracy, then lower dev object log-MSE"}


def prepare_inputs(processor, obj, root, device):
    return prepare_messages(processor, build_messages_for_object(obj, root), device)


def predict_log_weight(model, inputs, mass_head):
    """Predict a nonnegative, differentiable log1p weight."""
    import torch.nn.functional as F

    outputs = model(
        **inputs, output_hidden_states=True, return_dict=True,
        use_cache=False, logits_to_keep=1,
    )
    if not outputs.hidden_states:
        raise ValueError("The model did not return language hidden states")
    final_state = outputs.hidden_states[-1][:, -1, :]
    # softplus is smooth during training and guarantees a physically valid
    # nonnegative log1p mass without discarding gradients at zero.
    prediction = F.softplus(mass_head(final_state).squeeze(-1).float())
    if prediction.ndim != 1 or prediction.shape[0] != 1:
        raise ValueError("Pointwise regression expects batch size one")
    return prediction


def training_loss(model, processor, context, row, root, device):
    """MSE between predicted and gold log1p weight; prompt tokens have no loss."""
    import torch
    import torch.nn.functional as F

    inputs = prepare_inputs(processor, row, root, device)
    prediction = predict_log_weight(model, inputs, context)
    target = torch.tensor([math.log1p(row["weight"])], device=device, dtype=torch.float32)
    return F.mse_loss(prediction, target)


def predict_weight(model, processor, inputs, mass_head):
    """Return a positive weight in pounds from the scalar regression output."""
    predicted_log_weight = predict_log_weight(model, inputs, mass_head).item()
    if not math.isfinite(predicted_log_weight):
        raise ValueError("Pointwise model produced a non-finite log weight")
    weight = max(MIN_WEIGHT_LB, math.expm1(predicted_log_weight))
    if not math.isfinite(weight):
        raise ValueError("Pointwise model produced a non-finite weight")
    return weight


def object_error_metrics(objects, weights):
    """Diagnostic regression metrics over the unique objects in a pair slice."""
    squared_log_errors, absolute_errors, squared_errors = [], [], []
    for obj in objects:
        predicted, gold = weights[obj["id"]], obj["weight"]
        squared_log_errors.append((math.log1p(predicted) - math.log1p(gold)) ** 2)
        absolute_errors.append(abs(predicted - gold))
        squared_errors.append((predicted - gold) ** 2)
    count = len(objects)
    return {
        "object_log_mse": sum(squared_log_errors) / count,
        "object_mae_lb": sum(absolute_errors) / count,
        "object_rmse_lb": math.sqrt(sum(squared_errors) / count),
    }


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
    metrics.update(object_error_metrics(objects, weights))
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
    objects = unique_objects(subset)
    metrics["objects"] = len(objects)
    metrics.update(object_error_metrics(objects, weights))
    return metrics


def selection_key(metrics):
    return metrics["accuracy"], -metrics["object_log_mse"]
