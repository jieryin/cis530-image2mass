"""Two-image A/B supervision with the original first-answer probability MSE."""

from __future__ import annotations

import time
from pathlib import Path

from openai_strong_baseline.runner import write_json, write_jsonl
from prompts.pairwise import PROMPT, build_messages
from score import score
from ..data import load_object_record, pair_object
from ..model import prepare_messages, synchronize


NAME = "pairwise"
COUNT_UNIT = "pairs"


def training_rows(pairs):
    return pairs


def build_messages_for_pair(pair, root):
    """Load a validated pair's images; only images and dimensions enter the prompt.

    The caller owns the returned image objects.
    """
    records = []
    try:
        for side in ("a", "b"):
            records.append(load_object_record(pair_object(pair, side), root))
        return build_messages(records[0]["image"], pair["dims_a"],
                              records[1]["image"], pair["dims_b"])
    except Exception:
        for record in records:
            record["image"].close()
        raise


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


def setup(model, processor):
    return answer_token_ids(processor)


def metadata(context=None):
    result = {"probability_space": "binary", "answer_policy": "leading_choice_only_v1",
              "tie_label": "B", "loss": "answer_probability_mse",
              "selection": "dev accuracy, then lower dev MSE"}
    if context is not None:
        result["answer_token_ids"] = list(context)
    return result


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


def prepare_inputs(processor, pair, root, device):
    return prepare_messages(processor, build_messages_for_pair(pair, root), device)


def predict_probability(model, inputs, choices):
    # No labels= argument: HF's built-in cross-entropy is never computed.
    outputs = model(**inputs, logits_to_keep=1, use_cache=False)
    return answer_probability(outputs.logits, choices)


def training_loss(model, processor, context, row, root, device):
    import torch

    inputs = prepare_inputs(processor, row, root, device)
    outputs = model(**inputs, logits_to_keep=1, use_cache=False)
    target = torch.tensor([float(row["label"] == "A")], device=device)
    loss, _ = answer_mse(outputs.logits, target, context)
    return loss


def evaluate_rows(model, processor, choices, rows, root, device):
    import torch

    model.eval()
    predictions, probabilities = {}, {}
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for pair in rows:
            inputs = prepare_inputs(processor, pair, root, device)
            probability = predict_probability(model, inputs, choices).item()
            if not 0 <= probability <= 1:
                raise ValueError(f"Non-finite probability for {pair['pair_id']}")
            pair_id = pair["pair_id"]
            probabilities[pair_id] = probability
            predictions[pair_id] = "A" if probability > 0.5 else "B"
    synchronize(device)
    metrics = subset_metrics(rows, predictions, probabilities)
    metrics["elapsed_seconds"] = time.perf_counter() - started
    return predictions, probabilities, metrics


def save_predictions(directory, predictions, probabilities, metrics):
    directory = Path(directory)
    write_jsonl(directory / "preds.jsonl", (
        {"pair_id": pair_id, "pred": prediction, "p_a": probabilities[pair_id]}
        for pair_id, prediction in predictions.items()
    ))
    write_json(directory / "scores.json", metrics)


def subset_metrics(subset, predictions, probabilities):
    selected = {row["pair_id"]: predictions[row["pair_id"]] for row in subset}
    gold = {row["pair_id"]: row["label"] for row in subset}
    metrics = score(gold, selected)
    metrics["mse"] = sum((probabilities[key] - (gold[key] == "A")) ** 2 for key in gold) / len(gold)
    return metrics


def selection_key(metrics):
    return metrics["accuracy"], -metrics["mse"]
