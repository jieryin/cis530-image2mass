"""Build the original pointwise prompt and parse strictly positive weights."""

from __future__ import annotations

import json

from openai_strong_baseline.api import BaselineError, dimensions, positive_number
from openai_strong_baseline.pointwise.predict import PROMPT


def build_messages(image, dims):
    """Return one user message containing dimensions adjacent to its image."""
    dims = dimensions(dims, "Object input dimensions")
    if image.mode != "RGB":
        image = image.convert("RGB")
    return [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "text", "text": f"Physical dimensions in inches = {json.dumps(dims)}. Image follows."},
        {"type": "image", "image": image},
    ]}]


def parse_output(raw):
    """Use the OpenAI numeric policy: a JSON number, finite and above zero."""
    if not isinstance(raw, str):
        raise BaselineError("Qwen pointwise response must contain only a positive number")
    try:
        value = json.loads(raw.strip())
    except (ValueError, TypeError) as exc:
        raise BaselineError("Qwen pointwise response must contain only a positive number") from exc
    return positive_number(value, "Predicted weight_lb")


def compare(weight_a, weight_b):
    """Equal valid estimates select B, matching the shared evaluation policy."""
    weight_a = positive_number(weight_a, "Object A predicted weight_lb")
    weight_b = positive_number(weight_b, "Object B predicted weight_lb")
    return "A" if weight_a > weight_b else "B"
