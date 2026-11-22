"""Parse Qwen weight estimates; re-export the shared prompt for compatibility."""

from __future__ import annotations

import json

from prompts._shared import BaselineError, positive_number
from prompts.pointwise import PROMPT, build_messages


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
