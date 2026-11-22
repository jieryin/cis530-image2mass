"""Estimate one object's weight from its image and physical dimensions."""

from __future__ import annotations

import json

from prompts.pointwise import PROMPT, build_openai_content

from ..api import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_REASONING_EFFORT,
    BaselineError,
    positive_number,
    request_text,
)


async def predict(client, model, image, dims, detail="high", *,
                  reasoning_effort=DEFAULT_REASONING_EFFORT,
                  max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
    """Ask for one numeric estimate without supplying the record's true weight."""
    content = build_openai_content(image, dims, detail)
    result = await request_text(
        client, model, content,
        reasoning_effort=reasoning_effort, max_output_tokens=max_output_tokens,
    )
    try:
        value = json.loads(result["raw_output"])
    except ValueError as exc:
        raise BaselineError("Model response must contain only a positive number") from exc
    weight = positive_number(value, "Predicted weight_lb")
    return {"weight_lb": weight, **result}
