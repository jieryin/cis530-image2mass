"""Compare two objects directly from their images and physical dimensions."""

from __future__ import annotations

import json

from ..api import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_REASONING_EFFORT,
    BaselineError,
    dimensions,
    image_content,
    request_text,
)


PROMPT = (
    "Compare objects A and B using their images and physical dimensions in inches. "
    "Consider each object's size, likely material, and construction to decide which is heavier. "
    "Return A if object A is heavier, otherwise B. "
    "Return only A or B, with no explanation."
)


async def predict(client, model, image_a, dims_a, image_b, dims_b, detail="high", *,
                  reasoning_effort=DEFAULT_REASONING_EFFORT,
                  max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
    """Keep each object's dimensions adjacent to its image and parse one label."""
    content = [{"type": "input_text", "text": PROMPT}]
    for side, image, dims in (("A", image_a, dims_a), ("B", image_b, dims_b)):
        dims = dimensions(dims, f"Object {side} input dimensions")
        content.extend([
            {"type": "input_text", "text": f"Object {side}: dimensions in inches = {json.dumps(dims)}. Image follows."},
            image_content(image, detail),
        ])
    result = await request_text(
        client, model, content,
        reasoning_effort=reasoning_effort, max_output_tokens=max_output_tokens,
    )
    pred = result["raw_output"]
    if pred not in ("A", "B"):
        raise BaselineError("Model response must contain only A or B")
    return {"pred": pred, **result}
