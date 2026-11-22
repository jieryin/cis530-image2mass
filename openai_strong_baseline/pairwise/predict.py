"""Compare two objects directly from their images and physical dimensions."""

from __future__ import annotations

from prompts.pairwise import PROMPT, build_openai_content

from ..api import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_REASONING_EFFORT,
    BaselineError,
    request_text,
)


async def predict(client, model, image_a, dims_a, image_b, dims_b, detail="high", *,
                  reasoning_effort=DEFAULT_REASONING_EFFORT,
                  max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
    """Keep each object's dimensions adjacent to its image and parse one label."""
    content = build_openai_content(image_a, dims_a, image_b, dims_b, detail)
    result = await request_text(
        client, model, content,
        reasoning_effort=reasoning_effort, max_output_tokens=max_output_tokens,
    )
    pred = result["raw_output"]
    if pred not in ("A", "B"):
        raise BaselineError("Model response must contain only A or B")
    return {"pred": pred, **result}
