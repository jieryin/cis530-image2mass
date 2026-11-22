"""The unchanged heavier-object prompt and its shared message construction."""

from __future__ import annotations

import json

from ._shared import dimensions, openai_content, qwen_messages


PROMPT = (
    "Compare objects A and B using their images and physical dimensions in inches. "
    "Consider each object's size, likely material, and construction to decide which is heavier. "
    "Return A if object A is heavier, otherwise B. "
    "Return only A or B, with no explanation."
)


def _parts(image_a, dims_a, image_b, dims_b):
    """Accept only images and dimensions; keep each object's inputs together."""
    yield {"type": "text", "text": PROMPT}
    for side, image, dims in (("A", image_a, dims_a), ("B", image_b, dims_b)):
        dims = dimensions(dims, f"Object {side} input dimensions")
        yield {"type": "text", "text": f"Object {side}: dimensions in inches = {json.dumps(dims)}. Image follows."}
        yield {"type": "image", "image": image}


def build_messages(image_a, dims_a, image_b, dims_b):
    """Build Qwen messages shared by zero-shot prediction and fine-tuning."""
    return qwen_messages(_parts(image_a, dims_a, image_b, dims_b))


def build_openai_content(image_a, dims_a, image_b, dims_b, detail="high"):
    """Build the equivalent OpenAI content with the original image encoding."""
    return openai_content(_parts(image_a, dims_a, image_b, dims_b), detail)
