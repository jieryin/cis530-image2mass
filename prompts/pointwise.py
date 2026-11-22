"""The unchanged weight-estimation prompt and shared message construction."""

from __future__ import annotations

import json

from ._shared import dimensions, openai_content, qwen_messages


PROMPT = (
    "Estimate the product's weight in pounds from its image and physical dimensions in inches. "
    "Consider its size, likely material, and construction, and give your best estimate. "
    "Return only one positive number, with no units or explanation."
)


def _parts(image, dims):
    dims = dimensions(dims, "Object input dimensions")
    return [
        {"type": "text", "text": PROMPT},
        {"type": "text", "text": f"Physical dimensions in inches = {json.dumps(dims)}. Image follows."},
        {"type": "image", "image": image},
    ]


def build_messages(image, dims):
    """Build one Qwen user message with dimensions adjacent to its image."""
    return qwen_messages(_parts(image, dims))


def build_openai_content(image, dims, detail="high"):
    """Build the equivalent OpenAI content with the original image encoding."""
    return openai_content(_parts(image, dims), detail)
