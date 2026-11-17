"""Build the same pairwise prompt as the OpenAI baseline, using real images."""

from __future__ import annotations

import json

from openai_strong_baseline.api import BaselineError, dimensions
from openai_strong_baseline.pairwise.predict import PROMPT


def build_messages(image_a, dims_a, image_b, dims_b):
    """Only images and dimensions cross this boundary; metadata stays local."""
    content = [{"type": "text", "text": PROMPT}]
    for side, image, dims in (("A", image_a, dims_a), ("B", image_b, dims_b)):
        dims = dimensions(dims, f"Object {side} input dimensions")
        if image.mode != "RGB":
            image = image.convert("RGB")
        content.extend([
            {"type": "text", "text": f"Object {side}: dimensions in inches = {json.dumps(dims)}. Image follows."},
            {"type": "image", "image": image},
        ])
    return [{"role": "user", "content": content}]


def parse_output(raw):
    """Accept exactly one uppercase A/B label after trimming whitespace."""
    if not isinstance(raw, str) or raw.strip() not in ("A", "B"):
        raise BaselineError("Qwen pairwise response must contain only A or B")
    return raw.strip()
