"""Input validation and backend formatting shared by the task prompts."""

from __future__ import annotations

import base64
import io
import math


class BaselineError(ValueError):
    """Invalid inputs, incompatible cache, or unusable model prediction."""


def positive_number(value, description):
    """Return a finite positive float, rejecting booleans and numeric strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineError(f"{description} must be numeric")
    try:
        number = float(value)
    except OverflowError as exc:
        raise BaselineError(f"{description} must be finite and positive") from exc
    if not math.isfinite(number) or number <= 0:
        raise BaselineError(f"{description} must be finite and positive")
    return number


def dimensions(values, description):
    """Validate the three physical dimensions supplied to the model."""
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise BaselineError(f"{description} must contain three dimensions")
    return [positive_number(value, description) for value in values]


def image_content(image, detail="high"):
    """Encode a decoded image as an in-memory PNG input-image content block."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "type": "input_image",
        "image_url": f"data:image/png;base64,{encoded}",
        "detail": detail,
    }


def qwen_messages(parts):
    """Keep text unchanged and supply RGB images in one Qwen user message."""
    content = []
    for part in parts:
        if part["type"] == "text":
            content.append(part)
        else:
            image = part["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            content.append({"type": "image", "image": image})
    return [{"role": "user", "content": content}]


def openai_content(parts, detail="high"):
    """Translate the same ordered parts to Responses API content blocks."""
    return [
        {"type": "input_text", "text": part["text"]} if part["type"] == "text"
        else image_content(part["image"], detail)
        for part in parts
    ]
