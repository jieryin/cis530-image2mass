"""OpenAI response handling and compatibility exports for shared input helpers."""

from __future__ import annotations

from prompts._shared import BaselineError, dimensions, image_content, positive_number


DEFAULT_REASONING_EFFORT = "high"
DEFAULT_MAX_OUTPUT_TOKENS = 4096


async def request_text(client, model, content, *,
                       reasoning_effort=DEFAULT_REASONING_EFFORT,
                       max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
    """Await a plain-text response and retain metadata for the experiment cache."""
    settings = {"reasoning": {"effort": reasoning_effort}} if reasoning_effort != "default" else {}
    response = await client.responses.create(
        model=model,
        store=False,
        service_tier="default",
        max_output_tokens=max_output_tokens,
        **settings,
        input=[{"role": "user", "content": content}],
        text={"format": {"type": "text"}},
    )
    status = getattr(response, "status", None)
    if status != "completed":
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None)
        hint = (
            "; output token limit reached. Try a larger --max-output-tokens with a new cache"
            if reason == "max_output_tokens" else ""
        )
        raise BaselineError(f"Model response is {status!r}, not completed{hint}")
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "message":
            if any(getattr(part, "type", None) == "refusal" for part in item.content):
                raise BaselineError("Model refused the prediction")
    raw_output = getattr(response, "output_text", None)
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise BaselineError("Model response did not contain text")
    usage = getattr(response, "usage", None)
    return {
        "raw_output": raw_output.strip(),
        "response_id": getattr(response, "id", None),
        "response_model": getattr(response, "model", None),
        "usage": usage.model_dump(mode="json") if usage is not None else None,
    }
