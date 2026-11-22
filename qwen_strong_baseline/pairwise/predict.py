"""Parse Qwen pairwise outputs; re-export the shared prompt for compatibility."""

from __future__ import annotations

from prompts._shared import BaselineError
from prompts.pairwise import PROMPT, build_messages


def parse_output(raw):
    """Accept exactly one uppercase A/B label after trimming whitespace."""
    if not isinstance(raw, str) or raw.strip() not in ("A", "B"):
        raise BaselineError("Qwen pairwise response must contain only A or B")
    return raw.strip()
