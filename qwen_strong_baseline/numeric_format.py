"""A token-level positive JSON number grammar for pointwise generation.

The grammar constrains generation before decoding; it never repairs a response
or uses labels. It depends only on a tokenizer and can be tested without a model,
CUDA, or network access. Images, dimensions, and prompt text are unchanged.
"""

from __future__ import annotations

import json
import math
import re

from openai_strong_baseline.api import BaselineError


NUMERIC_FORMAT_POLICY = "positive_json_number_v1"
_NUMERIC_TOKEN = re.compile(r"[0-9.eE+\-]+\Z")


def _positive_complete(text):
    try:
        value = json.loads(text)
        return (type(value) in (int, float) and value > 0
                and math.isfinite(float(value)))
    except (ValueError, TypeError, OverflowError):
        return False


def _prefix_state(text):
    """Return a JSON-number prefix state, or None if no positive end is possible.

    Initial minus/plus and exponents on a zero mantissa cannot yield a positive
    JSON number. Completed exponent prefixes that already overflow/underflow
    cannot be rescued by adding exponent digits and are rejected as well.
    """
    state = "start"
    nonzero_mantissa = False
    for char in text:
        digit = "0" <= char <= "9"
        if state == "start":
            if char == "0":
                state = "zero"
            elif digit:
                state = "integer"
                nonzero_mantissa = True
            else:
                return None
        elif state in ("zero", "integer"):
            if digit and state == "integer":
                nonzero_mantissa = nonzero_mantissa or char != "0"
            elif char == ".":
                state = "decimal_point"
            elif char in "eE" and nonzero_mantissa:
                state = "exponent_marker"
            else:
                return None
        elif state == "decimal_point":
            if not digit:
                return None
            state = "fraction"
            nonzero_mantissa = nonzero_mantissa or char != "0"
        elif state == "fraction":
            if digit:
                nonzero_mantissa = nonzero_mantissa or char != "0"
            elif char in "eE" and nonzero_mantissa:
                state = "exponent_marker"
            else:
                return None
        elif state == "exponent_marker":
            if char in "+-":
                state = "exponent_sign"
            elif digit:
                state = "exponent_digits"
            else:
                return None
        elif state == "exponent_sign":
            if not digit:
                return None
            state = "exponent_digits"
        elif state == "exponent_digits":
            if not digit:
                return None
    if state == "exponent_digits" and not _positive_complete(text):
        # Adding exponent digits can only increase its absolute magnitude. A
        # negative exponent cannot rescue underflow; a nonnegative exponent
        # cannot rescue overflow. The opposite combinations remain viable for
        # unusually large/small mantissas even before a finite endpoint exists.
        mantissa, exponent = re.split("[eE]", text)
        try:
            value = float(text)
        except (ValueError, OverflowError):
            return None
        negative_exponent = exponent.startswith("-")
        if (value == 0 and negative_exponent) or (not math.isfinite(value) and not negative_exponent):
            return None
    return state


class PositiveNumberConstraint:
    """Build ``prefix_allowed_tokens_fn`` callbacks for positive JSON numbers.

    ``allowed_tokens(prefix)`` accepts a decoded generated-text prefix and
    returns token IDs. Whole numeric vocabulary tokens are considered, including
    multi-character pieces; restricting generation to individual digits would
    unnecessarily change the greedy choice among valid numeric continuations.
    """

    def __init__(self, tokenizer, eos_token_ids):
        self.tokenizer = tokenizer
        if type(eos_token_ids) is int:
            eos_token_ids = [eos_token_ids]
        if (not isinstance(eos_token_ids, (list, tuple, set)) or not eos_token_ids
                or any(type(item) is not int or item < 0 for item in eos_token_ids)):
            raise BaselineError("Numeric format constraint requires valid EOS token IDs")
        self.eos_token_ids = tuple(sorted(set(eos_token_ids)))
        candidates = {}
        for token, token_id in tokenizer.get_vocab().items():
            if (type(token_id) is not int or token_id < 0
                    or not isinstance(token, str) or not _NUMERIC_TOKEN.fullmatch(token)
                    or token_id in self.eos_token_ids):
                continue
            decoded = tokenizer.decode(
                [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False,
            )
            if decoded == token:
                candidates[token_id] = decoded
        if not candidates:
            raise BaselineError("Tokenizer contains no verified numeric tokens")
        self.numeric_tokens = tuple(sorted(candidates.items()))
        self._allowed_cache = {}
        if not self.allowed_tokens(""):
            raise BaselineError("Tokenizer cannot begin a positive JSON number")

    def allowed_tokens(self, prefix):
        """Mask invalid continuations; allow EOS only at finite positive values."""
        if not isinstance(prefix, str) or _prefix_state(prefix) is None:
            raise BaselineError(f"Invalid numeric generation prefix {prefix!r}")
        cached = self._allowed_cache.get(prefix)
        if cached is None:
            allowed = [token_id for token_id, text in self.numeric_tokens
                       if _prefix_state(prefix + text) is not None]
            if _positive_complete(prefix):
                allowed.extend(self.eos_token_ids)
            cached = tuple(sorted(set(allowed)))
            if not cached:
                raise BaselineError(f"Numeric grammar has no continuation for {prefix!r}")
            self._allowed_cache[prefix] = cached
        return list(cached)

    def for_prompt(self, prompt_length):
        """Return the official Transformers callback, excluding prompt tokens."""
        if type(prompt_length) is not int or prompt_length < 0:
            raise BaselineError("Numeric constraint prompt length must be nonnegative")

        def prefix_allowed_tokens_fn(batch_id, input_ids):
            generated = input_ids[prompt_length:]
            if hasattr(generated, "tolist"):
                generated = generated.tolist()
            prefix = self.tokenizer.decode(
                list(generated), skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            return self.allowed_tokens(prefix)

        return prefix_allowed_tokens_fn
