# Shared prompts

Both strong baselines and pairwise fine-tuning import prompt text and builders
from this package. The move preserves the original wording, whitespace,
dimension formatting, image order, and backend payloads.

| File | Responsibility |
|---|---|
| `pairwise.py` | A/B heavier-object prompt and shared two-object construction |
| `pointwise.py` | Numeric weight-estimation prompt and shared one-object construction |
| `_shared.py` | Dimension validation, Qwen RGB messages, OpenAI PNG encoding |

For Qwen zero-shot inference or fine-tuning:

```python
from prompts.pairwise import PROMPT, build_messages

messages = build_messages(image_a, dims_a, image_b, dims_b)
```

For OpenAI pairwise requests:

```python
from prompts.pairwise import build_openai_content

content = build_openai_content(image_a, dims_a, image_b, dims_b, detail="high")
```

Pointwise uses the same function names from `prompts.pointwise`, with one image
and one dimensions argument. The task's ordered text/image parts are built in
one place, then formatted for the backend. Qwen receives RGB image objects in a
single user message. OpenAI receives the original PNG data URLs and image-detail
setting; its response wrapper still supplies the user role.

Existing predictor-module `PROMPT` and Qwen `build_messages` names remain
compatibility exports. New callers should import directly from this package.
The shared helpers do not load model weights, import a model framework, or make
API requests. Output parsing remains in the baseline modules.

Qwen inference includes these source files in its run hashes. Fine-tuning records
the prompt text in checkpoint metadata.
The source move changes those hashes even though inputs are unchanged, so
source-pinned caches/checkpoints from older code require their original code to
resume. Existing saved predictions and results are not rewritten.
