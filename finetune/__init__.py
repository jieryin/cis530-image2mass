"""Shared Qwen LoRA training infrastructure with task-specific objectives."""

from importlib import import_module


TASK_NAMES = ("pairwise", "pointwise")


def get_task(name):
    """Load the selected task without loading model weights."""
    return import_module(f"{__name__}.{name}_finetune.task")
