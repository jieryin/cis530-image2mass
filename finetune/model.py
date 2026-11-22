"""Task-independent Qwen loading, LoRA adapters, and multimodal processing."""

from __future__ import annotations

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
MODEL_DEFAULTS = {"rank": 8, "alpha": 16, "dropout": 0.05, "max_pixels": 262144}


def select_device(name="auto", dtype="auto"):
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if dtype == "auto":
        dtype = "bfloat16" if name == "cuda" and torch.cuda.is_bf16_supported() else "float32"
    return torch.device(name), getattr(torch, dtype)


def synchronize(device):
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_report(device):
    import torch

    if device.type == "cuda":
        return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if device.type == "mps":
        # These APIs report CURRENT allocation, not peak allocation.
        return {"current_allocated_bytes": torch.mps.current_allocated_memory(),
                "driver_allocated_bytes": torch.mps.driver_allocated_memory()}
    return {}


def attach_lora(model, rank=MODEL_DEFAULTS["rank"], alpha=MODEL_DEFAULTS["alpha"],
                dropout=MODEL_DEFAULTS["dropout"]):
    from peft import LoraConfig, TaskType, get_peft_model

    # Exact full paths keep adapters out of the frozen vision encoder/merger.
    suffixes = ("q_proj", "k_proj", "v_proj", "o_proj")
    targets = [name for name, _ in model.named_modules()
               if ".language_model." in name and name.rsplit(".", 1)[-1] in suffixes]
    if not targets:
        raise ValueError("No Qwen language attention projections found for LoRA")
    return get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
        lora_dropout=dropout, target_modules=targets, bias="none",
    ))


def configure_image_pixels(processor, max_pixels=MODEL_DEFAULTS["max_pixels"]):
    """Set the fast processor itself; AutoProcessor can discard these kwargs."""
    image_processor = processor.image_processor
    image_processor.size = {"shortest_edge": min(65536, max_pixels), "longest_edge": max_pixels}
    image_processor.min_pixels = min(65536, max_pixels)
    image_processor.max_pixels = max_pixels


def load_model(device, dtype, *, adapter=None, trainable_adapter=False,
               max_pixels=MODEL_DEFAULTS["max_pixels"], rank=MODEL_DEFAULTS["rank"],
               alpha=MODEL_DEFAULTS["alpha"], dropout=MODEL_DEFAULTS["dropout"],
               checkpointing=True):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if adapter is None:
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
        configure_image_pixels(processor, max_pixels)
    else:
        processor = AutoProcessor.from_pretrained(adapter, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    model.config.use_cache = False
    if adapter is None:
        model = attach_lora(model, rank, alpha, dropout)
    else:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable_adapter)
    if adapter is None or trainable_adapter:
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in name or ".language_model." not in name for name in trainable):
            raise ValueError("Only language attention LoRA parameters may be trainable")
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, processor


def prepare_messages(processor, messages, device):
    """Tokenize one prompt; close the supplied image objects even on failure."""
    try:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
    finally:
        for message in messages:
            for part in message["content"]:
                if part["type"] == "image":
                    part["image"].close()
    inputs.pop("token_type_ids", None)
    return inputs.to(device)
