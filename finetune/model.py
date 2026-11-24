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


def attach_mass_head(model):
    """Attach the scalar head used by pointwise mass regression.

    The head predicts log1p(weight_lb) from the final assistant-prefix hidden
    state. It is attached before PEFT so ``modules_to_save`` can persist it
    beside the LoRA adapter.
    """
    import torch

    text_config = getattr(model.config, "text_config", model.config)
    hidden_size = getattr(text_config, "hidden_size", None)
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError("Could not determine the language-model hidden size")
    reference = next(model.parameters())
    model.mass_head = torch.nn.Linear(
        hidden_size, 1, bias=True, device=reference.device, dtype=reference.dtype,
    )
    # The final Qwen hidden state can have a large norm. A small initialization
    # keeps the initial scalar near zero; pointwise softplus then starts from a
    # stable prior of log1p(weight_lb) ~= log(2), or roughly one pound.
    with torch.no_grad():
        torch.nn.init.normal_(model.mass_head.weight, mean=0.0, std=1e-3)
        torch.nn.init.zeros_(model.mass_head.bias)
    return model.mass_head


def get_mass_head(model):
    """Return the pointwise head from either a base or PEFT-wrapped model."""
    candidates = [model]
    if hasattr(model, "get_base_model"):
        candidates.append(model.get_base_model())
    for candidate in candidates:
        head = getattr(candidate, "mass_head", None)
        if head is not None:
            return head
    raise ValueError("This model has no pointwise mass regression head")


def attach_lora(model, rank=MODEL_DEFAULTS["rank"], alpha=MODEL_DEFAULTS["alpha"],
                dropout=MODEL_DEFAULTS["dropout"], modules_to_save=None):
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
        modules_to_save=modules_to_save,
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
               checkpointing=True, pointwise_regression=False):
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
    if pointwise_regression:
        attach_mass_head(model)
    if adapter is None:
        model = attach_lora(
            model, rank, alpha, dropout,
            modules_to_save=["mass_head"] if pointwise_regression else None,
        )
    else:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable_adapter)
    if adapter is None or trainable_adapter:
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        def allowed(name):
            language_lora = "lora_" in name and ".language_model." in name
            saved_mass_head = pointwise_regression and ".mass_head." in name
            return language_lora or saved_mass_head

        if not trainable or any(not allowed(name) for name in trainable):
            raise ValueError("Only language attention LoRA and the optional mass head may be trainable")
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
