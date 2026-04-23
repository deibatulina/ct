from peft import LoraConfig, TaskType, get_peft_model


def resolve_lora_task_type(task_type):
    if isinstance(task_type, TaskType):
        return task_type

    if not isinstance(task_type, str):
        raise ValueError(f"Unsupported LoRA task type: {task_type}")

    try:
        return getattr(TaskType, task_type)
    except AttributeError as error:
        raise ValueError(f"Unsupported LoRA task type: {task_type}") from error


def build_lora_config(lora_config):
    if not lora_config or not lora_config.get("enabled", False):
        raise ValueError("LoRA config must be enabled for mlp_lora.")

    return LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config.get("dropout", 0.0),
        bias=lora_config.get("bias", "none"),
        task_type=resolve_lora_task_type(lora_config.get("task_type", "CAUSAL_LM")),
        target_modules=lora_config.get("target_modules"),
        fan_in_fan_out=lora_config.get("fan_in_fan_out", False),
    )


def apply_lora_to_language_model(language_model, lora_config):
    return get_peft_model(language_model, build_lora_config(lora_config))
