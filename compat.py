import sys
from typing import Any, Dict, Tuple


IS_MLX = sys.platform == "darwin"


if IS_MLX:
    # MLX-Tune (Apple Silicon) provides an Unsloth-compatible API surface.
    from mlx_tune import FastLanguageModel  # type: ignore
    from mlx_tune import SFTTrainer  # type: ignore
    from mlx_tune import train_on_responses_only  # type: ignore
else:
    # Unsloth (CUDA) + TRL trainer on NVIDIA GPUs.
    from unsloth import FastLanguageModel  # type: ignore
    from trl import SFTTrainer  # type: ignore
    from unsloth.chat_templates import train_on_responses_only  # type: ignore


def sanitize_lora_kwargs(lora_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep config portable across backends.

    - Unsloth uses the sentinel string "unsloth" for checkpointing mode.
    - MLX-Tune expects a boolean (or ignores the field).
    """
    out = dict(lora_cfg)
    if IS_MLX and out.get("use_gradient_checkpointing") == "unsloth":
        out["use_gradient_checkpointing"] = True
    return out


def sanitize_training_args_kwargs(training_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adjust a few TrainingArguments knobs that are CUDA/Unsloth-specific.
    """
    out = dict(training_cfg)

    if IS_MLX:
        # TRL/Transformers 'optim' values like "adamw_8bit" can be CUDA-only.
        if out.get("optim") == "adamw_8bit":
            out["optim"] = "adamw_torch"

    return out


def cuda_precision_kwargs(torch_mod) -> Dict[str, bool]:
    """
    Build fp16/bf16 kwargs for Transformers TrainingArguments.
    On MLX we skip these entirely.
    """
    if IS_MLX:
        return {}

    bf16_ok = bool(getattr(torch_mod.cuda, "is_bf16_supported", lambda: False)())
    return {
        "fp16": not bf16_ok,
        "bf16": bf16_ok,
    }

