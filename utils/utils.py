import os
import random

import torch

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def set_seed(seed: int, deterministic: bool = False):
    """
    Set the random seed for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enables full determinism, which costs roughly
            10-20% throughput. The sweep runs with the default, so results are
            reproducible up to cuDNN autotuning.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if HAS_NUMPY:
        np.random.seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass  # not every operation has a deterministic implementation
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device() -> str:
    """Get the available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_pad_token(tokenizer, model) -> int:
    """
    Ensure the tokenizer.pad_token_id is set and syncs it to model configs.
    If tokenizer.pad_token_id is None, set it to tokenizer.eos_token_id (int).

    Args:
        tokenizer: The tokenizer to check and set the pad_token_id.
        model: The model whose config needs to be updated with the pad_token_id.

    Returns:
        pad_id (int): The resolved pad token ID.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
            pad_id = tokenizer.pad_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            pad_id = tokenizer.pad_token_id
            if hasattr(model, 'resize_token_embeddings'):
                model.resize_token_embeddings(len(tokenizer))

        if hasattr(model, 'config'):
            model.config.pad_token_id = pad_id
        if hasattr(model, 'generation_config'):
            model.generation_config.pad_token_id = pad_id
    return pad_id


def ensure_dir(dir_path: str):
    """Ensure that a directory exists; if not, create it."""
    os.makedirs(dir_path, exist_ok=True)


def save_checkpoint(model, path: str, step: int, save_lora: bool = False):
    """
    Save checkpoint with projector weights and optionally LoRA adapter weights.

    Args:
        model: The ASRLLM model
        path: Path to save checkpoint
        step: Current training step
        save_lora: Whether to save LoRA adapter weights
    """
    checkpoint = {"step": step}

    proj = None
    if hasattr(model, "projector") and model.projector is not None:
        proj = model.projector
    elif hasattr(model, "encoder_projector") and model.encoder_projector is not None:
        proj = model.encoder_projector

    if proj is not None:
        checkpoint["projector"] = proj.state_dict()

    if save_lora and hasattr(model, "llm"):
        try:
            from peft import PeftModel
            if isinstance(model.llm, PeftModel):
                lora_state_dict = {}
                for name, param in model.llm.named_parameters():
                    if "lora_" in name or "modules_to_save" in name:
                        lora_state_dict[name] = param.cpu().clone()
                if lora_state_dict:
                    checkpoint["lora"] = lora_state_dict
                    print(f"LoRA adapter weights included in checkpoint ({len(lora_state_dict)} tensors)")
        except ImportError:
            pass
        except Exception as e:
            print(f"Warning: Could not save LoRA weights: {e}")

    torch.save(checkpoint, path)
    print(f"Checkpoint saved at step {step} to {path}")


def save_lora_adapter(model, output_dir: str, adapter_name: str = "lora_adapter"):
    """
    Save LoRA adapter weights separately with PEFT, so the adapter can later be
    loaded with PeftModel.from_pretrained().

    Args:
        model: The ASRLLM model with LoRA-enabled LLM
        output_dir: Directory to save the adapter
        adapter_name: Name for the adapter subdirectory
    """
    from peft import PeftModel

    if not hasattr(model, "llm"):
        raise ValueError("Model does not have 'llm' attribute")

    if not isinstance(model.llm, PeftModel):
        raise ValueError("LLM is not a PeftModel (LoRA not applied)")

    adapter_path = os.path.join(output_dir, adapter_name)
    os.makedirs(adapter_path, exist_ok=True)

    model.llm.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to {adapter_path}")
