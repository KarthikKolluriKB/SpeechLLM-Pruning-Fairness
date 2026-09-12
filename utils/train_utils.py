import json
import logging
import os
import random

logger = logging.getLogger(__name__)


def print_model_size(model, config) -> None:
    """
    Logs model name and the number of parameters of the model.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        config (dict): Configuration dictionary containing model details.
    """
    logger.info(f"Model: {config.model_name}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{config.model_name} has {total_params/ 1e6} Million trainable parameters")


def print_module_size(module, module_name: str) -> None:
    """
    Logs the module name, the number of parameters of a specific module.

    Args:
        module (torch.nn.Module): The module to be evaluated.
        module_name (str): Name of the module.
    """
    logger.info(f"Module: {module_name}")
    total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    logger.info(f"{module_name} has {total_params/ 1e6} Million trainable parameters")


def save_and_print_examples(
        hyp_texts: list[str],
        ref_texts: list[str],
        output_path: str,
        epoch: int,
        n_save: int = 10,
        n_print: int = 5,
        run=None,
        seed: int = 42
) -> None:
    """
    Save random hypothesis/reference pairs to JSONL and print a few examples.

    Args:
        hyp_texts: List of hypothesis (predicted) texts
        ref_texts: List of reference (ground truth) texts
        output_path: Directory to save the JSONL file
        epoch: Current epoch number
        n_save: Number of examples to save to file
        n_print: Number of examples to print to console
        run: Optional wandb run object for logging
        seed: Random seed for reproducible sampling
    """
    if len(hyp_texts) == 0 or len(ref_texts) == 0:
        print(f"[Epoch {epoch}] No examples to save/print.")
        return

    assert len(hyp_texts) == len(ref_texts), "Hypothesis and reference lists must have same length"

    random.seed(seed + epoch)  # different sample each epoch, still reproducible

    n_available = len(hyp_texts)
    n_to_sample = min(n_save, n_available)
    sample_indices = random.sample(range(n_available), n_to_sample)

    examples = []
    for idx in sample_indices:
        example = {
            "epoch": epoch,
            "sample_index": idx,
            "hyp": hyp_texts[idx],
            "ref": ref_texts[idx]
        }
        examples.append(example)

    os.makedirs(output_path, exist_ok=True)
    jsonl_path = os.path.join(output_path, f"epoch_{epoch:03d}_examples.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    logger.info(f"[Epoch {epoch}] Saved {n_to_sample} examples to {jsonl_path}")

    n_to_print = min(n_print, n_to_sample)
    print_indices = random.sample(range(n_to_sample), n_to_print)

    print(f"\n{'='*80}")
    print(f"Epoch {epoch} - ASR Examples (showing {n_to_print} of {n_to_sample} saved):")
    print(f"{'='*80}")

    for i, idx in enumerate(print_indices, 1):
        example = examples[idx]
        print(f"\n[{i}] Sample #{example['sample_index']}:")
        print(f"  REF: {example['ref']}")
        print(f"  HYP: {example['hyp']}")

    if run is not None:
        try:
            import wandb
            table = wandb.Table(columns=["epoch", "sample_index", "hyp", "ref"])
            for example in examples:
                table.add_data(
                    example["epoch"],
                    example["sample_index"],
                    example["hyp"],
                    example["ref"]
                )
            run.log({f"examples/epoch_{epoch:03d}": table}, commit=False)  # avoid creating an extra step
            logger.info(f"[Epoch {epoch}] Logged examples to wandb")
        except ImportError:
            logger.warning("wandb is not installed. Skipping wandb logging.")
        except Exception as e:
            logger.warning(f"Failed to log examples to wandb: {e}")
    print(f"{'='*80}\n")
