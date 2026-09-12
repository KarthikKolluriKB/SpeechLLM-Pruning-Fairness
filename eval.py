"""
Evaluation script for SLAM-ASR model.
Tests trained model on test set using actual generation (not teacher forcing).

Features:
    - WER and CER calculation
    - Beam search support (configurable num_beams)
    - LoRA adapter loading support
    - Wandb logging

Usage:
    python eval.py --config configs/whisper_largev2/english/eval/baseline.yaml
    python eval.py --config configs/whisper_largev2/english/eval/ablation_30L.yaml --num_beams 2
    python eval.py --config configs/fairspeech/whisper_largev2/eval/baseline.yaml

Reports corpus-level WER and CER and writes every hypothesis/reference pair to
JSONL. Per-group WER for the demographic analysis is computed separately by
scoring/evaluate_subgroup_wer.py.
"""

import argparse
import json
import os
import gc
import torch

from tqdm import tqdm
from omegaconf import OmegaConf
from types import SimpleNamespace
from torch.utils.data import DataLoader

from models.model import model_builder
from datamodule.dataset import get_speech_dataset
from utils.metrics import compute_wer, compute_cer, count_encoder_parameters
from utils.wand_config import init_wandb
from utils.log_config import get_logger
from utils.utils import ensure_dir

logger = get_logger(log_dir="logs", filename="eval.log")


def truncate_for_generation(input_ids, attention_mask, labels, modality_mask, device):
    """
    Truncate inputs to only include audio + prompt (remove answer).
    This is required for proper generation - we don't want the answer in the input.

    Returns:
        Truncated tensors ready for generation
    """
    gen_input_ids_list = []
    gen_attention_mask_list = []
    gen_modality_mask_list = []

    for i in range(labels.shape[0]):
        # Find where the answer starts (first non-ignored label)
        label_row = labels[i]
        answer_start_positions = (label_row != -100).nonzero(as_tuple=True)[0]

        if len(answer_start_positions) > 0:
            answer_start = answer_start_positions[0].item()
        else:
            answer_start = label_row.shape[0]

        # Truncate to only audio + prompt (exclude answer)
        gen_input_ids_list.append(input_ids[i, :answer_start])
        gen_attention_mask_list.append(attention_mask[i, :answer_start])
        gen_modality_mask_list.append(modality_mask[i, :answer_start])

    # Pad truncated sequences to same length (left-pad)
    max_gen_len = max(len(seq) for seq in gen_input_ids_list)
    gen_input_ids = torch.zeros(len(gen_input_ids_list), max_gen_len, dtype=input_ids.dtype, device=device)
    gen_attention_mask = torch.zeros(len(gen_attention_mask_list), max_gen_len, dtype=attention_mask.dtype, device=device)
    gen_modality_mask = torch.zeros(len(gen_modality_mask_list), max_gen_len, dtype=modality_mask.dtype, device=device)

    for i, (ids, mask, mod_mask) in enumerate(zip(gen_input_ids_list, gen_attention_mask_list, gen_modality_mask_list)):
        seq_len = len(ids)
        # Left-pad
        gen_input_ids[i, max_gen_len - seq_len:] = ids
        gen_attention_mask[i, max_gen_len - seq_len:] = mask
        gen_modality_mask[i, max_gen_len - seq_len:] = mod_mask

    return gen_input_ids, gen_attention_mask, gen_modality_mask


def save_examples_jsonl(hyp_texts, ref_texts, output_path, filename="test_examples.jsonl"):
    """Save all examples to a JSONL file."""
    ensure_dir(output_path)
    filepath = os.path.join(output_path, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        for i, (hyp, ref) in enumerate(zip(hyp_texts, ref_texts)):
            example = {
                "id": i,
                "reference": ref,
                "hypothesis": hyp,
            }
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(hyp_texts)} examples to {filepath}")
    return filepath


def log_examples_to_wandb(hyp_texts, ref_texts, run, num_examples=50):
    """Log example predictions to wandb as a table."""
    import wandb

    # Create a table with examples
    table = wandb.Table(columns=["ID", "Reference", "Hypothesis", "Match"])

    for i in range(min(num_examples, len(hyp_texts))):
        match = "✓" if hyp_texts[i].strip() == ref_texts[i].strip() else ""
        table.add_data(i, ref_texts[i], hyp_texts[i], match)

    run.log({"test/examples": table})
    logger.info(f"Logged {min(num_examples, len(hyp_texts))} examples to wandb")


@torch.no_grad()
def run_eval(args):
    """
    Run evaluation on the test set using actual generation.
    """
    run = None
    model = None
    device = None

    try:
        # 1. Load config and check files exist
        if not os.path.exists(args.cfg_path):
            raise FileNotFoundError(f"Config file not found: {args.cfg_path}")
        if not os.path.exists(args.ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt_path}")

        cfg = OmegaConf.load(args.cfg_path)
        train_cfg = cfg.train
        model_cfg = cfg.model
        data_cfg = cfg.data
        wandb_cfg = cfg.log if hasattr(cfg, 'log') else None
        eval_cfg = cfg.eval if hasattr(cfg, 'eval') else None

        # precedence: CLI args, then the eval block of the config, then defaults
        max_new_tokens = args.max_new_tokens or (
            eval_cfg.max_new_tokens if eval_cfg and hasattr(eval_cfg, 'max_new_tokens') else 128
        )
        repetition_penalty = args.repetition_penalty or (
            eval_cfg.repetition_penalty if eval_cfg and hasattr(eval_cfg, 'repetition_penalty') else 1.0
        )
        num_beams = args.num_beams or (
            eval_cfg.num_beams if eval_cfg and hasattr(eval_cfg, 'num_beams') else 1
        )
        length_penalty = eval_cfg.length_penalty if eval_cfg and hasattr(eval_cfg, 'length_penalty') else 1.0
        do_sample = eval_cfg.do_sample if eval_cfg and hasattr(eval_cfg, 'do_sample') else False
        temperature = eval_cfg.temperature if eval_cfg and hasattr(eval_cfg, 'temperature') else 1.0
        batch_size = args.batch_size or (
            eval_cfg.batch_size if eval_cfg and hasattr(eval_cfg, 'batch_size') else 8
        )

        logger.info(f"Generation settings: num_beams={num_beams}, max_new_tokens={max_new_tokens}, "
                   f"repetition_penalty={repetition_penalty}, length_penalty={length_penalty}, "
                   f"batch_size={batch_size}")

        # 2. Initialize wandb if enabled
        if wandb_cfg and wandb_cfg.use_wandb:
            run = init_wandb(
                use_wand=True,
                project=wandb_cfg.wandb_project_name,
                run_name=args.wandb_exp_name or f"eval_{wandb_cfg.wandb_exp_name}",
                tags=["eval", "asr-llm", "test"],
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            logger.info(f"Initialized wandb run: {run.url}")

        # 3. Build Model and load weights
        logger.info(f"Loading model with checkpoint: {args.ckpt_path}")
        model, tokenizer = model_builder(train_cfg, model_cfg, data_config=data_cfg)

        # Load checkpoint (handles both projector-only and projector+LoRA formats)
        checkpoint = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)

        # Load projector weights
        if 'projector' in checkpoint:
            # Nested format: {"step": ..., "projector": state_dict, "lora": ...}
            projector_state = checkpoint['projector']
            logger.info(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        else:
            # Direct state_dict format (legacy)
            projector_state = checkpoint
        model.projector.load_state_dict(projector_state)
        logger.info("Loaded projector weights successfully")

        # Load LoRA weights if present and LoRA is enabled
        use_lora = getattr(train_cfg, 'use_lora', False)
        if use_lora and 'lora' in checkpoint:
            try:
                model.llm.load_state_dict(checkpoint['lora'], strict=False)
                logger.info("Loaded LoRA adapter weights successfully")
            except Exception as e:
                logger.warning(f"Could not load LoRA weights from checkpoint: {e}")
        elif use_lora:
            logger.warning("LoRA is enabled but no LoRA weights found in checkpoint")

        # 4. Move model to device
        device = torch.device(args.device)
        model = model.to(device)
        model.eval()
        logger.info(f"Model moved to device: {device}")

        # Log encoder efficiency metrics
        num_layers = getattr(model_cfg, 'encoder_num_layers', None)
        encoder_params = count_encoder_parameters(model.encoder, num_layers=num_layers)

        efficiency_summary = f"""
{'='*60}
ENCODER EFFICIENCY METRICS
{'='*60}
Model:              Whisper-{model_cfg.encoder_model}
Layers Used:        {encoder_params['num_layers_used']} / {encoder_params['num_layers_total']}
Used Params:        {encoder_params['used_params']:,} ({encoder_params['used_params']/1e6:.2f}M)
LoRA Enabled:       {use_lora}
{'='*60}
"""
        logger.info(efficiency_summary)

        if run is not None:
            run.summary["efficiency/encoder_layers_used"] = encoder_params['num_layers_used']
            run.summary["efficiency/encoder_params_used_M"] = round(encoder_params['used_params'] / 1e6, 2)
            run.summary["efficiency/lora_enabled"] = use_lora

        # 5. Load Test Dataset
        test_dataset = get_speech_dataset(data_cfg, tokenizer, split=args.split)
        logger.info(f"Loaded {len(test_dataset)} samples from {args.split} split")

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=test_dataset.collator,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False,
        )

        # 6. Determine precision
        use_autocast = train_cfg.mixed_precision and device.type == 'cuda'
        amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        enc_dtype = amp_dtype if use_autocast else torch.float32

        # 7. Evaluation Loop with Generation
        all_hyp_texts = []
        all_ref_texts = []
        all_wer_scores = []
        all_cer_scores = []  # CER tracking

        logger.info(f"Starting evaluation with generation (num_beams={num_beams})...")

        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audio_mel = batch["audio_mel"].to(device).to(enc_dtype)
            modality_mask = batch["modality_mask"].to(device)

            # Get reference texts - handle both training and inference mode
            if "targets" in batch:
                # Inference mode: targets is the reference text list directly
                ref_texts = batch["targets"]
                if isinstance(ref_texts, torch.Tensor):
                    ref_texts = [str(t) for t in ref_texts]
            elif "labels" in batch:
                # Training mode: decode from labels
                labels = batch["labels"].to(device)
                ref_texts = []
                for label in labels:
                    valid_tokens = label[label != -100]
                    if len(valid_tokens) > 0:
                        ref_text = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()
                        ref_texts.append(ref_text)
            else:
                logger.error("Batch has neither 'targets' nor 'labels' - cannot get reference texts")
                continue

            # Generate transcriptions
            if use_autocast:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        audio_mel=audio_mel,
                        modality_mask=modality_mask,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        do_sample=do_sample,
                        repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty,
                        temperature=temperature,
                    )
            else:
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audio_mel=audio_mel,
                    modality_mask=modality_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    temperature=temperature,
                )

            # Decode generated texts
            hyp_texts = []
            for gen_ids in generated_ids:
                hyp_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                hyp_texts.append(hyp_text)

            # Compute WER and CER for this batch
            if hyp_texts and ref_texts:
                min_len = min(len(hyp_texts), len(ref_texts))
                batch_hyp = hyp_texts[:min_len]
                batch_ref = ref_texts[:min_len]

                if batch_hyp and batch_ref:
                    batch_wer = compute_wer(batch_hyp, batch_ref)
                    batch_cer = compute_cer(batch_hyp, batch_ref)  # CER calculation
                    all_wer_scores.append(batch_wer)
                    all_cer_scores.append(batch_cer)
                    all_hyp_texts.extend(batch_hyp)
                    all_ref_texts.extend(batch_ref)

            # Print some examples from first batch
            if batch_idx == 0:
                logger.info("\n" + "="*60)
                logger.info("SAMPLE PREDICTIONS (First Batch):")
                logger.info("="*60)
                for i in range(min(5, len(hyp_texts))):
                    logger.info(f"\n[{i+1}]")
                    logger.info(f"  REF: {ref_texts[i]}")
                    logger.info(f"  HYP: {hyp_texts[i]}")

            # Memory cleanup
            del input_ids, attention_mask, audio_mel, modality_mask, generated_ids
            if batch_idx % 20 == 0 and device.type == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()

        # 8. Calculate final metrics
        if all_wer_scores:
            avg_wer = sum(all_wer_scores) / len(all_wer_scores)
            avg_cer = sum(all_cer_scores) / len(all_cer_scores)
            word_acc = 1.0 - avg_wer if avg_wer <= 1.0 else 0.0
            char_acc = 1.0 - avg_cer if avg_cer <= 1.0 else 0.0

            # Final results
            logger.info("\n" + "="*60)
            logger.info("FINAL EVALUATION RESULTS:")
            logger.info("="*60)
            logger.info(f"Total samples:    {len(all_hyp_texts)}")
            logger.info(f"Beam size:        {num_beams}")
            logger.info(f"WER:              {avg_wer:.4f} ({avg_wer*100:.2f}%)")
            logger.info(f"CER:              {avg_cer:.4f} ({avg_cer*100:.2f}%)")
            logger.info(f"Word Accuracy:    {word_acc:.4f} ({word_acc*100:.2f}%)")
            logger.info(f"Char Accuracy:    {char_acc:.4f} ({char_acc*100:.2f}%)")
            logger.info("="*60)

            # Log to wandb
            if run is not None:
                run.log({
                    "test/wer": avg_wer,
                    "test/cer": avg_cer,
                    "test/word_accuracy": word_acc,
                    "test/char_accuracy": char_acc,
                    "test/num_samples": len(all_hyp_texts),
                    "test/num_beams": num_beams,
                })
                run.summary["test/final_wer"] = avg_wer
                run.summary["test/final_cer"] = avg_cer
                run.summary["test/final_word_accuracy"] = word_acc
                run.summary["test/final_char_accuracy"] = char_acc
                run.summary["test/num_beams"] = num_beams

                # Log examples table
                log_examples_to_wandb(all_hyp_texts, all_ref_texts, run, num_examples=50)

            # Save predictions to JSONL
            output_dir = args.output_dir or (eval_cfg.output_dir if eval_cfg else "eval_results")
            ensure_dir(output_dir)
            save_examples_jsonl(all_hyp_texts, all_ref_texts, output_dir)

            # Also save a summary file
            summary_path = os.path.join(output_dir, "eval_summary.json")
            summary = {
                "checkpoint": args.ckpt_path,
                "config": args.cfg_path,
                "split": args.split,
                "num_samples": len(all_hyp_texts),
                "num_beams": num_beams,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty,
                "use_lora": use_lora,
                "wer": avg_wer,
                "cer": avg_cer,
                "word_accuracy": word_acc,
                "char_accuracy": char_acc,
            }
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved summary to {summary_path}")

        else:
            logger.error("No samples were processed!")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise

    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA out of memory: {e}")
        logger.error("Try reducing batch size or num_beams")
        if device and device.type == 'cuda':
            torch.cuda.empty_cache()
        raise

    except Exception as e:
        logger.exception(f"Error during evaluation: {e}")
        raise

    finally:
        # Cleanup
        if device and device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        if run is not None:
            try:
                run.finish()
            except Exception as e:
                logger.warning(f"Failed to finish wandb run: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ASR-LLM model on test set")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the config file (alternative to --cfg_path).",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default=None,
        help="Path to the config file.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the model checkpoint (projector weights). If not provided, uses eval.projector_path from config.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation. Overrides config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Data split to evaluate on.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the evaluation on.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results. Overrides config.",
    )
    parser.add_argument(
        "--wandb_exp_name",
        type=str,
        default=None,
        help="Wandb experiment name.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of new tokens to generate. Overrides config.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="Repetition penalty for generation. Overrides config.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams for beam search. Overrides config.",
    )
    args = parser.parse_args()

    # Handle --config as alias for --cfg_path
    if args.config and not args.cfg_path:
        args.cfg_path = args.config

    # Validate cfg_path is provided
    if not args.cfg_path:
        parser.error("--config or --cfg_path is required")

    # If ckpt_path not provided, load from config
    if not args.ckpt_path:
        cfg = OmegaConf.load(args.cfg_path)
        if hasattr(cfg, 'eval') and hasattr(cfg.eval, 'projector_path'):
            args.ckpt_path = cfg.eval.projector_path
            print(f"Using projector_path from config: {args.ckpt_path}")
        else:
            parser.error("--ckpt_path is required (or set eval.projector_path in config)")

    # Load output_dir from config if not specified
    if args.output_dir is None:
        cfg = OmegaConf.load(args.cfg_path)
        if hasattr(cfg, 'eval') and hasattr(cfg.eval, 'output_dir'):
            args.output_dir = cfg.eval.output_dir
        else:
            args.output_dir = "eval_results"

    return args


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
