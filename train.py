"""
SLAM-ASR Training Script

Training Modes:
    1. Projector-only: Only train the projector (fastest)
       Config: freeze_encoder=true, freeze_llm=true, use_lora=false

    2. Projector + LoRA: Train projector and LoRA adapters on LLM
       Config: freeze_encoder=true, use_lora=true
       Note: freeze_llm is ignored when use_lora=true (PEFT handles freezing)
"""

import os
import time
import argparse
import gc
import math
import sys

# Fix CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from omegaconf import OmegaConf

from utils.utils import set_seed, get_device, resolve_pad_token, ensure_dir, save_checkpoint, save_lora_adapter
from utils.log_config import get_logger
from utils.wand_config import init_wandb
from models.model import model_builder
from datamodule.dataset import get_speech_dataset
from utils.metrics import compute_wer, compute_cer, count_encoder_parameters
from utils.train_utils import print_model_size, save_and_print_examples


class EarlyStopChecker:
    """Simple early stopping checker."""
    def __init__(self, mode="min", patience=5, min_delta=0.001):
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def check(self, value):
        """Returns True if training should stop."""
        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def evaluate(cfg, model, dataloader, device, enc_dtype, tokenizer):
    """Evaluate model on validation set."""
    model.eval()

    # Disable gradient checkpointing during eval
    grad_ckpt_was_enabled = False
    if hasattr(model.llm, 'is_gradient_checkpointing') and model.llm.is_gradient_checkpointing:
        grad_ckpt_was_enabled = True
        model.llm.gradient_checkpointing_disable()

    total_loss, n_batches = 0.0, 0
    all_accuracies = []
    all_wer_scores = []
    all_cer_scores = []
    all_hyp_texts, all_ref_texts = [], []

    use_autocast = bool(cfg.train.mixed_precision and torch.cuda.is_available())
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    max_eval_batches = cfg.train.get("max_eval_batches", None)
    max_new_tokens = cfg.train.get("max_new_tokens", 128)
    repetition_penalty = cfg.train.get("repetition_penalty", 1.2)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_eval_batches is not None and batch_idx >= max_eval_batches:
                break

            if batch_idx % 50 == 0:
                print(f"  Eval batch {batch_idx}/{len(dataloader)}...", end="\r")

            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            audio_mel = batch["audio_mel"].to(device).to(enc_dtype)
            modality_mask = batch['modality_mask'].to(device)

            # Get reference texts
            labels_cpu = labels.detach().cpu()
            ref_texts = []
            for label in labels_cpu:
                valid_tokens = label[label != -100]
                if len(valid_tokens) > 0:
                    ref_text = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()
                    ref_texts.append(ref_text)

            # Truncate inputs for generation (remove answer portion)
            gen_input_ids_list = []
            gen_attention_mask_list = []
            gen_modality_mask_list = []

            for i in range(labels.shape[0]):
                label_row = labels[i]
                answer_start_positions = (label_row != -100).nonzero(as_tuple=True)[0]

                if len(answer_start_positions) > 0:
                    answer_start = answer_start_positions[0].item()
                else:
                    answer_start = label_row.shape[0]

                gen_input_ids_list.append(input_ids[i, :answer_start])
                gen_attention_mask_list.append(attention_mask[i, :answer_start])
                gen_modality_mask_list.append(modality_mask[i, :answer_start])

            # Pad truncated sequences
            max_gen_len = max(len(seq) for seq in gen_input_ids_list)
            gen_input_ids = torch.zeros(len(gen_input_ids_list), max_gen_len, dtype=input_ids.dtype, device=device)
            gen_attention_mask = torch.zeros(len(gen_attention_mask_list), max_gen_len, dtype=attention_mask.dtype, device=device)
            gen_modality_mask = torch.zeros(len(gen_modality_mask_list), max_gen_len, dtype=modality_mask.dtype, device=device)

            for i, (ids, mask, mod_mask) in enumerate(zip(gen_input_ids_list, gen_attention_mask_list, gen_modality_mask_list)):
                seq_len = len(ids)
                gen_input_ids[i, max_gen_len - seq_len:] = ids
                gen_attention_mask[i, max_gen_len - seq_len:] = mask
                gen_modality_mask[i, max_gen_len - seq_len:] = mod_mask

            # Forward pass for loss
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    model_outputs, metrics = model.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        audio_mel=audio_mel,
                        modality_mask=modality_mask,
                        inference_mode=False
                    )
            else:
                model_outputs, metrics = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    audio_mel=audio_mel,
                    modality_mask=modality_mask,
                    inference_mode=False
                )

            if model_outputs.loss is not None:
                total_loss += model_outputs.loss.item()
            if "acc" in metrics:
                all_accuracies.append(metrics["acc"])

            del model_outputs

            # Generate for WER
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    generated_ids = model.generate(
                        input_ids=gen_input_ids,
                        attention_mask=gen_attention_mask,
                        audio_mel=audio_mel,
                        modality_mask=gen_modality_mask,
                        max_new_tokens=max_new_tokens,
                        num_beams=1,
                        do_sample=False,
                        repetition_penalty=repetition_penalty,
                    )
            else:
                generated_ids = model.generate(
                    input_ids=gen_input_ids,
                    attention_mask=gen_attention_mask,
                    audio_mel=audio_mel,
                    modality_mask=gen_modality_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    repetition_penalty=repetition_penalty,
                )

            # Decode
            hyp_texts = []
            for gen_ids in generated_ids:
                hyp_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                hyp_texts.append(hyp_text)

            n_batches += 1

            # Compute WER/CER
            if hyp_texts and ref_texts:
                min_len = min(len(hyp_texts), len(ref_texts))
                batch_hyp = hyp_texts[:min_len]
                batch_ref = ref_texts[:min_len]

                if batch_hyp and batch_ref:
                    batch_wer = compute_wer(batch_hyp, batch_ref)
                    batch_cer = compute_cer(batch_hyp, batch_ref)
                    all_wer_scores.append(batch_wer)
                    all_cer_scores.append(batch_cer)
                    all_hyp_texts.extend(batch_hyp)
                    all_ref_texts.extend(batch_ref)

            # Cleanup
            del input_ids, attention_mask, labels, audio_mel, modality_mask
            del gen_input_ids, gen_attention_mask, gen_modality_mask
            del generated_ids, labels_cpu
            gc.collect()
            torch.cuda.empty_cache()

    # Re-enable gradient checkpointing
    if grad_ckpt_was_enabled:
        model.llm.gradient_checkpointing_enable()

    # Calculate metrics
    val_loss = total_loss / max(n_batches, 1)
    val_acc = sum(all_accuracies) / max(len(all_accuracies), 1) if all_accuracies else 0.0
    val_wer_score = sum(all_wer_scores) / max(len(all_wer_scores), 1) if all_wer_scores else 1.0
    val_cer_score = sum(all_cer_scores) / max(len(all_cer_scores), 1) if all_cer_scores else 1.0
    val_word_acc = 1.0 - val_wer_score if val_wer_score <= 1.0 else 0.0

    model.train()

    return val_loss, val_acc, val_wer_score, val_cer_score, val_word_acc, all_hyp_texts, all_ref_texts


def print_trainable_parameters(model, logger):
    """Print detailed breakdown of trainable vs frozen parameters."""
    logger.info("PARAMETER BREAKDOWN BY MODULE")

    trainable_params = 0
    frozen_params = 0

    # Encoder
    enc_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    enc_frozen = sum(p.numel() for p in model.encoder.parameters() if not p.requires_grad)
    logger.info(f"Encoder:    Trainable={enc_trainable:>12,} | Frozen={enc_frozen:>12,}")
    trainable_params += enc_trainable
    frozen_params += enc_frozen

    # Projector
    proj_trainable = sum(p.numel() for p in model.projector.parameters() if p.requires_grad)
    proj_frozen = sum(p.numel() for p in model.projector.parameters() if not p.requires_grad)
    logger.info(f"Projector:  Trainable={proj_trainable:>12,} | Frozen={proj_frozen:>12,}")
    trainable_params += proj_trainable
    frozen_params += proj_frozen

    # LLM
    llm_trainable = sum(p.numel() for p in model.llm.parameters() if p.requires_grad)
    llm_frozen = sum(p.numel() for p in model.llm.parameters() if not p.requires_grad)
    logger.info(f"LLM:        Trainable={llm_trainable:>12,} | Frozen={llm_frozen:>12,}")
    trainable_params += llm_trainable
    frozen_params += llm_frozen

    logger.info(f"TOTAL:      Trainable={trainable_params:>12,} | Frozen={frozen_params:>12,}")
    if trainable_params + frozen_params > 0:
        logger.info(f"            ({100*trainable_params/(trainable_params+frozen_params):.2f}% trainable)")

    return trainable_params, frozen_params


def main():
    """Main training loop."""
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    # Setup directories and logging
    ensure_dir(cfg.train.output_dir)
    log_dir = cfg.log.log_dir if cfg.log.log_dir else "."
    ensure_dir(log_dir)
    logger = get_logger(log_dir=log_dir, filename=os.path.basename(cfg.log.log_filename))
    logger.info(f"Loaded config from {args.config}")

    # Seed and device
    set_seed(cfg.train.seed)
    device = get_device()
    logger.info(f"Device: {device}")

    use_lora = getattr(cfg.train, 'use_lora', False)
    if use_lora:
        logger.info("TRAINING MODE: Projector + LoRA")
        lora_cfg = getattr(cfg.train, 'lora', None)
        if lora_cfg:
            logger.info(f"  LoRA r={getattr(lora_cfg, 'r', 'default')}")
            logger.info(f"  LoRA alpha={getattr(lora_cfg, 'alpha', 'default')}")
    else:
        logger.info("TRAINING MODE: Projector Only")

    model, tokenizer = model_builder(cfg.train, cfg.model)
    pad_id = resolve_pad_token(tokenizer, model.llm)
    logger.info(f"Resolved pad_token_id: {pad_id}")

    model.to(device)

    # Enable gradient checkpointing
    if hasattr(model.llm, 'gradient_checkpointing_enable'):
        model.llm.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing for LLM")

    # Note: Freezing is already handled in model_builder() -> setup_llm()
    # We just collect the trainable parameters here

    projector_params = [p for p in model.projector.parameters() if p.requires_grad]
    logger.info(f"Projector trainable params: {sum(p.numel() for p in projector_params):,}")

    lora_params = []
    if use_lora:
        for name, p in model.llm.named_parameters():
            if p.requires_grad:
                lora_params.append(p)
        logger.info(f"LoRA trainable params: {sum(p.numel() for p in lora_params):,}")

    # Print full breakdown
    print_trainable_parameters(model, logger)

    all_trainable_params = projector_params + lora_params

    if not all_trainable_params:
        raise ValueError("No trainable parameters found! Check your config.")

    if use_lora and lora_params:
        # Separate learning rates for projector and LoRA
        lora_lr_multiplier = getattr(cfg.train, 'lora_lr_multiplier', 1.0)
        param_groups = []

        if projector_params:
            param_groups.append({
                "params": projector_params,
                "lr": cfg.train.lr,
                "name": "projector"
            })

        if lora_params:
            param_groups.append({
                "params": lora_params,
                "lr": cfg.train.lr * lora_lr_multiplier,
                "name": "lora"
            })

        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)
        logger.info(f"Optimizer: AdamW with projector_lr={cfg.train.lr}, lora_lr={cfg.train.lr * lora_lr_multiplier}")
    else:
        optimizer = torch.optim.AdamW(
            projector_params,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay
        )
        logger.info(f"Optimizer: AdamW with lr={cfg.train.lr}")

    scheduler = None
    if hasattr(cfg, 'scheduler') and cfg.scheduler is not None:
        warmup_steps = cfg.scheduler.get('warmup_steps', 0)
        total_steps = cfg.scheduler.get('total_training_steps', cfg.train.get('total_steps', 10000))
        min_lr_ratio = cfg.scheduler.get('min_lr_ratio', 0.1)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        logger.info(f"Scheduler: cosine_with_warmup | warmup={warmup_steps} | total={total_steps}")

    train_ds = get_speech_dataset(cfg.data, tokenizer, split=cfg.data.get("train_split", "train"))
    train_dataloader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=train_ds.collator,
        num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.train.num_workers > 0
    )
    logger.info(f"Train: {len(train_ds)} samples, {len(train_dataloader)} batches")

    val_ds = get_speech_dataset(cfg.data, tokenizer, split=cfg.data.get("val_split", "validation"))
    val_dataloader = DataLoader(
        val_ds,
        batch_size=cfg.train.get('val_batch_size', cfg.train.batch_size),
        shuffle=False,
        collate_fn=val_ds.collator,
        num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.train.num_workers > 0
    )
    logger.info(f"Val: {len(val_ds)} samples, {len(val_dataloader)} batches")

    run = None
    if cfg.log.use_wandb:
        run = init_wandb(
            use_wand=cfg.log.use_wandb,
            project=cfg.log.wandb_project_name,
            run_name=cfg.log.wandb_exp_name,
            tags=["slam-asr", "lora" if use_lora else "projector-only"],
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        logger.info(f"W&B run: {run.url}")

    use_autocast = bool(cfg.train.mixed_precision and torch.cuda.is_available())
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(enabled=(use_autocast and amp_dtype == torch.float16))
    enc_dtype = amp_dtype if use_autocast else torch.float32
    logger.info(f"Mixed precision: {use_autocast}, dtype={amp_dtype}")

    early_stop = None
    if hasattr(cfg, "early_stopping") and cfg.early_stopping is not None:
        patience = cfg.early_stopping.get("patience", 5)
        min_delta = cfg.early_stopping.get("min_delta", 0.001)
        early_stop = EarlyStopChecker(mode="min", patience=patience, min_delta=min_delta)
        logger.info(f"Early stopping: patience={patience}, min_delta={min_delta}")

    global_step = 0
    best_val_wer = float("inf")
    best_val_cer = float("inf")
    best_train_wer = float("inf")
    best_val_path = None
    training_start_time = time.time()

    logger.info("STARTING TRAINING")

    for epoch in range(cfg.train.num_epochs):
        model.train()
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            global_step += 1

            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            audio_mel = batch["audio_mel"].to(device).to(enc_dtype)
            modality_mask = batch['modality_mask'].to(device)

            # Forward
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs, metrics = model.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        audio_mel=audio_mel,
                        modality_mask=modality_mask
                    )
                    loss = outputs.loss
            else:
                outputs, metrics = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    audio_mel=audio_mel,
                    modality_mask=modality_mask
                )
                loss = outputs.loss

            batch_wer = metrics.get("wer", -1.0)
            acc = metrics.get("acc", 0.0)
            word_acc = 1.0 - batch_wer if batch_wer >= 0.0 else 0.0

            # Backward
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if cfg.train.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(all_trainable_params, cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.train.grad_clip is not None:
                    clip_grad_norm_(all_trainable_params, cfg.train.grad_clip)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            loss_value = loss.item()

            # Cleanup
            del input_ids, attention_mask, labels, audio_mel, modality_mask, outputs, loss
            gc.collect()
            torch.cuda.empty_cache()

            # Logging
            if global_step % cfg.log.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                if batch_wer < best_train_wer:
                    best_train_wer = batch_wer

                logger.info(f"Epoch={epoch} | Step={global_step} | WER={batch_wer:.4f} | Loss={loss_value:.4f} | Acc={acc:.4f} | LR={lr:.2e}")

                if run is not None:
                    run.log({
                        "train/wer": batch_wer,
                        "train/loss": loss_value,
                        "train/acc": acc,
                        "train/lr": lr,
                        "train/epoch": epoch,
                    }, step=global_step)

        val_loss, val_acc, val_wer, val_cer, val_word_acc, hyp_texts, ref_texts = evaluate(
            cfg, model, val_dataloader, device, enc_dtype, tokenizer
        )

        if val_cer < best_val_cer:
            best_val_cer = val_cer

        logger.info(f"Epoch {epoch} | Val WER: {val_wer:.4f} | Val CER: {val_cer:.4f} | Val Loss: {val_loss:.4f}")

        if run is not None:
            run.log({
                "val/wer": val_wer,
                "val/cer": val_cer,
                "val/loss": val_loss,
                "val/acc": val_acc,
            }, step=global_step)

        # Save examples
        save_and_print_examples(
            hyp_texts=hyp_texts,
            ref_texts=ref_texts,
            output_path=cfg.train.output_dir,
            epoch=epoch,
            n_save=10,
            n_print=5,
            run=run,
            seed=cfg.train.seed
        )

        # Model selection uses aggregate validation WER only; no demographic
        # label enters training or selection.
        if val_wer < best_val_wer:
            best_val_wer = val_wer
            best_val_path = os.path.join(cfg.train.output_dir, "checkpoint_best_wer.pt")
            save_checkpoint(model, best_val_path, global_step, save_lora=use_lora)
            logger.info(f"New best model! WER={val_wer:.4f} -> {best_val_path}")

            if use_lora:
                try:
                    save_lora_adapter(model, cfg.train.output_dir, adapter_name="lora_adapter_best")
                except Exception as e:
                    logger.warning(f"Could not save LoRA adapter: {e}")

        # Early stopping
        if early_stop is not None:
            monitor = cfg.early_stopping.get('monitor', 'val/wer')
            monitor_value = val_loss if monitor == 'val/loss' else val_wer
            if early_stop.check(monitor_value):
                logger.info(f"Early stopping at epoch {epoch}")
                break

    final_path = os.path.join(cfg.train.output_dir, "checkpoint_final.pt")
    save_checkpoint(model, final_path, global_step, save_lora=use_lora)
    logger.info(f"Final checkpoint: {final_path}")

    if use_lora:
        try:
            save_lora_adapter(model, cfg.train.output_dir, adapter_name="lora_adapter_final")
        except Exception as e:
            logger.warning(f"Could not save final LoRA adapter: {e}")

    total_time = (time.time() - training_start_time) / 60

    logger.info("TRAINING SUMMARY")
    logger.info(f"Mode: {'Projector + LoRA' if use_lora else 'Projector Only'}")
    logger.info(f"Best Training WER: {best_train_wer:.4f} ({best_train_wer*100:.2f}%)")
    logger.info(f"Best Validation WER: {best_val_wer:.4f} ({best_val_wer*100:.2f}%)")
    logger.info(f"Best Validation CER: {best_val_cer:.4f} ({best_val_cer*100:.2f}%)")
    logger.info(f"Total Time: {total_time:.1f} min ({total_time/60:.2f} hours)")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
