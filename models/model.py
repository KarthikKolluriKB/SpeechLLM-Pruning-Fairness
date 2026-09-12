"""
SLAM-ASR Model: Speech-Language Model for Automatic Speech Recognition

Architecture:
    - Encoder: Whisper (frozen or trainable)
    - Projector: Maps encoder outputs to LLM embedding space (trainable)
    - LLM: Qwen2.5-3B (frozen, or LoRA fine-tuned)

Training Modes:
    1. Projector-only: freeze_encoder=true, freeze_llm=true, use_lora=false
    2. Projector + LoRA: freeze_encoder=true, use_lora=true
"""

import torch
import torch.nn as nn
import logging
from torch.nn import CrossEntropyLoss
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from models.encoder import WhisperWrappedEncoder
from utils.metrics import compute_accuracy, compute_wer, decode_texts_from_outputs
from utils.train_utils import print_model_size, print_module_size

logger = logging.getLogger(__name__)


def model_builder(train_config, model_config, **kwargs):
    """
    Build the complete SLAM-ASR model.

    Args:
        train_config: Training configuration (from cfg.train)
        model_config: Model configuration (from cfg.model)
        **kwargs: Additional arguments (e.g., ckpt_path for loading weights)

    Returns:
        model: ASRLLM model
        tokenizer: Tokenizer for the LLM
    """
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)
    encoder = setup_encoder(train_config, model_config, **kwargs)
    llm = setup_llm(train_config, model_config, **kwargs)
    projector = setup_projector(train_config, model_config, **kwargs)

    model = ASRLLM(
        encoder,
        llm,
        projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs
    )

    ckpt_path = kwargs.get("ckpt_path", None)
    if ckpt_path is not None:
        logger.info(f"Loading checkpoint from {ckpt_path}")
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")

        if 'projector' in ckpt_dict:
            model.projector.load_state_dict(ckpt_dict['projector'], strict=True)
            logger.info("Loaded projector weights from checkpoint")

        use_lora = getattr(train_config, 'use_lora', False)
        if use_lora and 'lora' in ckpt_dict:
            try:
                model.llm.load_state_dict(ckpt_dict['lora'], strict=False)
                logger.info("Loaded LoRA adapter weights from checkpoint")
            except Exception as e:
                logger.warning(f"Could not load LoRA weights: {e}")

    print_model_size(model, train_config)

    return model, tokenizer


def setup_tokenizer(train_config, model_config, **kwargs):
    """Setup tokenizer for the LLM."""
    tokenizer = AutoTokenizer.from_pretrained(model_config.llm_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_encoder(train_config, model_config, **kwargs):
    """
    Setup Whisper encoder.

    Freezing is handled here based on train_config.freeze_encoder.
    """
    encoder_name = model_config.encoder_model_name
    encoder = WhisperWrappedEncoder.load(model_config)
    print_module_size(encoder, encoder_name)

    if train_config.freeze_encoder:
        for name, params in encoder.named_parameters():
            params.requires_grad = False
        encoder.eval()
        logger.info("Encoder: FROZEN")
    else:
        logger.info("Encoder: TRAINABLE")

    print_module_size(encoder, encoder_name)
    return encoder


def setup_llm(train_config, model_config, **kwargs):
    """
    Setup LLM with optional LoRA.

    Training modes:
        1. use_lora=True: Apply LoRA adapters (PEFT auto-freezes base params)
        2. use_lora=False, freeze_llm=True: Freeze all LLM params
        3. use_lora=False, freeze_llm=False: Full LLM fine-tuning (expensive!)
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_config.llm_model,
        torch_dtype=torch.bfloat16 if train_config.mixed_precision else torch.float32,
        attn_implementation="sdpa",
        load_in_8bit=True if train_config.quantization else None,
        device_map="auto" if train_config.quantization else None,
    )
    print_module_size(model, model_config.llm_model_name)

    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    use_lora = getattr(train_config, 'use_lora', False)

    if use_lora:
        # PEFT freezes the base parameters when the adapters are attached
        logger.info("Applying LoRA to the LLM")

        lora_config = getattr(train_config, 'lora', None)

        if lora_config is None:
            logger.warning("No LoRA config found, using defaults")
            lora_r = 16
            lora_alpha = 32
            lora_dropout = 0.05
            lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        else:
            lora_r = getattr(lora_config, 'r', 16)
            lora_alpha = getattr(lora_config, 'alpha', 32)
            lora_dropout = getattr(lora_config, 'dropout', 0.05)
            lora_target_modules = list(getattr(lora_config, 'target_modules',
                                        ["q_proj", "v_proj", "k_proj", "o_proj"]))

        logger.info(f"LoRA Config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        logger.info(f"LoRA Target Modules: {lora_target_modules}")

        # Create and apply LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, peft_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"LoRA Trainable Parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        model.print_trainable_parameters()

    elif train_config.freeze_llm:
        logger.info("LLM: FROZEN (no LoRA)")
        for name, param in model.named_parameters():
            param.requires_grad = False
        model.eval()

    else:
        logger.info("LLM: FULL FINE-TUNING (all parameters trainable)")

    return model


def setup_projector(train_config, model_config, **kwargs):
    """Setup projector to map encoder outputs to LLM embedding space."""
    projector_type = model_config.projector.lower() if model_config.projector else ""

    if projector_type in ["linear", "concatlinear"]:
        from models.projector import EncoderProjectorConcat
        projector = EncoderProjectorConcat(model_config)
    elif projector_type in ["cov1d-linear", "conv1d-linear", "cov1d", "conv1d"]:
        from models.projector import EncoderProjectorCov1d
        projector = EncoderProjectorCov1d(model_config)
    elif projector_type in ["q-former", "qformer"]:
        from models.projector import EncoderProjectorQFormer
        projector = EncoderProjectorQFormer(model_config)
    else:
        raise ValueError(f"Unknown projector type: '{model_config.projector}'. "
                        f"Supported: linear, concatLinear, cov1d-linear, q-former")

    print_module_size(projector, model_config.projector)
    return projector


class ASRLLM(nn.Module):
    """
    Speech-Language Model for ASR.

    Combines:
        - Whisper encoder for audio feature extraction
        - Projector for modality alignment
        - LLM for text generation
    """

    def __init__(
        self,
        encoder: nn.Module,
        llm: nn.Module,
        projector: Optional[nn.Module],
        tokenizer,
        train_config,
        model_config,
        **kwargs
    ):
        super().__init__()

        self.encoder = encoder
        self.llm = llm
        self.projector = projector
        self.tokenizer = tokenizer
        self.train_config = train_config
        self.model_config = model_config
        self.dataset_config = kwargs.get("data_config", None)

        self.metric = kwargs.get("metric", True)
        self.label_smoothing = getattr(train_config, 'label_smoothing', 0.0)
        if self.label_smoothing > 0:
            logger.info(f"Label smoothing enabled: {self.label_smoothing}")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        audio_mel = kwargs.get("audio_mel", None)
        audio_mel_mask = kwargs.get("audio_mel_mask", None)
        audio_mel_post_mask = kwargs.get("audio_mel_post_mask", None)
        modality_mask = kwargs.get("modality_mask", None)

        if getattr(self.train_config, "freeze_encoder", False):
            self.encoder.eval()
            with torch.no_grad():
                encoder_outputs = self.encoder.extract_variable_length_features(
                    audio_mel.permute(0, 2, 1)
                )
        else:
            encoder_outputs = self.encoder.extract_variable_length_features(
                audio_mel.permute(0, 2, 1)
            )

        if audio_mel_post_mask is None:
            audio_mel_post_mask = torch.ones(
                encoder_outputs.size()[:-1],
                dtype=torch.long,
                device=encoder_outputs.device
            )

        encoder_outputs = self._apply_projector(encoder_outputs, audio_mel_post_mask)

        if input_ids is not None:
            input_ids[input_ids == -1] = 0  # placeholder ids marking the audio span

            # the embedding layer sits at a different depth per architecture
            if hasattr(self.llm, 'model') and hasattr(self.llm.model, "embed_tokens"):
                inputs_embeds = self.llm.model.embed_tokens(input_ids)
            elif hasattr(self.llm, "model") and hasattr(self.llm.model, "model") and hasattr(self.llm.model.model, "embed_tokens"):
                inputs_embeds = self.llm.model.model.embed_tokens(input_ids)
            else:
                inputs_embeds = self.llm.model.model.model.embed_tokens(input_ids)

        # splice the projected audio embeddings into the text embedding sequence
        if modality_mask is not None:
            modality_mask_start_indices = (modality_mask == True).float().argmax(dim=1)
            modality_lengths = torch.clamp(
                modality_mask.sum(dim=1),
                max=encoder_outputs.shape[1]
            ).tolist()

            encoder_outs_pad = torch.zeros_like(inputs_embeds)
            for i in range(encoder_outputs.shape[0]):
                encoder_outs_pad[
                    i,
                    modality_mask_start_indices[i]:modality_mask_start_indices[i]+modality_lengths[i]
                ] = encoder_outputs[i][:modality_lengths[i]]

            inputs_embeds = encoder_outs_pad + inputs_embeds * (~modality_mask[:, :, None])

        if kwargs.get("inference_mode", False):
            return inputs_embeds, attention_mask

        if labels is not None and self.label_smoothing > 0:
            # Explicit loss branch: label smoothing is applied here because the
            # LLM's built-in `labels=...` loss does not expose it.
            model_outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=None,
                use_cache=False
            )

            logits = model_outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=self.label_smoothing
            )
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            model_outputs.loss = loss
        else:
            model_outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False
            )

        metrics = {}
        if self.metric and labels is not None:
            with torch.no_grad():
                logits_cpu = model_outputs.logits.detach().cpu()
                labels_cpu = labels.detach().cpu()

                preds = torch.argmax(logits_cpu, dim=-1)
                acc = compute_accuracy(preds[:, :-1], labels_cpu[:, 1:], ignore_label=-100)
                metrics["acc"] = float(acc.item())

                hyp_texts, ref_texts = decode_texts_from_outputs(
                    logits=logits_cpu,
                    labels=labels_cpu,
                    tokenizer=self.tokenizer,
                    ignore_label=-100
                )
                wer_score = compute_wer(hyp_texts=hyp_texts, ref_texts=ref_texts)
                metrics["wer"] = float(wer_score)

                del preds, labels_cpu, logits_cpu

        return model_outputs, metrics

    def _apply_projector(self, encoder_outputs, audio_mel_post_mask):
        """Apply projector to encoder outputs."""
        projector_type = self.model_config.projector.lower() if self.model_config.projector else ""

        if projector_type in ["q-former", "qformer"]:
            encoder_outputs = self.projector(encoder_outputs, audio_mel_post_mask)
        else:
            encoder_outputs = self.projector(encoder_outputs)

        return encoder_outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Generate text from audio input."""
        kwargs["inference_mode"] = True

        if inputs_embeds is None:
            inputs_embeds, attention_mask = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=kwargs.get("max_new_tokens", 200),
            num_beams=kwargs.get("num_beams", 4),
            do_sample=kwargs.get("do_sample", False),
            min_length=kwargs.get("min_length", 1),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            attention_mask=attention_mask,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        return model_outputs
