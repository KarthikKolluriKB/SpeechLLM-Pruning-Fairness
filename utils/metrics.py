"""
Text normalization, WER/CER, and encoder parameter counting.

WER and CER are computed with jiwer over a Whisper-style basic normalization
(lowercase, bracketed content removed, punctuation and symbols stripped,
whitespace collapsed).
"""

from typing import List, Tuple, Dict
import re
import unicodedata

import torch
import jiwer
from transformers import AutoTokenizer


class BasicTextNormalizer:
    """
    Text normalizer for ASR evaluation, after OpenAI Whisper's
    BasicTextNormalizer (whisper/normalizers/basic.py).

    NFKC normalization keeps letters outside ASCII (e.g. æ, ø, å) intact
    rather than folding them to their base character.
    """

    def __call__(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[<\[][^>\]]*[>\]]", "", text)
        text = re.sub(r"\(([^)]+?)\)", "", text)
        text = unicodedata.normalize("NFKC", text)

        # Drop marks, symbols and punctuation; keep letters and numbers
        text = "".join(
            " " if unicodedata.category(c)[0] in "MSP" else c
            for c in text
        )

        text = re.sub(r"\s+", " ", text)
        return text.strip()


_normalizer = BasicTextNormalizer()


def normalize_text(text: str) -> str:
    """Apply ASR text normalization."""
    return _normalizer(text)


WER_TRANSFORM = jiwer.Compose([
    jiwer.SubstituteRegexes({
        r"[<\[][^>\]]*[>\]]": "",
        r"\(([^)]+?)\)": "",
    }),
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords(),
])


def compute_accuracy(pad_outputs: torch.LongTensor,
                     pad_targets: torch.LongTensor,
                     ignore_label: int) -> torch.Tensor:
    """Token-level accuracy.

    Args:
        pad_outputs (LongTensor): Prediction tensors (B, Lmax).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).
    """
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_outputs.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    return numerator.float() / denominator.float()


def decode_texts_from_outputs(logits: torch.Tensor,
                            labels: torch.Tensor,
                            tokenizer: AutoTokenizer,
                            ignore_label: int = -100) -> Tuple[List[str], List[str]]:
    """Decode model outputs and labels into texts.

    Args:
        logits (torch.Tensor): Prediction logits (B, L, V).
        labels (torch.Tensor): Target label tensors (B, L).
        tokenizer (AutoTokenizer): Tokenizer for decoding indices to text.
        ignore_label (int): Label to ignore in decoding.

    Returns:
        Tuple[List[str], List[str]]: Tuple of (hypothesis texts, reference texts).
    """
    pred_ids = torch.argmax(logits, dim=-1)

    seq_diff = pred_ids.size(1) - labels.size(1)
    if seq_diff:
        raise ValueError(f"Prediction and label sequence lengths do not match: {pred_ids.size(1)} vs {labels.size(1)}")

    valid_mask = (labels != ignore_label)

    hyp_texts, ref_texts = [], []
    for pred, label, mask in zip(pred_ids, labels, valid_mask):
        valid_pred = pred[mask]
        valid_label = label[mask]

        if len(valid_pred) == 0 or len(valid_label) == 0:
            continue

        try:
            pred_text = tokenizer.decode(valid_pred, skip_special_tokens=True).strip()
            label_text = tokenizer.decode(valid_label, skip_special_tokens=True).strip()

            if pred_text and label_text:
                hyp_texts.append(pred_text)
                ref_texts.append(label_text)
        except:
            raise ValueError("Decoding failed for predictions or labels.")

    return hyp_texts, ref_texts


def compute_wer(hyp_texts: List[str], ref_texts: List[str], normalize: bool = True) -> float:
    """Word Error Rate from hypothesis and reference texts.

    Args:
        hyp_texts (List[str]): List of hypothesis (predicted) texts.
        ref_texts (List[str]): List of reference (ground truth) texts.
        normalize (bool): Apply the basic normalization above before scoring.

    Returns:
        float: Word Error Rate (0.0 = perfect, 1.0 = 100% error).
    """
    if not hyp_texts or not ref_texts:
        return 0.0

    if normalize:
        output = jiwer.process_words(
            ref_texts,
            hyp_texts,
            reference_transform=WER_TRANSFORM,
            hypothesis_transform=WER_TRANSFORM
        )
        return output.wer
    else:
        return jiwer.wer(ref_texts, hyp_texts)


def compute_cer(hyp_texts: List[str], ref_texts: List[str], normalize: bool = True) -> float:
    """Character Error Rate from hypothesis and reference texts.

    Args:
        hyp_texts (List[str]): List of hypothesis (predicted) texts.
        ref_texts (List[str]): List of reference (ground truth) texts.
        normalize (bool): Apply the basic normalization above before scoring.

    Returns:
        float: Character Error Rate (0.0 = perfect, 1.0 = 100% error).
    """
    if not hyp_texts or not ref_texts:
        return 0.0

    if normalize:
        normalized_hyp = [normalize_text(text) for text in hyp_texts]
        normalized_ref = [normalize_text(text) for text in ref_texts]
        return jiwer.cer(normalized_ref, normalized_hyp)
    else:
        return jiwer.cer(ref_texts, hyp_texts)


def count_encoder_parameters(encoder, num_layers: int = None) -> Dict[str, int]:
    """
    Count Whisper encoder parameters, split by pruned and retained layers.

    Args:
        encoder: Whisper encoder module.
        num_layers: Number of layers retained (None = all layers).

    Returns:
        Dict with total, used and pruned parameter counts, the per-component
        counts (conv, positional embedding, final layer norm), and the layer
        counts used to compute them.
    """
    conv_params = sum(p.numel() for p in encoder.conv1.parameters())
    conv_params += sum(p.numel() for p in encoder.conv2.parameters())

    pos_params = encoder.positional_embedding.numel()
    ln_params = sum(p.numel() for p in encoder.ln_post.parameters())

    total_blocks = len(encoder.blocks)
    block_params_list = [sum(p.numel() for p in block.parameters()) for block in encoder.blocks]
    total_block_params = sum(block_params_list)

    if num_layers is None:
        num_layers = total_blocks

    used_block_params = sum(block_params_list[:num_layers])

    total_params = conv_params + pos_params + ln_params + total_block_params
    used_params = conv_params + pos_params + ln_params + used_block_params

    return {
        "total_params": total_params,
        "used_params": used_params,
        "pruned_params": total_params - used_params,
        "conv_params": conv_params,
        "pos_embedding_params": pos_params,
        "ln_post_params": ln_params,
        "block_params_per_layer": block_params_list[0] if block_params_list else 0,
        "num_layers_used": num_layers,
        "num_layers_total": total_blocks,
    }
