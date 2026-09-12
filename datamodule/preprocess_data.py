"""
Preprocessing utilities for Common Voice dataset.

Handles:
- Text preprocessing (lowercase, punctuation removal)
- Audio loading and resampling
- Duration filtering
"""

import re
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MIN_DURATION = 0.5   # seconds
DEFAULT_MAX_DURATION = 30.0  # seconds (Whisper limit)


def preprocess_transcription(text: str) -> str:
    """
    Preprocess transcription text.

    - Convert to lowercase
    - Remove all punctuation (keep apostrophes for contractions)
    - Collapse multiple spaces

    Args:
        text: Raw transcription text

    Returns:
        Preprocessed text
    """
    if not text:
        return ""

    text = text.lower()

    # Remove punctuation (keep apostrophes for contractions like "don't")
    text = re.sub(r"[^\w\s']", " ", text)

    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_transcript(tsv_path: Path) -> list[dict]:
    """
    Load transcript from TSV file.

    Args:
        tsv_path: Path to TSV file

    Returns:
        List of sample dictionaries with keys: path, sentence, client_id, etc.
    """
    samples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            samples.append(dict(row))
    return samples


def load_audio(
    audio_path: Path,
    target_sr: int = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """
    Load audio file and resample if needed.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio_array, sr = sf.read(str(audio_path))

    if sr != target_sr:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio_array.astype(np.float32), sr


def process_sample(
    sample: dict,
    audio_dir: Path,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
) -> Optional[dict]:
    """
    Process a single sample: load audio, filter by duration, preprocess text.

    Args:
        sample: Sample dict from transcript (must have 'path', 'sentence', 'client_id')
        audio_dir: Directory containing audio files
        target_sr: Target sample rate
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds

    Returns:
        Processed sample dict or None if filtered out
    """
    # Get audio file path
    audio_filename = sample.get("path", "")
    if not audio_filename:
        return None

    audio_path = audio_dir / audio_filename
    if not audio_path.exists():
        return None

    # Get transcription
    raw_text = sample.get("sentence", "").strip()
    if not raw_text:
        return None

    # Load audio
    try:
        audio_array, sr = load_audio(audio_path, target_sr)
        duration = len(audio_array) / sr
    except Exception:
        return None

    # Filter by duration
    if duration < min_duration or duration > max_duration:
        return None

    # Preprocess transcription
    preprocessed_text = preprocess_transcription(raw_text)

    return {
        "audio_array": audio_array,
        "sampling_rate": sr,
        "raw_transcription": raw_text,
        "transcription": preprocessed_text,
        "duration": round(duration, 3),
        "speaker_id": sample.get("client_id", "")[:16],
    }


def process_split(
    transcript: list[dict],
    audio_dir: Path,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    show_progress: bool = True,
) -> tuple[list[dict], dict]:
    """
    Process all samples in a split.

    Args:
        transcript: List of sample dicts from transcript
        audio_dir: Directory containing audio files
        target_sr: Target sample rate
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        show_progress: Whether to show progress bar

    Returns:
        Tuple of (processed_samples, stats_dict)
    """
    from tqdm.auto import tqdm

    processed_samples = []
    stats = {
        "total": 0,
        "valid": 0,
        "too_short": 0,
        "too_long": 0,
        "missing": 0,
        "empty": 0,
    }

    audio_dir = Path(audio_dir)
    iterator = tqdm(transcript, desc="Processing") if show_progress else transcript

    for sample in iterator:
        stats["total"] += 1

        # Get audio file path
        audio_filename = sample.get("path", "")
        if not audio_filename:
            stats["missing"] += 1
            continue

        audio_path = audio_dir / audio_filename
        if not audio_path.exists():
            stats["missing"] += 1
            continue

        # Get transcription
        raw_text = sample.get("sentence", "").strip()
        if not raw_text:
            stats["empty"] += 1
            continue

        # Load audio
        try:
            audio_array, sr = load_audio(audio_path, target_sr)
            duration = len(audio_array) / sr
        except Exception:
            stats["missing"] += 1
            continue

        # Filter by duration
        if duration < min_duration:
            stats["too_short"] += 1
            continue
        if duration > max_duration:
            stats["too_long"] += 1
            continue

        # Preprocess transcription
        preprocessed_text = preprocess_transcription(raw_text)

        processed_samples.append({
            "audio_array": audio_array,
            "sampling_rate": sr,
            "raw_transcription": raw_text,
            "transcription": preprocessed_text,
            "duration": round(duration, 3),
            "speaker_id": sample.get("client_id", "")[:16],
        })
        stats["valid"] += 1

    return processed_samples, stats
