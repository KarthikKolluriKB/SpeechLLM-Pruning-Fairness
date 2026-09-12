"""
Main script to download, preprocess, and save Common Voice dataset as HuggingFace Dataset.

This script orchestrates:
1. Downloading transcript and audio files from HuggingFace
2. Preprocessing audio (loading, resampling) and text (lowercase, no punctuation)
3. Filtering by duration
4. Optionally limiting training data to a maximum number of hours
5. Saving as HuggingFace Dataset with pre-computed audio arrays

Usage:
    python -m datamodule.hf_data --language da --output-dir data/cv22_hf
    python -m datamodule.hf_data --language en --output-dir data/cv22_hf
    python -m datamodule.hf_data --language nl --output-dir data/cv22_hf
    python -m datamodule.hf_data --language en --output-dir data/cv22_hf --max-hours 100
"""

import gc
import random
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, Features, Value, Sequence, concatenate_datasets

from datamodule.download_data import download_transcript, download_and_extract_audio
from datamodule.preprocess_data import load_transcript, process_split


# Buffer multiplier: download more audio than needed to account for filtering
DOWNLOAD_BUFFER_MULTIPLIER = 2.0  # Download 100% extra to ensure enough valid samples


# Default configuration
DEFAULT_SPLITS = ["train", "dev", "test"]
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MIN_DURATION = 0.5   # seconds
DEFAULT_MAX_DURATION = 30.0  # seconds

# Batch size for dataset creation (to avoid OOM)
DEFAULT_BATCH_SIZE = 4000


def get_dataset_features() -> Features:
    """
    Get the HuggingFace Features schema for the dataset.
    """
    return Features({
        "audio_array": Sequence(Value("float32")),
        "sampling_rate": Value("int32"),
        "raw_transcription": Value("string"),
        "transcription": Value("string"),
        "duration": Value("float32"),
        "speaker_id": Value("string"),
    })


def limit_samples_by_hours(
    processed_samples: list,
    max_hours: float,
    shuffle: bool = True,
    seed: int = 42,
) -> list:
    """
    Limit processed samples to a maximum number of hours.
    
    Samples are shuffled before selection to ensure a representative subset.
    Selection continues until adding the next sample would exceed max_hours.
    
    Args:
        processed_samples: List of processed sample dictionaries (must have 'duration' key)
        max_hours: Maximum total duration in hours
        shuffle: Whether to shuffle samples before selection (default: True)
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        List of samples with total duration <= max_hours
    """
    if not processed_samples:
        return []
    
    max_seconds = max_hours * 3600
    
    # Shuffle samples for representative selection
    if shuffle:
        samples = processed_samples.copy()
        random.seed(seed)
        random.shuffle(samples)
    else:
        samples = processed_samples
    
    selected_samples = []
    total_duration = 0.0
    
    for sample in samples:
        sample_duration = sample["duration"]
        if total_duration + sample_duration <= max_seconds:
            selected_samples.append(sample)
            total_duration += sample_duration
        
        # Early exit if we've reached the target
        if total_duration >= max_seconds:
            break
    
    actual_hours = total_duration / 3600
    print(f"[Limit] Selected {len(selected_samples)} samples ({actual_hours:.2f} hours) "
          f"from {len(processed_samples)} samples (target: {max_hours} hours)")
    
    return selected_samples


def create_dataset_in_batches(
    processed_samples: list,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> Dataset:
    """
    Create HuggingFace Dataset in batches to avoid OOM errors.
    
    For large datasets (e.g., 50+ hours of audio), creating a Dataset
    from a single list can exceed memory limits. This function processes
    samples in smaller batches and concatenates them.
    
    Args:
        processed_samples: List of processed sample dictionaries
        batch_size: Number of samples per batch (default: 4000)
        
    Returns:
        HuggingFace Dataset
    """
    features = get_dataset_features()
    total = len(processed_samples)
    
    # For small datasets, process directly
    if total <= batch_size:
        print(f"[Dataset] Creating dataset with {total} samples (single batch)")
        return Dataset.from_list(processed_samples, features=features)
    
    # For large datasets, process in batches
    num_batches = (total + batch_size - 1) // batch_size
    print(f"[Dataset] Creating dataset in {num_batches} batches (batch_size={batch_size})")
    
    datasets = []
    
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        batch_num = i // batch_size + 1
        print(f"[Dataset] Processing batch {batch_num}/{num_batches}: samples {i} to {end}")
        
        # Extract batch
        batch = processed_samples[i:end]
        
        # Create dataset for this batch
        ds = Dataset.from_list(batch, features=features)
        datasets.append(ds)
        
        # Clear batch from memory
        del batch
        gc.collect()
    
    # Concatenate all batches
    print(f"[Dataset] Concatenating {len(datasets)} batches...")
    final_dataset = concatenate_datasets(datasets)
    
    # Clear intermediate datasets
    del datasets
    gc.collect()
    
    print(f"[Dataset] Created dataset with {len(final_dataset)} samples")
    return final_dataset


def prepare_dataset(
    language: str,
    output_dir: Path,
    splits: Optional[list[str]] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    repo_id: str = "fsicoli/common_voice_22_0",
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_train_hours: Optional[float] = None,
    limit_seed: int = 42,
) -> DatasetDict:
    """
    Download, preprocess, and create HuggingFace DatasetDict for a language.
    
    Args:
        language: Language code (e.g., 'da', 'en', 'nl')
        output_dir: Base output directory
        splits: List of splits to process (default: ['train', 'dev', 'test'])
        sample_rate: Target audio sample rate
        min_duration: Minimum audio duration in seconds
        max_duration: Maximum audio duration in seconds
        repo_id: HuggingFace dataset repository ID
        batch_size: Batch size for dataset creation (to avoid OOM)
        max_train_hours: Maximum hours of training data (None = use all data)
        limit_seed: Random seed for reproducible sample selection when limiting hours
        
    Returns:
        HuggingFace DatasetDict with all splits
    """
    if splits is None:
        splits = DEFAULT_SPLITS
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Temporary directory for extracted audio
    temp_dir = output_dir / "_temp" / language
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    datasets_dict = {}
    total_stats = {"total": 0, "valid": 0}
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing: {language} - {split}")
        print('='*60)
        
        # Download transcript
        transcript_path = download_transcript(language, split, repo_id)
        transcript = load_transcript(transcript_path)
        print(f"[Transcript] Loaded {len(transcript)} entries")
        
        # Download and extract audio
        # For training split with hour limit, download with buffer to ensure enough valid samples
        download_target_hours = None
        if split == "train" and max_train_hours is not None:
            download_target_hours = max_train_hours * DOWNLOAD_BUFFER_MULTIPLIER
            print(f"[Download] Target: {max_train_hours} hours (downloading ~{download_target_hours:.1f} hours with buffer)")
        
        audio_dir = download_and_extract_audio(
            language, split, temp_dir, repo_id,
            target_hours=download_target_hours
        )
        
        # Process samples
        print(f"\n[Processing] Processing {len(transcript)} samples...")
        processed_samples, stats = process_split(
            transcript,
            audio_dir,
            target_sr=sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
        )
        
        # Print stats before limiting
        print(f"\n[Stats] {split} (before limiting):")
        print(f"  Total: {stats['total']}")
        print(f"  Valid: {stats['valid']}")
        print(f"  Too short (<{min_duration}s): {stats['too_short']}")
        print(f"  Too long (>{max_duration}s): {stats['too_long']}")
        print(f"  Missing/Error: {stats['missing']}")
        print(f"  Empty text: {stats['empty']}")
        
        total_hours_before = sum(s["duration"] for s in processed_samples) / 3600
        print(f"  Total duration: {total_hours_before:.2f} hours")
        
        # Apply hour limit only to training data
        if split == "train" and max_train_hours is not None:
            print(f"\n[Limit] Applying {max_train_hours} hour limit to training data...")
            processed_samples = limit_samples_by_hours(
                processed_samples,
                max_hours=max_train_hours,
                shuffle=True,
                seed=limit_seed,
            )
            stats["valid"] = len(processed_samples)
        
        total_hours = sum(s["duration"] for s in processed_samples) / 3600
        print(f"  Final duration: {total_hours:.2f} hours ({len(processed_samples)} samples)")
        
        # Create HuggingFace Dataset in batches to avoid OOM
        dataset = create_dataset_in_batches(processed_samples, batch_size=batch_size)
        
        # Free memory immediately
        del processed_samples
        gc.collect()
        
        # Rename 'dev' to 'validation' for consistency
        split_name = "validation" if split == "dev" else split
        datasets_dict[split_name] = dataset
        
        total_stats["total"] += stats["total"]
        total_stats["valid"] += stats["valid"]
    
    print(f"\n{'='*60}")
    print(f"All splits processed for {language}!")
    print('='*60)
    for name, ds in datasets_dict.items():
        print(f"  {name}: {len(ds)} samples")
    
    return DatasetDict(datasets_dict)


def save_dataset(
    dataset_dict: DatasetDict,
    output_dir: Path,
    language: str,
) -> Path:
    """
    Save DatasetDict to disk.
    
    Args:
        dataset_dict: HuggingFace DatasetDict to save
        output_dir: Base output directory
        language: Language code for subdirectory
        
    Returns:
        Path where dataset was saved
    """
    save_path = Path(output_dir) / language
    print(f"\n[Save] Saving dataset to: {save_path}")
    
    dataset_dict.save_to_disk(str(save_path))
    
    print(f"[Save] Dataset saved successfully!")
    return save_path


def prepare_and_save(
    language: str,
    output_dir: str = "data/cv22_hf",
    splits: Optional[list[str]] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_train_hours: Optional[float] = None,
    limit_seed: int = 42,
) -> Path:
    """
    Convenience function to prepare and save dataset in one call.
    
    Args:
        language: Language code (e.g., 'da', 'en', 'nl')
        output_dir: Output directory path
        splits: List of splits to process
        sample_rate: Target audio sample rate
        min_duration: Minimum audio duration
        max_duration: Maximum audio duration
        batch_size: Batch size for dataset creation (to avoid OOM)
        max_train_hours: Maximum hours of training data (None = use all data)
        limit_seed: Random seed for reproducible sample selection when limiting hours
        
    Returns:
        Path where dataset was saved
    """
    output_path = Path(output_dir)
    
    # Prepare dataset
    dataset_dict = prepare_dataset(
        language=language,
        output_dir=output_path,
        splits=splits,
        sample_rate=sample_rate,
        min_duration=min_duration,
        max_duration=max_duration,
        batch_size=batch_size,
        max_train_hours=max_train_hours,
        limit_seed=limit_seed,
    )
    
    # Save dataset
    save_path = save_dataset(dataset_dict, output_path, language)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Language: {language}")
    print(f"Output: {save_path.absolute()}")
    if max_train_hours is not None:
        print(f"Training data limit: {max_train_hours} hours")
        # Check actual hours in train split
        train_ds = dataset_dict.get('train', None)
        if train_ds is not None:
            actual_train_hours = sum(train_ds['duration']) / 3600
            print(f"Actual training hours after filtering/limiting: {actual_train_hours:.2f}")
            if actual_train_hours < max_train_hours:
                print(f"[WARNING] Only {actual_train_hours:.2f} hours of training data available after filtering and limiting. Consider increasing DOWNLOAD_BUFFER_MULTIPLIER or checking data quality.")
    print(f"\nDataset features:")
    print(f"  - audio_array: Pre-computed audio as float32 array")
    print(f"  - sampling_rate: {sample_rate} Hz")
    print(f"  - raw_transcription: Original text (with punctuation)")
    print(f"  - transcription: Preprocessed text (lowercase, no punctuation)")
    print(f"  - duration: Audio duration in seconds")
    print(f"  - speaker_id: Anonymized speaker ID")
    print(f"\nTo load:")
    print(f"  from datasets import load_from_disk")
    print(f"  dataset = load_from_disk('{save_path}')")

    return save_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download and preprocess Common Voice dataset"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="da",
        help="Language code (da=Danish, en=English, nl=Dutch, etc.)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/cv22_hf",
        help="Output directory for HuggingFace dataset"
    )
    parser.add_argument(
        "--splits", "-s",
        type=str,
        nargs="+",
        default=["train", "dev", "test"],
        help="Splits to process (default: train dev test)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target audio sample rate (default: 16000)"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum audio duration in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for dataset creation to avoid OOM (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="Maximum hours of training data (default: None = use all data). "
             "Only applies to training split; dev and test are kept complete."
    )
    parser.add_argument(
        "--limit-seed",
        type=int,
        default=42,
        help="Random seed for reproducible sample selection when using --max-hours (default: 42)"
    )
    
    args = parser.parse_args()
    
    prepare_and_save(
        language=args.language,
        output_dir=args.output_dir,
        splits=args.splits,
        sample_rate=args.sample_rate,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        batch_size=args.batch_size,
        max_train_hours=args.max_hours,
        limit_seed=args.limit_seed,
    )