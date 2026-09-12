"""
Download Common Voice dataset from HuggingFace.

Downloads transcript TSV files and audio tar files from the 
fsicoli/common_voice_22_0 dataset repository.

Supports downloading multiple tar files for large datasets (e.g., English)
until a target number of hours is reached.
"""

import os
import tarfile
from pathlib import Path
from tqdm.auto import tqdm

from huggingface_hub import hf_hub_download, list_repo_files


# Default dataset repository
DEFAULT_REPO = "fsicoli/common_voice_22_0"


def download_transcript(
    language: str,
    split: str,
    repo_id: str = DEFAULT_REPO,
) -> Path:
    """
    Download transcript TSV file for a language and split.
    
    Args:
        language: Language code (e.g., 'da', 'en', 'nl')
        split: Dataset split ('train', 'dev', 'test')
        repo_id: HuggingFace dataset repository ID
        
    Returns:
        Path to downloaded TSV file
    """
    filename = f"transcript/{language}/{split}.tsv"
    print(f"[Download] Downloading transcript: {filename}")
    
    tsv_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
    )
    
    return Path(tsv_path)


def list_audio_tar_files(
    language: str,
    split: str,
    repo_id: str = DEFAULT_REPO,
) -> list[str]:
    """
    List all available audio tar files for a language and split.
    
    Args:
        language: Language code (e.g., 'da', 'en', 'nl')
        split: Dataset split ('train', 'dev', 'test')
        repo_id: HuggingFace dataset repository ID
        
    Returns:
        Sorted list of tar file paths in the repository
    """
    # List all files in the repo
    all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    
    # Filter for audio tar files matching our language and split
    # Pattern: audio/{lang}/{split}/{lang}_{split}_{N}.tar
    prefix = f"audio/{language}/{split}/{language}_{split}_"
    tar_files = [f for f in all_files if f.startswith(prefix) and f.endswith(".tar")]
    
    # Sort by index number
    def get_index(filename):
        # Extract number from filename like "audio/en/train/en_train_5.tar"
        basename = os.path.basename(filename)
        # basename is like "en_train_5.tar"
        num_part = basename.replace(f"{language}_{split}_", "").replace(".tar", "")
        return int(num_part)
    
    tar_files.sort(key=get_index)
    
    return tar_files


def extract_tar_file(
    tar_path: str,
    audio_dir: Path,
) -> int:
    """
    Extract mp3 files from a tar archive.
    
    Args:
        tar_path: Path to the tar file
        audio_dir: Directory to extract audio files to
        
    Returns:
        Number of files extracted
    """
    extracted_count = 0
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc=f"Extracting {os.path.basename(tar_path)}"):
            if member.isfile() and member.name.endswith('.mp3'):
                # Extract just the filename, not full path
                basename = os.path.basename(member.name)
                dest_path = audio_dir / basename
                try:
                    with tar.extractfile(member) as src:
                        if src is not None:
                            with open(dest_path, 'wb') as dst:
                                dst.write(src.read())
                            extracted_count += 1
                except Exception as e:
                    print(f"[Warning] Failed to extract {member.name}: {e}")
    
    return extracted_count


def estimate_hours_from_files(num_files: int, avg_duration_sec: float = 6.0) -> float:
    """
    Estimate total hours from number of audio files.
    
    Args:
        num_files: Number of audio files
        avg_duration_sec: Average duration per file in seconds (default: 6.0)
        
    Returns:
        Estimated hours
    """
    return (num_files * avg_duration_sec) / 3600


def download_and_extract_audio(
    language: str,
    split: str,
    output_dir: Path,
    repo_id: str = DEFAULT_REPO,
    target_hours: float | None = None,
    avg_duration_sec: float = 6.0,
) -> Path:
    """
    Download and extract audio tar files for a language and split.
    
    For large datasets with multiple tar files, this function can stop
    downloading once enough audio has been collected to meet target_hours.
    
    Args:
        language: Language code (e.g., 'da', 'en', 'nl')
        split: Dataset split ('train', 'dev', 'test')
        output_dir: Directory to extract audio files to
        repo_id: HuggingFace dataset repository ID
        target_hours: Target hours of audio to download (None = download all)
        avg_duration_sec: Average duration per file for estimation (default: 6.0s)
        
    Returns:
        Path to directory containing extracted audio files
    """
    # Create output directory
    audio_dir = output_dir / "audio" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Check existing files
    existing_files = list(audio_dir.glob("*.mp3"))
    existing_count = len(existing_files)
    
    if existing_count > 0:
        existing_hours = estimate_hours_from_files(existing_count, avg_duration_sec)
        print(f"[Download] Found {existing_count} existing files (~{existing_hours:.1f} hours)")
        
        if target_hours is not None and existing_hours >= target_hours:
            print(f"[Download] Already have enough audio for target ({target_hours} hours)")
            return audio_dir
    
    # List available tar files
    tar_files = list_audio_tar_files(language, split, repo_id)
    print(f"[Download] Found {len(tar_files)} tar files for {language}/{split}")
    
    if not tar_files:
        print(f"[Warning] No tar files found for {language}/{split}")
        return audio_dir
    
    # Download and extract tar files
    total_extracted = existing_count
    
    for i, tar_filename in enumerate(tar_files):
        # Check if we have enough audio
        if target_hours is not None:
            estimated_hours = estimate_hours_from_files(total_extracted, avg_duration_sec)
            if estimated_hours >= target_hours:
                print(f"[Download] Reached target: ~{estimated_hours:.1f} hours >= {target_hours} hours")
                break
        
        print(f"\n[Download] Downloading tar file {i+1}/{len(tar_files)}: {tar_filename}")
        
        try:
            tar_path = hf_hub_download(
                repo_id=repo_id,
                filename=tar_filename,
                repo_type="dataset",
            )
            
            print(f"[Download] Extracting to: {audio_dir}")
            extracted = extract_tar_file(tar_path, audio_dir)
            total_extracted += extracted
            
            estimated_hours = estimate_hours_from_files(total_extracted, avg_duration_sec)
            print(f"[Download] Extracted {extracted} files. Total: {total_extracted} (~{estimated_hours:.1f} hours)")
            
        except Exception as e:
            print(f"[Error] Failed to download/extract {tar_filename}: {e}")
            continue
    
    # Final verification
    mp3_files = list(audio_dir.glob("*.mp3"))
    final_hours = estimate_hours_from_files(len(mp3_files), avg_duration_sec)
    print(f"\n[Download] Final: {len(mp3_files)} .mp3 files (~{final_hours:.1f} hours) in {audio_dir}")
    
    return audio_dir


def download_split(
    language: str,
    split: str,
    output_dir: Path,
    repo_id: str = DEFAULT_REPO,
    target_hours: float | None = None,
) -> tuple[Path, Path]:
    """
    Download both transcript and audio for a split.
    
    Args:
        language: Language code (e.g., 'da', 'en', 'nl')
        split: Dataset split ('train', 'dev', 'test')
        output_dir: Directory to extract audio files to
        repo_id: HuggingFace dataset repository ID
        target_hours: Target hours of audio to download (None = download all)
        
    Returns:
        Tuple of (transcript_path, audio_dir)
    """
    transcript_path = download_transcript(language, split, repo_id)
    audio_dir = download_and_extract_audio(
        language, split, output_dir, repo_id, target_hours=target_hours
    )
    return transcript_path, audio_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Common Voice dataset")
    parser.add_argument("--language", "-l", type=str, default="da",
                        help="Language code (da, en, nl, etc.)")
    parser.add_argument("--split", "-s", type=str, default="train",
                        choices=["train", "dev", "test"],
                        help="Dataset split")
    parser.add_argument("--output-dir", "-o", type=str, default="data/cv22_raw",
                        help="Output directory for extracted files")
    parser.add_argument("--target-hours", "-t", type=float, default=None,
                        help="Target hours of audio to download (default: all)")
    parser.add_argument("--list-only", action="store_true",
                        help="Only list available tar files, don't download")
    
    args = parser.parse_args()
    
    if args.list_only:
        tar_files = list_audio_tar_files(args.language, args.split)
        print(f"\nAvailable tar files for {args.language}/{args.split}:")
        for f in tar_files:
            print(f"  {f}")
        print(f"\nTotal: {len(tar_files)} tar files")
    else:
        output_dir = Path(args.output_dir)
        transcript_path, audio_dir = download_split(
            args.language, args.split, output_dir,
            target_hours=args.target_hours
        )
        
        print(f"\nDownload complete!")
        print(f"  Transcript: {transcript_path}")
        print(f"  Audio dir: {audio_dir}")