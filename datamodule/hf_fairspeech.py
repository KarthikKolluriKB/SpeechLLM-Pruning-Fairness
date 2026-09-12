"""
Build an HF dataset for Meta's ASR Fairness Evaluation Dataset (paper name:
Fair-Speech) in the project's standard SLAM-ASR schema.

Source (already staged by scripts/download_fairspeech.py):
    data/fairspeech/metadata.tsv   (26,471 rows)
    data/fairspeech/audio/*.wav    (26,475 files; ~4 extras likely macOS junk)

Source schema:
    hash_name           32-hex string, matches <hash_name>.wav under audio/
    transcription       'hey facebook answer the call'  etc.
    age                 '18 - 22'  etc.  (string ranges, NOT numeric)
    gender              'female' / 'male'
    first_language      'English'  etc.
    socioeconomic_bkgd  'Low' / 'Medium' / 'High'  (self-reported)
    ethnicity           'White' / ...

Output schema (data/fairspeech_hf/):
    audio_array, sampling_rate, raw_transcription, transcription, duration,
    speaker_id (= hash_name), gender, age, l1 (= first_language lowercased),
    ses (= socioeconomic_bkgd lowercased), ethnicity (lowercased), accent ('missing')

LICENCE — IMPORTANT (Meta's terms):
    Do NOT redistribute the produced HF dataset. The output stays under
    data/ and per-utterance evaluation output under results/, both of which
    are gitignored. Summary-level metrics (per-group WER, etc.) may be
    published.

Usage:
    python -m datamodule.hf_fairspeech
    python -m datamodule.hf_fairspeech --input_dir data/fairspeech \
        --output_dir data/fairspeech_hf --resample 16000

Audio files are nominally 16 kHz (Whisper-friendly) but the resample step
runs only if the actual sample rate differs from --resample.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


def _norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_gender(g: str) -> str:
    g = (g or "").strip().lower()
    if g.startswith("m"):
        return "male"
    if g.startswith("f"):
        return "female"
    if not g:
        return "missing"
    return "other"


def _lower_or_missing(v: str | None) -> str:
    v = (v or "").strip().lower()
    return v or "missing"


def _index_audio_files(audio_dir: Path) -> dict[str, Path]:
    """Walk the audio/ tree ONCE and build {hash_name: path} so the per-row
    lookup is O(1). Tolerates either a flat layout (audio/<hash>.wav) or
    one nested layer (audio/<sub>/<hash>.wav). Skips macOS resource forks.
    """
    print(f"[Fair-Speech] Indexing audio files under {audio_dir} ...")
    index: dict[str, Path] = {}
    for p in audio_dir.rglob("*.wav"):
        name = p.name
        if name.startswith("._"):
            continue
        stem = p.stem  # filename without .wav
        # If multiple files share a stem, prefer the shallower one.
        if stem not in index or len(p.parts) < len(index[stem].parts):
            index[stem] = p
    print(f"[Fair-Speech] Indexed {len(index)} unique audio stems.")
    return index


def _flush_batch(buf: list[dict], features, datasets_pkg):
    if not buf:
        return None
    return datasets_pkg.Dataset.from_list(buf, features=features)


def build(input_dir: Path, output_dir: Path, target_sr: int = 16000,
          batch_size: int = 2000) -> None:
    """Batched build to keep peak memory bounded.

    26k Fair-Speech rows × ~7s × 16 kHz × float32 ≈ 12 GB in Python; the
    project's CV22 builder uses the same batched-then-concatenate pattern.
    """
    import datasets as ds_pkg
    from datasets import DatasetDict, Features, Sequence, Value, concatenate_datasets
    import soundfile as sf

    meta_path = input_dir / "metadata.tsv"
    audio_dir = input_dir / "audio"
    if not meta_path.exists():
        raise SystemExit(f"Missing metadata.tsv at {meta_path}")
    if not audio_dir.exists():
        raise SystemExit(f"Missing audio/ folder at {audio_dir}")

    print(f"[Fair-Speech] Reading metadata: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        meta_rows = list(reader)
    print(f"[Fair-Speech] Metadata rows: {len(meta_rows)}")

    audio_index = _index_audio_files(audio_dir)

    features = Features({
        "audio_array": Sequence(Value("float32")),
        "sampling_rate": Value("int32"),
        "raw_transcription": Value("string"),
        "transcription": Value("string"),
        "duration": Value("float32"),
        "speaker_id": Value("string"),
        "gender": Value("string"),
        "age": Value("string"),
        "l1": Value("string"),
        "ses": Value("string"),
        "ethnicity": Value("string"),
        "accent": Value("string"),
    })

    batches: list = []
    buf: list[dict] = []
    stats = {"total": len(meta_rows), "kept": 0, "audio_missing": 0, "decode_error": 0,
             "empty_text": 0, "resampled": 0}
    total_dur = 0.0
    import gc

    def _drain():
        nonlocal buf
        ds_batch = _flush_batch(buf, features, ds_pkg)
        if ds_batch is not None:
            batches.append(ds_batch)
            print(f"  flushed batch of {len(buf)} rows (batches so far: {len(batches)})")
        buf = []
        gc.collect()

    for i, r in enumerate(meta_rows):
        hash_name = (r.get("hash_name") or "").strip()
        raw_text = (r.get("transcription") or "").strip()
        if not hash_name or not raw_text:
            stats["empty_text"] += 1
            continue
        wav_path = audio_index.get(hash_name)
        if wav_path is None:
            stats["audio_missing"] += 1
            continue
        try:
            arr, sr = sf.read(str(wav_path), dtype="float32")
        except Exception as e:
            stats["decode_error"] += 1
            if stats["decode_error"] <= 3:
                print(f"  decode error on {wav_path.name}: {e}")
            continue
        if sr != target_sr:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr).astype(np.float32)
            sr = target_sr
            stats["resampled"] += 1

        duration = float(len(arr) / sr)
        total_dur += duration

        buf.append({
            "audio_array": arr.tolist(),
            "sampling_rate": sr,
            "raw_transcription": raw_text,
            "transcription": _norm_text(raw_text),
            "duration": duration,
            "speaker_id": hash_name,
            "gender": _norm_gender(r.get("gender")),
            "age": _lower_or_missing(r.get("age")),
            "l1": _lower_or_missing(r.get("first_language")),
            "ses": _lower_or_missing(r.get("socioeconomic_bkgd")),
            "ethnicity": _lower_or_missing(r.get("ethnicity")),
            "accent": "missing",
        })
        stats["kept"] += 1

        if len(buf) >= batch_size:
            _drain()
        if (i + 1) % 1000 == 0:
            print(f"  ... {i+1}/{len(meta_rows)} rows processed, "
                  f"{stats['kept']} kept, {stats['audio_missing']} missing audio")

    _drain()

    print("[Fair-Speech] Build stats:")
    for k, v in stats.items():
        print(f"  {k:<14} {v}")
    print(f"  total audio:   {total_dur/3600.0:.2f} h  ({stats['kept']} rows)")

    if not batches:
        raise SystemExit("Zero rows survived — check input_dir and the metadata schema.")

    print(f"[Fair-Speech] Concatenating {len(batches)} batches ...")
    final = concatenate_datasets(batches)
    out_dict = DatasetDict({"test": final})

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Fair-Speech] Saving to {output_dir}")
    out_dict.save_to_disk(str(output_dir))
    print(f"[Fair-Speech] Done. test split: {len(final)} rows.")


def parse_args():
    p = argparse.ArgumentParser(description="Build an HF dataset for Meta Fair-Speech.")
    p.add_argument("--input_dir", type=Path, default=Path("data/fairspeech"),
                   help="Staged Fair-Speech directory (output of download_fairspeech.py).")
    p.add_argument("--output_dir", type=Path, default=Path("data/fairspeech_hf"),
                   help="Where to save the SLAM-ASR-shaped HF dataset.")
    p.add_argument("--resample", type=int, default=16000,
                   help="Target sample rate (Whisper expects 16000).")
    p.add_argument("--batch_size", type=int, default=2000,
                   help="Rows per HF Dataset.from_list batch. Lower if you OOM.")
    return p.parse_args()


def main():
    args = parse_args()
    build(args.input_dir, args.output_dir, target_sr=args.resample, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
