"""Decode one configuration and score WER/CER per demographic group.

Same decoding path as eval.py, but writes a per-utterance CSV and per-group
summaries with bootstrap CIs. Groups below 200 utterances or 30 minutes of
audio are written with point estimates only. See the README for usage.
"""

from __future__ import annotations

import argparse
import csv
import gc
import re
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datamodule.dataset import get_speech_dataset  # noqa: E402
from models.model import model_builder  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from bootstrap_ci import wer_with_ci, cer_with_ci  # noqa: E402


GENDER_BUCKETS = ["male", "female", "other", "missing"]
ANALYSABLE_MIN_SECONDS = 30 * 60
ANALYSABLE_MIN_UTTS = 200

DEFAULT_PER_SEED_DIR = PROJECT_ROOT / "results/per_group_wer"


def _norm_gender(raw: str | None) -> str:
    if raw is None:
        return "missing"
    g = raw.strip().lower()
    if g in {"", "nan", "none"}:
        return "missing"
    if g.startswith("male") or g == "m" or g == "male_masculine":
        return "male"
    if g.startswith("female") or g == "f" or g == "female_feminine":
        return "female"
    return "other"


def _normalise_sentence(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_cv_demographics(tsv_path: Path | None,
                         language: str = "en") -> dict[tuple[str, str], dict]:
    """Map (client_id[:16], normalised sentence) to demographics from test.tsv.

    The preprocessed CV dataset truncates client_id to 16 chars and drops the
    demographic columns, hence the join. The TSV must be the same language as
    the audio; it is downloaded if tsv_path is omitted.
    """
    if tsv_path is None:
        from huggingface_hub import hf_hub_download
        tsv_path = Path(hf_hub_download(
            repo_id="fsicoli/common_voice_22_0",
            filename=f"transcript/{language}/test.tsv",
            repo_type="dataset",
        ))
    out: dict[tuple[str, str], dict] = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            client_id = row.get("client_id", "") or ""
            sent = _normalise_sentence(row.get("sentence", ""))
            key = (client_id[:16], sent)
            out[key] = {
                "client_id_full": client_id,
                "gender": _norm_gender(row.get("gender")),
                "age": (row.get("age") or "").strip().lower() or "missing",
                "accent": (row.get("accents") or row.get("accent") or "").strip().lower() or "missing",
                "path": row.get("path", ""),
            }
    return out


def join_demographics_from_hf_columns(
    rows: list[dict],
    hf_dataset,
    demographic_columns: list[str],
) -> tuple[list[dict], int]:
    """Read demographic columns straight from the HF dataset rows.

    For datasets that carry demographics inline. gender goes through
    _norm_gender so 'm'/'Male'/'male' all land in the same bucket. Returns
    (rows, number of rows with no gender value).
    """
    n_missing = 0
    for r in rows:
        idx = r["row_idx"]
        sample = hf_dataset[idx]
        for col in demographic_columns:
            raw = sample.get(col, None)
            if col == "gender":
                r["gender"] = _norm_gender(raw if isinstance(raw, str) else None)
            else:
                v = (raw or "")
                r[col] = v.strip().lower() if isinstance(v, str) else "missing"
                if r[col] == "":
                    r[col] = "missing"
        # the summary code expects these three to exist
        for required in ("gender", "age", "accent"):
            r.setdefault(required, "missing")
        r.setdefault("client_id_full", r.get("speaker_id", ""))
        if r["gender"] == "missing":
            n_missing += 1
    return rows, n_missing


@torch.no_grad()
def run_inference(
    cfg,
    checkpoint_path: Path,
    device: torch.device,
    split: str,
) -> list[dict]:
    train_cfg = cfg.train
    model_cfg = cfg.model
    data_cfg = cfg.data
    eval_cfg = cfg.eval if hasattr(cfg, "eval") else None

    if hasattr(data_cfg, "inference_mode"):
        data_cfg.inference_mode = True

    max_new_tokens = getattr(eval_cfg, "max_new_tokens", 128) if eval_cfg else 128
    num_beams = getattr(eval_cfg, "num_beams", 1) if eval_cfg else 1
    do_sample = getattr(eval_cfg, "do_sample", False) if eval_cfg else False
    repetition_penalty = getattr(eval_cfg, "repetition_penalty", 1.0) if eval_cfg else 1.0
    length_penalty = getattr(eval_cfg, "length_penalty", 1.0) if eval_cfg else 1.0
    temperature = getattr(eval_cfg, "temperature", 1.0) if eval_cfg else 1.0
    batch_size = getattr(eval_cfg, "batch_size", 16) if eval_cfg else 16

    print(f"[Eval] gen settings: num_beams={num_beams}, max_new_tokens={max_new_tokens}, "
          f"rep_pen={repetition_penalty}, len_pen={length_penalty}, batch_size={batch_size}")

    print(f"[Eval] Building model from config; loading checkpoint: {checkpoint_path}")
    model, tokenizer = model_builder(train_cfg, model_cfg, data_config=data_cfg)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    projector_state = ckpt["projector"] if isinstance(ckpt, dict) and "projector" in ckpt else ckpt
    model.projector.load_state_dict(projector_state)
    use_lora = getattr(train_cfg, "use_lora", False)
    if use_lora and isinstance(ckpt, dict) and "lora" in ckpt:
        try:
            model.llm.load_state_dict(ckpt["lora"], strict=False)
            print("[Eval] Loaded LoRA adapter weights.")
        except Exception as e:
            print(f"[Eval] WARNING: could not load LoRA weights: {e}")

    model = model.to(device)
    model.eval()

    test_dataset = get_speech_dataset(data_cfg, tokenizer, split=split)
    print(f"[Eval] Test dataset: {len(test_dataset)} samples (split={split})")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.collator,
        num_workers=getattr(eval_cfg, "num_workers", 4) if eval_cfg else 4,
        pin_memory=(device.type == "cuda"),
    )

    use_autocast = bool(getattr(train_cfg, "mixed_precision", False)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    enc_dtype = amp_dtype if use_autocast else torch.float32

    rows: list[dict] = []
    # row index into valid_indices, for the speaker_id/duration lookup
    valid_indices = test_dataset.valid_indices
    hf_dataset = test_dataset.hf_dataset

    cursor = 0  # position in the (filtered) dataset; advances by batch size
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Inference")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        audio_mel = batch["audio_mel"].to(device).to(enc_dtype)
        modality_mask = batch["modality_mask"].to(device)

        ref_texts = batch.get("targets")
        keys = batch.get("keys")
        if ref_texts is None or keys is None:
            raise RuntimeError("Batch missing `targets`/`keys`; set data.inference_mode=true.")

        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                gen_ids = model.generate(
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
            gen_ids = model.generate(
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

        hyp_texts = [tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen_ids]

        for i, (hyp, ref, key) in enumerate(zip(hyp_texts, ref_texts, keys)):
            real_idx = valid_indices[cursor + i]
            sample_row = hf_dataset[real_idx]
            rows.append({
                "row_idx": int(real_idx),
                "key": str(key),
                "speaker_id": str(sample_row.get("speaker_id", "")),
                "duration_s": float(sample_row.get("duration", 0.0)),
                "reference": ref,
                "hypothesis": hyp,
            })
        cursor += len(hyp_texts)

        del input_ids, attention_mask, audio_mel, modality_mask, gen_ids
        if batch_idx % 20 == 0 and device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    return rows


def join_demographics(rows: list[dict], demo_map: dict[tuple[str, str], dict]) -> tuple[list[dict], int]:
    """Add gender/age/accent/client_id_full to each row. Returns (rows, n_unmatched)."""
    n_unmatched = 0
    for r in rows:
        key = (r["speaker_id"][:16], _normalise_sentence(r["reference"]))
        demo = demo_map.get(key)
        if demo is None:
            n_unmatched += 1
            r.update({"gender": "missing", "age": "missing", "accent": "missing", "client_id_full": ""})
        else:
            r.update({
                "gender": demo["gender"],
                "age": demo["age"],
                "accent": demo["accent"],
                "client_id_full": demo["client_id_full"],
            })
    return rows, n_unmatched


def write_per_utterance_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # write every populated axis, otherwise recovering it means decoding again
    present_demo = [c for c in KNOWN_AXES if any(c in r for r in rows)]
    fields = ["row_idx", "key", "speaker_id", "client_id_full"] + present_demo \
        + ["duration_s", "reference", "hypothesis"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _empty_metric() -> dict:
    return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}


# axes the summary understands, in display order
KNOWN_AXES = ["gender", "age", "accent", "l1", "ses", "ethnicity"]


def _summarise_subset(refs, hyps, n_bootstrap, do_bootstrap):
    """WER/CER for one bucket of utterances, with a CI when do_bootstrap."""
    nb = n_bootstrap if do_bootstrap else 0
    if not refs:
        return _empty_metric(), _empty_metric()
    w = wer_with_ci(refs, hyps, n_bootstrap=nb, seed=42)
    c = cer_with_ci(refs, hyps, n_bootstrap=nb, seed=42)
    return (
        {"point": w["wer"], "ci_low": w["ci_low"], "ci_high": w["ci_high"]},
        {"point": c["cer"], "ci_low": c["ci_low"], "ci_high": c["ci_high"]},
    )


def per_gender_summary(
    rows: list[dict],
    condition: str,
    seed: int,
    n_bootstrap: int = 1000,
) -> tuple[list[dict], dict]:
    by_gender: dict[str, list[dict]] = {g: [] for g in GENDER_BUCKETS}
    for r in rows:
        by_gender.setdefault(r["gender"], []).append(r)

    summary_rows = []
    for g in GENDER_BUCKETS:
        subset = by_gender.get(g, [])
        n = len(subset)
        dur = sum(float(r["duration_s"]) for r in subset)
        analysable = (dur >= ANALYSABLE_MIN_SECONDS) and (n >= ANALYSABLE_MIN_UTTS)
        wer, cer = _summarise_subset(
            [r["reference"] for r in subset],
            [r["hypothesis"] for r in subset],
            n_bootstrap, do_bootstrap=analysable,
        )
        summary_rows.append({
            "condition": condition, "seed": seed, "gender": g,
            "n_utts": n, "duration_hours": round(dur / 3600.0, 4),
            "wer": wer["point"], "wer_ci_low": wer["ci_low"], "wer_ci_high": wer["ci_high"],
            "cer": cer["point"], "cer_ci_low": cer["ci_low"], "cer_ci_high": cer["ci_high"],
            "analysable": "yes" if analysable else "no",
        })

    agg_wer, agg_cer = _summarise_subset(
        [r["reference"] for r in rows],
        [r["hypothesis"] for r in rows],
        n_bootstrap, do_bootstrap=True,
    )
    aggregate_row = {
        "condition": condition, "seed": seed, "gender": "ALL",
        "n_utts": len(rows),
        "duration_hours": round(sum(float(r["duration_s"]) for r in rows) / 3600.0, 4),
        "wer": agg_wer["point"], "wer_ci_low": agg_wer["ci_low"], "wer_ci_high": agg_wer["ci_high"],
        "cer": agg_cer["point"], "cer_ci_low": agg_cer["ci_low"], "cer_ci_high": agg_cer["ci_high"],
        "analysable": "yes",
    }
    summary_rows.append(aggregate_row)
    return summary_rows, aggregate_row


def write_summary_csv(summary_rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["condition", "seed", "gender", "n_utts", "duration_hours",
              "wer", "wer_ci_low", "wer_ci_high",
              "cer", "cer_ci_low", "cer_ci_high",
              "analysable"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)


MULTIAXIS_FIELDS = ["condition", "seed", "axis", "group", "n_utts", "duration_hours",
                    "wer", "wer_ci_low", "wer_ci_high",
                    "cer", "cer_ci_low", "cer_ci_high", "analysable"]


def _axis_groups(rows: list[dict], axis: str) -> tuple[dict[str, list[dict]], list[str]]:
    """Bucket rows by `axis`. Gender keeps its canonical order, other axes sort."""
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        val = r.get(axis, "missing")
        val = val if (isinstance(val, str) and val) else "missing"
        groups[val].append(r)
    if axis == "gender":
        order = [g for g in GENDER_BUCKETS if g in groups] + \
                sorted(g for g in groups if g not in GENDER_BUCKETS)
    else:
        order = sorted(groups.keys())
    return groups, order


def multiaxis_summary(
    rows: list[dict],
    axes: list[str],
    condition: str,
    seed: int,
    n_bootstrap: int = 1000,
) -> list[dict]:
    """One row per (axis, group), plus an ALL row per axis."""
    out: list[dict] = []
    # same for every axis, so compute it once
    total_dur_all = sum(float(r["duration_s"]) for r in rows)
    agg_wer, agg_cer = _summarise_subset(
        [r["reference"] for r in rows], [r["hypothesis"] for r in rows],
        n_bootstrap, do_bootstrap=bool(rows),
    )
    for axis in axes:
        groups, order = _axis_groups(rows, axis)
        for g in order:
            subset = groups[g]
            n = len(subset)
            dur = sum(float(r["duration_s"]) for r in subset)
            analysable = (dur >= ANALYSABLE_MIN_SECONDS) and (n >= ANALYSABLE_MIN_UTTS)
            wer, cer = _summarise_subset(
                [r["reference"] for r in subset],
                [r["hypothesis"] for r in subset],
                n_bootstrap, do_bootstrap=analysable,
            )
            out.append({
                "condition": condition, "seed": seed, "axis": axis, "group": g,
                "n_utts": n, "duration_hours": round(dur / 3600.0, 4),
                "wer": wer["point"], "wer_ci_low": wer["ci_low"], "wer_ci_high": wer["ci_high"],
                "cer": cer["point"], "cer_ci_low": cer["ci_low"], "cer_ci_high": cer["ci_high"],
                "analysable": "yes" if analysable else "no",
            })
        out.append({
            "condition": condition, "seed": seed, "axis": axis, "group": "ALL",
            "n_utts": len(rows), "duration_hours": round(total_dur_all / 3600.0, 4),
            "wer": agg_wer["point"], "wer_ci_low": agg_wer["ci_low"], "wer_ci_high": agg_wer["ci_high"],
            "cer": agg_cer["point"], "cer_ci_low": agg_cer["ci_low"], "cer_ci_high": agg_cer["ci_high"],
            "analysable": "yes",
        })
    return out


def write_multiaxis_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MULTIAXIS_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MULTIAXIS_FIELDS})


def parse_args():
    p = argparse.ArgumentParser(description="Per-subgroup WER evaluation for the bias-pruning experiment.")
    p.add_argument("--config", type=Path, required=True,
                   help="Path to eval YAML config (e.g. configs/whisper_medium/english/eval/baseline.yaml).")
    p.add_argument("--checkpoint_path", type=Path, required=True,
                   help="Path to the projector checkpoint (.pt).")
    p.add_argument("--prune_depth", type=int, required=True,
                   help="Top-down prune depth. 0 == unpruned. For whisper-small 12L total: "
                        "prune_depth=N means (12-N) layers kept.")
    p.add_argument("--seed", type=int, required=True,
                   help="Training seed of this checkpoint (for filename + bookkeeping only).")
    p.add_argument("--condition", type=str, required=True,
                   help="Free-form condition label used in output filenames (e.g. 'unpruned', "
                        "'pruned_2L', 'depth_05_kept_07').")
    p.add_argument("--language", default="en")
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hf_dataset_path", type=Path, default=None,
                   help="Override data.hf_dataset_path in the YAML config. Use this to point "
                        "the inference loader at a dataset built by "
                        "datamodule/hf_<dataset>.py (e.g., data/fairspeech_hf/).")
    p.add_argument("--demographic_source", choices=["cv22_tsv", "hf_columns"],
                   default="cv22_tsv",
                   help="cv22_tsv: join from upstream CV22 transcript/en/test.tsv (CV22 only). "
                        "hf_columns: read demographic columns directly from the HF dataset rows.")
    p.add_argument("--demographic_columns", nargs="+",
                   default=["gender", "age", "accent", "l1", "ses", "ethnicity"],
                   help="Column names to pull from each HF dataset row when "
                        "--demographic_source=hf_columns. Missing columns become 'missing'.")
    p.add_argument("--cv_test_tsv", type=Path, default=None,
                   help="Path to upstream CV22 transcript/en/test.tsv. Downloaded if omitted. "
                        "Only used when --demographic_source=cv22_tsv.")
    p.add_argument("--output_path", type=Path, default=None,
                   help="Per-utterance results CSV. Default: results/per_utterance/{condition}_{seed}.csv. Give --condition a corpus-specific label: files are named from it alone, so two corpora evaluated at the same depth would otherwise overwrite each other.")
    p.add_argument("--per_seed_dir", type=Path, default=DEFAULT_PER_SEED_DIR)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override eval.batch_size from the YAML config (e.g. raise it for "
                        "smaller models that under-use the GPU). If unset, uses the config value.")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")
    if not args.checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint_path}")

    cfg = OmegaConf.load(str(args.config))
    if args.hf_dataset_path is not None:
        # lets one eval config serve several datasets
        cfg.data.hf_dataset_path = str(args.hf_dataset_path)
        print(f"[Eval] Override: cfg.data.hf_dataset_path = {cfg.data.hf_dataset_path}")
    if args.batch_size is not None:
        # force_add covers configs with no eval.batch_size key
        OmegaConf.update(cfg, "eval.batch_size", args.batch_size, force_add=True)
        print(f"[Eval] Override: cfg.eval.batch_size = {args.batch_size}")
    device = torch.device(args.device)

    rows = run_inference(cfg, args.checkpoint_path, device, split=args.split)
    print(f"[Eval] Inference produced {len(rows)} rows.")

    if args.demographic_source == "cv22_tsv":
        print(f"[Eval] Loading CV22 demographics from upstream "
              f"{args.language}/test.tsv ...")
        demo_map = load_cv_demographics(args.cv_test_tsv, args.language)
        rows, n_unmatched = join_demographics(rows, demo_map)
        if n_unmatched:
            print(f"[Eval] WARNING: {n_unmatched} of {len(rows)} utterances did not match a "
                  "CV22 demo row (speaker_id prefix + normalised sentence). "
                  "These are labelled gender=missing.")
    else:  # hf_columns
        print(f"[Eval] Reading demographics from HF dataset columns: "
              f"{args.demographic_columns}")
        from datasets import load_from_disk
        # reloaded here: run_inference keeps its own copy scoped internally
        hf_dataset_path = cfg.data.hf_dataset_path
        ds = load_from_disk(hf_dataset_path)
        split_name = args.split
        if split_name in ("val", "dev"):
            split_name = "validation"
        if split_name not in ds:
            raise SystemExit(f"Split '{split_name}' not found in {hf_dataset_path}. "
                             f"Available: {list(ds.keys())}")
        rows, n_missing = join_demographics_from_hf_columns(
            rows, ds[split_name], args.demographic_columns,
        )
        if n_missing:
            print(f"[Eval] {n_missing} of {len(rows)} rows had no `gender` value in the dataset; "
                  "they are labelled gender=missing.")

    out_utt = args.output_path or (
        PROJECT_ROOT / "results/per_utterance" / f"{args.condition}_seed{args.seed}.csv"
    )
    write_per_utterance_csv(rows, out_utt)
    print(f"[Eval] Wrote per-utterance CSV: {out_utt}")

    summary_rows, aggregate_row = per_gender_summary(
        rows, condition=args.condition, seed=args.seed, n_bootstrap=args.n_bootstrap,
    )
    summary_path = args.per_seed_dir / f"{args.condition}_seed{args.seed}.csv"
    write_summary_csv(summary_rows, summary_path)
    print(f"[Eval] Wrote per-gender summary: {summary_path}")

    # gender plus every other populated axis
    if args.demographic_source == "hf_columns":
        requested = args.demographic_columns
    else:  # cv22_tsv carries gender/age/accent
        requested = ["gender", "age", "accent"]
    multi_axes = [a for a in requested
                  if a in KNOWN_AXES and any(a in r for r in rows)]
    multiaxis_path = args.per_seed_dir / f"{args.condition}_seed{args.seed}_multiaxis.csv"
    multi_rows = multiaxis_summary(
        rows, multi_axes, condition=args.condition, seed=args.seed,
        n_bootstrap=args.n_bootstrap,
    )
    write_multiaxis_csv(multi_rows, multiaxis_path)
    print(f"[Eval] Wrote multi-axis summary ({', '.join(multi_axes)}): {multiaxis_path}")

    print("\n=== Per-gender WER / CER (this seed) ===")
    print(f"{'gender':<8} {'n':>6} {'hours':>7} "
          f"{'WER':>8} {'WER 95% CI':>20} "
          f"{'CER':>8} {'CER 95% CI':>20} {'analysable':>11}")
    for r in summary_rows:
        wci = f"[{r['wer_ci_low']*100:.2f}, {r['wer_ci_high']*100:.2f}]"
        cci = f"[{r['cer_ci_low']*100:.2f}, {r['cer_ci_high']*100:.2f}]"
        print(f"{r['gender']:<8} {r['n_utts']:>6d} {r['duration_hours']:>7.2f} "
              f"{r['wer']*100:>7.2f}% {wci:>20} "
              f"{r['cer']*100:>7.2f}% {cci:>20} {r['analysable']:>11}")

    print(f"\nAggregate WER = {aggregate_row['wer']*100:.2f}% | "
          f"Aggregate CER = {aggregate_row['cer']*100:.2f}% on {aggregate_row['n_utts']} utts.")


if __name__ == "__main__":
    main()
