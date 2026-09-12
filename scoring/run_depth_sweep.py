"""Run evaluate_subgroup_wer.py over a depth sweep, one depth at a time.

Defaults to large-v2 English on Fair-Speech at depths 0, 2, 4, 6, 8, 10.
Checkpoints come from the layout the training configs write:

    {checkpoint_root}/baseline/checkpoint_best_wer.pt
    {checkpoint_root}/ablation_{kept}L/checkpoint_best_wer.pt

Use --checkpoint_overrides for files that live elsewhere, --dry_run to print
the plan without running anything.
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_SCRIPT = Path(__file__).resolve().parent / "evaluate_subgroup_wer.py"


# Registry of evaluation datasets the sweep can target. Each entry tells
# evaluate_subgroup_wer.py where the HF dataset lives, which demographic axes it
# ships, and how to extract them. Add a dataset here once its builder under
# datamodule/hf_*.py has run.
DATASETS = {
    "cv22": {
        "hf_dataset_path": "data/cv22_hf/en",
        "demographic_source": "cv22_tsv",
        "demographic_columns": None,    # cv22_tsv path does its own join
        "tag_suffix": "cv22",
    },
    # Danish / Dutch Common Voice for the cross-lingual comparison.
    # angle. Same cv22_tsv demographic join (gender/age/accent only; CV never
    # collected race/SES), but you MUST pass --cv_test_tsv pointing at the
    # matching-language transcript/<lang>/test.tsv (the auto-download default is
    # English-only). hf_dataset_path assumes the da/nl HF builds live alongside en.
    "cv22_da": {
        "hf_dataset_path": "data/cv22_hf/da",
        "demographic_source": "cv22_tsv",
        "demographic_columns": None,
        "tag_suffix": "cv22_da",
    },
    "cv22_nl": {
        "hf_dataset_path": "data/cv22_hf/nl",
        "demographic_source": "cv22_tsv",
        "demographic_columns": None,
        "tag_suffix": "cv22_nl",
    },
    "fairspeech": {
        "hf_dataset_path": "data/fairspeech_hf",
        "demographic_source": "hf_columns",
        "demographic_columns": ["gender", "age", "l1", "ses", "ethnicity"],
        "tag_suffix": "fairspeech",
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Sweep evaluate_subgroup_wer.py over pruning depths.")
    p.add_argument("--model_dir", type=Path,
                   default=PROJECT_ROOT / "configs/fairspeech/whisper_largev2/eval",
                   help="Folder with baseline.yaml + ablation_NL.yaml eval configs.")
    p.add_argument("--total_layers", type=int, default=32,
                   help="Total encoder layers for this model (whisper-small=12, medium=24, large-v2=32).")
    p.add_argument("--depths", type=int, nargs="+", default=None,
                   help="Prune depths to run (0 = unpruned). Defaults to the paper's 0, 2, 4, 6, 8, 10.")
    p.add_argument("--checkpoint_root", type=Path,
                   default=PROJECT_ROOT / "outputs/whisper_largev2/english",
                   help="Folder holding baseline/ and ablation_NL/ checkpoint subfolders.")
    p.add_argument("--baseline_checkpoint", type=Path, default=None,
                   help="Unpruned baseline checkpoint. Defaults to "
                        "{checkpoint_root}/baseline/checkpoint_best_wer.pt.")
    p.add_argument("--checkpoint_overrides", nargs="*", default=[],
                   help="Per-depth overrides, format 'depth=/abs/path/checkpoint.pt'. Example: "
                        "--checkpoint_overrides 2=/scratch/ckpts/depth2.pt 4=/scratch/ckpts/depth4.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", choices=sorted(DATASETS.keys()), default="fairspeech",
                   help="Which evaluation dataset to point evaluate_subgroup_wer.py at. "
                        "Pick from the registry at the top of this script.")
    p.add_argument("--device", default=None,
                   help="Pass to evaluate_subgroup_wer.py. If unset, that script picks its own default. "
                        "Prefer CUDA_VISIBLE_DEVICES=N instead so each child sees just one GPU.")
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override eval.batch_size for every depth (raise it for smaller "
                        "models to use the GPU better). Passed through to evaluate_subgroup_wer.py.")
    p.add_argument("--cv_test_tsv", type=Path, default=None)
    p.add_argument("--per_seed_dir", type=Path, default=None,
                   help="Override where per-group summary CSVs are written. Set a "
                        "dataset-specific folder so a Fair-Speech sweep does not overwrite a "
                        "Common Voice sweep that uses the same d{NN}_keep{NN} condition names.")
    p.add_argument("--per_utt_dir", type=Path, default=None,
                   help="Override where per-utterance CSVs are written (one per depth, named "
                        "{condition}_seed{seed}.csv). Like --per_seed_dir, set this per dataset to "
                        "avoid filename collisions across sweeps.")
    p.add_argument("--continue_on_error", action="store_true",
                   help="Don't stop the sweep when one depth fails.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the commands that would run without executing them.")
    p.add_argument("--gap_seconds", type=int, default=10,
                   help="Pause between depths to let CUDA settle.")
    return p.parse_args()


def _condition_name(prune_depth: int, layers_kept: int) -> str:
    """Sortable condition tag, used as the output filename."""
    return f"d{prune_depth:02d}_keep{layers_kept:02d}"


def _eval_config_for(depth: int, model_dir: Path, total_layers: int) -> Path:
    if depth == 0:
        return model_dir / "baseline.yaml"
    layers_kept = total_layers - depth
    return model_dir / f"ablation_{layers_kept}L.yaml"


def _checkpoint_for(depth: int, args, overrides: dict[int, Path]) -> Path:
    if depth in overrides:
        return overrides[depth]
    if depth == 0:
        return args.baseline_checkpoint
    layers_kept = args.total_layers - depth
    return args.checkpoint_root / f"ablation_{layers_kept}L" / "checkpoint_best_wer.pt"


def _parse_overrides(items: list[str]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--checkpoint_overrides item must be 'depth=path', got: {item!r}")
        depth_s, path_s = item.split("=", 1)
        out[int(depth_s)] = Path(path_s)
    return out


def main():
    args = parse_args()
    overrides = _parse_overrides(args.checkpoint_overrides)
    depths = args.depths if args.depths is not None else [0, 2, 4, 6, 8, 10]
    if args.baseline_checkpoint is None:
        args.baseline_checkpoint = args.checkpoint_root / "baseline" / "checkpoint_best_wer.pt"

    if not DEFAULT_EVAL_SCRIPT.exists():
        raise SystemExit(f"Missing eval script: {DEFAULT_EVAL_SCRIPT}")

    dataset_cfg = DATASETS[args.dataset]
    hf_dataset_path = PROJECT_ROOT / dataset_cfg["hf_dataset_path"]
    if args.dataset != "cv22" and not hf_dataset_path.exists():
        print(f"[Sweep] WARNING: HF dataset for '{args.dataset}' not found at {hf_dataset_path}. "
              f"Run the matching builder under datamodule/hf_{args.dataset}.py first.")

    print(f"[Sweep] dataset:        {args.dataset}  ({hf_dataset_path})")
    print(f"[Sweep] depths:         {depths}")
    print(f"[Sweep] eval script:    {DEFAULT_EVAL_SCRIPT}")
    print(f"[Sweep] demographics:   {dataset_cfg['demographic_source']}"
          + (f", columns={dataset_cfg['demographic_columns']}" if dataset_cfg['demographic_columns'] else ""))

    # plan first, so missing checkpoints show up before any GPU work
    plan: list[dict] = []
    for d in depths:
        cfg = _eval_config_for(d, args.model_dir, args.total_layers)
        ckpt = _checkpoint_for(d, args, overrides)
        layers_kept = args.total_layers - d
        condition = _condition_name(d, layers_kept)
        plan.append({"depth": d, "layers_kept": layers_kept, "config": cfg,
                     "checkpoint": ckpt, "condition": condition})

    print(f"\n{'depth':>5} {'kept':>5}  {'condition':<14} {'config':<60} {'ckpt exists':<11}")
    for p_ in plan:
        print(f"{p_['depth']:>5d} {p_['layers_kept']:>5d}  {p_['condition']:<14} "
              f"{str(p_['config']):<60} {'yes' if p_['checkpoint'].exists() else 'NO':<11}")

    missing = [p_ for p_ in plan if not p_["checkpoint"].exists()]
    if missing and not args.dry_run:
        print(f"\n[Sweep] WARNING: {len(missing)} checkpoint(s) missing; those depths will be skipped.")

    if args.dry_run:
        print("\n[Sweep] DRY RUN: no commands executed.")
        return

    results: list[dict] = []
    for p_ in plan:
        if not p_["checkpoint"].exists():
            print(f"\n[Sweep] SKIP {p_['condition']}: checkpoint missing: {p_['checkpoint']}")
            results.append({"depth": p_["depth"], "status": "skip-missing-ckpt"})
            continue

        cmd = [
            sys.executable, str(DEFAULT_EVAL_SCRIPT),
            "--config", str(p_["config"]),
            "--checkpoint_path", str(p_["checkpoint"]),
            "--prune_depth", str(p_["depth"]),
            "--seed", str(args.seed),
            "--condition", p_["condition"],
            "--n_bootstrap", str(args.n_bootstrap),
            "--hf_dataset_path", str(hf_dataset_path),
            "--demographic_source", dataset_cfg["demographic_source"],
        ]
        if dataset_cfg["demographic_columns"]:
            cmd += ["--demographic_columns", *dataset_cfg["demographic_columns"]]
        if args.cv_test_tsv:
            cmd += ["--cv_test_tsv", str(args.cv_test_tsv)]
        if args.per_seed_dir:
            cmd += ["--per_seed_dir", str(args.per_seed_dir)]
        if args.per_utt_dir:
            cmd += ["--output_path", str(args.per_utt_dir / f"{p_['condition']}_seed{args.seed}.csv")]
        if args.batch_size:
            cmd += ["--batch_size", str(args.batch_size)]
        if args.device:
            cmd += ["--device", args.device]

        print(f"\n[Sweep] ===== depth={p_['depth']} kept={p_['layers_kept']} =====")
        print("        " + " ".join(cmd))
        start = time.time()
        try:
            subprocess.run(cmd, check=True)
            status = "ok"
        except subprocess.CalledProcessError as e:
            status = f"failed ({e.returncode})"
            print(f"[Sweep] FAILED depth={p_['depth']}: return code {e.returncode}")
            if not args.continue_on_error:
                print("[Sweep] Stopping sweep. Pass --continue_on_error to keep going.")
                results.append({"depth": p_["depth"], "status": status,
                                 "wall_time_s": time.time() - start})
                break

        wall = time.time() - start
        results.append({"depth": p_["depth"], "status": status, "wall_time_s": wall})
        print(f"[Sweep] depth={p_['depth']} done in {wall/60:.1f} min, status={status}")

        # let CUDA settle before the next depth
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        if args.gap_seconds > 0:
            time.sleep(args.gap_seconds)

    print(f"\n{'='*60}\nSWEEP SUMMARY ({len(results)} runs)\n{'='*60}")
    print(f"{'depth':>5}  {'status':<20}  {'wall_time':>10}")
    for r in results:
        wt = f"{r.get('wall_time_s', 0)/60:.1f} min" if r.get("wall_time_s") else "n/a"
        print(f"{r['depth']:>5d}  {r['status']:<20}  {wt:>10}")


if __name__ == "__main__":
    main()
