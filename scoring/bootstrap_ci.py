"""Corpus WER/CER with utterance-level bootstrap confidence intervals."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

try:
    import jiwer
except ImportError as e:
    raise SystemExit("jiwer is required. Install it with `pip install jiwer`.") from e


WORD_TRANSFORM = jiwer.Compose([
    jiwer.SubstituteRegexes({
        r"[<\[][^>\]]*[>\]]": "",
        r"\(([^)]+?)\)": "",
    }),
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

CHAR_TRANSFORM = jiwer.Compose([
    jiwer.SubstituteRegexes({
        r"[<\[][^>\]]*[>\]]": "",
        r"\(([^)]+?)\)": "",
    }),
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfChars(),
])


def _to_lists(seq) -> List[str]:
    if isinstance(seq, np.ndarray):
        return [str(x) for x in seq.tolist()]
    return list(seq)


def _per_utt_stats(
    references: Sequence[str],
    hypotheses: Sequence[str],
    unit: str = "word",
    transform=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-utterance (errors, reference tokens), word- or character-level."""
    if unit not in {"word", "char"}:
        raise ValueError(f"unit must be 'word' or 'char', got {unit!r}")

    if transform is None:
        transform = WORD_TRANSFORM if unit == "word" else CHAR_TRANSFORM
    process = jiwer.process_words if unit == "word" else jiwer.process_characters

    refs = _to_lists(references)
    hyps = _to_lists(hypotheses)
    if len(refs) != len(hyps):
        raise ValueError(f"reference and hypothesis lengths differ: {len(refs)} vs {len(hyps)}")

    errors = np.zeros(len(refs), dtype=np.int64)
    n_tokens = np.zeros(len(refs), dtype=np.int64)
    for i, (r, h) in enumerate(zip(refs, hyps)):
        out = process(
            [r], [h],
            reference_transform=transform,
            hypothesis_transform=transform,
        )
        errors[i] = int(out.substitutions + out.deletions + out.insertions)
        n_tokens[i] = int(out.hits + out.substitutions + out.deletions)
    return errors, n_tokens


def _corpus_rate(errors: np.ndarray, n_tokens: np.ndarray) -> float:
    total = int(n_tokens.sum())
    if total == 0:
        return 0.0
    return float(errors.sum()) / float(total)


def metric_with_ci(
    references: Sequence[str],
    hypotheses: Sequence[str],
    unit: str = "word",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    transform=None,
) -> dict:
    """WER (unit='word') or CER (unit='char') with a percentile bootstrap CI.

    The point estimate comes from the full set, not from the mean of the
    resamples. Returns point, ci_low, ci_high, n_utts, n_bootstrap, alpha, unit,
    plus a `wer` (or `cer`) key aliasing point.
    """
    errors, n_tokens = _per_utt_stats(references, hypotheses, unit=unit, transform=transform)
    n = len(errors)
    point = _corpus_rate(errors, n_tokens)
    alias_key = "wer" if unit == "word" else "cer"

    if n == 0 or n_bootstrap <= 0:
        return {
            "point": point, alias_key: point,
            "ci_low": point, "ci_high": point,
            "n_utts": n, "n_bootstrap": 0, "alpha": alpha, "unit": unit,
        }

    rng = np.random.default_rng(seed)
    boots = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots[i] = _corpus_rate(errors[idx], n_tokens[idx])

    lo = float(np.percentile(boots, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "point": point, alias_key: point,
        "ci_low": lo, "ci_high": hi,
        "n_utts": n, "n_bootstrap": int(n_bootstrap), "alpha": alpha, "unit": unit,
    }


def wer_with_ci(references, hypotheses, n_bootstrap=1000, alpha=0.05, seed=42, transform=None):
    return metric_with_ci(references, hypotheses, unit="word",
                          n_bootstrap=n_bootstrap, alpha=alpha, seed=seed, transform=transform)


def cer_with_ci(references, hypotheses, n_bootstrap=1000, alpha=0.05, seed=42, transform=None):
    return metric_with_ci(references, hypotheses, unit="char",
                          n_bootstrap=n_bootstrap, alpha=alpha, seed=seed, transform=transform)
