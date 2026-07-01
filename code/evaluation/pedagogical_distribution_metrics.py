#!/usr/bin/env python3
"""
pedagogical_distribution_metrics.py

Two related evaluations from the paper:

(A) "Pedagogical distribution fidelity" (Section 6.2, Figure 2): does a
    model's mix of pedagogical roles (Definition/Concept/Explanation/...)
    match the distribution found in authentic human-led lectures?
        - Jensen-Shannon Divergence between gold and predicted tag
          distributions (lower = closer to human distribution)
        - Token-to-Tag Ratio: avg. tokens per pedagogical tag (higher =
          more verbose per instructional unit / worse pacing)
        - Tag Consistency: agreement between gold and predicted tags on
          aligned segments (Cohen's kappa)

(B) "Discourse history impact on coherence" (Section 6.3, Figure 3): how
    does conditioning on previous-slide history (H_{i-1}) affect referential
    consistency and long-range coherence?
        - DiscoScore (Zhao et al., 2023) as a coherence metric
        - Win Rate from pairwise preference judgments (human or LLM-judge),
          with vs. without discourse history

Usage:
    # (A) pedagogical distribution fidelity, from a CSV of aligned
    # gold-vs-predicted tags (one row per segment)
    python3 pedagogical_distribution_metrics.py distribution \\
        --input tags.csv --gold-col gold_label --pred-col pred_label \\
        --tokens-col pred_token_count --group-col language \\
        --output distribution_summary.csv

    # (B) discourse-history win rate, from pairwise judgments
    python3 pedagogical_distribution_metrics.py win-rate \\
        --input judgments.csv --winner-col winner \\
        --with-context-col used_context --group-col model \\
        --output winrate_summary.csv
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pedagogical_distribution_metrics")

PEDAGOGICAL_TAGS = [
    "Definition", "Concept", "Explanation", "Elaboration", "Example",
    "Contrast", "Recap", "Question", "Opening", "Organization",
]


# --------------------------------------------------------------------------
# (A.1) Jensen-Shannon Divergence between tag distributions
# --------------------------------------------------------------------------

def tag_distribution(tags: list[str], tag_set: list[str] = PEDAGOGICAL_TAGS, smoothing: float = 1e-6) -> np.ndarray:
    """Normalized frequency distribution over tag_set, with add-smoothing so
    JS divergence stays well-defined even if a tag never occurs."""
    counts = np.array([tags.count(t) for t in tag_set], dtype=float) + smoothing
    return counts / counts.sum()


def js_divergence(gold_tags: list[str], pred_tags: list[str], tag_set: list[str] = PEDAGOGICAL_TAGS) -> float:
    """Jensen-Shannon divergence (base-2, in [0, 1]) between the gold and
    predicted pedagogical-tag distributions for a set of segments."""
    from scipy.spatial.distance import jensenshannon
    p = tag_distribution(gold_tags, tag_set)
    q = tag_distribution(pred_tags, tag_set)
    js_dist = jensenshannon(p, q, base=2)  # returns sqrt(JS divergence)
    return float(js_dist ** 2)


# --------------------------------------------------------------------------
# (A.2) Token-to-Tag Ratio
# --------------------------------------------------------------------------

def token_to_tag_ratio(token_counts: list[int], tags: list[str]) -> float:
    """Average number of tokens produced per pedagogical tag/segment.
    High values indicate verbose, poorly-paced instructional units."""
    if not tags:
        return float("nan")
    return sum(token_counts) / len(tags)


# --------------------------------------------------------------------------
# (A.3) Tag Consistency (agreement on aligned gold/predicted segments)
# --------------------------------------------------------------------------

def tag_consistency(gold_tags: list[str], pred_tags: list[str]) -> dict:
    """Accuracy and Cohen's kappa between aligned gold/predicted tag
    sequences (same segmentation assumed; length mismatch is an error)."""
    if len(gold_tags) != len(pred_tags):
        raise ValueError(
            f"gold_tags and pred_tags must be aligned/same length "
            f"({len(gold_tags)} vs {len(pred_tags)}); align segments before calling this."
        )
    from sklearn.metrics import accuracy_score, cohen_kappa_score
    accuracy = accuracy_score(gold_tags, pred_tags)
    kappa = cohen_kappa_score(gold_tags, pred_tags)
    return {"accuracy": accuracy, "cohen_kappa": kappa}


# --------------------------------------------------------------------------
# (A) Orchestration: distribution fidelity summary (Figure 2 style)
# --------------------------------------------------------------------------

def summarize_distribution_fidelity(
    df: pd.DataFrame,
    gold_col: str,
    pred_col: str,
    tokens_col: Optional[str] = None,
    group_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    group_cols = group_cols or []
    rows = []
    groups = df.groupby(group_cols) if group_cols else [((), df)]
    for key, group in groups:
        key = key if isinstance(key, tuple) else (key,)
        gold_tags = list(group[gold_col])
        pred_tags = list(group[pred_col])
        row = dict(zip(group_cols, key))
        row["n_segments"] = len(group)
        row["js_divergence"] = js_divergence(gold_tags, pred_tags)
        if tokens_col:
            row["token_to_tag_ratio"] = token_to_tag_ratio(list(group[tokens_col]), pred_tags)
        try:
            consistency = tag_consistency(gold_tags, pred_tags)
            row["tag_consistency_acc"] = consistency["accuracy"]
            row["tag_consistency_kappa"] = consistency["cohen_kappa"]
        except ValueError as e:
            log.warning("Skipping tag consistency for group %s: %s", key, e)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# (B.1) DiscoScore wrapper
# --------------------------------------------------------------------------

def compute_discoscore(hypotheses: list[str], references: list[str]) -> list[float]:
    """Thin wrapper around the DiscoScore package (Zhao et al., 2023) for
    reference-based discourse coherence scoring. Requires:
        pip install disco-score
    and (like BERTScore) a one-time model download on first use."""
    try:
        from discoscore import DiscoScore  # package name per Zhao et al. reference impl
    except ImportError as e:
        raise ImportError(
            "DiscoScore is not installed. Run: pip install disco-score --break-system-packages "
            "(see https://github.com/AIPHES/DiscoScore for setup)."
        ) from e
    scorer = DiscoScore(model_name="bert-base-uncased")
    return [scorer.score(hyp, [ref]) for hyp, ref in zip(hypotheses, references)]


# --------------------------------------------------------------------------
# (B.2) Win Rate from pairwise judgments
# --------------------------------------------------------------------------

def compute_win_rate(
    df: pd.DataFrame,
    winner_col: str,
    condition_col: str,
    positive_label: str = "generation",
    group_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """`winner_col` should contain which side won a pairwise comparison
    (e.g. 'generation' vs 'reference', or 'with_context' vs 'without_context').
    Win rate = fraction of comparisons where positive_label won, per group
    (e.g. per model, per with/without-context condition)."""
    group_cols = group_cols or []
    all_group_cols = group_cols + [condition_col] if condition_col not in group_cols else group_cols
    rows = []
    groups = df.groupby(all_group_cols) if all_group_cols else [((), df)]
    for key, group in groups:
        key = key if isinstance(key, tuple) else (key,)
        win_rate = (group[winner_col] == positive_label).mean()
        row = dict(zip(all_group_cols, key))
        row["n_comparisons"] = len(group)
        row["win_rate"] = win_rate
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pedagogical distribution fidelity + discourse-history coherence metrics")
    sub = p.add_subparsers(dest="command", required=True)

    dist = sub.add_parser("distribution", help="JS divergence / Token-to-Tag Ratio / Tag Consistency")
    dist.add_argument("--input", required=True, help="CSV of aligned gold/predicted tag rows")
    dist.add_argument("--gold-col", default="gold_label")
    dist.add_argument("--pred-col", default="pred_label")
    dist.add_argument("--tokens-col", default=None, help="Column with predicted-segment token counts")
    dist.add_argument("--group-col", action="append", default=None, help="Repeatable; e.g. --group-col model --group-col language")
    dist.add_argument("--output", required=True)

    wr = sub.add_parser("win-rate", help="Win rate from pairwise preference judgments")
    wr.add_argument("--input", required=True, help="CSV of pairwise judgment rows")
    wr.add_argument("--winner-col", default="winner")
    wr.add_argument("--condition-col", default="used_context", help="Column marking with/without-context condition")
    wr.add_argument("--positive-label", default="generation")
    wr.add_argument("--group-col", action="append", default=None)
    wr.add_argument("--output", required=True)

    ds = sub.add_parser("discoscore", help="DiscoScore coherence between hypotheses and references")
    ds.add_argument("--input", required=True, help="CSV with hypothesis/reference columns")
    ds.add_argument("--hyp-col", default="hypothesis")
    ds.add_argument("--ref-col", default="reference")
    ds.add_argument("--output", required=True)

    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    df = pd.read_csv(args.input)

    if args.command == "distribution":
        result = summarize_distribution_fidelity(
            df, args.gold_col, args.pred_col, args.tokens_col, args.group_col,
        )
    elif args.command == "win-rate":
        result = compute_win_rate(
            df, args.winner_col, args.condition_col, args.positive_label, args.group_col,
        )
    elif args.command == "discoscore":
        scores = compute_discoscore(list(df[args.hyp_col]), list(df[args.ref_col]))
        result = df.copy()
        result["discoscore"] = scores
    else:
        raise SystemExit(f"Unknown command: {args.command}")

    result.to_csv(args.output, index=False)
    log.info("Wrote %d rows to %s", len(result), args.output)


if __name__ == "__main__":
    main()
