#!/usr/bin/env python3
"""
content_fidelity_metrics.py

"Slide Coverage and Pedagogical Expansion" evaluation (paper Section 6.1,
Table 1 / Table 2): how well a generated narration covers the source slide
content while adding pedagogically useful elaboration.

Metrics:
    - ROUGE-1 / ROUGE-2 / ROUGE-L, reported as both Recall and F1
      (high recall + low F1 = verbose/over-generating; the paper uses this
      exact pattern to diagnose "linguistic recall without pedagogical
      precision")
    - METEOR
    - BERTScore (precision/recall/F1)

Usage (reproduces a Table-1-style per-model, per-language breakdown):
    python3 content_fidelity_metrics.py --input generations.csv \\
        --pred-col generation --ref-col slide_text \\
        --model-col model --lang-col language --output table1.csv

Input CSV must have: a prediction column (model's generated narration),
a reference column (slide/source text it should cover), and optionally
model/language columns for grouped reporting.

Delta table (Table 2 style: score change after adding pedagogy signals):
    python3 content_fidelity_metrics.py --diff \\
        --baseline baseline_generations.csv --treatment pedagogy_generations.csv \\
        --pred-col generation --ref-col slide_text --model-col model \\
        --output table2_delta.csv
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("content_fidelity_metrics")


# --------------------------------------------------------------------------
# ROUGE-1/2/L (Recall + F1)
# --------------------------------------------------------------------------

def compute_rouge(pred: str, ref: str) -> dict:
    """Returns rouge1/rouge2/rougeL, each as {'recall': ..., 'f1': ...}."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ref, pred)  # rouge_scorer.score(target, prediction)
    return {
        f"rouge{k[5:]}": {"recall": v.recall, "f1": v.fmeasure}
        for k, v in scores.items()
    }


# --------------------------------------------------------------------------
# METEOR
# --------------------------------------------------------------------------

_METEOR_READY = False


def _ensure_nltk_data() -> None:
    global _METEOR_READY
    if _METEOR_READY:
        return
    import nltk
    for pkg in ("wordnet", "omw-1.4", "punkt", "punkt_tab"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not fetch NLTK resource '%s': %s", pkg, e)
    _METEOR_READY = True


def compute_meteor(pred: str, ref: str) -> float:
    _ensure_nltk_data()
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    try:
        pred_toks = word_tokenize(pred)
        ref_toks = word_tokenize(ref)
    except LookupError:
        pred_toks, ref_toks = pred.split(), ref.split()
    return meteor_score([ref_toks], pred_toks)


# --------------------------------------------------------------------------
# BERTScore
# --------------------------------------------------------------------------

_BERTSCORE_MODELS = {
    "en": "roberta-large",
    "hi": "bert-base-multilingual-cased",
    "bn": "bert-base-multilingual-cased",
}


def compute_bertscore(preds: list[str], refs: list[str], lang: str = "en") -> dict:
    """Batched (much faster than per-example). Requires downloading a
    pretrained model from Hugging Face on first use -- needs network access
    to huggingface.co."""
    try:
        from bert_score import score as bert_score_fn
    except ImportError as e:
        raise ImportError(
            "bert_score is not installed. Run: pip install bert_score --break-system-packages"
        ) from e
    model_type = _BERTSCORE_MODELS.get(lang, "bert-base-multilingual-cased")
    try:
        P, R, F1 = bert_score_fn(preds, refs, model_type=model_type, lang=lang, verbose=False)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"BERTScore failed (often a model-download/network issue): {e}"
        ) from e
    return {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}


# --------------------------------------------------------------------------
# Per-row and aggregate evaluation
# --------------------------------------------------------------------------

def evaluate_pair(pred: str, ref: str) -> dict:
    """Row-level scores (ROUGE + METEOR; BERTScore is batched separately
    since it's far cheaper to run once over the whole set)."""
    rouge = compute_rouge(pred, ref)
    meteor = compute_meteor(pred, ref)
    return {
        "rouge1_recall": rouge["rouge1"]["recall"], "rouge1_f1": rouge["rouge1"]["f1"],
        "rouge2_recall": rouge["rouge2"]["recall"], "rouge2_f1": rouge["rouge2"]["f1"],
        "rougeL_recall": rouge["rougeL"]["recall"], "rougeL_f1": rouge["rougeL"]["f1"],
        "meteor": meteor,
    }


def evaluate_dataframe(
    df: pd.DataFrame,
    pred_col: str,
    ref_col: str,
    lang_col: Optional[str] = None,
    compute_bs: bool = True,
) -> pd.DataFrame:
    """Adds per-row rouge/meteor columns and, if compute_bs, a bertscore_f1
    column (computed in a single batched call per language group)."""
    rows = [evaluate_pair(p, r) for p, r in zip(df[pred_col], df[ref_col])]
    scored = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    if compute_bs:
        scored["bertscore_precision"] = None
        scored["bertscore_recall"] = None
        scored["bertscore_f1"] = None
        groups = scored.groupby(lang_col) if lang_col else [(None, scored)]
        for lang, group in groups:
            try:
                from bert_score import score as bert_score_fn
                model_type = _BERTSCORE_MODELS.get(lang, "bert-base-multilingual-cased") if lang else "roberta-large"
                P, R, F1 = bert_score_fn(
                    list(group[pred_col]), list(group[ref_col]),
                    model_type=model_type, lang=lang or "en", verbose=False,
                )
                scored.loc[group.index, "bertscore_precision"] = P.tolist()
                scored.loc[group.index, "bertscore_recall"] = R.tolist()
                scored.loc[group.index, "bertscore_f1"] = F1.tolist()
            except Exception as e:  # noqa: BLE001
                log.warning("BERTScore skipped for group '%s' (likely no network access to download "
                            "the model): %s", lang, e)
    return scored


def summarize_table1(
    scored_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Collapses row-level scores into the Table-1 style summary: mean
    R1/R2/RL (recall & f1), mean METEOR, mean BERTScore F1, per group
    (e.g. per model x language)."""
    agg_map = {
        "rouge1_recall": "mean", "rouge1_f1": "mean",
        "rouge2_recall": "mean", "rouge2_f1": "mean",
        "rougeL_recall": "mean", "rougeL_f1": "mean",
        "meteor": "mean",
    }
    if "bertscore_f1" in scored_df.columns:
        agg_map["bertscore_f1"] = "mean"
    summary = scored_df.groupby(group_cols).agg(agg_map).reset_index()
    summary = summary.rename(columns={"bertscore_f1": "bertscore"})
    return summary


def summarize_table2_delta(
    baseline_scored: pd.DataFrame,
    treatment_scored: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Table-2 style delta: treatment score minus baseline score, per group,
    for R1/R2/RL (recall & f1), METEOR, BERTScore."""
    base_summary = summarize_table1(baseline_scored, group_cols).set_index(group_cols)
    treat_summary = summarize_table1(treatment_scored, group_cols).set_index(group_cols)
    delta = (treat_summary - base_summary).add_prefix("delta_").reset_index()
    return delta


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Content fidelity (ROUGE/METEOR/BERTScore) evaluation")
    p.add_argument("--input", help="CSV with prediction/reference columns (single-run mode)")
    p.add_argument("--diff", action="store_true", help="Compute Table-2 style delta between two runs")
    p.add_argument("--baseline", help="CSV for baseline run (--diff mode)")
    p.add_argument("--treatment", help="CSV for treatment run (--diff mode)")
    p.add_argument("--pred-col", default="prediction")
    p.add_argument("--ref-col", default="reference")
    p.add_argument("--model-col", default=None)
    p.add_argument("--lang-col", default=None)
    p.add_argument("--no-bertscore", action="store_true", help="Skip BERTScore (no network / faster iteration)")
    p.add_argument("--output", required=True)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    group_cols = [c for c in (args.model_col, args.lang_col) if c]

    if args.diff:
        if not args.baseline or not args.treatment:
            raise SystemExit("--diff requires --baseline and --treatment")
        base_df = pd.read_csv(args.baseline)
        treat_df = pd.read_csv(args.treatment)
        base_scored = evaluate_dataframe(base_df, args.pred_col, args.ref_col, args.lang_col, compute_bs=not args.no_bertscore)
        treat_scored = evaluate_dataframe(treat_df, args.pred_col, args.ref_col, args.lang_col, compute_bs=not args.no_bertscore)
        result = summarize_table2_delta(base_scored, treat_scored, group_cols)
    else:
        if not args.input:
            raise SystemExit("--input is required unless --diff is set")
        df = pd.read_csv(args.input)
        scored = evaluate_dataframe(df, args.pred_col, args.ref_col, args.lang_col, compute_bs=not args.no_bertscore)
        result = summarize_table1(scored, group_cols) if group_cols else scored

    result.to_csv(args.output, index=False)
    log.info("Wrote %d rows to %s", len(result), args.output)


if __name__ == "__main__":
    main()
