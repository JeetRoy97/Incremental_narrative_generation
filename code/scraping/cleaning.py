#!/usr/bin/env python3
"""
lecture_pipeline.py

End-to-end pipeline for building an incremental lecture-narration dataset
from CSV/TSV slide-transcript pairs:

  1. Clean slide text (dedup, remove headers/footers, whitespace normalize)
  2. Clean transcript text (remove timestamps/disfluencies, punctuation normalize)
  3. Temporally align transcript segments to slides (per lecture, in order)
  4. Rewrite transcripts for narrative coherence via an LLM API (pluggable,
     defaults to a Gemini-style REST call)

Input:  CSV or TSV with columns (minimum):
    course_id, week, lecture_id, slide_id, slide_order, slide_text,
    transcript_text, timestamp_start, timestamp_end, language
  (timestamp_start / timestamp_end are optional; if absent, alignment
   falls back to slide_order)

Output: CSV/TSV with additional columns:
    slide_text_clean, transcript_text_clean, transcript_text_rewritten

Usage:
    python lecture_pipeline.py --input data.csv --output out.csv \
        --stages clean align rewrite --sep , --api-key-env GEMINI_API_KEY

    # skip the LLM call (e.g. no API key / offline dry run)
    python lecture_pipeline.py --input data.csv --output out.csv --stages clean align
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lecture_pipeline")

REQUIRED_COLUMNS = [
    "course_id", "week", "lecture_id", "slide_id", "slide_order",
    "slide_text", "transcript_text",
]
OPTIONAL_COLUMNS = ["timestamp_start", "timestamp_end", "language"]


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def load_data(path: str, sep: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV/TSV file, auto-detecting delimiter from extension if not given."""
    if sep is None:
        sep = "\t" if path.lower().endswith(".tsv") else ","
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")
    for c in OPTIONAL_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    log.info("Loaded %d rows from %s", len(df), path)
    return df


def save_data(df: pd.DataFrame, path: str, sep: Optional[str] = None) -> None:
    if sep is None:
        sep = "\t" if path.lower().endswith(".tsv") else ","
    df.to_csv(path, sep=sep, index=False)
    log.info("Wrote %d rows to %s", len(df), path)


# --------------------------------------------------------------------------
# Stage 1: Slide cleaning
# --------------------------------------------------------------------------

_HEADER_FOOTER_PATTERNS = [
    r"^\s*NPTEL\b.*$",
    r"^\s*Page\s*\d+(\s*/\s*\d+)?\s*$",
    r"^\s*Slide\s*\d+\s*$",
    r"^\s*www\.\S+\s*$",
    r"^\s*\d{1,2}/\d{1,2}/\d{2,4}\s*$",   # stray dates
]
_HEADER_FOOTER_RE = re.compile("|".join(_HEADER_FOOTER_PATTERNS), re.IGNORECASE)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_slide_text(text: str) -> str:
    if not text:
        return ""
    lines = text.split("\n")
    kept = [ln for ln in lines if not _HEADER_FOOTER_RE.match(ln.strip())]
    cleaned = "\n".join(kept)
    return _normalize_whitespace(cleaned)


def dedup_slides(df: pd.DataFrame, text_col: str = "slide_text_clean") -> pd.DataFrame:
    """Drop consecutive slides within the same lecture that are duplicates or
    near-empty, preserving the first occurrence and original ordering."""
    df = df.sort_values(["course_id", "lecture_id", "slide_order"]).reset_index(drop=True)

    def _is_low_content(t: str) -> bool:
        return len(re.sub(r"\W", "", t)) < 3

    keep_mask = []
    prev_key = None
    prev_text = None
    for _, row in df.iterrows():
        key = (row["course_id"], row["lecture_id"])
        text = row[text_col]
        is_dup = (key == prev_key) and (text == prev_text) and text != ""
        keep_mask.append(not is_dup and not _is_low_content(text))
        prev_key, prev_text = key, text
    n_dropped = len(df) - sum(keep_mask)
    if n_dropped:
        log.info("Dropped %d duplicate/low-content slides", n_dropped)
    return df[pd.Series(keep_mask, index=df.index)].reset_index(drop=True)


# --------------------------------------------------------------------------
# Stage 2: Transcript cleaning
# --------------------------------------------------------------------------

_TIMESTAMP_RE = re.compile(r"\[?\(?\b\d{1,2}:\d{2}(:\d{2})?\b\)?\]?")
_DISFLUENCY_RE = re.compile(
    r"\b(um+|uh+|erm+|you know|i mean|like,|so,\s*so|okay so)\b\s*",
    re.IGNORECASE,
)
_FRAGMENT_JOIN_RE = re.compile(r"(\w)-\s*\n\s*(\w)")  # hyphenated line-break fragments
_MULTI_PUNCT_RE = re.compile(r"([.?!]){2,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.?!;:])")


def clean_transcript_text(text: str) -> str:
    if not text:
        return ""
    text = _TIMESTAMP_RE.sub("", text)
    text = _FRAGMENT_JOIN_RE.sub(r"\1\2", text)
    text = _DISFLUENCY_RE.sub("", text)
    text = _MULTI_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _normalize_whitespace(text)
    # ensure sentence-terminal punctuation and capitalized start
    if text and text[-1] not in ".?!":
        text += "."
    if text:
        text = text[0].upper() + text[1:]
    return text


def filter_non_pedagogical(df: pd.DataFrame, text_col: str = "transcript_text_clean",
                            min_words: int = 4) -> pd.DataFrame:
    """Flag/remove segments that are conversational filler or too short to be
    informative (e.g. 'okay any questions', 'yes', 'thank you')."""
    filler_exact = {"yes", "no", "okay", "ok", "thanks", "thank you", "any questions"}

    def _is_filler(t: str) -> bool:
        words = t.strip().lower().rstrip(".?!").split()
        if len(words) == 0:
            return True
        if len(words) < min_words and " ".join(words) in filler_exact:
            return True
        return False

    mask = ~df[text_col].apply(_is_filler)
    n_dropped = (~mask).sum()
    if n_dropped:
        log.info("Filtered %d non-pedagogical/filler transcript segments", n_dropped)
    return df[mask].reset_index(drop=True)


# --------------------------------------------------------------------------
# Stage 3: Temporal alignment
# --------------------------------------------------------------------------

def _to_seconds(ts: str) -> Optional[float]:
    if not ts:
        return None
    parts = ts.strip().split(":")
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return None
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts[-3:]
    return h * 3600 + m * 60 + s


def align_transcripts_to_slides(df: pd.DataFrame) -> pd.DataFrame:
    """Merge transcript segments that fall within a slide's [start, end)
    window into a single ordered narration per slide. Falls back to
    slide_order-based 1:1 alignment when timestamps are unavailable."""
    df = df.copy()
    df["_start_sec"] = df["timestamp_start"].apply(_to_seconds)
    df["_end_sec"] = df["timestamp_end"].apply(_to_seconds)

    has_timestamps = df["_start_sec"].notna().any()
    out_rows = []

    for (course_id, lecture_id), group in df.groupby(["course_id", "lecture_id"], sort=False):
        group = group.sort_values("slide_order")
        if has_timestamps and group["_start_sec"].notna().all():
            # merge any transcript rows sharing the same slide_id/time-window
            for slide_id, sub in group.groupby("slide_id", sort=False):
                sub = sub.sort_values("_start_sec")
                merged_transcript = " ".join(
                    t for t in sub["transcript_text_clean"] if t
                )
                row = sub.iloc[0].copy()
                row["transcript_text_clean"] = merged_transcript
                out_rows.append(row)
        else:
            # already 1:1 per slide_order; keep as-is but ensure sort order
            for _, row in group.iterrows():
                out_rows.append(row)

    aligned = pd.DataFrame(out_rows).drop(columns=["_start_sec", "_end_sec"])
    aligned = aligned.sort_values(["course_id", "lecture_id", "slide_order"]).reset_index(drop=True)
    log.info("Aligned into %d slide-narration rows", len(aligned))
    return aligned


# --------------------------------------------------------------------------
# Stage 4: Narrative rewriting (pluggable LLM call)
# --------------------------------------------------------------------------




@dataclass
class RewriteConfig:
    api_key: Optional[str] = None
    model: str = "gemini-2.5-pro"
    endpoint: str = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    max_retries: int = 3
    retry_backoff_sec: float = 2.0
    request_timeout_sec: float = 30.0
    sleep_between_calls_sec: float = 0.0


def default_gemini_call(prompt: str, cfg: RewriteConfig) -> str:
    """Minimal Gemini REST call. Swap this out for your provider of choice
    (Anthropic, OpenAI, a local model, etc.) by passing a different
    `call_fn` to rewrite_narratives()."""
    import requests  # local import so the module works without the dep if unused

    url = cfg.endpoint.format(model=cfg.model)
    resp = requests.post(
        url,
        params={"key": cfg.api_key},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=cfg.request_timeout_sec,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def rewrite_narratives(
    df: pd.DataFrame,
    cfg: RewriteConfig,
    call_fn: Callable[[str, RewriteConfig], str] = default_gemini_call,
    slide_text_col: str = "slide_text_clean",
    transcript_col: str = "transcript_text_clean",
    out_col: str = "transcript_text_rewritten",
) -> pd.DataFrame:
    """Rewrite each transcript segment in lecture order, feeding the previous
    slide's rewritten narration as context so coherence/flow carries across
    slides (this is the "incremental" part of the pipeline)."""
    df = df.sort_values(["course_id", "lecture_id", "slide_order"]).reset_index(drop=True)
    rewritten = [""] * len(df)
    prev_key = None
    prev_context = ""

    for i, row in df.iterrows():
        key = (row["course_id"], row["lecture_id"])
        if key != prev_key:
            prev_context = ""
        prompt = REWRITE_PROMPT_TEMPLATE.format(
            prev_context=prev_context or "(start of lecture)",
            slide_text=row[slide_text_col],
            transcript_text=row[transcript_col],
        )

        result = row[transcript_col]  # fallback: keep original if all retries fail
        for attempt in range(1, cfg.max_retries + 1):
            try:
                result = call_fn(prompt, cfg)
                break
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "Rewrite failed for %s slide_order=%s (attempt %d/%d): %s",
                    key, row["slide_order"], attempt, cfg.max_retries, e,
                )
                if attempt < cfg.max_retries:
                    time.sleep(cfg.retry_backoff_sec * attempt)

        rewritten[i] = result
        prev_context = result
        prev_key = key
        if cfg.sleep_between_calls_sec:
            time.sleep(cfg.sleep_between_calls_sec)

    df[out_col] = rewritten
    return df


# --------------------------------------------------------------------------
# Pipeline orchestration
# --------------------------------------------------------------------------

def run_pipeline(
    df: pd.DataFrame,
    stages: list[str],
    rewrite_cfg: Optional[RewriteConfig] = None,
    call_fn: Callable[[str, RewriteConfig], str] = default_gemini_call,
) -> pd.DataFrame:
    if "clean" in stages:
        log.info("Stage: clean")
        df["slide_text_clean"] = df["slide_text"].apply(clean_slide_text)
        df["transcript_text_clean"] = df["transcript_text"].apply(clean_transcript_text)
        df = dedup_slides(df)
        df = filter_non_pedagogical(df)
    else:
        df["slide_text_clean"] = df.get("slide_text_clean", df["slide_text"])
        df["transcript_text_clean"] = df.get("transcript_text_clean", df["transcript_text"])

    if "align" in stages:
        log.info("Stage: align")
        df = align_transcripts_to_slides(df)

    if "rewrite" in stages:
        log.info("Stage: rewrite")
        cfg = rewrite_cfg or RewriteConfig()
        if not cfg.api_key:
            log.warning("No API key provided; skipping rewrite stage.")
        else:
            df = rewrite_narratives(df, cfg, call_fn=call_fn)

    return df


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Slide-to-lecture narration data pipeline")
    p.add_argument("--input", required=True, help="Path to input CSV/TSV")
    p.add_argument("--output", required=True, help="Path to write output CSV/TSV")
    p.add_argument("--sep", default=None, help="Delimiter override (default: auto by extension)")
    p.add_argument(
        "--stages", nargs="+", default=["clean", "align", "rewrite"],
        choices=["clean", "align", "rewrite"],
        help="Which pipeline stages to run, in order (default: all three)",
    )
    p.add_argument("--api-key-env", default="GEMINI_API_KEY",
                    help="Env var holding the LLM API key (used only for the rewrite stage)")
    p.add_argument("--model", default="gemini-2.5-pro", help="Model name for the rewrite stage")
    p.add_argument("--sleep-between-calls", type=float, default=0.0,
                    help="Seconds to sleep between rewrite API calls (rate limiting)")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    df = load_data(args.input, sep=args.sep)

    rewrite_cfg = RewriteConfig(
        api_key=os.environ.get(args.api_key_env),
        model=args.model,
        sleep_between_calls_sec=args.sleep_between_calls,
    )

    out = run_pipeline(df, stages=args.stages, rewrite_cfg=rewrite_cfg)
    save_data(out, args.output, sep=args.sep)


if __name__ == "__main__":
    main()