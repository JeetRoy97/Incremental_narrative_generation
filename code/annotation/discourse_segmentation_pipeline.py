#!/usr/bin/env python3
"""
discourse_segmentation_pipeline.py

Rhetorical segmentation + multi-model verification pipeline for lecture
transcripts, implementing the exact annotation prompts used in the paper:

  1. Primary annotator: segments a full lecture transcript (as indexed
     sentences s1, s2, ...) into contiguous rhetorical spans, each with a
     single dominant pedagogical label. Prompt = "Rhetorical Segmentation
     Annotation Guidelines" (Table 7).
  2. Verifier ensemble: audits each segment against a pedagogical risk
     rubric (boundary misalignment, role confusion, conceptual ambiguity,
     over-/under-segmentation, label drift), returning VALID/FLAGGED per
     segment. Prompt = "Multi-Model Verification and Pedagogical Quality
     Control" (Table 10).
  3. Routing: any segment FLAGGED by >=1 verifier goes to a human-review
     queue; everything else is accepted.

The pipeline is provider-agnostic: each model is a `ModelClient`
(name + `call_fn(prompt, cfg) -> str`), so "primary" and each "verifier"
can point at whatever API/local model you actually use (e.g. Gemini for
the primary annotator; GPT-4o-mini, LLaMA-3-8B, Gemma-3-12B, Qwen-3-8B as
verifiers).

Input:  CSV/TSV with one row per lecture:
    course_id, lecture_id, transcript_text
  (transcript_text is the full raw/cleaned lecture transcript for that
   lecture; it is sentence-split internally. If you already have
   sentence-level rows, pass --presegmented and use columns
   course_id, lecture_id, sentence_id, sentence_text instead.)

Few-shot examples: JSON file, list of:
    {
      "sentences": {"s1": "...", "s2": "...", ...},
      "segments": [
        {"segment_id": 1, "start": "s1", "end": "s3", "label": "Opening"},
        ...
      ]
    }

Output:
    <output>.csv          -- all segments: course_id, lecture_id, segment_id,
                              start, end, label, segment_text, status, issue_type,
                              justification, flagged_by
    <output>.review.csv   -- subset with status == FLAGGED (human review queue)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("discourse_segmentation")

LABEL_SET: dict[str, str] = {
    "Definition": "formal and precise meaning of a term",
    "Concept": "introduces or explains an idea informally",
    "Example": "specific instance illustrating a concept",
    "Explanation": "describes mechanisms or reasoning",
    "Elaboration": "expands or refines prior content",
    "Contrast": "compares differences between ideas",
    "Organization": "lecture flow management or transitions",
    "Recap": "summary of previously covered material",
    "Question": "explicit question posed",
    "Opening": "lecture introduction or objectives",
}
LABELS = list(LABEL_SET.keys())

ISSUE_TYPES = [
    "Boundary Misalignment", "Role Confusion", "Conceptual Ambiguity",
    "Over-segmentation", "Under-segmentation", "Label Drift", "None",
]


# --------------------------------------------------------------------------
# Model client abstraction (provider-agnostic)
# --------------------------------------------------------------------------

@dataclass
class RequestConfig:
    api_key: Optional[str] = None
    model: str = ""
    endpoint: str = ""
    max_retries: int = 3
    retry_backoff_sec: float = 2.0
    request_timeout_sec: float = 60.0
    extra: dict = field(default_factory=dict)


CallFn = Callable[[str, RequestConfig], str]


@dataclass
class ModelClient:
    name: str
    cfg: RequestConfig
    call_fn: CallFn

    def generate(self, prompt: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                return self.call_fn(prompt, self.cfg)
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning("[%s] call failed (attempt %d/%d): %s",
                            self.name, attempt, self.cfg.max_retries, e)
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_backoff_sec * attempt)
        raise RuntimeError(f"{self.name} failed after {self.cfg.max_retries} attempts") from last_err


def gemini_call(prompt: str, cfg: RequestConfig) -> str:
    import requests
    url = cfg.endpoint or f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.model}:generateContent"
    resp = requests.post(
        url, params={"key": cfg.api_key},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=cfg.request_timeout_sec,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def openai_compatible_call(prompt: str, cfg: RequestConfig) -> str:
    """Works for OpenAI and OpenAI-compatible endpoints (vLLM/Ollama/etc.
    serving LLaMA-3, Gemma-3, Qwen-3, ...)."""
    import requests
    url = cfg.endpoint or "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
    resp = requests.post(
        url, headers=headers,
        json={
            "model": cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": cfg.extra.get("temperature", 0.0),
        },
        timeout=cfg.request_timeout_sec,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


CALL_FN_REGISTRY: dict[str, CallFn] = {
    "gemini": gemini_call,
    "openai": openai_compatible_call,
    "local": openai_compatible_call,
}


def build_client(spec: str, api_key_env: str, endpoint_env: str) -> ModelClient:
    """spec format: 'provider:model_name', e.g. 'openai:gpt-4o-mini'."""
    provider, _, model = spec.partition(":")
    if provider not in CALL_FN_REGISTRY:
        raise ValueError(f"Unknown provider '{provider}' in '{spec}'. Choices: {list(CALL_FN_REGISTRY)}")
    cfg = RequestConfig(
        api_key=os.environ.get(api_key_env, ""),
        model=model or provider,
        endpoint=os.environ.get(endpoint_env, ""),
    )
    return ModelClient(name=spec, cfg=cfg, call_fn=CALL_FN_REGISTRY[provider])


# --------------------------------------------------------------------------
# Sentence splitting / indexing
# --------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.?!])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def index_sentences(sentences: list[str]) -> dict[str, str]:
    return {f"s{i+1}": s for i, s in enumerate(sentences)}


def format_indexed_block(indexed: dict[str, str]) -> str:
    return "\n".join(f"{sid}: {text}" for sid, text in indexed.items())


# --------------------------------------------------------------------------
# Stage 1: Rhetorical segmentation (Table 7 prompt, verbatim structure)
# --------------------------------------------------------------------------

SEGMENTATION_PROMPT_TEMPLATE = """You are an expert annotator for lecture understanding datasets. Your task is to
perform rhetorical segmentation of a lecture transcript. Rhetorical segmentation
refers to dividing a transcript into contiguous spans of sentences, where each
span corresponds to a single dominant pedagogical or communicative function.
This annotation will be used for downstream tasks such as lecture understanding,
summarization, and educational discourse analysis. Therefore, consistency,
coherence, and functional accuracy are critical.

INPUT FORMAT
A lecture transcript is provided as a sequence of indexed sentences:
s1: ...
s2: ...
s3: ...

ANNOTATION PROCESS (FOLLOW IN ORDER)
When performing segmentation, follow these mental steps:
1. Read the full transcript to understand global topic flow.
2. Identify major shifts in pedagogical function (e.g., definition -> explanation -> example).
3. Group consecutive sentences that serve the same communicative purpose.
4. Ensure each segment is semantically coherent and self-contained.
5. Assign the most dominant rhetorical label to each segment.
6. Verify that every sentence is assigned exactly once and that no overlaps exist.

SEGMENTATION PRINCIPLES
- Segments must be contiguous (only consecutive sentences allowed).
- Each sentence must belong to exactly one segment.
- Prefer coarse-grained segmentation unless a clear functional shift occurs.
- Do NOT split segments based on minor stylistic or lexical changes.
- Use semantic and discourse-level understanding, not surface keywords.

LABEL SET (choose EXACTLY ONE per segment)
{label_set_block}

QUALITY CONSTRAINTS
- Assign exactly ONE label per segment.
- Every sentence must be included in exactly one segment.
- Maintain original sentence order.
- Avoid unnecessary fragmentation.
- Ensure segments are pedagogically meaningful and coherent.

FEW-SHOT EXAMPLES
{few_shot_block}

OUTPUT FORMAT (STRICT)
Return a JSON list of segments:
[
  {{"segment_id": 1, "start": "s1", "end": "s5", "label": "<ONE LABEL>"}}
]

FINAL NOTES
- Ensure full coverage of the transcript (no missing sentences).
- Ensure strict sequential ordering.
- Output must be valid JSON, and nothing else.

TRANSCRIPT TO SEGMENT:
{sentences_block}"""


def _label_set_block() -> str:
    return "\n".join(f"- {label}: {desc}" for label, desc in LABEL_SET.items())


def _format_few_shot_examples(examples: list[dict]) -> str:
    if not examples:
        return "(no few-shot examples provided)"
    blocks = []
    for i, ex in enumerate(examples, start=1):
        sent_block = format_indexed_block(ex["sentences"])
        seg_block = json.dumps(ex["segments"], indent=2)
        blocks.append(f"Example {i}:\nInput sentences:\n{sent_block}\nSegmented output:\n{seg_block}")
    return "\n\n".join(blocks)


def load_few_shot_examples(path: Optional[str]) -> list[dict]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    for ex in examples:
        if "sentences" not in ex or "segments" not in ex:
            raise ValueError("Each few-shot example needs 'sentences' and 'segments' keys")
    return examples


def _extract_json(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    match = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in model output: {raw[:200]!r}")
    return json.loads(match.group(0))


def _validate_and_repair_segments(segments: list[dict], indexed: dict[str, str]) -> list[dict]:
    """Ensure full, non-overlapping, sequential sentence coverage. Drops
    segments referencing unknown sentence ids; fills any coverage gaps by
    appending an 'Unlabeled' segment so no sentence is silently lost."""
    all_ids = list(indexed.keys())
    id_pos = {sid: i for i, sid in enumerate(all_ids)}
    valid = []
    for seg in segments:
        if seg.get("start") in id_pos and seg.get("end") in id_pos and id_pos[seg["end"]] >= id_pos[seg["start"]]:
            if seg.get("label") not in LABELS:
                seg["label"] = "Unlabeled"
            valid.append(seg)
        else:
            log.warning("Dropping malformed segment: %s", seg)

    valid.sort(key=lambda s: id_pos[s["start"]])
    covered = [False] * len(all_ids)
    for seg in valid:
        for i in range(id_pos[seg["start"]], id_pos[seg["end"]] + 1):
            covered[i] = True

    repaired = list(valid)
    gap_start = None
    for i, is_covered in enumerate(covered + [True]):  # sentinel to flush trailing gap
        if not is_covered and gap_start is None:
            gap_start = i
        elif is_covered and gap_start is not None:
            repaired.append({
                "segment_id": f"gap_{gap_start}",
                "start": all_ids[gap_start], "end": all_ids[i - 1], "label": "Unlabeled",
            })
            gap_start = None

    repaired.sort(key=lambda s: id_pos[s["start"]])
    for i, seg in enumerate(repaired, start=1):
        seg["segment_id"] = i
    return repaired


def segment_transcript(
    sentences: list[str],
    primary_client: ModelClient,
    few_shot_examples: list[dict],
) -> list[dict]:
    indexed = index_sentences(sentences)
    prompt = SEGMENTATION_PROMPT_TEMPLATE.format(
        label_set_block=_label_set_block(),
        few_shot_block=_format_few_shot_examples(few_shot_examples),
        sentences_block=format_indexed_block(indexed),
    )
    raw = primary_client.generate(prompt)
    try:
        segments = _extract_json(raw)
    except (ValueError, json.JSONDecodeError) as e:
        log.error("Failed to parse segmentation output, falling back to one segment: %s", e)
        segments = [{"segment_id": 1, "start": "s1", "end": f"s{len(sentences)}", "label": "Unlabeled"}]
    segments = _validate_and_repair_segments(segments, indexed)
    for seg in segments:
        span = [indexed[f"s{i}"] for i in range(
            int(seg["start"][1:]), int(seg["end"][1:]) + 1)]
        seg["segment_text"] = " ".join(span)
    return segments


# --------------------------------------------------------------------------
# Stage 2: Multi-model verification (Table 10 prompt, verbatim structure)
# --------------------------------------------------------------------------

VERIFICATION_PROMPT_TEMPLATE = """You are an expert evaluator for lecture discourse segmentation quality assurance.
Your task is to verify the correctness of segment-level annotations produced
by an automated system.
These annotations are used for downstream educational understanding tasks,
so high precision in segmentation boundaries and label assignment is required.

INPUT:
A segmented lecture transcript, where each segment contains:
- start sentence index
- end sentence index
- rhetorical label
- the sentences of that segment (for context, adjacent segments are also shown)

{segments_block}

VERIFICATION OBJECTIVE:
Assess whether each segment is:
1. Semantically coherent
2. Correctly bounded (no boundary misalignment)
3. Correctly labeled with respect to dominant pedagogical function
4. Free from conceptual ambiguity or role confusion

PEDAGOGICAL RISK RUBRIC (ERROR MODES TO DETECT):
- Boundary Misalignment: incorrect start/end sentence grouping
- Role Confusion: incorrect label assignment (e.g., Explanation vs Concept)
- Conceptual Ambiguity: unclear or mixed pedagogical function within a segment
- Over-segmentation: unnecessary splitting of coherent discourse
- Under-segmentation: merging distinct rhetorical functions
- Label Drift: label does not match dominant discourse intent

VERIFICATION PROCESS (FOLLOW IN ORDER):
1. Read the full segment and surrounding context.
2. Check coherence of sentence span.
3. Validate whether the label matches the dominant rhetorical function.
4. Compare against adjacent segments for consistency.
5. Flag any segment violating the rubric.

OUTPUT REQUIREMENTS:
For each segment, return:
- segment id
- status: VALID, FLAGGED
- issue type (if FLAGGED)
- short justification
If no issues are found, mark as VALID.

Return ONLY a JSON list, nothing else:
[
  {{"segment_id": 1, "status": "VALID", "issue_type": "None", "justification": "..."}}
]"""


def _format_segments_for_verification(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        lines.append(
            f'Segment {seg["segment_id"]}: start={seg["start"]}, end={seg["end"]}, '
            f'label={seg["label"]}\nText: "{seg["segment_text"]}"\n'
        )
    return "\n".join(lines)


def _parse_verification_output(raw: str, n_segments: int) -> list[dict]:
    try:
        parsed = _extract_json(raw)
    except (ValueError, json.JSONDecodeError) as e:
        log.error("Failed to parse verifier output: %s", e)
        parsed = []
    by_id = {}
    for item in parsed:
        sid = item.get("segment_id")
        by_id[sid] = {
            "status": item.get("status", "FLAGGED") if item.get("status") in ("VALID", "FLAGGED") else "FLAGGED",
            "issue_type": item.get("issue_type", "None"),
            "justification": item.get("justification", ""),
        }
    # any segment the verifier didn't return a verdict for is treated as FLAGGED
    # (fail-safe: silence from a verifier should not silently pass a segment)
    return [
        by_id.get(i, {"status": "FLAGGED", "issue_type": "verifier_no_response", "justification": ""})
        for i in range(1, n_segments + 1)
    ]


def verify_segments(segments: list[dict], verifiers: list[ModelClient]) -> list[dict]:
    segments_block = _format_segments_for_verification(segments)
    prompt = VERIFICATION_PROMPT_TEMPLATE.format(segments_block=segments_block)

    per_verifier_results: dict[str, list[dict]] = {}
    for verifier in verifiers:
        try:
            raw = verifier.generate(prompt)
            per_verifier_results[verifier.name] = _parse_verification_output(raw, len(segments))
        except Exception as e:  # noqa: BLE001
            log.error("Verifier %s failed entirely: %s", verifier.name, e)
            per_verifier_results[verifier.name] = [
                {"status": "FLAGGED", "issue_type": "verifier_error", "justification": str(e)}
                for _ in segments
            ]

    for idx, seg in enumerate(segments):
        votes = {name: results[idx] for name, results in per_verifier_results.items()}
        flagged_by = [name for name, v in votes.items() if v["status"] == "FLAGGED"]
        seg["verifier_votes"] = json.dumps(votes)
        seg["flagged_by"] = ";".join(flagged_by)
        seg["status"] = "FLAGGED" if flagged_by else "VALID"
        # surface the first flagging verifier's issue/justification for convenience
        if flagged_by:
            first = votes[flagged_by[0]]
            seg["issue_type"] = first["issue_type"]
            seg["justification"] = first["justification"]
        else:
            seg["issue_type"] = "None"
            seg["justification"] = ""
    return segments


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_pipeline_for_lecture(
    course_id: str,
    lecture_id: str,
    transcript_text: str,
    primary_client: ModelClient,
    verifiers: list[ModelClient],
    few_shot_examples: list[dict],
) -> pd.DataFrame:
    sentences = split_sentences(transcript_text)
    if not sentences:
        return pd.DataFrame()
    segments = segment_transcript(sentences, primary_client, few_shot_examples)
    segments = verify_segments(segments, verifiers)
    df = pd.DataFrame(segments)
    df.insert(0, "lecture_id", lecture_id)
    df.insert(0, "course_id", course_id)
    return df


def run_pipeline(
    df: pd.DataFrame,
    primary_client: ModelClient,
    verifiers: list[ModelClient],
    few_shot_examples: list[dict],
    text_col: str = "transcript_text",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    for _, row in df.iterrows():
        log.info("Segmenting lecture %s / %s", row["course_id"], row["lecture_id"])
        lecture_df = run_pipeline_for_lecture(
            row["course_id"], row["lecture_id"], row[text_col],
            primary_client, verifiers, few_shot_examples,
        )
        all_rows.append(lecture_df)

    result = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if result.empty:
        return result, result

    accepted = result[result["status"] == "VALID"].reset_index(drop=True)
    review = result[result["status"] == "FLAGGED"].reset_index(drop=True)
    log.info("Segmentation complete: %d segments total, %d flagged for human review",
              len(result), len(review))
    return accepted, review


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def load_data(path: str, sep: Optional[str] = None) -> pd.DataFrame:
    if sep is None:
        sep = "\t" if path.lower().endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)


def save_data(df: pd.DataFrame, path: str, sep: Optional[str] = None) -> None:
    if sep is None:
        sep = "\t" if path.lower().endswith(".tsv") else ","
    df.to_csv(path, sep=sep, index=False)
    log.info("Wrote %d rows to %s", len(df), path)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rhetorical segmentation + multi-model verification pipeline")
    p.add_argument("--input", required=True, help="CSV/TSV with course_id, lecture_id, transcript_text (one row per lecture)")
    p.add_argument("--output", required=True, help="Path to write accepted (VALID) segments CSV/TSV")
    p.add_argument("--sep", default=None)
    p.add_argument("--text-col", default="transcript_text")
    p.add_argument("--few-shot-examples", default=None, help="JSON file of few-shot segmentation examples")
    p.add_argument("--primary", default="gemini:gemini-2.5-pro",
                    help="Primary annotator as 'provider:model'")
    p.add_argument("--primary-api-key-env", default="GEMINI_API_KEY")
    p.add_argument("--primary-endpoint-env", default="PRIMARY_ENDPOINT")
    p.add_argument("--verifiers", nargs="+",
                    default=["openai:gpt-4o-mini", "local:llama-3-8b", "local:gemma-3-12b", "local:qwen-3-8b"],
                    help="Verifier models as 'provider:model' entries")
    p.add_argument("--verifier-api-key-env", default="VERIFIER_API_KEY")
    p.add_argument("--verifier-endpoint-env", default="VERIFIER_ENDPOINT")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    df = load_data(args.input, sep=args.sep)
    few_shot_examples = load_few_shot_examples(args.few_shot_examples)

    primary_client = build_client(args.primary, args.primary_api_key_env, args.primary_endpoint_env)
    verifiers = [
        build_client(spec, args.verifier_api_key_env, args.verifier_endpoint_env)
        for spec in args.verifiers
    ]

    accepted, review = run_pipeline(df, primary_client, verifiers, few_shot_examples, text_col=args.text_col)

    save_data(accepted, args.output, sep=args.sep)
    review_path = re.sub(r"(\.\w+)$", r".review\1", args.output)
    save_data(review, review_path, sep=args.sep)


if __name__ == "__main__":
    main()
