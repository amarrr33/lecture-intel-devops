"""Generation pipeline: text cleaning, fusion, material generation, and output formatting."""
from typing import Dict, Any, List, Optional
import re
import json
import logging

from app.providers.base_provider import BaseProvider
from app.json_utils import extract_json, safe_json_loads
from app.dedup import deduplicate_by_embedding

logger = logging.getLogger(__name__)

# ---------------------------
# Text cleaning
# ---------------------------

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t).strip()

    # Fix common OCR typos
    t = re.sub(r"\bscalears\b", "scalars", t, flags=re.I)

    # De-dup adjacent repeated fragments
    parts = [p.strip() for p in re.split(r"[|\u2022\n]+", t) if p.strip()]
    dedup: List[str] = []
    for p in parts:
        if not dedup or p.lower() != dedup[-1].lower():
            dedup.append(p)

    t = " ".join(dedup)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ---------------------------
# Quality check (fixed: no longer over-aggressive)
# ---------------------------

_PROMPT_ECHO_RE = re.compile(
    r"^\s*#{1,3}\s*(role|task|rules|constraints|instructions)\b",
    re.IGNORECASE | re.MULTILINE,
)


def looks_bad(out: str) -> bool:
    """Check if LLM output is empty, trivial, or prompt echo."""
    o = (out or "").strip()
    if not o:
        return True
    if len(o) < 60:
        return True
    if o.lower() in {"true", "false", "insufficient content", "none", "null"}:
        return True
    if _PROMPT_ECHO_RE.match(o):
        return True
    return False


def _safe_json_loads(s: str) -> Optional[Any]:
    """Backward compat wrapper."""
    return safe_json_loads(s)

# ---------------------------
# Output helpers
# ---------------------------

def flashcards_to_markdown(cards: List[Dict[str, Any]]) -> str:
    if not cards:
        return ""
    lines = ["| Question | Answer | Topic | Memory Anchor | Difficulty |", "|---|---|---|---|---|"]
    for c in cards:
        q = str(c.get("question", "")).replace("\n", " ").replace("|", "/").strip()
        a = str(c.get("answer", "")).replace("\n", " ").replace("|", "/").strip()
        t = str(c.get("topic", "")).replace("\n", " ").replace("|", "/").strip()
        m = str(c.get("memory_anchor", "")).replace("\n", " ").replace("|", "/").strip()
        d = str(c.get("difficulty", "2")).strip()
        lines.append(f"| {q} | {a} | {t} | {m} | {d} |")
    return "\n".join(lines)

def exam_pack_to_markdown(exam_pack: Dict[str, List[Dict[str, Any]]]) -> str:
    if not exam_pack:
        return ""

    sections = []
    for marks_cat, questions in exam_pack.items():
        if not questions:
            continue

        sections.append(f"## {marks_cat} Mark Questions")
        for i, q_obj in enumerate(questions, 1):
            q_text = q_obj.get("q", "")
            a_text = q_obj.get("a", "")
            marks = q_obj.get("marks", "?")

            sections.append(f"**Q{i}:** {q_text} ({marks} Marks)")
            sections.append(f"**Answer:**\n{a_text}")

            mp = q_obj.get("marking_points", [])
            if mp:
                sections.append("\n*Marking Scheme:*")
                for point in mp:
                    sections.append(f"- {point}")

            sections.append("\n---\n")

    return "\n".join(sections)

# ---------------------------
# Fusion (uses alignment data when available)
# ---------------------------

def fuse_topic_chunk(
    topic_text: str,
    slide_ids: List[str],
    slides_by_id: Dict[str, str],
) -> str:
    """
    Fuse a topic's transcript text with its aligned slide content.
    This replaces the naive _fuse_inputs that concatenated everything.
    """
    transcript_text = clean_text(topic_text)
    slide_parts = []
    for sid in slide_ids:
        st = clean_text(slides_by_id.get(sid, ""))
        if st:
            slide_parts.append(f"[{sid}] {st}")
    slide_text = "\n".join(slide_parts)

    if slide_text and transcript_text:
        return (
            f"--- SLIDE CONTENT ---\n{slide_text}\n\n"
            f"--- LECTURE SPEECH ---\n{transcript_text}"
        )
    elif transcript_text:
        return f"--- LECTURE SPEECH ---\n{transcript_text}"
    elif slide_text:
        return f"--- SLIDE CONTENT ---\n{slide_text}"
    return ""


def _fuse_inputs(transcript_text: str, slides: List[Dict[str, Any]]) -> str:
    """Legacy fusion fallback when alignment is not available."""
    transcript_text = clean_text(transcript_text)
    slide_text = clean_text("\n".join(s.get("text", "") for s in slides if s.get("text")))

    if slide_text and transcript_text:
        return f"--- SLIDE CONTENT ---\n{slide_text}\n\n--- LECTURE TRANSCRIPT ---\n{transcript_text}"
    elif transcript_text:
        return transcript_text
    elif slide_text:
        return slide_text
    return ""

# ---------------------------
# Generation pipeline (topic-level)
# ---------------------------

def generate_lecture_materials(
    transcript: Dict[str, Any],
    slides: List[Dict[str, Any]],
    provider: BaseProvider,
    topic_chunks: Optional[List[Any]] = None,
    slides_by_id: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Generate all lecture materials.

    If topic_chunks (from segment.py) are provided, generates per-topic.
    Otherwise falls back to naive chunking for backward compatibility.
    """
    # Build fused chunks
    fused_chunks: List[str] = []

    if topic_chunks and slides_by_id is not None:
        # New path: topic-level fusion with alignment
        for tc in topic_chunks:
            chunk_text = fuse_topic_chunk(
                tc.text if hasattr(tc, "text") else tc.get("text", ""),
                tc.slide_ids if hasattr(tc, "slide_ids") else tc.get("slide_ids", []),
                slides_by_id,
            )
            if chunk_text.strip():
                fused_chunks.append(chunk_text)
    else:
        # Legacy fallback: naive concatenation + word chunking
        transcript_text = transcript.get("text", "") or ""
        fused_text = _fuse_inputs(transcript_text, slides)
        if fused_text.strip():
            from app.tokens import split_to_token_chunks
            fused_chunks = split_to_token_chunks(fused_text, chunk_tokens=1200, overlap_tokens=150)

    if not fused_chunks:
        return {"summary": "Insufficient content", "flashcards": [], "qa_questions": {}, "notes": ""}

    all_notes_parts: List[str] = []
    all_summary_parts: List[str] = []
    all_cards: List[Dict[str, Any]] = []
    qa_accum = {"1-4 Marks (Recall)": [], "5-10 Marks (Application)": [], "15-20 Marks (Synthesis)": []}

    for i, chunk in enumerate(fused_chunks):
        logger.info("Generating for chunk %d/%d...", i + 1, len(fused_chunks))

        c_notes = provider.generate_notes(chunk)
        if c_notes and not looks_bad(c_notes):
            all_notes_parts.append(c_notes)

        c_summary = provider.generate_summary(chunk)
        if c_summary and not looks_bad(c_summary):
            all_summary_parts.append(c_summary)

        c_cards = provider.generate_flashcards(chunk, n=15)
        if c_cards:
            all_cards.extend(c_cards)

        c_qa = provider.generate_questions_with_answers(chunk)
        if c_qa:
            for k in qa_accum:
                qa_accum[k].extend(c_qa.get(k, []))

    # Deduplicate flashcards and questions
    if len(all_cards) > 1:
        all_cards = deduplicate_by_embedding(all_cards, key="question", threshold=0.85)
    for k in qa_accum:
        if len(qa_accum[k]) > 1:
            qa_accum[k] = deduplicate_by_embedding(qa_accum[k], key="q", threshold=0.85)

    final_notes = "\n\n---\n\n".join(all_notes_parts) if all_notes_parts else "Insufficient content"
    final_summary = "\n\n---\n\n".join(all_summary_parts) if all_summary_parts else "Insufficient content"

    return {
        "notes": final_notes,
        "summary": final_summary,
        "flashcards": all_cards,
        "qa_questions": qa_accum,
    }


def generate_subject_materials(subject_corpus: str, syllabus: str, provider: BaseProvider) -> Dict[str, Any]:
    """Generates the final global study pack for the entire subject corpus."""
    if not subject_corpus:
        return {"summary": "Insufficient content", "flashcards": [], "qa_questions": {}}

    summary = provider.generate_subject_summary(subject_corpus, syllabus)
    cards = provider.generate_subject_flashcards(subject_corpus, syllabus, n=50)
    qa = provider.generate_subject_questions_with_answers(subject_corpus, syllabus)

    return {
        "summary": summary,
        "flashcards": cards,
        "qa_questions": qa,
    }
