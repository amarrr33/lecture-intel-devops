from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Any, List


# ---------------------------
# JSON SAVE
# ---------------------------

def save_json(obj: Dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


# ---------------------------
# ALIGNMENT MARKDOWN
# ---------------------------

def save_alignment_markdown(alignment: Dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = alignment.get("stats", {})

    lines = [
        "# Slide ↔ Speech Alignment Report\n",
        f"- Slides: **{stats.get('num_slides', 0)}**",
        f"- Transcript segments: **{stats.get('num_segments', 0)}**",
        f"- Avg confidence: **{stats.get('avg_confidence', 0):.3f}**\n"
    ]

    for a in alignment.get("alignments", []):
        seg = a.get("segment", {})
        lines.append(f"## {a.get('slide_id')}")
        lines.append(f"- Segment idx: `{a.get('best_segment_idx')}`")
        lines.append(f"- Cosine: **{a.get('cosine', 0):.3f}**")
        lines.append(f"- Confidence: **{a.get('confidence', 0):.3f}**")
        lines.append(f"- Time: **{seg.get('start', 0):.1f}s → {seg.get('end', 0):.1f}s**")
        lines.append(f"- Text: {seg.get('text', '')}\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------
# NOTES MARKDOWN
# ---------------------------

def save_notes_markdown(notes: List[Dict[str, Any]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Slide-wise Notes\n"]

    for n in notes:
        lines.append(f"## {n.get('slide_id')} (conf={n.get('confidence', 0):.2f})")
        lines.append(n.get("note", ""))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------
# SUMMARY MARKDOWN
# ---------------------------

def save_summary(summary: str, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not summary:
        summary = "_No summary generated._"

    lines = [
        "# Lecture Summary\n",
        summary
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------
# FLASHCARDS
# ---------------------------

def save_flashcards_json(cards: List[Dict[str, str]], out_path: str | Path) -> None:
    save_json(cards, out_path)


def save_flashcards_csv(cards: List[Dict[str, str]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["question,answer,topic,memory_anchor"]

    for c in cards:
        q = c.get("q", "").replace(",", " ")
        a = c.get("a", "").replace(",", " ")
        t = c.get("topic", "").replace(",", " ")
        m = c.get("memory_anchor", "").replace(",", " ")
        lines.append(f"{q},{a},{t},{m}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_flashcards_markdown(cards: List[Dict[str, str]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not cards:
        out_path.write_text("_No flashcards generated._", encoding="utf-8")
        return

    lines = [
        "# Flashcards\n",
        "| Question | Answer | Topic | Memory Anchor |",
        "|---|---|---|---|"
    ]

    for c in cards:
        q = c.get("q", "").replace("\n", " ")
        a = c.get("a", "").replace("\n", " ")
        t = c.get("topic", "").replace("\n", " ")
        m = c.get("memory_anchor", "").replace("\n", " ")
        lines.append(f"| {q} | {a} | {t} | {m} |")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------
# EXAM PACK (Q + A)
# ---------------------------

def save_exam_pack(qa: Dict[str, List[Dict[str, str]]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not qa:
        out_path.write_text("_No questions generated._", encoding="utf-8")
        return

    lines = ["# Exam Question Bank\n"]

    for section, items in qa.items():
        lines.append(f"\n## {section} Marks\n")

        for i, q in enumerate(items, 1):
            lines.append(f"### Q{i}. {q.get('q')}")
            lines.append("**Answer:**")
            lines.append(q.get("answer_points", ""))
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------
# METADATA
# ---------------------------

def save_metadata(metadata: Dict[str, Any], out_path: str | Path) -> None:
    save_json(metadata, out_path)