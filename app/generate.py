from __future__ import annotations

from typing import Dict, Any, List, Optional
import re
import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------
# CLEANING
# ---------------------------

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------
# QUALITY CHECK
# ---------------------------

_BAD_MARKERS = ["###", "instruction", "task:", "output:", "true", "false"]

def looks_bad(out: str) -> bool:
    if not out or len(out.strip()) < 60:
        return True
    lo = out.lower()
    if any(m in lo for m in _BAD_MARKERS):
        return True
    return False


def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except:
        return None


# ---------------------------
# GEMINI
# ---------------------------

def _gemini_available():
    return bool(os.getenv("GEMINI_API_KEY"))

def _gemini_generate(prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


def use_gemini():
    return os.getenv("GEN_PROVIDER", "t5") == "gemini" and _gemini_available()


# ---------------------------
# T5
# ---------------------------

class T5Generator:

    def __init__(self, model_name="google/flan-t5-base", device="cpu"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def run(self, prompt: str, max_tokens=512):
        inp = self.tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inp,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                no_repeat_ngram_size=3,
            )

        return self.tok.decode(out[0], skip_special_tokens=True)


_T5 = None

def get_t5():
    global _T5
    if _T5 is None:
        _T5 = T5Generator()
    return _T5


# ---------------------------
# TOPIC-AWARE NOTES (NEW 🔥)
# ---------------------------

def generate_topic_notes(topic: str, slide: str, speech: str) -> str:

    combined = f"""
TOPIC: {topic}

SLIDE CONTENT:
{slide}

LECTURE EXPLANATION:
{speech}
"""

    prompt = f"""
You are a top university professor.

Create HIGH-QUALITY exam notes using BOTH slide + lecture explanation.

Rules:
- Combine both sources
- Add missing explanation if slide is vague
- Add examples if lecture mentions them
- Do NOT copy text blindly

OUTPUT FORMAT:
Title: {topic}
- detailed explanation
- key reasoning
- examples if possible

Key Terms:
- term: definition
- term: definition

CONTENT:
{combined}
"""

    # Prefer Gemini
    if use_gemini():
        out = _gemini_generate(prompt)
        if not looks_bad(out):
            return out

    # fallback T5
    gen = get_t5()
    out = gen.run(prompt, 400)
    return "" if looks_bad(out) else out


# ---------------------------
# SLIDEWISE NOTES (UPDATED)
# ---------------------------

def generate_slidewise_notes(slides, alignment, topics=None):

    notes = []
    by_id = {s["slide_id"]: clean_text(s.get("text")) for s in slides}

    for a in alignment.get("alignments", []):

        sid = a["slide_id"]
        slide_text = by_id.get(sid, "")
        speech = clean_text(a["segment"]["text"])

        # 🔥 NEW: topic selection
        topic = sid
        if topics:
            topic = topics.get(sid, sid)

        note = generate_topic_notes(topic, slide_text, speech)

        if not note:
            continue

        notes.append({
            "slide_id": sid,
            "note": note,
            "confidence": a.get("confidence", 0)
        })

    return notes


# ---------------------------
# LECTURE MATERIALS (STRONG)
# ---------------------------

def generate_lecture_materials(transcript: Dict[str, Any]):

    text = clean_text(transcript.get("text", ""))[:12000]

    if not text:
        return {}

    # ---------- SUMMARY ----------
    summary_prompt = f"""
Create a COMPLETE STUDY SUMMARY.

DO NOT just summarize — EXPAND concepts.

Include:
- explanations
- examples
- real-world intuition

FORMAT:
Key Concepts:
- ...

Examples:
- ...

Exam Takeaways:
- ...

TEXT:
{text}
"""

    # ---------- FLASHCARDS ----------
    flash_prompt = f"""
Generate 20 HIGH-QUALITY flashcards.

Return JSON:
[
{{"q":"","a":"","topic":"","memory_anchor":""}}
]

TEXT:
{text}
"""

    # ---------- QUESTIONS ----------
    qa_prompt = f"""
Generate exam questions WITH answers.

Return JSON:
{{
"1-4": [],
"5": [],
"10": [],
"15": []
}}

TEXT:
{text}
"""

    if use_gemini():
        summary = _gemini_generate(summary_prompt)
        flash = _safe_json_loads(_gemini_generate(flash_prompt))
        qa = _safe_json_loads(_gemini_generate(qa_prompt))

        return {
            "summary": summary,
            "flashcards": flash or [],
            "qa_questions": qa or {}
        }

    # fallback T5
    gen = get_t5()

    summary = gen.run(summary_prompt, 600)
    flash = []
    qa = {}

    return {
        "summary": summary if not looks_bad(summary) else "",
        "flashcards": flash,
        "qa_questions": qa
    }


# ---------------------------
# FLASHCARD MARKDOWN
# ---------------------------

def flashcards_to_markdown(cards):

    if not cards:
        return ""

    lines = ["| Q | A | Topic | Memory |", "|---|---|---|---|"]

    for c in cards:
        lines.append(f"| {c['q']} | {c['a']} | {c['topic']} | {c['memory_anchor']} |")

    return "\n".join(lines)