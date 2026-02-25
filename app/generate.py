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
# SMART FUSION (KEY FIX 🚀)
# ---------------------------

def combine_slide_and_speech(slide: str, speech: str) -> str:
    """
    CORE IDEA:
    - Slides = structure
    - Speech = depth
    """

    slide = clean_text(slide)
    speech = clean_text(speech)

    if not slide and not speech:
        return ""

    # prioritize slide keywords
    if len(slide) > 20:
        return f"""
Topic from slide:
{slide}

Detailed explanation from lecture:
{speech}
""".strip()

    # fallback if slide weak
    return speech


# ---------------------------
# QUALITY CHECK
# ---------------------------

def is_bad(out: str) -> bool:
    if not out or len(out.strip()) < 40:
        return True
    low = out.lower()
    if "task" in low or "instruction" in low:
        return True
    return False


# ---------------------------
# GEMINI (OPTIONAL)
# ---------------------------

def use_gemini() -> bool:
    return os.getenv("GEN_PROVIDER", "t5") == "gemini" and os.getenv("GEMINI_API_KEY")


def gemini_generate(prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    return model.generate_content(prompt).text.strip()


# ---------------------------
# T5 GENERATOR
# ---------------------------

class T5:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def run(self, prompt: str, max_tokens=300):
        inputs = self.tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        return self.tok.decode(out[0], skip_special_tokens=True)


_T5 = None

def get_t5():
    global _T5
    if _T5 is None:
        _T5 = T5()
    return _T5


# ---------------------------
# NOTES
# ---------------------------

def generate_notes(slide: str, speech: str) -> str:
    text = combine_slide_and_speech(slide, speech)

    prompt = f"""
Create detailed study notes.

Content:
{text}

Rules:
- Give proper explanation
- Include WHY and HOW
- Add examples if possible
- No headings like TASK
"""

    if use_gemini():
        out = gemini_generate(prompt)
    else:
        out = get_t5().run(prompt)

    return "" if is_bad(out) else out


# ---------------------------
# SUMMARY
# ---------------------------

def generate_summary(text: str) -> str:
    prompt = f"""
Write a detailed lecture summary.

IMPORTANT:
- Explain concepts clearly
- Expand topics (don't just repeat)
- Add missing explanation if needed

Lecture:
{text[:6000]}
"""

    if use_gemini():
        out = gemini_generate(prompt)
    else:
        out = get_t5().run(prompt, 500)

    return "" if is_bad(out) else out


# ---------------------------
# FLASHCARDS
# ---------------------------

def generate_flashcards(text: str) -> List[Dict]:
    prompt = f"""
Create 10 high quality flashcards.

Return JSON:
[
  {{"q":"...","a":"..."}}
]

Lecture:
{text[:5000]}
"""

    if use_gemini():
        out = gemini_generate(prompt)
    else:
        out = get_t5().run(prompt, 400)

    try:
        return json.loads(out)
    except:
        return []


# ---------------------------
# QUESTIONS + ANSWERS
# ---------------------------

def generate_qa(text: str) -> Dict:
    prompt = f"""
Create exam questions WITH answers.

Return JSON:
{{
 "2": [{{"q":"...","a":"..."}}],
 "5": [...],
 "10": [...]
}}

Lecture:
{text[:6000]}
"""

    if use_gemini():
        out = gemini_generate(prompt)
    else:
        out = get_t5().run(prompt, 500)

    try:
        return json.loads(out)
    except:
        return {}


# ---------------------------
# PIPELINE FUNCTIONS
# ---------------------------

def generate_slidewise_notes(slides, alignment):
    notes = []
    for a in alignment.get("alignments", []):
        slide = next((s["text"] for s in slides if s["slide_id"] == a["slide_id"]), "")
        speech = a.get("segment", {}).get("text", "")

        note = generate_notes(slide, speech)

        notes.append({
            "slide_id": a["slide_id"],
            "note": note
        })

    return notes


def generate_lecture_materials(transcript: Dict[str, Any]):
    text = transcript.get("text", "")

    return {
        "summary": generate_summary(text),
        "flashcards": generate_flashcards(text),
        "questions": generate_qa(text),
    }