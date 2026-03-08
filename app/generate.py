from __future__ import annotations
from typing import Dict, Any, List
import re, json, os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from google import genai


# --------------------------------------------------
# DEBUG MODE
# --------------------------------------------------

DEBUG = True

def debug_print(title, data, max_len=400):
    if not DEBUG:
        return

    print("\n" + "="*60)
    print(f"DEBUG: {title}")
    print("="*60)

    if isinstance(data, str):
        print(data[:max_len])
    else:
        print(data)


# --------------------------------------------------
# GEMINI
# --------------------------------------------------

def use_gemini():
    return bool(os.getenv("GEMINI_API_KEY"))


def gemini_generate(prompt: str) -> str:
    try:

        debug_print("PROMPT LENGTH", len(prompt))
        debug_print("PROMPT SAMPLE", prompt)

        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        out = (response.text or "").strip()

        debug_print("GEMINI OUTPUT", out)

        return out

    except Exception as e:
        print(f"⚠️ Gemini failed → fallback to empty: {e}")
        return ""


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(t: str):
    if not t:
        return ""
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------
# QUALITY CHECK
# --------------------------------------------------

def looks_bad(out: str):
    if not out or len(out.strip()) < 80:
        return True

    if any(x in out.lower() for x in ["instruction", "task:", "output:", "###"]):
        return True

    return False


def safe_json(s: str):
    if not s:
        return None

    try:
        # remove markdown code blocks
        s = s.strip()

        if s.startswith("```"):
            s = s.split("```")[1]

            # remove optional "json"
            if s.startswith("json"):
                s = s[4:]

        s = s.strip()

        return json.loads(s)

    except Exception as e:
        print("JSON PARSE FAILED:", e)
        return None


# --------------------------------------------------
# T5 FALLBACK MODEL
# --------------------------------------------------

class T5Generator:

    def __init__(self):

        self.tok = AutoTokenizer.from_pretrained("google/flan-t5-base")

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-base"
        )

    def run(self, prompt, max_tokens=512):

        inp = self.tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():

            out = self.model.generate(
                **inp,
                max_new_tokens=max_tokens
            )

        return self.tok.decode(out[0], skip_special_tokens=True)


_T5 = None

def get_t5():

    global _T5

    if _T5 is None:
        _T5 = T5Generator()

    return _T5


# --------------------------------------------------
# SLIDE NOTES (T5 ONLY — NO GEMINI)
# --------------------------------------------------

def generate_topic_notes(topic, slide, speech):

    slide = slide[:500]
    speech = speech[:300]

    prompt = f"""
Create short exam notes.

Topic: {topic}

Use slide as main source.

Explain clearly.

SLIDE:
{slide}

LECTURE:
{speech}
"""

    gen = get_t5()

    out = gen.run(prompt, 200)

    return "" if looks_bad(out) else out


# --------------------------------------------------
# SLIDEWISE NOTES
# --------------------------------------------------

def generate_slidewise_notes(slides, alignment):

    notes = []

    by_id = {
        s["slide_id"]: clean_text(s.get("text"))
        for s in slides
    }

    for a in alignment.get("alignments", []):

        sid = a["slide_id"]

        slide_text = by_id.get(sid, "")

        speech = clean_text(a["segment"]["text"])

        topic = a.get("topic") or sid

        note = generate_topic_notes(topic, slide_text, speech)

        if note:
            notes.append({
                "slide_id": sid,
                "note": note,
                "confidence": a.get("confidence", 0)
            })

    return notes


# --------------------------------------------------
# GLOBAL LECTURE MATERIALS (GEMINI USED HERE)
# --------------------------------------------------

def generate_lecture_materials(transcript: Dict[str, Any]):

    segments = transcript.get("segments", [])

    chunks = []
    current = ""

    for s in segments:

        current += " " + s["text"]

        if len(current) > 2000:
            chunks.append(current)
            current = ""

    if current:
        chunks.append(current)

    text = chunks[0] if chunks else ""

    debug_print("TRANSCRIPT CHUNK SIZE", len(text))
    debug_print("TRANSCRIPT SAMPLE", text)

    if not text:
        return {
            "summary": "",
            "flashcards": [],
            "qa_questions": {}
        }

    # ---------------- SUMMARY ----------------

    summary_prompt = f"""
Create structured study summary.

Sections:
Key Concepts
Examples
Exam Takeaways

{text}
"""

    summary = gemini_generate(summary_prompt)

    # ---------------- FLASHCARDS ----------------

    flash_prompt = f"""
Return ONLY JSON.

Create 10 flashcards.

[
{{"q":"question","a":"answer","topic":"topic","memory_anchor":"hint"}}
]

TEXT:
{text}
"""

    flash_raw = gemini_generate(flash_prompt)

    debug_print("FLASH RAW", flash_raw)

    flashcards = safe_json(flash_raw)

    # ---------------- QUESTIONS ----------------

    qa_prompt = f"""
Return ONLY JSON.

{{
"1-4":[{{"q":"","answer_points":"- ...","marks":"4"}}],
"5":[{{"q":"","answer_points":"- ...","marks":"5"}}],
"10":[{{"q":"","answer_points":"- ...","marks":"10"}}]
}}

TEXT:
{text}
"""

    qa_raw = gemini_generate(qa_prompt)

    debug_print("QA RAW", qa_raw)

    qa = safe_json(qa_raw)

    return {
        "summary": summary,
        "flashcards": flashcards or [],
        "qa_questions": qa or {}
    }


# --------------------------------------------------
# MARKDOWN
# --------------------------------------------------

def flashcards_to_markdown(cards):

    if not cards:
        return ""

    lines = [
        "| Q | A | Topic | Memory |",
        "|---|---|---|---|"
    ]

    for c in cards:

        lines.append(
            f"| {c['q']} | {c['a']} | {c['topic']} | {c['memory_anchor']} |"
        )

    return "\n".join(lines)