from __future__ import annotations
from typing import Dict, Any, List
import re, json, os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ✅ NEW SDK
from google import genai


# ---------------- GEMINI (FIXED ✅)
from google import genai

# ---------------- GEMINI ----------------
def use_gemini():
    return bool(os.getenv("GEMINI_API_KEY"))

def gemini_generate(prompt: str) -> str:
    try:
        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",   # ✅ CORRECT MODEL
            contents=prompt,
        )

        return (response.text or "").strip()

    except Exception as e:
        print(f"⚠️ Gemini failed → fallback to T5: {e}")
        return ""


# ---------------- CLEAN ----------------
def clean_text(t: str) -> str:
    if not t:
        return ""
    return re.sub(r"\s+", " ", t).strip()


# ---------------- QUALITY ----------------
def looks_bad(out: str) -> bool:
    if not out or len(out.strip()) < 80:
        return True
    if any(x in out.lower() for x in ["instruction", "task:", "output:", "###"]):
        return True
    return False


def safe_json(s: str):
    try:
        return json.loads(s)
    except:
        return None


# ---------------- T5 (fallback) ----------------
class T5Generator:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def run(self, prompt, max_tokens=512):
        inp = self.tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_tokens)
        return self.tok.decode(out[0], skip_special_tokens=True)


_T5 = None
def get_t5():
    global _T5
    if _T5 is None:
        _T5 = T5Generator()
    return _T5


# ---------------- NOTES ----------------
def generate_topic_notes(topic, slide, speech):

    prompt = f"""
Create exam-ready notes.

Title: {topic}
- explanation
- reasoning
- examples

Key Terms:
- term: definition

SLIDE:
{slide}

LECTURE:
{speech}
"""

    if use_gemini():
        try:
            out = gemini_generate(prompt)
            if not looks_bad(out):
                return out
        except Exception as e:
            print("⚠️ Gemini failed → fallback to T5:", e)

    gen = get_t5()
    out = gen.run(prompt, 400)
    return "" if looks_bad(out) else out


# ---------------- SLIDEWISE ----------------
def generate_slidewise_notes(slides, alignment):

    notes = []
    by_id = {s["slide_id"]: clean_text(s.get("text")) for s in slides}

    for a in alignment.get("alignments", []):

        sid = a["slide_id"]
        slide_text = by_id.get(sid, "")
        speech = clean_text(a["segment"]["text"])

        # ✅ FIXED KEY
        topic = a.get("topic") or sid

        note = generate_topic_notes(topic, slide_text, speech)

        if note:
            notes.append({
                "slide_id": sid,
                "note": note,
                "confidence": a.get("confidence", 0)
            })

    return notes


# ---------------- MATERIALS ----------------
def generate_lecture_materials(transcript: Dict[str, Any]):

    text = clean_text(transcript.get("text", ""))

    # ✅ TOKEN CONTROL
    sentences = text.split(".")
    text = " ".join(sentences[:200])

    if not text:
        return {"summary": "", "flashcards": [], "qa_questions": {}}

    summary_prompt = f"""
Create structured study summary.

Key Concepts:
- ...

Examples:
- ...

Exam Takeaways:
- ...

{text}
"""

    if use_gemini():
        try:
            summary = gemini_generate(summary_prompt)

            flash = safe_json(gemini_generate(f"""
Return JSON:
[{{"q":"","a":"","topic":"","memory_anchor":""}}]

{text}
"""))

            qa = safe_json(gemini_generate(f"""
Return JSON:
{{"1-4": [], "5": [], "10": []}}

{text}
"""))

            return {
                "summary": summary,
                "flashcards": flash or [],
                "qa_questions": qa or {}
            }

        except Exception as e:
            print("⚠️ Gemini failed → fallback:", e)

    # fallback
    gen = get_t5()
    summary = gen.run(summary_prompt, 600)

    flashcards = []
    for c in sentences[:20]:
        if len(c.strip()) < 40:
            continue
        flashcards.append({
            "q": f"What is: {c[:40]}?",
            "a": c.strip(),
            "topic": "General",
            "memory_anchor": "Recall sentence"
        })

    return {
        "summary": summary if not looks_bad(summary) else "",
        "flashcards": flashcards,
        "qa_questions": {}
    }


# ---------------- MARKDOWN ----------------
def flashcards_to_markdown(cards):
    if not cards:
        return ""
    lines = ["| Q | A | Topic | Memory |", "|---|---|---|---|"]
    for c in cards:
        lines.append(f"| {c['q']} | {c['a']} | {c['topic']} | {c['memory_anchor']} |")
    return "\n".join(lines)