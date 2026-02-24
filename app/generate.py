from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import re
import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ---------------------------
# Text cleaning + chunking
# ---------------------------

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t).strip()

    # Fix a couple common OCR-ish typos (extend if you want)
    t = re.sub(r"\bscalears\b", "scalars", t, flags=re.I)

    # De-dup adjacent repeated fragments
    parts = [p.strip() for p in re.split(r"[|•\n]+", t) if p.strip()]
    dedup: List[str] = []
    for p in parts:
        if not dedup or p.lower() != dedup[-1].lower():
            dedup.append(p)

    t = " ".join(dedup)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def split_into_word_chunks(text: str, chunk_words: int = 850, overlap_words: int = 120) -> List[str]:
    text = clean_text(text)
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    i = 0
    while i < len(words):
        j = min(len(words), i + chunk_words)
        chunk = " ".join(words[i:j]).strip()
        if chunk:
            chunks.append(chunk)
        if j >= len(words):
            break
        i = max(0, j - overlap_words)
    return chunks


# ---------------------------
# Anti-echo / quality checks
# ---------------------------

_BAD_MARKERS = [
    "generate ", "lecture content:", "slide content:", "instructions:",
    "format:", "rules:", "lecture:", "task:", "role:", "output",
    "### role", "### task", "### rules", "### output"
]

def looks_bad(out: str) -> bool:
    o = (out or "").strip()
    if not o:
        return True

    lo = o.lower()

    # too short
    if len(o) < 60:
        return True

    # classic garbage
    if lo in {"true", "false"}:
        return True

    # prompt echo / headings leaking into output
    if any(m in lo for m in _BAD_MARKERS):
        return True

    return False


def _safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


# ---------------------------
# Optional: Gemini (best quality) with T5 fallback
# ---------------------------

def _gemini_available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def _gemini_generate(prompt: str) -> str:
    """
    Requires: pip install google-generativeai
    Uses GEMINI_API_KEY from environment/.env (DO NOT hardcode).
    """
    import google.generativeai as genai  # type: ignore
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


def _prefer_gemini() -> bool:
    return os.getenv("GEN_PROVIDER", "t5").lower() == "gemini" and _gemini_available()


# ---------------------------
# Generator (FLAN-T5)
# ---------------------------

class T5Generator:
    """
    FLAN-T5 for instruction-following.
    Also: delimiters + strict formats + retry on echo/garbage.
    """
    def __init__(self, model_name: str = "google/flan-t5-base", device: str = "cpu"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def _run_once(self, prompt: str, max_tokens: int) -> str:
        inp = self.tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        # IMPORTANT: if temperature is set, enable sampling or remove temperature.
        with torch.no_grad():
            out = self.model.generate(
                **inp,
                max_new_tokens=max_tokens,
                num_beams=2,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                no_repeat_ngram_size=3,
            )

        return self.tok.decode(out[0], skip_special_tokens=True).strip()

    def _run(self, prompt: str, max_tokens: int, retry_prompt: Optional[str] = None) -> str:
        out = self._run_once(prompt, max_tokens)
        if looks_bad(out) and retry_prompt:
            out2 = self._run_once(retry_prompt, max_tokens)
            if not looks_bad(out2):
                return out2
        return out

    # ---------------- NOTES (slide-wise) ----------------

    def generate_notes(self, slide: str, speech: str, max_tokens: int = 360) -> str:
        slide = clean_text(slide)
        speech = clean_text(speech)

        if len(slide) < 20 and len(speech) < 20:
            return ""

        prompt = f"""
### ROLE
You are a university teaching assistant making exam-ready notes for ANY subject.

### TASK
Create concise study notes for ONE slide.

### STYLE
Detailed Outline + Key Terms

### OUTPUT FORMAT (MUST FOLLOW)
Title: <core concept>
- <5 to 8 bullets explaining what/why/how>
Key Terms:
- <term>: <1 line definition>
- <term>: <1 line definition>

### CONSTRAINTS
- Output ONLY in the format above
- No "Slide Content:" / "Lecture Explanation:" / "Instructions:"
- No True/False
- If content is insufficient, write: Title: Insufficient content

### DATA
Slide: {slide}
Speech: {speech}
""".strip()

        retry = f"""
Output ONLY:
Title: ...
- ...
Key Terms:
- ...: ...

DATA:
{slide}\n{speech}
""".strip()

        out = self._run(prompt, max_tokens=max_tokens, retry_prompt=retry)
        return "" if looks_bad(out) else out

    # ---------------- SUMMARY (chunked map→reduce) ----------------

    def _summarize_chunk(self, chunk_text: str) -> str:
        chunk_text = clean_text(chunk_text)

        prompt = f"""
### TASK
Summarize this lecture section into 10–14 detailed bullet points.

### RULES
- Output ONLY bullet points
- Include examples / key details mentioned
- Make bullets informative (not one-liners)
- No headings, no prompt echo

### TEXT
{chunk_text}
""".strip()

        retry = f"""
Only output 10–14 bullet points. No headings.

TEXT:
{chunk_text}
""".strip()

        out = self._run(prompt, max_tokens=320, retry_prompt=retry)
        return "" if looks_bad(out) else out

    def generate_summary(self, lecture_text: str) -> str:
        lecture_text = clean_text(lecture_text)
        chunks = split_into_word_chunks(lecture_text, chunk_words=850, overlap_words=120)
        if not chunks:
            return ""

        partials = [self._summarize_chunk(c) for c in chunks]
        partials = [p for p in partials if p.strip()]
        if not partials:
            return ""

        merged = "\n".join(partials)

        prompt = f"""
### TASK
Create a FINAL detailed lecture summary from the bullet summaries.

### OUTPUT FORMAT (MUST FOLLOW)
Key Concepts:
- ...
- ...
Important Examples / Applications:
- ...
Exam Takeaways:
- ...

### RULES
- Make it detailed (not 2 lines)
- Output ONLY in the format above
- No prompt echo / no headings other than the 3 section titles above

### BULLET SUMMARIES
{merged}
""".strip()

        retry = f"""
Output ONLY:
Key Concepts:
- ...
Important Examples / Applications:
- ...
Exam Takeaways:
- ...

BULLETS:
{merged}
""".strip()

        out = self._run(prompt, max_tokens=900, retry_prompt=retry)
        return "" if looks_bad(out) else out

    # ---------------- FLASHCARDS (JSON) ----------------

    def generate_flashcards(self, lecture_text: str, n: int = 12) -> List[Dict[str, str]]:
        lecture_text = clean_text(lecture_text)
        chunks = split_into_word_chunks(lecture_text, chunk_words=750, overlap_words=100)
        use_text = "\n".join(chunks[:2]) if chunks else lecture_text

        prompt = f"""
### TASK
Generate exactly {n} flashcards from the lecture.

### OUTPUT (MUST BE VALID JSON ARRAY)
[
  {{"q":"...","a":"...","topic":"...","memory_anchor":"..."}},
  ...
]

### RULES
- Exactly {n} items
- q: clear question
- a: 2–5 sentences, specific, correct
- topic: short label
- memory_anchor: quick trick/analogy/mnemonic
- No extra text, no markdown, no prompt echo, no True/False

### LECTURE
{use_text}
""".strip()

        retry = f"""
Return ONLY valid JSON array with exactly {n} objects.
Each object MUST have keys: q, a, topic, memory_anchor.

TEXT:
{use_text}
""".strip()

        out = self._run(prompt, max_tokens=950, retry_prompt=retry)

        parsed = _safe_json_loads(out)
        if not isinstance(parsed, list) or len(parsed) != n:
            return []

        cleaned: List[Dict[str, str]] = []
        for item in parsed:
            if not isinstance(item, dict):
                return []
            q = str(item.get("q", "")).strip()
            a = str(item.get("a", "")).strip()
            topic = str(item.get("topic", "")).strip()
            ma = str(item.get("memory_anchor", "")).strip()
            if not (q and a and topic and ma):
                return []
            cleaned.append({"q": q, "a": a, "topic": topic, "memory_anchor": ma})

        return cleaned

    # ---------------- QUESTIONS WITH ANSWERS (JSON) ----------------

    def _qa_block(self, lecture_text: str, marks: int, count: int) -> List[Dict[str, str]]:
        lecture_text = clean_text(lecture_text)
        chunks = split_into_word_chunks(lecture_text, chunk_words=850, overlap_words=100)
        use_text = "\n".join(chunks[:2]) if chunks else lecture_text

        if marks <= 4:
            bloom = "Remember / Understand"
            guidance = "Short questions testing definitions, key points. Answers should be crisp."
        elif marks <= 10:
            bloom = "Apply / Analyze"
            guidance = "Needs reasoning and examples. Answers should be structured."
        else:
            bloom = "Evaluate / Create"
            guidance = "Long-answer, multi-step. Answers must include points to write in exam."

        prompt = f"""
### ROLE
Act as a university professor for ANY subject.

### TASK
Create EXACTLY {count} exam-style questions of {marks} marks EACH, WITH ANSWERS.

### DIFFICULTY
Bloom Level: {bloom}
Guidance: {guidance}

### OUTPUT (MUST BE VALID JSON ARRAY)
[
  {{"q":"...","answer_points":"- point\\n- point\\n- point","marks":"{marks}"}},
  ...
]

### RULES
- Exactly {count} items
- q must be specific to the lecture
- answer_points must be bullet points (hyphen bullets)
- No MCQ, no True/False
- No extra text

### LECTURE
{use_text}
""".strip()

        retry = f"""
Return ONLY valid JSON array with exactly {count} items.
Each item keys: q, answer_points, marks.

LECTURE:
{use_text}
""".strip()

        out = self._run(prompt, max_tokens=950, retry_prompt=retry)
        parsed = _safe_json_loads(out)
        if not isinstance(parsed, list) or len(parsed) != count:
            return []

        out_items: List[Dict[str, str]] = []
        for it in parsed:
            if not isinstance(it, dict):
                return []
            q = str(it.get("q", "")).strip()
            ans = str(it.get("answer_points", "")).strip()
            mk = str(it.get("marks", str(marks))).strip()
            if not q or not ans:
                return []
            out_items.append({"q": q, "answer_points": ans, "marks": mk})
        return out_items

    def generate_questions_with_answers(self, lecture_text: str) -> Dict[str, List[Dict[str, str]]]:
        return {
            "1-4": self._qa_block(lecture_text, marks=4,  count=10),
            "5":   self._qa_block(lecture_text, marks=5,  count=8),
            "10":  self._qa_block(lecture_text, marks=10, count=6),
            "15":  self._qa_block(lecture_text, marks=15, count=4),
            "20":  self._qa_block(lecture_text, marks=20, count=3),
        }


# ---- SINGLETON (loads model once for speed) ----
_T5: Optional[T5Generator] = None

def get_t5(model_name: str = "google/flan-t5-base", device: str = "cpu") -> T5Generator:
    global _T5
    if _T5 is None:
        _T5 = T5Generator(model_name=model_name, device=device)
    return _T5


# ---------------------------
# Output helpers
# ---------------------------

def flashcards_to_markdown(cards: List[Dict[str, str]]) -> str:
    if not cards:
        return ""
    lines = ["| Question | Answer | Topic | Memory Anchor |", "|---|---|---|---|"]
    for c in cards:
        q = c["q"].replace("\n", " ").strip()
        a = c["a"].replace("\n", " ").strip()
        t = c["topic"].replace("\n", " ").strip()
        m = c["memory_anchor"].replace("\n", " ").strip()
        lines.append(f"| {q} | {a} | {t} | {m} |")
    return "\n".join(lines)


# ---------------------------
# Pipeline functions
# ---------------------------

def generate_slidewise_notes(
    slides: List[Dict[str, str]],
    alignment: Dict[str, Any],
    min_confidence: float = 0.55,
) -> List[Dict[str, Any]]:
    gen = get_t5()
    notes: List[Dict[str, Any]] = []

    by_id = {s["slide_id"]: clean_text(s.get("text") or "") for s in slides}

    for a in alignment.get("alignments", []):
        sid = a.get("slide_id")
        if not sid:
            continue

        conf = float(a.get("confidence", 0.0))
        slide_text = by_id.get(sid, "")
        speech_text = clean_text((a.get("segment", {}) or {}).get("text") or "")

        note = gen.generate_notes(slide_text, speech_text)

        # Avoid writing junk for low-confidence + empty outputs
        if not note and conf < min_confidence:
            continue

        notes.append({
            "slide_id": sid,
            "note": note if note else "(low confidence: could not generate clean notes)",
            "confidence": conf,
            "time": [
                float((a.get("segment", {}) or {}).get("start", 0.0)),
                float((a.get("segment", {}) or {}).get("end", 0.0)),
            ],
        })

    return notes


def generate_lecture_materials(transcript: Dict[str, Any]) -> Dict[str, Any]:
    lecture_text = clean_text(transcript.get("text", "") or "")
    if not lecture_text:
        return {"summary": "", "flashcards": [], "qa_questions": {}}

    # Prefer Gemini if enabled; fallback to T5 always
    if _prefer_gemini():
        # Gemini prompts are already strict JSON / structured; still safe-check outputs
        summary_prompt = f"""
You are a university lecturer. Write a detailed lecture summary for ANY subject.

OUTPUT FORMAT:
Key Concepts:
- ...
Important Examples / Applications:
- ...
Exam Takeaways:
- ...

LECTURE:
{lecture_text[:8000]}
""".strip()

        flash_prompt = f"""
Create exactly 12 flashcards for ANY subject from this lecture.

Return ONLY valid JSON array:
[
  {{"q":"...","a":"...","topic":"...","memory_anchor":"..."}},
  ...
]

LECTURE:
{lecture_text[:8000]}
""".strip()

        qa_prompt = f"""
Create exam questions WITH answers for ANY subject from this lecture.

Return ONLY valid JSON object with keys "1-4","5","10","15","20".
Each value is a JSON array of objects: {{"q":"...","answer_points":"- ...\\n- ...","marks":"..."}}.

Counts:
1-4: 10 items (4 marks)
5: 8 items
10: 6 items
15: 4 items
20: 3 items

LECTURE:
{lecture_text[:8000]}
""".strip()

        summary = _gemini_generate(summary_prompt)
        flash_raw = _gemini_generate(flash_prompt)
        qa_raw = _gemini_generate(qa_prompt)

        cards = _safe_json_loads(flash_raw)
        if not isinstance(cards, list):
            cards = []

        qa = _safe_json_loads(qa_raw)
        if not isinstance(qa, dict):
            qa = {}

        return {
            "summary": summary if not looks_bad(summary) else "",
            "flashcards": cards if isinstance(cards, list) else [],
            "qa_questions": qa if isinstance(qa, dict) else {},
        }

    # T5 path
    gen = get_t5(model_name=os.getenv("T5_MODEL", "google/flan-t5-base"))

    summary = gen.generate_summary(lecture_text)
    cards = gen.generate_flashcards(lecture_text, n=12)
    qa = gen.generate_questions_with_answers(lecture_text)

    return {
        "summary": summary,
        "flashcards": cards,
        "qa_questions": qa,
    }