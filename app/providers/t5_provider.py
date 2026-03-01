import os
import json
from typing import Dict, Any, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from app.providers.base_provider import BaseProvider
from app.generate import split_into_word_chunks, clean_text, looks_bad, _safe_json_loads

# ---- SINGLETON (loads model once for speed) ----
_T5_MODEL = None
_T5_TOK = None

class T5Provider(BaseProvider):
    """
    FLAN-T5 offline provider. LIMITED QUALITY.

    Limitations:
    - Max 512 input tokens (most lecture content gets truncated)
    - Cannot reliably generate valid JSON
    - Summary quality is poor for academic content
    - Use only when Gemini API is completely unavailable
    - For better offline results, consider Ollama + Mistral 7B instead
    """
    def __init__(self, model_name: str = "google/flan-t5-base", device: str = "cpu"):
        self._model_name = os.getenv("T5_MODEL", model_name)
        self.device = device
        
        global _T5_MODEL, _T5_TOK
        if _T5_MODEL is None or _T5_TOK is None:
            _T5_TOK = AutoTokenizer.from_pretrained(self._model_name)
            _T5_MODEL = AutoModelForSeq2SeqLM.from_pretrained(self._model_name).to(self.device)
            
        self.tok = _T5_TOK
        self.model = _T5_MODEL

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def used_internet(self) -> bool:
        return False

    def _run_once(self, prompt: str, max_tokens: int) -> str:
        inp = self.tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inp,
                max_new_tokens=max_tokens,
                num_beams=2,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                no_repeat_ngram_size=4,
                repetition_penalty=1.2,
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

    def generate_notes(self, lecture_text: str, speech_text: str = "", max_tokens: int = 400) -> str:
        lecture_text = clean_text(lecture_text)
        if speech_text:
            lecture_text = f"{lecture_text} {clean_text(speech_text)}"

        if len(lecture_text) < 50:
            return "Insufficient content"

        prompt = f"""
### ROLE
You are a university teaching assistant making exam-ready notes for ANY subject / any technical or academic lecture.

### TASK
Create concise study notes for ONE slide.

### STYLE
Detailed Outline + Key Terms + Example + Common Mistake

### FORMAT
Title: <core concept>
- <Explanation bullets: WHAT/WHY/HOW>
Key Terms:
- <term>: <definition>

### CONSTRAINTS
- Output ONLY in the format above
- No "Slide Content:" / "Instructions:"
- If content is insufficient, write: Insufficient content

### LECTURE CHUNK
{lecture_text}
""".strip()

        retry = f"""
Output ONLY:
Title: ...
- ...
Key Terms:
- ...: ...

DATA:
{lecture_text}
""".strip()

        out = self._run(prompt, max_tokens=max_tokens, retry_prompt=retry)
        return "Insufficient content" if looks_bad(out) else out

    # ---------------- SUMMARY (chunked map→reduce) ----------------

    def _summarize_chunk(self, chunk_text: str) -> str:
        chunk_text = clean_text(chunk_text)

        prompt = f"""
### TASK
Summarize this lecture section into 10–14 detailed bullet points. Include examples / key details mentioned.

### RULES
- Output ONLY bullet points
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
            return "Insufficient content"

        partials = [self._summarize_chunk(c) for c in chunks]
        partials = [p for p in partials if p.strip()]
        if not partials:
            return "Insufficient content"

        merged = "\n".join(partials)

        prompt = f"""
### TASK
Create a FINAL detailed lecture summary from the bullet summaries for ANY subject.

### OUTPUT FORMAT (MUST FOLLOW)
Overview:
<Deep textbook-style explanation summarizing the chunk>

Core Principles & Logical Intuition:
- <Principle>: <Thorough explanation and intuition>

Examples / Extensions:
- <Example demonstrating the concept>

### RULES
- Make it detailed (not 2 lines)
- Output ONLY in the format above
- No prompt echo

### BULLET SUMMARIES
{merged}
""".strip()

        retry = f"""
Output ONLY:
Overview:
...
Core Principles & Logical Intuition:
- ...
Examples / Extensions:
- ...

BULLETS:
{merged}
""".strip()

        out = self._run(prompt, max_tokens=1000, retry_prompt=retry)
        return "Insufficient content" if looks_bad(out) else out

    # ---------------- FLASHCARDS (JSON) ----------------

    def generate_flashcards(self, lecture_text: str, n: int = 20) -> List[Dict[str, str]]:
        lecture_text = clean_text(lecture_text)
        chunks = split_into_word_chunks(lecture_text, chunk_words=750, overlap_words=100)
        use_text = "\n".join(chunks[:3]) if chunks else lecture_text

        prompt = f"""
### TASK
Generate exactly {n} flashcards for ANY subject / any technical or academic lecture.

### OUTPUT (MUST BE VALID JSON ARRAY)
[
  {{"q":"...","a":"...","topic":"...","memory_anchor":"...","difficulty":2}},
  ...
]

### RULES
- Exactly {n} items
- q: clear, specific conceptual question 
- a: 2–5 sentences, specific, precise
- topic: short label
- memory_anchor: quick trick/analogy/mnemonic
- difficulty: 1, 2, or 3
- No extra text, no markdown, no prompt echo

### LECTURE
{use_text}
""".strip()

        retry = f"""
Return ONLY valid JSON array with exactly {n} objects.
Each object MUST have keys: q, a, topic, memory_anchor, difficulty. No incomplete fields.

TEXT:
{use_text}
""".strip()

        out = self._run(prompt, max_tokens=1200, retry_prompt=retry)

        parsed = _safe_json_loads(out)
        if not isinstance(parsed, list) or len(parsed) < 1:
             # try falling back to smaller count if it couldn't do 20
            out_smaller = self._run(prompt.replace(str(n), "10"), max_tokens=1000, retry_prompt=retry.replace(str(n), "10"))
            parsed_smaller = _safe_json_loads(out_smaller)
            if not isinstance(parsed_smaller, list):
                return []
            parsed = parsed_smaller

        cleaned: List[Dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            q = str(item.get("q", item.get("question", ""))).strip()
            a = str(item.get("a", item.get("answer", ""))).strip()
            topic = str(item.get("topic", "")).strip()
            ma = str(item.get("memory_anchor", "")).strip()
            diff = item.get("difficulty", 2)
            try:
                diff = int(diff)
            except:
                diff = 2
                
            if not (q and a and topic and ma):
                continue
            cleaned.append({"topic": topic, "question": q, "answer": a, "memory_anchor": ma, "difficulty": diff})

        return cleaned

    # ---------------- QUESTIONS WITH ANSWERS (JSON) ----------------

    def _qa_block(self, lecture_text: str, marks: int, count: int) -> List[Dict[str, Any]]:
        lecture_text = clean_text(lecture_text)
        chunks = split_into_word_chunks(lecture_text, chunk_words=850, overlap_words=100)
        use_text = "\n".join(chunks[:3]) if chunks else lecture_text

        if marks <= 4:
            bloom = "Remember / Understand"
            guidance = "Short questions testing definitions, key points. Answers should be crisp."
        elif marks <= 10:
            bloom = "Apply / Analyze"
            guidance = "Needs reasoning and examples. Answers should be structured points."
        else:
            bloom = "Evaluate / Create"
            guidance = "Long-answer, deep analysis, multi-step. Answers must include points to write in exam."

        prompt = f"""
### ROLE
Act as a university professor for ANY subject / any technical or academic lecture.

### TASK
Create EXACTLY {count} exam-style questions of {marks} marks EACH, WITH ANSWERS.

### DIFFICULTY
Bloom Level: {bloom}
Guidance: {guidance}

### OUTPUT (MUST BE VALID JSON ARRAY)
[
  {{"q":"...","a":"...","marks":{marks},"marking_points":["point 1", "point 2"]}},
  ...
]

### RULES
- Exactly {count} items
- q must be specific to the lecture (no irrelevant/generic questions)
- marking_points must be an array of strings (tiny marking scheme)
- No MCQ, no True/False
- No extra text

### LECTURE
{use_text}
""".strip()

        retry = f"""
Return ONLY valid JSON array with exactly {count} items.
Each item keys: q, a, marks, marking_points (array of strings).

LECTURE:
{use_text}
""".strip()

        out = self._run(prompt, max_tokens=1000, retry_prompt=retry)
        parsed = _safe_json_loads(out)
        
        # fallback
        if not isinstance(parsed, list) or len(parsed) == 0:
            out2 = self._run(prompt.replace(f"EXACTLY {count}", f"EXACTLY {max(1, count//2)}"), max_tokens=1000, retry_prompt=retry)
            parsed2 = _safe_json_loads(out2)
            if not isinstance(parsed2, list):
                return []
            parsed = parsed2

        out_items: List[Dict[str, Any]] = []
        for it in parsed:
            if not isinstance(it, dict):
                continue
            q = str(it.get("q", "")).strip()
            ans = str(it.get("a", it.get("answer", ""))).strip()
            mk = it.get("marks", marks)
            mp = it.get("marking_points", [])
            if isinstance(mp, str):
                mp = [mp]
            if not isinstance(mp, list):
                mp = []
                
            if not q or not ans:
                continue
            out_items.append({"q": q, "a": ans, "marks": mk, "marking_points": [str(x) for x in mp]})
        return out_items

    def generate_questions_with_answers(self, lecture_text: str) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "1-4 Marks (Recall)": self._qa_block(lecture_text, marks=2,  count=5),
            "5-10 Marks (Application)":   self._qa_block(lecture_text, marks=8,  count=4),
            "15-20 Marks (Synthesis)":  self._qa_block(lecture_text, marks=15, count=2),
        }

    # ---------------- MEGA-PROMPTS (Subject-Level) ----------------
    
    def generate_subject_summary(self, subject_corpus: str, syllabus: str) -> str:
        raise NotImplementedError("T5 cannot handle 100k+ tokens for Mega-Prompts.")

    def generate_subject_flashcards(self, subject_corpus: str, syllabus: str, n: int = 50) -> List[Dict[str, str]]:
        raise NotImplementedError("T5 cannot handle 100k+ tokens for Mega-Prompts.")

    def generate_subject_questions_with_answers(self, subject_corpus: str, syllabus: str) -> Dict[str, List[Dict[str, Any]]]:
        raise NotImplementedError("T5 cannot handle 100k+ tokens for Mega-Prompts.")
