"""Gemini API provider for lecture material generation.

Features:
- Bounded retry with exponential backoff (no infinite loops)
- Gemini JSON mode for structured outputs
- System/user message separation
- Per-task temperature control
- Token-aware prompt construction
"""
import os
import re
import logging
import time
from typing import Dict, Any, List, Optional

from app.providers.base_provider import BaseProvider
from app.json_utils import extract_json
from app.tokens import truncate_to_token_budget, get_model_limit

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
BASE_SLEEP = 15

# --- Prompt-echo detection (replaces the over-aggressive looks_bad) ---

_PROMPT_ECHO_RE = re.compile(
    r"^\s*#{1,3}\s*(role|task|rules|output|constraints|format|instructions)\b",
    re.IGNORECASE | re.MULTILINE,
)


def _is_bad_output(text: str) -> bool:
    """Check if LLM output is empty, trivial, or prompt echo."""
    if not text or not text.strip():
        return True
    t = text.strip()
    if len(t) < 60:
        return True
    if t.lower() in {"true", "false", "insufficient content", "none", "null"}:
        return True
    # Only flag if output STARTS with prompt-echo headers
    if _PROMPT_ECHO_RE.match(t):
        return True
    return False


class GeminiProvider(BaseProvider):
    """Google Gemini API provider with production-grade reliability."""

    def __init__(self):
        self._model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        from google import genai
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self._token_limit = get_model_limit(self._model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def used_internet(self) -> bool:
        return True

    # ----------------------------------------------------------------
    # Core generation with bounded retry + exponential backoff
    # ----------------------------------------------------------------

    def _generate(
        self,
        prompt: str,
        system_instruction: str = "",
        temperature: float = 0.3,
        json_mode: bool = False,
        retry_prompt: str = "",
    ) -> str:
        from google.genai import types
        from google.genai.errors import ClientError, ServerError

        config_kwargs: Dict[str, Any] = {"temperature": temperature}
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(
            system_instruction=system_instruction if system_instruction else None,
            **config_kwargs,
        )

        last_error: Optional[Exception] = None

        for attempt in range(MAX_RETRIES):
            try:
                resp = self.client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=config,
                )
                text = (resp.text or "").strip()

                if _is_bad_output(text) and retry_prompt and attempt == 0:
                    resp2 = self.client.models.generate_content(
                        model=self._model_name,
                        contents=retry_prompt,
                        config=config,
                    )
                    text2 = (resp2.text or "").strip()
                    if not _is_bad_output(text2):
                        return text2

                return text

            except ClientError as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                    sleep_time = BASE_SLEEP * (2 ** attempt)
                    logger.warning(
                        "Rate limited (429). Attempt %d/%d. Sleeping %ds...",
                        attempt + 1, MAX_RETRIES, sleep_time,
                    )
                    time.sleep(sleep_time)
                    last_error = e
                    continue
                raise

            except ServerError as e:
                err_str = str(e).lower()
                if any(k in err_str for k in ("503", "500", "unavailable", "overloaded")):
                    sleep_time = BASE_SLEEP * (2 ** attempt)
                    logger.warning(
                        "Server error. Attempt %d/%d. Sleeping %ds...",
                        attempt + 1, MAX_RETRIES, sleep_time,
                    )
                    time.sleep(sleep_time)
                    last_error = e
                    continue
                raise

        logger.error("Gemini API failed after %d retries. Last error: %s", MAX_RETRIES, last_error)
        return ""

    # ----------------------------------------------------------------
    # Token-aware text preparation
    # ----------------------------------------------------------------

    def _prep_text(self, text: str, reserve: int = 3000) -> str:
        return truncate_to_token_budget(text, self._token_limit, reserve=reserve)

    # ----------------------------------------------------------------
    # NOTES
    # ----------------------------------------------------------------

    NOTES_SYSTEM = """You are a university teaching assistant creating exam-ready study notes.
Output ONLY in this format:

Title: <core concept>
- <Explanation bullets: WHAT / WHY / HOW>
Key Terms:
- <term>: <definition>

If the content is too short or nonsensical, write exactly: Insufficient content
Do NOT echo the prompt. Do NOT add headers like "Instructions:" or "Role:"."""

    def generate_notes(self, lecture_text: str, speech_text: str = "", max_tokens: int = 400) -> str:
        from app.generate import clean_text
        lecture_text = clean_text(lecture_text)
        speech_text = clean_text(speech_text) if speech_text else ""

        combined = lecture_text
        if speech_text:
            combined = f"{lecture_text}\n\nSpeech context: {speech_text}"

        if len(combined) < 50:
            return "Insufficient content"

        content = self._prep_text(combined)
        prompt = f"Create concise study notes for this lecture excerpt:\n\n{content}"

        out = self._generate(
            prompt=prompt,
            system_instruction=self.NOTES_SYSTEM,
            temperature=0.3,
        )
        return "Insufficient content" if _is_bad_output(out) else out

    # ----------------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------------

    SUMMARY_SYSTEM = """You are an expert textbook author writing a detailed summary.
Create a deep, textbook-style explanation of the concepts — do NOT merely paraphrase.
Expand concepts clearly. Preserve the logical progression. Include examples.

Output format:
Overview:
<Deep explanation of the topics covered>

Core Principles:
- <Principle>: <Thorough explanation and intuition>

Examples:
- <Example demonstrating the concept>

If content is blank or nonsensical, write: Insufficient content
Do NOT echo the prompt. Do NOT add "Here is the summary"."""

    def generate_summary(self, lecture_text: str) -> str:
        if len(lecture_text) < 50:
            return "Insufficient content"

        content = self._prep_text(lecture_text)
        prompt = f"Summarize this lecture content:\n\n{content}"

        retry = f"The previous output was invalid. Summarize this text into Overview + Core Principles + Examples:\n\n{content[:2000]}"

        out = self._generate(
            prompt=prompt,
            system_instruction=self.SUMMARY_SYSTEM,
            temperature=0.3,
            retry_prompt=retry,
        )
        return "Insufficient content" if _is_bad_output(out) else out

    # ----------------------------------------------------------------
    # FLASHCARDS (JSON mode)
    # ----------------------------------------------------------------

    FLASHCARDS_SYSTEM = """You generate flashcards for academic study.
Each flashcard MUST have: question, answer, topic, memory_anchor, difficulty (1-3).

Output a JSON array. Example:
[{"question": "What is...", "answer": "It is...", "topic": "Topic A", "memory_anchor": "Think of it like...", "difficulty": 2}]

Rules:
- Flashcards must be conceptual, not trivial.
- 'memory_anchor' should be a quick analogy or mnemonic.
- Never return an empty array if there is educational content.
- If content is truly insufficient, return: []"""

    def generate_flashcards(self, lecture_text: str, n: int = 15) -> List[Dict[str, str]]:
        content = self._prep_text(lecture_text)
        prompt = f"Generate up to {n} flashcards from this lecture content:\n\n{content}"

        retry = (
            f"Your previous response was not valid JSON. "
            f"Return ONLY a JSON array of flashcard objects. Content:\n\n{content[:2000]}"
        )

        out = self._generate(
            prompt=prompt,
            system_instruction=self.FLASHCARDS_SYSTEM,
            temperature=0.5,
            json_mode=True,
            retry_prompt=retry,
        )

        parsed = extract_json(out)
        if not isinstance(parsed, list):
            return []

        from app.schemas import validate_flashcards
        valid = validate_flashcards(parsed)
        return [fc.model_dump() for fc in valid]

    # ----------------------------------------------------------------
    # QUESTIONS WITH ANSWERS (JSON mode)
    # ----------------------------------------------------------------

    QA_SYSTEM = """You create exam-style questions WITH detailed answers.
Generate questions across Bloom's Taxonomy levels:
- Marks 1-4: Recall / Understanding (Define, Identify)
- Marks 5-10: Application / Analysis (Solve, Compare, Explain how)
- Marks 15-20: Evaluation / Synthesis (Design, Critique, Prove)

Output a JSON array. Example:
[{"marks": 2, "question": "Define...", "answer": "...is defined as..."}]

Rules:
- Include FULL answers, not just question stubs.
- Generate at least 3-5 questions spanning all levels.
- No MCQ, no True/False.
- If content is truly insufficient, return: []"""

    def generate_questions_with_answers(self, lecture_text: str) -> Dict[str, List[Dict[str, Any]]]:
        content = self._prep_text(lecture_text)
        prompt = f"Create an exam pack of questions with answers for:\n\n{content}"

        retry = (
            f"Your previous response was not valid JSON. "
            f"Return ONLY a JSON array of question objects. Content:\n\n{content[:2000]}"
        )

        out = self._generate(
            prompt=prompt,
            system_instruction=self.QA_SYSTEM,
            temperature=0.4,
            json_mode=True,
            retry_prompt=retry,
        )

        parsed = extract_json(out)
        if not isinstance(parsed, list):
            return {"1-4 Marks (Recall)": [], "5-10 Marks (Application)": [], "15-20 Marks (Synthesis)": []}

        recall, application, synthesis = [], [], []

        from app.schemas import validate_exam_questions
        valid = validate_exam_questions(parsed)

        for eq in valid:
            obj = eq.model_dump()
            if eq.marks <= 4:
                recall.append(obj)
            elif eq.marks <= 10:
                application.append(obj)
            else:
                synthesis.append(obj)

        return {
            "1-4 Marks (Recall)": recall,
            "5-10 Marks (Application)": application,
            "15-20 Marks (Synthesis)": synthesis,
        }

    # ----------------------------------------------------------------
    # MEGA-PROMPTS (Subject-Level)
    # ----------------------------------------------------------------

    MEGA_SUMMARY_SYSTEM = """You are a university lecturer creating a comprehensive master study guide.
Combine the lecture corpus into ONE detailed study guide.
If a syllabus topic is not covered in the corpus, note it as "Not covered in lecture material" — do NOT fabricate content.

Output format:
Overview:
<Broad introduction to the subject>

Module-by-Module Breakdown:
- <Module Name>:
  <Detailed explanation + concepts + examples>

Master Glossary:
- <term>: <definition>"""

    def generate_subject_summary(self, subject_corpus: str, syllabus: str) -> str:
        corpus = self._prep_text(subject_corpus, reserve=5000)
        prompt = f"SYLLABUS:\n{syllabus}\n\nCORPUS:\n{corpus}"

        retry = f"Previous output was invalid. Create the study guide. SYLLABUS:\n{syllabus}\n\nCORPUS:\n{corpus[:3000]}"

        out = self._generate(
            prompt=prompt,
            system_instruction=self.MEGA_SUMMARY_SYSTEM,
            temperature=0.3,
            retry_prompt=retry,
        )
        return "Insufficient content" if _is_bad_output(out) else out

    MEGA_FLASHCARDS_SYSTEM = """Generate flashcards spanning an entire subject.
If a syllabus topic is not covered in the corpus, note it as "Not covered" — do NOT invent facts.
Output a JSON array of objects with keys: topic, question, answer, memory_anchor, difficulty."""

    def generate_subject_flashcards(self, subject_corpus: str, syllabus: str, n: int = 50) -> List[Dict[str, str]]:
        corpus = self._prep_text(subject_corpus, reserve=5000)
        prompt = f"Generate exactly {n} flashcards.\n\nSYLLABUS:\n{syllabus}\n\nCORPUS:\n{corpus}"

        retry = f"Return ONLY a JSON array of {n} flashcard objects.\n\nSYLLABUS:\n{syllabus}\n\nCORPUS:\n{corpus[:3000]}"

        out = self._generate(
            prompt=prompt,
            system_instruction=self.MEGA_FLASHCARDS_SYSTEM,
            temperature=0.5,
            json_mode=True,
            retry_prompt=retry,
        )

        parsed = extract_json(out)
        if not isinstance(parsed, list):
            return []

        from app.schemas import validate_flashcards
        valid = validate_flashcards(parsed)
        return [fc.model_dump() for fc in valid]

    MEGA_QA_SYSTEM = """Create a comprehensive exam bank WITH answers for an entire semester.
Output a JSON object with keys "1-2", "5", "10", "15".
Each value is an array of: {"q": "...", "a": "...", "marks": <int>, "marking_points": ["...", "..."]}.
If a syllabus topic is not covered, note it — do NOT fabricate answers."""

    def generate_subject_questions_with_answers(self, subject_corpus: str, syllabus: str) -> Dict[str, List[Dict[str, Any]]]:
        corpus = self._prep_text(subject_corpus, reserve=5000)
        prompt = f"SYLLABUS:\n{syllabus}\n\nCORPUS:\n{corpus}"

        retry = f"Return ONLY the JSON object. SYLLABUS:\n{syllabus}\n\nCORPUS:\n{corpus[:3000]}"

        out = self._generate(
            prompt=prompt,
            system_instruction=self.MEGA_QA_SYSTEM,
            temperature=0.4,
            json_mode=True,
            retry_prompt=retry,
        )

        parsed = extract_json(out)
        if not isinstance(parsed, dict):
            return {"1-2": [], "5": [], "10": [], "15": []}

        from app.schemas import validate_exam_questions

        def clean_section(key: str, default_marks: int) -> List[Dict[str, Any]]:
            arr = parsed.get(key, [])
            if not isinstance(arr, list):
                return []
            valid = validate_exam_questions(arr, default_marks=default_marks)
            return [eq.model_dump() for eq in valid]

        return {
            "1-2": clean_section("1-2", 2),
            "5": clean_section("5", 5),
            "10": clean_section("10", 10),
            "15": clean_section("15", 15),
        }
