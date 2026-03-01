"""
Pydantic schemas for pipeline outputs.

All LLM-generated structured outputs are validated against these
schemas before being written to disk.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class Flashcard(BaseModel):
    topic: str
    question: str
    answer: str
    memory_anchor: str = ""
    difficulty: int = Field(default=2, ge=1, le=3)

    @field_validator("difficulty", mode="before")
    @classmethod
    def coerce_difficulty(cls, v: Any) -> int:
        try:
            return max(1, min(3, int(v)))
        except (TypeError, ValueError):
            return 2


class ExamQuestion(BaseModel):
    q: str
    a: str
    marks: int = Field(default=2, ge=1)
    marking_points: List[str] = Field(default_factory=list)

    @field_validator("marking_points", mode="before")
    @classmethod
    def coerce_marking_points(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [v] if v.strip() else []
        if isinstance(v, list):
            return [str(x) for x in v if x]
        return []


class TopicSegment(BaseModel):
    topic_id: str
    text: str
    start_time: float = 0.0
    end_time: float = 0.0
    slide_ids: List[str] = Field(default_factory=list)


class ExamPack(BaseModel):
    """Exam questions grouped by mark category."""
    recall: List[ExamQuestion] = Field(default_factory=list, alias="1-4 Marks (Recall)")
    application: List[ExamQuestion] = Field(default_factory=list, alias="5-10 Marks (Application)")
    synthesis: List[ExamQuestion] = Field(default_factory=list, alias="15-20 Marks (Synthesis)")

    model_config = {"populate_by_name": True}


class LectureMaterials(BaseModel):
    notes: str = ""
    summary: str = ""
    flashcards: List[Flashcard] = Field(default_factory=list)
    qa_questions: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)


# --- Validation helpers ---

def validate_flashcards(raw: List[Dict[str, Any]]) -> List[Flashcard]:
    """Validate and clean a list of raw flashcard dicts."""
    valid = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        # Normalize key variations
        q = str(item.get("question", item.get("q", ""))).strip()
        a = str(item.get("answer", item.get("a", ""))).strip()
        topic = str(item.get("topic", "")).strip()
        if not (q and a and topic):
            continue
        try:
            fc = Flashcard(
                topic=topic,
                question=q,
                answer=a,
                memory_anchor=str(item.get("memory_anchor", "")).strip(),
                difficulty=item.get("difficulty", 2),
            )
            valid.append(fc)
        except Exception:
            continue

    logger.debug("Validated %d/%d flashcards", len(valid), len(raw))
    return valid


def validate_exam_questions(raw: List[Dict[str, Any]], default_marks: int = 2) -> List[ExamQuestion]:
    """Validate and clean a list of raw exam question dicts."""
    valid = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        q = str(item.get("q", item.get("question", ""))).strip()
        a = str(item.get("a", item.get("answer", ""))).strip()
        if not (q and a):
            continue
        try:
            eq = ExamQuestion(
                q=q,
                a=a,
                marks=item.get("marks", default_marks),
                marking_points=item.get("marking_points", []),
            )
            valid.append(eq)
        except Exception:
            continue

    logger.debug("Validated %d/%d exam questions", len(valid), len(raw))
    return valid
