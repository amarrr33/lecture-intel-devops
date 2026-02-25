from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Any

from app.audio import transcribe_whisper
from app.slides import load_slides
from app.align import align_slides_to_transcript
from app.generate import (
    generate_slidewise_notes,
    generate_lecture_materials
)

app = FastAPI(title="Lecture Intelligence System (Full Pipeline)")

# -------------------------
# Request Schema
# -------------------------

class ProcessRequest(BaseModel):
    audio_path: str
    slides_path: str
    whisper_model: str = "base"
    use_ocr_if_empty: bool = False


# -------------------------
# Health Check
# -------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------
# FULL PIPELINE ENDPOINT
# -------------------------

@app.post("/process")
def process_lecture(req: ProcessRequest) -> Dict[str, Any]:
    """
    FULL PIPELINE:
    1) Audio → Transcript
    2) Slides → Text
    3) Align Slides ↔ Transcript
    4) Generate Notes (slide-wise)
    5) Generate Summary + Flashcards + Questions
    """

    # -------------------------
    # 1. TRANSCRIBE AUDIO
    # -------------------------
    transcript = transcribe_whisper(
        req.audio_path,
        model_size=req.whisper_model
    )

    # -------------------------
    # 2. LOAD SLIDES
    # -------------------------
    slides = load_slides(
        req.slides_path,
        use_ocr_if_empty=req.use_ocr_if_empty
    )

    slides_payload = [
        {"slide_id": s.slide_id, "text": s.text}
        for s in slides
    ]

    # -------------------------
    # 3. ALIGNMENT
    # -------------------------
    aligned = align_slides_to_transcript(
        slides_payload,
        transcript["segments"]
    )

    # -------------------------
    # 4. SLIDE-WISE NOTES (USES BOTH SLIDE + SPEECH)
    # -------------------------
    notes = generate_slidewise_notes(
        slides_payload,
        aligned
    )

    # -------------------------
    # 5. LECTURE MATERIALS (USES FULL TRANSCRIPT)
    # -------------------------
    materials = generate_lecture_materials(transcript)

    # -------------------------
    # FINAL RESPONSE
    # -------------------------
    return {
        "transcript": transcript,
        "slides": slides_payload,
        "alignment": aligned,
        "notes": notes,
        "summary": materials.get("summary", ""),
        "flashcards": materials.get("flashcards", []),
        "questions": materials.get("qa_questions", {}),
    }