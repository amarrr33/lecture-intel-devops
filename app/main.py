from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime
import uuid
import shutil

from app.audio import transcribe_whisper
from app.slides import load_slides
from app.align import align_slides_to_transcript
from app.generate import (
    generate_slidewise_notes,
    generate_lecture_materials
)
from app.smart_run import download_youtube_audio, merge_audios
from app.output import (
    save_json,
    save_alignment_markdown,
    save_notes_markdown,
    save_summary,
    save_flashcards_json,
    save_flashcards_markdown,
    save_exam_pack,
    save_metadata,
)

app = FastAPI(title="Lecture Intelligence System (Full Pipeline)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request Schema
# -------------------------

class ProcessRequest(BaseModel):
    audio_path: str
    slides_path: str
    whisper_model: str = "base"
    use_ocr_if_empty: bool = False
    output_dir: str | None = None


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
        model_size=req.whisper_model,
        out_dir=req.output_dir,
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

def _make_run_output_dir() -> Path:
    runs_root = Path("data") / "outputs"
    runs_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{stamp}_{uuid.uuid4().hex[:6]}"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _download_and_merge_links(links: List[str], run_dir: Path) -> Path:
    audio_parts_dir = run_dir / "audio_parts"
    audio_parts_dir.mkdir(parents=True, exist_ok=True)

    downloaded: List[Path] = []
    failed: List[str] = []

    for link in links:
        try:
            path = download_youtube_audio(link, audio_parts_dir)
            if path.exists():
                downloaded.append(path)
            else:
                failed.append(link)
        except Exception:
            failed.append(link)

    if not downloaded:
        raise HTTPException(
            status_code=400,
            detail="Could not download audio from provided YouTube links.",
        )

    combined_audio = run_dir / "combined_audio.mp3"
    merge_audios(downloaded, combined_audio)

    # Keep quick debug visibility for user.
    save_json(
        {
            "requested_links": links,
            "downloaded_count": len(downloaded),
            "failed_count": len(failed),
            "failed_links": failed,
            "combined_audio": str(combined_audio),
        },
        run_dir / "link_processing.json",
    )

    return combined_audio


@app.post("/process_upload")
async def process_upload(
    ppt: UploadFile = File(...),
    links: str = Form(default="[]"),
) -> Dict[str, Any]:
    """
    Frontend-compatible endpoint:
    - receives multipart form with `ppt` and `links`
    - runs existing processing pipeline
    """
    try:
        parsed_links = json.loads(links) if links else []
        if not isinstance(parsed_links, list):
            raise ValueError("`links` must be a JSON array")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid links payload: {exc}") from exc

    run_dir = _make_run_output_dir()
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(ppt.filename or "slides.pdf").name
    slides_path = inputs_dir / safe_name
    payload = await ppt.read()
    slides_path.write_bytes(payload)

    if not parsed_links:
        raise HTTPException(
            status_code=400,
            detail="At least one YouTube link is required.",
        )

    combined_audio_path = _download_and_merge_links(parsed_links, run_dir)

    req = ProcessRequest(
        audio_path=str(combined_audio_path),
        slides_path=str(slides_path),
        whisper_model="base",
        use_ocr_if_empty=False,
        output_dir=str(run_dir),
    )
    result = process_lecture(req)

    # Persist full artifacts for every request.
    save_json(result.get("transcript", {}), run_dir / "transcript.json")
    save_json({"slides": result.get("slides", [])}, run_dir / "slides.json")
    save_json(result.get("alignment", {}), run_dir / "alignment.json")
    save_json(result.get("notes", []), run_dir / "notes.json")
    save_summary(result.get("summary", ""), run_dir / "summary.md")
    save_flashcards_json(result.get("flashcards", []), run_dir / "flashcards.json")
    save_json(result.get("questions", {}), run_dir / "questions.json")
    save_alignment_markdown(result.get("alignment", {}), run_dir / "alignment.md")
    save_notes_markdown(result.get("notes", []), run_dir / "notes.md")
    save_flashcards_markdown(result.get("flashcards", []), run_dir / "flashcards.md")
    if isinstance(result.get("questions"), dict):
        save_exam_pack(result["questions"], run_dir / "exam_pack.md")
    else:
        save_exam_pack({}, run_dir / "exam_pack.md")

    save_metadata(
        {
            "slides_file": str(slides_path),
            "combined_audio": str(combined_audio_path),
            "links": parsed_links,
            "output_folder": str(run_dir),
        },
        run_dir / "metadata.json",
    )

    # Keep original uploaded slides copy in outputs folder root.
    shutil.copy(slides_path, run_dir / safe_name)

    return {
        **result,
        "output_folder": str(run_dir),
        "combined_audio_file": str(combined_audio_path),
    }