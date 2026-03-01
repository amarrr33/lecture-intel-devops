"""Main pipeline runner: extraction → alignment → segmentation → generation."""
from pathlib import Path
import json
import logging
import time
from typing import List, Optional

from app.dataset import scan_dataset, outputs_dir
from app.audio import transcribe_whisper
from app.slides import load_slides
from app.align import align_slides_to_transcript
from app.segment import segment_transcript
from app.generate import (
    generate_lecture_materials,
    flashcards_to_markdown,
    exam_pack_to_markdown,
    clean_text,
)
from app.providers import get_provider

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/lectures")
SYLLABUS_FILE = Path("data/syllabus.txt")
WHISPER_MODEL = "small"  # Upgraded from "base" for better academic transcription


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    if text and text.strip():
        path.write_text(text, encoding="utf-8")


def main(target_lectures: Optional[List[int]] = None):
    lectures = scan_dataset(DATA_DIR)

    if target_lectures:
        lectures = [lf for lf in lectures if lf.lecture_num in target_lectures]

    if not lectures:
        logger.warning("No lectures found in %s", DATA_DIR)
        return

    # Initialize provider
    provider = get_provider()
    logger.info("Provider: %s (%s)", provider.__class__.__name__, provider.model_name)

    syllabus_text = (
        SYLLABUS_FILE.read_text(encoding="utf-8")
        if SYLLABUS_FILE.exists()
        else "No syllabus provided."
    )

    all_transcripts: List[str] = []

    # =====================================================================
    # PASS 1: Extraction (Transcription + Slide OCR)
    # =====================================================================
    logger.info("--- PASS 1: Data Extraction (%d lectures) ---", len(lectures))

    for lf in lectures:
        out = outputs_dir(lf.folder)
        out.mkdir(parents=True, exist_ok=True)

        logger.info("Extracting Lecture %02d...", lf.lecture_num)

        # --- Transcript (cached) ---
        tpath = out / "transcript.json"
        if lf.audio:
            if not tpath.exists():
                transcript = transcribe_whisper(lf.audio, model_size=WHISPER_MODEL)
                _write_json(tpath, transcript)

        # --- Slides (cached) ---
        spath = out / "slides.json"
        if lf.slides and not spath.exists():
            slides = load_slides(lf.slides, use_ocr_if_empty=True)
            slides_payload = [{"slide_id": s.slide_id, "text": s.text} for s in slides]
            _write_json(spath, slides_payload)

        # Collect transcripts for subject-level generation
        if tpath.exists():
            td = json.loads(tpath.read_text(encoding="utf-8"))
            text = td.get("text", "")
            if text:
                all_transcripts.append(f"--- LECTURE {lf.lecture_num:02d} ---\n{text}\n")

    # =====================================================================
    # PASS 2: Alignment + Segmentation + Generation (per lecture)
    # =====================================================================
    logger.info("--- PASS 2: Alignment + Generation (%d lectures) ---", len(lectures))

    for lf in lectures:
        out = outputs_dir(lf.folder)
        tpath = out / "transcript.json"
        spath = out / "slides.json"

        # Check for existing completed output (checkpoint/resume)
        meta_path = out / "metadata.json"
        if meta_path.exists():
            logger.info("Lecture %02d already processed (checkpoint). Skipping.", lf.lecture_num)
            continue

        transcript_data = {}
        slides_data = []

        if tpath.exists():
            transcript_data = json.loads(tpath.read_text(encoding="utf-8"))
        if spath.exists():
            slides_data = json.loads(spath.read_text(encoding="utf-8"))

        if not transcript_data and not slides_data:
            logger.warning("Skipping Lecture %02d - No data.", lf.lecture_num)
            continue

        logger.info("Processing Lecture %02d...", lf.lecture_num)
        start_time = time.time()

        # --- Step 1: Alignment ---
        alignment = None
        segments = transcript_data.get("segments", [])
        if segments and slides_data:
            try:
                alignment = align_slides_to_transcript(slides_data, segments)
                _write_json(out / "alignment.json", alignment)
                logger.info(
                    "Aligned %d slides to %d segments (avg_conf=%.3f)",
                    len(slides_data), len(segments),
                    alignment.get("stats", {}).get("avg_confidence", 0),
                )
            except Exception as e:
                logger.warning("Alignment failed for Lecture %02d: %s", lf.lecture_num, e)
                alignment = None

        # --- Step 2: Topic Segmentation ---
        topic_chunks = None
        if segments:
            try:
                topic_chunks = segment_transcript(
                    segments,
                    alignment=alignment,
                    min_words=80,
                )
                _write_json(
                    out / "topics.json",
                    [tc.to_dict() for tc in topic_chunks],
                )
                logger.info("Segmented into %d topics", len(topic_chunks))
            except Exception as e:
                logger.warning("Segmentation failed for Lecture %02d: %s", lf.lecture_num, e)
                topic_chunks = None

        # --- Step 3: Build slides lookup ---
        slides_by_id = {
            s["slide_id"]: s.get("text", "") for s in slides_data
        } if slides_data else {}

        # --- Step 4: Generate Materials ---
        materials = generate_lecture_materials(
            transcript_data,
            slides_data,
            provider,
            topic_chunks=topic_chunks,
            slides_by_id=slides_by_id,
        )

        # --- Step 5: Write Outputs ---
        _write_text(out / "summary.md", materials.get("summary", ""))

        flashcards = materials.get("flashcards", []) or []
        _write_json(out / "flashcards.json", flashcards)
        _write_text(out / "flashcards.md", flashcards_to_markdown(flashcards))

        qa = materials.get("qa_questions", {}) or {}
        _write_json(out / "exam_pack.json", qa)
        _write_text(out / "exam_pack.md", exam_pack_to_markdown(qa))

        _write_text(out / "notes.md", materials.get("notes", ""))

        elapsed = time.time() - start_time
        meta = {
            "mode": "topic_level_pipeline",
            "lecture_id": lf.lecture_num,
            "provider_used": provider.__class__.__name__,
            "model_used": provider.model_name,
            "internet_usage_flag": provider.used_internet,
            "num_topics": len(topic_chunks) if topic_chunks else 0,
            "num_slides": len(slides_data),
            "num_segments": len(segments),
            "alignment_avg_confidence": (
                alignment.get("stats", {}).get("avg_confidence", 0)
                if alignment else None
            ),
            "num_flashcards": len(flashcards),
            "num_questions": sum(len(v) for v in qa.values()),
            "runtime_seconds": round(elapsed, 2),
        }
        _write_json(out / "metadata.json", meta)

        logger.info(
            "Lecture %02d complete in %.1fs (%d flashcards, %d questions)",
            lf.lecture_num, elapsed, len(flashcards), meta["num_questions"],
        )

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Configure structured logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    target_lectures = [1, 2]
    main(target_lectures=target_lectures)
