from pathlib import Path
import json
import time

from app.dataset import scan_dataset, outputs_dir
from app.audio import transcribe_whisper
from app.slides import load_slides
from app.align import align_slides_to_transcript
from app.generate import (
    generate_slidewise_notes,
    generate_lecture_materials,
)

from app.output import (
    save_json,
    save_alignment_markdown,
    save_notes_markdown,
    save_summary,
    save_flashcards_json,
    save_flashcards_csv,
    save_flashcards_markdown,
    save_exam_pack,
    save_metadata,
)

DATA_DIR = Path("data/lectures")


def main(limit: int = 0):
    lectures = scan_dataset(DATA_DIR)
    if limit:
        lectures = lectures[:limit]

    for lf in lectures:
        start_time = time.time()

        out = outputs_dir(lf.folder)
        out.mkdir(parents=True, exist_ok=True)

        print(f"\n🚀 Processing Lecture {lf.lecture_num:02d}")

        # ---------------------------
        # METADATA INIT
        # ---------------------------
        metadata = {
            "lecture": lf.lecture_num,
            "audio": str(lf.audio) if lf.audio else None,
            "slides": str(lf.slides) if lf.slides else None,
            "mode": "both" if (lf.audio and lf.slides)
                    else ("audio_only" if lf.audio else ("slides_only" if lf.slides else "empty")),
            "model_used": None,
            "used_internet": False,
            "runtime_sec": None,
        }

        transcript = None
        slides_payload = None
        aligned = None

        # ---------------------------
        # TRANSCRIPT (cached)
        # ---------------------------
        tpath = out / "transcript.json"
        if lf.audio:
            try:
                if tpath.exists():
                    transcript = json.loads(tpath.read_text(encoding="utf-8"))
                else:
                    transcript = transcribe_whisper(lf.audio, model_size="base")
                    save_json(transcript, tpath)
            except Exception as e:
                print(f"❌ Transcript failed: {e}")
                transcript = None

        # ---------------------------
        # SLIDES (cached)
        # ---------------------------
        spath = out / "slides.json"
        if lf.slides:
            try:
                if spath.exists():
                    slides_payload = json.loads(spath.read_text(encoding="utf-8"))
                else:
                    slides = load_slides(lf.slides, use_ocr_if_empty=True)
                    slides_payload = [{"slide_id": s.slide_id, "text": s.text} for s in slides]
                    save_json(slides_payload, spath)
            except Exception as e:
                print(f"❌ Slides failed: {e}")
                slides_payload = None

        # ---------------------------
        # ALIGNMENT
        # ---------------------------
        if transcript and slides_payload:
            apath = out / "alignment.json"

            try:
                if apath.exists():
                    aligned = json.loads(apath.read_text(encoding="utf-8"))
                else:
                    aligned = align_slides_to_transcript(
                        slides_payload,
                        transcript["segments"]
                    )
                    save_json(aligned, apath)

                save_alignment_markdown(aligned, out / "alignment.md")

            except Exception as e:
                print(f"❌ Alignment failed: {e}")
                aligned = None

        # ---------------------------
        # NOTES (slide-wise)
        # ---------------------------
        if aligned and slides_payload:
            try:
                notes = generate_slidewise_notes(slides_payload, aligned)

                if notes:
                    save_json(notes, out / "notes.json")
                    save_notes_markdown(notes, out / "notes.md")
                else:
                    print("⚠️ Notes empty")

            except Exception as e:
                print(f"❌ Notes failed: {e}")

        # ---------------------------
        # MATERIALS (summary + flashcards + QA)
        # ---------------------------
        if transcript:
            try:
                materials = generate_lecture_materials(transcript)

                summary = materials.get("summary", "")
                flashcards = materials.get("flashcards", [])
                qa = materials.get("qa_questions", {})

                # detect provider
                metadata["model_used"] = "gemini" if summary else "t5"
                metadata["used_internet"] = metadata["model_used"] == "gemini"

                # ---- SUMMARY ----
                if summary.strip():
                    save_summary(summary, out / "summary.md")
                else:
                    print("⚠️ Empty summary")

                # ---- FLASHCARDS ----
                if flashcards:
                    save_flashcards_json(flashcards, out / "flashcards.json")
                    save_flashcards_csv(flashcards, out / "flashcards.csv")
                    save_flashcards_markdown(flashcards, out / "flashcards.md")
                else:
                    print("⚠️ Flashcards empty")

                # ---- QA ----
                if qa:
                    save_json(qa, out / "exam_pack.json")
                    save_exam_pack(qa, out / "exam_pack.md")
                else:
                    print("⚠️ Questions empty")

            except Exception as e:
                print(f"❌ Materials generation failed: {e}")

        # ---------------------------
        # FINAL METADATA
        # ---------------------------
        metadata["runtime_sec"] = round(time.time() - start_time, 2)
        save_metadata(metadata, out / "metadata.json")

        print(f"✅ Lecture {lf.lecture_num:02d} done → {out}")


if __name__ == "__main__":
    main()