from pathlib import Path
import json

from app.dataset import scan_dataset, outputs_dir
from app.audio import transcribe_whisper
from app.slides import load_slides
from app.align import align_slides_to_transcript

# ✅ USE FINAL PIPELINE
from app.topic_segmenter import extract_topics

from app.generate import (
    generate_slidewise_notes,
    generate_lecture_materials,
    flashcards_to_markdown,
)

DATA_DIR = Path("data/lectures")


# ---------------------------
# TOPIC HELPERS
# ---------------------------

def build_topic_map(topics):
    return [
        {
            "start": t["start"],
            "end": t["end"],
            "text": t["text"],
        }
        for t in topics
    ]


def get_topic_for_segment(segment, topic_map):
    s = segment.get("start", 0.0)

    for t in topic_map:
        if t["start"] <= s <= t["end"]:
            return t["text"]

    return ""


# ---------------------------
# MAIN PIPELINE
# ---------------------------

def main(limit: int = 0):

    lectures = scan_dataset(DATA_DIR)
    if limit:
        lectures = lectures[:limit]

    for lf in lectures:

        out = outputs_dir(lf.folder)
        out.mkdir(parents=True, exist_ok=True)

        # ---------------- METADATA ----------------
        meta = {
            "lecture": lf.lecture_num,
            "audio": lf.audio.name if lf.audio else None,
            "slides": lf.slides.name if lf.slides else None,
            "mode": "both" if (lf.audio and lf.slides)
                    else ("audio_only" if lf.audio
                    else ("slides_only" if lf.slides else "empty")),
        }

        (out / "metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        # ---------------- TRANSCRIPT ----------------
        transcript = None
        if lf.audio:
            tpath = out / "transcript.json"

            if tpath.exists():
                transcript = json.loads(tpath.read_text(encoding="utf-8"))
            else:
                transcript = transcribe_whisper(lf.audio, model_size="base")
                tpath.write_text(json.dumps(transcript, indent=2), encoding="utf-8")

        # ---------------- SLIDES ----------------
        slides_payload = None
        if lf.slides:
            spath = out / "slides.json"

            if spath.exists():
                slides_payload = json.loads(spath.read_text(encoding="utf-8"))
            else:
                slides = load_slides(lf.slides, use_ocr_if_empty=False)
                slides_payload = [{"slide_id": s.slide_id, "text": s.text} for s in slides]
                spath.write_text(json.dumps(slides_payload, indent=2), encoding="utf-8")

        # ---------------- 🔥 TOPIC SEGMENTATION ----------------
        topics = None
        topic_map = []

        if transcript:
            tpath = out / "topics.json"

            if tpath.exists():
                topics = json.loads(tpath.read_text(encoding="utf-8"))
            else:
                topics = extract_topics(transcript["segments"])
                tpath.write_text(json.dumps(topics, indent=2), encoding="utf-8")

            topic_map = build_topic_map(topics)

        # ---------------- ALIGNMENT ----------------
        aligned = None

        if transcript and slides_payload:
            apath = out / "alignment.json"

            if apath.exists():
                aligned = json.loads(apath.read_text(encoding="utf-8"))
            else:
                aligned = align_slides_to_transcript(
                    slides_payload,
                    transcript["segments"]
                )
                apath.write_text(json.dumps(aligned, indent=2), encoding="utf-8")

        # ---------------- 🔥 FUSION + NOTES ----------------
        if aligned:

            enriched_alignments = []

            for a in aligned["alignments"]:
                seg = a["segment"]

                topic_text = get_topic_for_segment(seg, topic_map)

                enriched_alignments.append({
                    **a,
                    "topic": topic_text  # ✅ IMPORTANT CHANGE
                })

            # pass enriched alignment
            notes = generate_slidewise_notes(
                slides_payload,
                {"alignments": enriched_alignments}
            )

            (out / "notes.json").write_text(
                json.dumps(notes, indent=2),
                encoding="utf-8"
            )

            (out / "notes.md").write_text(
                "\n\n".join(
                    f"### {n['slide_id']} (conf={n['confidence']:.2f})\n{n['note']}"
                    for n in notes
                ),
                encoding="utf-8",
            )

        # ---------------- GLOBAL MATERIALS ----------------
        if transcript:

            materials = generate_lecture_materials(transcript)

            summary = materials.get("summary", "") or ""
            if summary.strip():
                (out / "summary.md").write_text(summary, encoding="utf-8")

            flashcards = materials.get("flashcards", []) or []
            (out / "flashcards.json").write_text(
                json.dumps(flashcards, indent=2),
                encoding="utf-8"
            )

            fc_md = flashcards_to_markdown(flashcards)
            if fc_md.strip():
                (out / "flashcards.md").write_text(fc_md, encoding="utf-8")

            qa = materials.get("qa_questions", {}) or {}
            (out / "questions_with_answers.json").write_text(
                json.dumps(qa, indent=2),
                encoding="utf-8"
            )

        print(f"✅ Lecture {lf.lecture_num:02d} done → {out}")


if __name__ == "__main__":
    main()