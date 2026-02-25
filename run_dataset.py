from pathlib import Path
import json

from app.dataset import scan_dataset, outputs_dir
from app.audio import transcribe_whisper
from app.slides import load_slides
from app.align import align_slides_to_transcript
from app.topic_segmenter import segment_topics  # ✅ NEW
from app.generate import (
    generate_slidewise_notes,
    generate_lecture_materials,
    flashcards_to_markdown,
)

DATA_DIR = Path("data/lectures")


def build_topic_map(topics):
    """
    Map time → topic text
    """
    mapping = []
    for t in topics:
        mapping.append({
            "start": t["start"],
            "end": t["end"],
            "text": t["text"]
        })
    return mapping


def get_topic_for_segment(segment, topic_map):
    """
    Find which topic a segment belongs to
    """
    s = segment["start"]
    for t in topic_map:
        if t["start"] <= s <= t["end"]:
            return t["text"]
    return ""


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
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))

        # ---------------- TRANSCRIPT ----------------
        transcript = None
        if lf.audio:
            tpath = out / "transcript.json"
            if tpath.exists():
                transcript = json.loads(tpath.read_text())
            else:
                transcript = transcribe_whisper(lf.audio, model_size="base")
                tpath.write_text(json.dumps(transcript, indent=2))

        # ---------------- SLIDES ----------------
        slides_payload = None
        if lf.slides:
            spath = out / "slides.json"
            if spath.exists():
                slides_payload = json.loads(spath.read_text())
            else:
                slides = load_slides(lf.slides)
                slides_payload = [{"slide_id": s.slide_id, "text": s.text} for s in slides]
                spath.write_text(json.dumps(slides_payload, indent=2))

        # ---------------- TOPIC SEGMENTATION 🔥 ----------------
        topics = None
        if transcript:
            topics = segment_topics(transcript["segments"])
            (out / "topics.json").write_text(json.dumps(topics, indent=2))

        topic_map = build_topic_map(topics) if topics else []

        # ---------------- ALIGNMENT ----------------
        aligned = None
        if transcript and slides_payload:
            apath = out / "alignment.json"
            if apath.exists():
                aligned = json.loads(apath.read_text())
            else:
                aligned = align_slides_to_transcript(slides_payload, transcript["segments"])
                apath.write_text(json.dumps(aligned, indent=2))

        # ---------------- FUSION + NOTES 🔥 ----------------
        if aligned:
            enriched_alignments = []

            for a in aligned["alignments"]:
                seg = a["segment"]
                topic_text = get_topic_for_segment(seg, topic_map)

                enriched_alignments.append({
                    **a,
                    "topic_text": topic_text
                })

            notes = generate_slidewise_notes(slides_payload, {"alignments": enriched_alignments})

            (out / "notes.json").write_text(json.dumps(notes, indent=2))
            (out / "notes.md").write_text(
                "\n\n".join(f"### {n['slide_id']}\n{n['note']}" for n in notes)
            )

        # ---------------- GLOBAL MATERIALS ----------------
        if transcript:
            materials = generate_lecture_materials(transcript)

            if materials["summary"]:
                (out / "summary.md").write_text(materials["summary"])

            (out / "flashcards.json").write_text(json.dumps(materials["flashcards"], indent=2))

            fc_md = flashcards_to_markdown(materials["flashcards"])
            if fc_md:
                (out / "flashcards.md").write_text(fc_md)

            (out / "questions_with_answers.json").write_text(
                json.dumps(materials["qa_questions"], indent=2)
            )

        print(f"✅ Lecture {lf.lecture_num} done → {out}")


if __name__ == "__main__":
    main()