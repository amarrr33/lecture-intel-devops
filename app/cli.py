from __future__ import annotations

import argparse
from pathlib import Path

from app.audio import download_audio, transcribe_whisper
from app.slides import load_slides
from app.align import align_slides_to_transcript
from app.output import save_json, save_markdown
from app.generate import generate_slidewise_notes


def main():
    p = argparse.ArgumentParser("lecture-intel-full")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--youtube", type=str)
    src.add_argument("--audio", type=str)

    p.add_argument("--slides", type=str, required=True)
    p.add_argument("--whisper-model", type=str, default="base")
    p.add_argument("--use-ocr-if-empty", action="store_true")
    p.add_argument("--outdir", type=str, default="runs/out")

    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("🚀 Starting pipeline...")

    # ---------------------------
    # 1) AUDIO → TRANSCRIPT
    # ---------------------------
    print("🎤 Processing audio...")

    if args.youtube:
        audio_path = download_audio(args.youtube, out_dir=str(outdir / "audio"))
    else:
        audio_path = Path(args.audio)

    transcript = transcribe_whisper(audio_path, model_size=args.whisper_model)

    if not transcript["segments"]:
        raise RuntimeError("❌ Whisper failed: no transcript segments found")

    # ---------------------------
    # 2) SLIDES → TEXT
    # ---------------------------
    print("📊 Extracting slides...")

    slides = load_slides(args.slides, use_ocr_if_empty=args.use_ocr_if_empty)

    slides_payload = [
        {"slide_id": s.slide_id, "text": s.text or ""}
        for s in slides
    ]

    # ---------------------------
    # 3) ALIGN (NEW VERSION)
    # ---------------------------
    print("🔗 Aligning slides with speech (context-aware)...")

    aligned = align_slides_to_transcript(
        slides_payload,
        transcript["segments"]
    )

    # ---------------------------
    # 4) GENERATE NOTES
    # ---------------------------
    print("🧠 Generating notes...")

    notes = generate_slidewise_notes(slides_payload, aligned)

    # ---------------------------
    # 🔥 FALLBACK: IF NOTES TOO WEAK
    # ---------------------------
    empty_notes = sum(1 for n in notes if not n["note"].strip())

    if empty_notes > len(notes) * 0.5:
        print("⚠️ Many notes are weak → using transcript fallback")

        full_text = transcript["text"]

        notes = [{
            "slide_id": "FULL_LECTURE",
            "note": full_text[:3000],
            "confidence": 1.0,
            "time": [0, 0]
        }]

    # ---------------------------
    # 5) SAVE OUTPUTS
    # ---------------------------
    print("💾 Saving outputs...")

    payload = {
        "transcript": transcript,
        "slides": slides_payload,
        "alignment": aligned,
        "notes": notes
    }

    save_json(payload, outdir / "result.json")
    save_markdown(aligned, outdir / "alignment.md")

    # notes markdown
    notes_md = ["# Slide-wise Notes\n"]

    for n in notes:
        notes_md.append(f"## {n['slide_id']}  (conf={n['confidence']:.3f})")
        notes_md.append(n["note"])
        notes_md.append("")

    (outdir / "notes.md").write_text("\n".join(notes_md), encoding="utf-8")

    print(f"""
✅ DONE

📄 JSON: {outdir/'result.json'}
🔗 Alignment: {outdir/'alignment.md'}
📝 Notes: {outdir/'notes.md'}
""")


if __name__ == "__main__":
    main()