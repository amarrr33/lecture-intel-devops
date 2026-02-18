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
    src.add_argument("--youtube", type=str, help="YouTube link")
    src.add_argument("--audio", type=str, help="Path to audio file (.wav/.mp3)")

    p.add_argument("--slides", type=str, required=True, help="Path to slides (.pptx or .pdf)")
    p.add_argument("--whisper-model", type=str, default="base", choices=["tiny","base","small","medium"])
    p.add_argument("--use-ocr-if-empty", action="store_true")
    p.add_argument("--outdir", type=str, default="runs/out")

    # full stack options
    p.add_argument("--confidence-model", type=str, default="runs/models/confidence_mlp.pt")
    p.add_argument("--confidence-threshold", type=float, default=0.55)
    p.add_argument("--refine-rounds", type=int, default=2)
    p.add_argument("--refine-window", type=int, default=6)

    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) audio -> transcript
    if args.youtube:
        audio_path = download_audio(args.youtube, out_dir=str(outdir / "audio"))
    else:
        audio_path = Path(args.audio)
    transcript = transcribe_whisper(audio_path, model_size=args.whisper_model)

    # 2) slides -> text
    slides = load_slides(args.slides, use_ocr_if_empty=args.use_ocr_if_empty)
    slides_payload = [{"slide_id": s.slide_id, "text": s.text} for s in slides]

    # 3) align (MiniLM + MLP + refinement)
    aligned = align_slides_to_transcript(
        slides_payload,
        transcript["segments"],
        confidence_model_path=args.confidence_model if Path(args.confidence_model).exists() else None,
        confidence_threshold=args.confidence_threshold,
        refine_rounds=args.refine_rounds,
        refine_window=args.refine_window,
    )

    # 4) generate notes (T5-small)
    notes = generate_slidewise_notes(slides_payload, aligned)

    # 5) save outputs
    payload = {"transcript": transcript, "slides": slides_payload, "alignment": aligned, "notes": notes}
    save_json(payload, outdir / "result.json")
    save_markdown(aligned, outdir / "alignment.md")

    # also save notes as simple markdown
    notes_md = ["# Slide-wise Notes (T5-small)\n"]
    for n in notes:
        notes_md.append(f"## {n['slide_id']}  (conf={n['confidence']:.3f})")
        notes_md.append(n["note"])
        notes_md.append("")
    (outdir / "notes.md").write_text("\n".join(notes_md), encoding="utf-8")

    print(f"\n✅ Done.\n- JSON: {outdir/'result.json'}\n- Align: {outdir/'alignment.md'}\n- Notes: {outdir/'notes.md'}\n")

if __name__ == "__main__":
    main()
