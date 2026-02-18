from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

from app.audio import transcribe_whisper
from app.slides import load_slides
from app.align import align_slides_to_transcript

app = FastAPI(title="Low-Compute Multimodal Lecture Intelligence System")

class AlignRequest(BaseModel):
    audio_path: str
    slides_path: str
    whisper_model: str = "base"
    use_ocr_if_empty: bool = False

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/align")
def align(req: AlignRequest):
    transcript = transcribe_whisper(req.audio_path, model_size=req.whisper_model)
    slides = load_slides(req.slides_path, use_ocr_if_empty=req.use_ocr_if_empty)
    slides_payload = [{"slide_id": s.slide_id, "text": s.text} for s in slides]
    aligned = align_slides_to_transcript(slides_payload, transcript["segments"])
    return {"transcript": transcript, "slides": slides_payload, "alignment": aligned}
