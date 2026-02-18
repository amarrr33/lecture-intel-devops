from __future__ import annotations
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class T5Generator:
    def __init__(self, model_name: str = "t5-small", device: str = "cpu"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def generate_notes(self, slide_text: str, speech_text: str, max_new_tokens: int = 120) -> str:
        prompt = (
            "summarize lecture notes:\n"
            f"slide: {slide_text}\n"
            f"speech: {speech_text}\n"
        )
        inp = self.tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                no_repeat_ngram_size=3,
            )
        return self.tok.decode(out[0], skip_special_tokens=True).strip()

def generate_slidewise_notes(slides: List[Dict[str, str]], alignment: Dict[str, Any]) -> List[Dict[str, Any]]:
    gen = T5Generator(model_name="t5-small", device="cpu")
    notes = []
    by_id = {s["slide_id"]: s["text"] for s in slides}

    for a in alignment["alignments"]:
        sid = a["slide_id"]
        slide_text = by_id.get(sid, "")
        speech_text = a["segment"]["text"]
        note = gen.generate_notes(slide_text, speech_text)
        notes.append({
            "slide_id": sid,
            "note": note,
            "confidence": a["confidence"],
            "time": [a["segment"]["start"], a["segment"]["end"]],
        })
    return notes
