from __future__ import annotations
from app.embeddings import embed_texts
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

# ---- singleton model ----
_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL


def embed(texts: List[str]) -> np.ndarray:
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ---------------------------
# 🔥 BETTER CHUNKING (KEY FIX)
# ---------------------------
def build_context_chunks(segments: List[Dict[str, Any]], window: int = 5) -> List[Dict]:
    """
    Instead of 1 segment → use sliding window context (VERY IMPORTANT)
    """
    chunks = []

    for i in range(len(segments)):
        lo = max(0, i - window)
        hi = min(len(segments), i + window)

        text = " ".join(s["text"] for s in segments[lo:hi]).strip()

        chunks.append({
            "start": segments[lo]["start"],
            "end": segments[hi - 1]["end"],
            "text": text,
            "center_idx": i
        })

    return chunks


# ---------------------------
# MAIN ALIGNMENT
# ---------------------------
def align_slides_to_transcript(
    slides: List[Dict[str, str]],
    segments: List[Dict[str, Any]],
) -> Dict[str, Any]:

    # clean empty segments
    segments = [s for s in segments if s.get("text", "").strip()]
    if not segments:
        raise ValueError("No transcript text found")

    # 🔥 USE CONTEXT CHUNKS
    chunks = build_context_chunks(segments, window=5)

    slide_texts = [s.get("text", "") or "(empty)" for s in slides]
    chunk_texts = [c["text"] for c in chunks]

    S = embed(slide_texts)
    C = embed(chunk_texts)

    sims = cosine_similarity(S, C)

    alignments = []

    for i, slide in enumerate(slides):
        best_idx = int(np.argmax(sims[i]))
        best_chunk = chunks[best_idx]

        confidence = float(sims[i][best_idx])

        alignments.append({
            "slide_id": slide["slide_id"],
            "confidence": confidence,
            "segment": {
                "start": best_chunk["start"],
                "end": best_chunk["end"],
                "text": best_chunk["text"]
            }
        })

    return {
        "alignments": alignments,
        "stats": {
            "num_slides": len(slides),
            "num_segments": len(segments),
            "num_chunks": len(chunks),
            "avg_confidence": float(np.mean([a["confidence"] for a in alignments]))
        }
    }