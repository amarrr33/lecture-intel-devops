from __future__ import annotations

from typing import List, Dict, Any
import numpy as np

from app.align import embed_texts  # ✅ only this

# ---------------------------
# cosine similarity
# ---------------------------

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ---------------------------
# MAIN SEGMENTATION
# ---------------------------

def segment_topics(
    segments: List[Dict[str, Any]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold: float = 0.72,
    min_chunk_size: int = 2,
    max_chunk_size: int = 12,
) -> List[Dict[str, Any]]:

    segs = [s for s in segments if (s.get("text") or "").strip()]
    if not segs:
        return []

    texts = [s["text"].strip() for s in segs]

    # embeddings
    E = embed_texts(texts, model_name=model_name)

    topics = []

    current_chunk = {
        "start": segs[0]["start"],
        "end": segs[0]["end"],
        "texts": [texts[0]],
        "embs": [E[0]],
    }

    def finalize_chunk(chunk):
        return {
            "start": chunk["start"],
            "end": chunk["end"],
            "text": " ".join(chunk["texts"]).strip(),
            "length": len(chunk["texts"]),
        }

    for i in range(1, len(segs)):
        cur_emb = E[i]
        prev_emb = np.mean(current_chunk["embs"], axis=0)

        sim = _cos_sim(prev_emb, cur_emb)

        should_split = (
            sim < similarity_threshold and len(current_chunk["texts"]) >= min_chunk_size
        ) or len(current_chunk["texts"]) >= max_chunk_size

        if should_split:
            topics.append(finalize_chunk(current_chunk))

            current_chunk = {
                "start": segs[i]["start"],
                "end": segs[i]["end"],
                "texts": [texts[i]],
                "embs": [cur_emb],
            }
        else:
            current_chunk["texts"].append(texts[i])
            current_chunk["embs"].append(cur_emb)
            current_chunk["end"] = segs[i]["end"]

    if current_chunk["texts"]:
        topics.append(finalize_chunk(current_chunk))

    return topics


# ---------------------------
# MERGE SMALL TOPICS
# ---------------------------

def merge_small_topics(topics, min_length=2):

    if not topics:
        return []

    merged = []
    buffer = None

    for t in topics:
        if t["length"] < min_length:
            if buffer is None:
                buffer = t
            else:
                buffer["text"] += " " + t["text"]
                buffer["end"] = t["end"]
                buffer["length"] += t["length"]
        else:
            if buffer:
                t["text"] = buffer["text"] + " " + t["text"]
                t["start"] = buffer["start"]
                t["length"] += buffer["length"]
                buffer = None
            merged.append(t)

    if buffer:
        merged.append(buffer)

    return merged


# ---------------------------
# FINAL ENTRY
# ---------------------------

def extract_topics(segments):

    raw = segment_topics(segments)
    final = merge_small_topics(raw)

    return final