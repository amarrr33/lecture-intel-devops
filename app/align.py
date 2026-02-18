from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    embs = m.encode(texts, normalize_embeddings=True)
    return np.asarray(embs, dtype=np.float32)

def _make_chunks(segments: List[Dict[str, Any]], chunk_size: int = 2) -> List[Dict[str, Any]]:
    """
    Merge adjacent transcript segments into larger chunks to help refinement.
    """
    out = []
    i = 0
    while i < len(segments):
        j = min(len(segments), i + chunk_size)
        text = " ".join(s.get("text","").strip() for s in segments[i:j]).strip()
        out.append({
            "start": float(segments[i].get("start", 0.0)),
            "end": float(segments[j-1].get("end", 0.0)),
            "text": text,
            "source_idxs": list(range(i, j))
        })
        i = j
    return out

def align_slides_to_transcript(
    slide_texts: List[Dict[str, str]],
    segments: List[Dict[str, Any]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    confidence_model_path: Optional[str] = None,
    confidence_threshold: float = 0.55,
    refine_rounds: int = 2,
    refine_window: int = 6,
) -> Dict[str, Any]:
    """
    Full alignment:
      1) MiniLM embeddings + cosine => initial best per slide
      2) MLP confidence => confidence score for chosen pair
      3) iterative refinement if confidence < threshold:
         - search in a +/- window around initial index
         - also try larger speech chunks (merged segments)
    """
    from app.confidence import load_confidence_model, features_from_pair

    segs = [s for s in segments if (s.get("text") or "").strip()]
    if not segs:
        raise ValueError("No transcript segments found (Whisper returned empty).")

    slide_strings = [s["text"] if s["text"] else "(empty)" for s in slide_texts]
    seg_strings = [s.get("text","").strip() for s in segs]

    S = embed_texts(slide_strings, model_name=model_name)
    T = embed_texts(seg_strings, model_name=model_name)

    sims = cosine_similarity(S, T)  # [num_slides, num_segments]
    best = sims.argmax(axis=1)
    best_scores = sims.max(axis=1)

    # Load MLP confidence model if provided
    cm = None
    in_dim = 1 + (S.shape[1] * 4)  # [cos] + |a-b| + a*b + a + b  => 1 + 4*d
    if confidence_model_path:
        try:
            cm = load_confidence_model(confidence_model_path, in_dim=in_dim, device="cpu")
        except Exception:
            cm = None

    def score_pair(i_slide: int, j_seg: int) -> Tuple[float, float]:
        """returns (cosine, confidence)"""
        cos = float(np.dot(S[i_slide], T[j_seg]))  # normalized
        if cm is None:
            return cos, cos  # fallback: use cosine as "confidence"
        feat = features_from_pair(S[i_slide], T[j_seg])[None, :]
        conf = float(cm.predict_proba(feat)[0])
        return cos, conf

    # Initial alignments
    alignments: List[Dict[str, Any]] = []
    for i, s in enumerate(slide_texts):
        j = int(best[i])
        cos, conf = score_pair(i, j)
        alignments.append({
            "slide_id": s["slide_id"],
            "best_segment_idx": j,
            "cosine": cos,
            "confidence": conf,
            "segment": {
                "start": float(segs[j].get("start", 0.0)),
                "end": float(segs[j].get("end", 0.0)),
                "text": segs[j].get("text","").strip(),
            }
        })

    # Iterative refinement
    for _round in range(refine_rounds):
        changed = 0
        # try merged chunks too
        merged = _make_chunks(segs, chunk_size=2)
        merged_texts = [m["text"] for m in merged]
        MT = embed_texts(merged_texts, model_name=model_name)

        for i in range(len(alignments)):
            if alignments[i]["confidence"] >= confidence_threshold:
                continue

            cur_j = int(alignments[i]["best_segment_idx"])
            lo = max(0, cur_j - refine_window)
            hi = min(len(segs), cur_j + refine_window + 1)

            # search best within window using confidence (or cosine fallback)
            best_j, best_conf, best_cos = cur_j, alignments[i]["confidence"], alignments[i]["cosine"]
            for j in range(lo, hi):
                cos, conf = score_pair(i, j)
                if conf > best_conf:
                    best_conf, best_cos, best_j = conf, cos, j

            # also try merged-chunk match (coarse)
            # pick best merged chunk by cosine with slide embedding
            sims_m = (S[i] @ MT.T)
            mj = int(sims_m.argmax())
            mcos = float(sims_m[mj])
            mconf = mcos  # merged confidence fallback
            if cm is not None:
                # build a fake "segment vector" from merged embedding
                # (use same feature function)
                from app.confidence import features_from_pair
                feat = features_from_pair(S[i], MT[mj])[None, :]
                mconf = float(cm.predict_proba(feat)[0])

            # choose between refined single segment vs merged chunk
            if mconf > best_conf:
                # write merged as segment
                alignments[i]["best_segment_idx"] = int(merged[mj]["source_idxs"][0])
                alignments[i]["cosine"] = float(mcos)
                alignments[i]["confidence"] = float(mconf)
                alignments[i]["segment"] = {
                    "start": float(merged[mj]["start"]),
                    "end": float(merged[mj]["end"]),
                    "text": merged[mj]["text"],
                }
                changed += 1
            elif best_j != cur_j:
                alignments[i]["best_segment_idx"] = int(best_j)
                alignments[i]["cosine"] = float(best_cos)
                alignments[i]["confidence"] = float(best_conf)
                alignments[i]["segment"] = {
                    "start": float(segs[best_j].get("start", 0.0)),
                    "end": float(segs[best_j].get("end", 0.0)),
                    "text": segs[best_j].get("text","").strip(),
                }
                changed += 1

        if changed == 0:
            break

    return {
        "alignments": alignments,
        "stats": {
            "num_slides": len(slide_texts),
            "num_segments": len(segs),
            "avg_confidence": float(np.mean([a["confidence"] for a in alignments])),
        }
    }
