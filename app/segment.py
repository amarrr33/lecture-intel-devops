"""
Topic segmentation for lecture transcripts.

Detects topic boundaries by measuring embedding similarity between
consecutive transcript segments. When similarity drops significantly,
a new topic begins.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.align import embed_texts

logger = logging.getLogger(__name__)

MIN_TOPIC_WORDS = 80
DEFAULT_WINDOW = 3


@dataclass
class TopicChunk:
    topic_id: str
    text: str
    start_time: float
    end_time: float
    segment_indices: List[int] = field(default_factory=list)
    slide_ids: List[str] = field(default_factory=list)

    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "segment_indices": self.segment_indices,
            "slide_ids": self.slide_ids,
        }


def _windowed_texts(segments: List[Dict[str, Any]], window: int) -> List[str]:
    """Create overlapping window texts for smoother similarity computation."""
    texts = []
    for i in range(len(segments)):
        lo = max(0, i - window // 2)
        hi = min(len(segments), i + window // 2 + 1)
        combined = " ".join(
            (s.get("text", "") or "").strip() for s in segments[lo:hi]
        )
        texts.append(combined)
    return texts


def detect_boundaries(
    segments: List[Dict[str, Any]],
    window: int = DEFAULT_WINDOW,
    threshold_factor: float = 1.0,
) -> List[int]:
    """
    Detect topic boundary indices in transcript segments.

    Uses sliding-window embeddings and flags boundaries where
    consecutive similarity drops below (mean - threshold_factor * std).

    Returns list of segment indices where new topics begin.
    """
    if len(segments) < 3:
        return [0]

    texts = _windowed_texts(segments, window)
    embeddings = embed_texts(texts)

    # Cosine similarity between consecutive windowed embeddings
    sims = []
    for i in range(len(embeddings) - 1):
        sim = float(np.dot(embeddings[i], embeddings[i + 1]))
        sims.append(sim)

    sims_arr = np.array(sims)
    mean_sim = float(np.mean(sims_arr))
    std_sim = float(np.std(sims_arr))
    threshold = mean_sim - threshold_factor * std_sim

    logger.debug(
        "Segmentation stats: mean=%.3f std=%.3f threshold=%.3f",
        mean_sim, std_sim, threshold,
    )

    # Index 0 always starts a topic
    boundaries = [0]
    for i, sim in enumerate(sims):
        if sim < threshold:
            boundaries.append(i + 1)

    return boundaries


def segment_transcript(
    segments: List[Dict[str, Any]],
    alignment: Optional[Dict[str, Any]] = None,
    window: int = DEFAULT_WINDOW,
    threshold_factor: float = 1.0,
    min_words: int = MIN_TOPIC_WORDS,
) -> List[TopicChunk]:
    """
    Segment transcript into topic chunks.

    Args:
        segments: Whisper transcript segments [{text, start, end}, ...]
        alignment: Optional alignment result from align.py (to attach slide_ids)
        window: Sliding window size for embedding smoothing
        threshold_factor: How many std below mean triggers a boundary
        min_words: Minimum words per topic; smaller chunks get merged into neighbors

    Returns:
        List of TopicChunk with text, timestamps, and associated slide_ids.
    """
    valid_segs = [s for s in segments if (s.get("text") or "").strip()]
    if not valid_segs:
        return []

    boundaries = detect_boundaries(valid_segs, window, threshold_factor)

    # Build raw topic chunks from boundaries
    raw_chunks: List[TopicChunk] = []
    for b_idx, start in enumerate(boundaries):
        end = boundaries[b_idx + 1] if b_idx + 1 < len(boundaries) else len(valid_segs)
        chunk_segs = valid_segs[start:end]
        if not chunk_segs:
            continue

        text = " ".join((s.get("text", "") or "").strip() for s in chunk_segs)
        raw_chunks.append(TopicChunk(
            topic_id=f"topic_{b_idx + 1:02d}",
            text=text.strip(),
            start_time=float(chunk_segs[0].get("start", 0.0)),
            end_time=float(chunk_segs[-1].get("end", 0.0)),
            segment_indices=list(range(start, end)),
        ))

    # Merge small chunks into their neighbors
    merged: List[TopicChunk] = []
    for chunk in raw_chunks:
        if merged and chunk.word_count() < min_words:
            # Merge into previous
            prev = merged[-1]
            prev.text = prev.text + " " + chunk.text
            prev.end_time = chunk.end_time
            prev.segment_indices.extend(chunk.segment_indices)
        else:
            merged.append(chunk)

    # If the last chunk is too small, merge it backward
    if len(merged) > 1 and merged[-1].word_count() < min_words:
        last = merged.pop()
        merged[-1].text = merged[-1].text + " " + last.text
        merged[-1].end_time = last.end_time
        merged[-1].segment_indices.extend(last.segment_indices)

    # Re-number topic IDs after merging
    for i, chunk in enumerate(merged):
        chunk.topic_id = f"topic_{i + 1:02d}"

    # Attach slide_ids from alignment if available
    if alignment:
        _attach_slide_ids(merged, alignment, valid_segs)

    logger.info(
        "Segmented %d transcript segments into %d topics",
        len(valid_segs), len(merged),
    )
    return merged


def _attach_slide_ids(
    topics: List[TopicChunk],
    alignment: Dict[str, Any],
    segments: List[Dict[str, Any]],
) -> None:
    """Map aligned slide_ids to their corresponding topic chunks."""
    # Build segment_idx -> topic_idx mapping
    seg_to_topic: Dict[int, int] = {}
    for t_idx, topic in enumerate(topics):
        for s_idx in topic.segment_indices:
            seg_to_topic[s_idx] = t_idx

    for a in alignment.get("alignments", []):
        seg_idx = a.get("best_segment_idx")
        slide_id = a.get("slide_id")
        if seg_idx is not None and slide_id:
            t_idx = seg_to_topic.get(seg_idx)
            if t_idx is not None and slide_id not in topics[t_idx].slide_ids:
                topics[t_idx].slide_ids.append(slide_id)
