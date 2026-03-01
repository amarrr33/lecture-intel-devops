"""
Embedding-based deduplication for generated content.

Prevents duplicate flashcards and questions that arise from
overlapping topic chunks during generation.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from app.align import embed_texts

logger = logging.getLogger(__name__)


def deduplicate_by_embedding(
    items: List[Dict[str, Any]],
    key: str = "question",
    threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """
    Deduplicate a list of dicts by embedding similarity on a text field.

    Uses greedy selection: keeps the first item, then skips any subsequent
    item whose embedding similarity to any kept item exceeds the threshold.

    Args:
        items: List of dicts to deduplicate.
        key: The dict key whose value is used for similarity comparison.
        threshold: Cosine similarity threshold above which items are duplicates.

    Returns:
        Deduplicated list (preserves original order).
    """
    if len(items) <= 1:
        return items

    texts = [str(item.get(key, "")).strip() for item in items]

    # Filter out items with empty text
    valid_indices = [i for i, t in enumerate(texts) if t]
    if not valid_indices:
        return items

    valid_texts = [texts[i] for i in valid_indices]
    embeddings = embed_texts(valid_texts)

    kept_indices: List[int] = []
    kept_embeddings: List[np.ndarray] = []

    for idx, emb in zip(valid_indices, embeddings):
        if not kept_embeddings:
            kept_indices.append(idx)
            kept_embeddings.append(emb)
            continue

        # Check similarity against all kept items
        sims = np.dot(np.array(kept_embeddings), emb)
        if float(np.max(sims)) < threshold:
            kept_indices.append(idx)
            kept_embeddings.append(emb)

    # Also include items that had empty keys (don't drop them)
    empty_indices = [i for i in range(len(items)) if i not in valid_indices]
    result_indices = sorted(set(kept_indices) | set(empty_indices))

    deduped = [items[i] for i in result_indices]

    removed = len(items) - len(deduped)
    if removed > 0:
        logger.info("Deduplication removed %d/%d items (key='%s')", removed, len(items), key)

    return deduped
