"""
Token counting and budget management for LLM prompts.

Uses a character-based approximation (1 token ≈ 4 chars for English)
which is reliable enough for budget management without adding a
heavy tokenizer dependency.
"""
from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

# Approximate chars per token for English text
CHARS_PER_TOKEN = 4

# Context limits (in tokens) - conservative practical limits
MODEL_LIMITS = {
    "gemini-2.5-flash": 200_000,   # 1M actual, 200K practical for quality
    "gemini-2.0-flash": 200_000,
    "gemini-1.5-flash": 200_000,
    "gemini-1.5-pro": 200_000,
    "google/flan-t5-base": 512,
    "google/flan-t5-large": 512,
    "default": 100_000,
}

# Reserve tokens for the prompt template / system instructions
PROMPT_OVERHEAD = 2_000


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def get_model_limit(model_name: str) -> int:
    """Get the practical token limit for a model."""
    for key, limit in MODEL_LIMITS.items():
        if key in model_name.lower():
            return limit
    return MODEL_LIMITS["default"]


def truncate_to_token_budget(
    text: str,
    max_tokens: int,
    reserve: int = PROMPT_OVERHEAD,
) -> str:
    """
    Truncate text to fit within a token budget.

    Truncates at sentence boundaries when possible.
    """
    available = max_tokens - reserve
    if available <= 0:
        return ""

    estimated = estimate_tokens(text)
    if estimated <= available:
        return text

    # Truncate to approximate character limit
    char_limit = available * CHARS_PER_TOKEN

    truncated = text[:char_limit]

    # Try to cut at the last sentence boundary
    for sep in [". ", ".\n", "! ", "? "]:
        last_sep = truncated.rfind(sep)
        if last_sep > char_limit * 0.7:  # Don't cut too much
            truncated = truncated[: last_sep + 1]
            break

    logger.debug(
        "Truncated text from ~%d to ~%d tokens",
        estimated, estimate_tokens(truncated),
    )
    return truncated


def split_to_token_chunks(
    text: str,
    chunk_tokens: int = 1200,
    overlap_tokens: int = 150,
) -> List[str]:
    """
    Split text into token-budgeted chunks with overlap.

    Splits at sentence boundaries when possible.
    """
    if not text or not text.strip():
        return []

    estimated = estimate_tokens(text)
    if estimated <= chunk_tokens:
        return [text]

    chunk_chars = chunk_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    chunks: List[str] = []
    pos = 0
    text_len = len(text)

    while pos < text_len:
        end = min(text_len, pos + chunk_chars)

        # Try to break at a sentence boundary
        if end < text_len:
            for sep in [". ", ".\n", "! ", "? ", "\n\n"]:
                last_sep = text[pos:end].rfind(sep)
                if last_sep > chunk_chars * 0.6:
                    end = pos + last_sep + len(sep)
                    break

        chunk = text[pos:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        # Advance with overlap
        pos = max(pos + 1, end - overlap_chars)

    return chunks
