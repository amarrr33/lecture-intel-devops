"""
Robust JSON extraction from LLM output.

Handles: markdown fences, trailing text, case variations,
partial JSON, and nested structures.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Pattern for ```json ... ``` blocks (case-insensitive, dotall)
_FENCED_JSON_RE = re.compile(
    r"```\s*(?:json)?\s*\n?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# Pattern for outermost JSON array or object
_JSON_ARRAY_RE = re.compile(r"(\[.*\])", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"(\{.*\})", re.DOTALL)


def extract_json(text: str) -> Optional[Any]:
    """
    Extract and parse JSON from LLM output text.

    Tries in order:
    1. Direct parse of full text
    2. Extract from ```json ... ``` fenced blocks
    3. Find outermost [...] or {...} in text
    4. Return None if all fail
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # 1. Try direct parse
    parsed = _try_parse(text)
    if parsed is not None:
        return parsed

    # 2. Try fenced code blocks
    for match in _FENCED_JSON_RE.finditer(text):
        inner = match.group(1).strip()
        parsed = _try_parse(inner)
        if parsed is not None:
            return parsed

    # 3. Try to find outermost array
    m = _JSON_ARRAY_RE.search(text)
    if m:
        parsed = _try_parse(m.group(1))
        if parsed is not None:
            return parsed

    # 4. Try to find outermost object
    m = _JSON_OBJECT_RE.search(text)
    if m:
        parsed = _try_parse(m.group(1))
        if parsed is not None:
            return parsed

    logger.warning("Failed to extract JSON from LLM output (length=%d)", len(text))
    return None


def _try_parse(s: str) -> Optional[Any]:
    """Attempt JSON parse, return None on failure."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


def safe_json_loads(s: str) -> Optional[Any]:
    """Backward-compatible wrapper. Prefer extract_json for LLM output."""
    return _try_parse(s)
