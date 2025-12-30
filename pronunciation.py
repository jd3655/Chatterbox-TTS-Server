"""Utilities for applying pronunciation dictionaries to text.

This module provides a single entry point, :func:`apply_pronunciation_dict`,
which replaces whole-word occurrences in text based on a provided mapping while
preserving bracketed tags such as ``[laugh]`` or ``[pause:0.3s]``.

Rules:
- Whole-word matches use a custom boundary definition with token characters
  ``[A-Za-z0-9']``.
- Bracketed spans (``[ ... ]``) are left untouched.
- Replacements are applied deterministically: longest keys first, then
  lexicographically.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

TOKEN_CHARS_CLASS = r"A-Za-z0-9'"
BRACKET_SPAN_PATTERN = re.compile(r"\[[^\]]*\]")


def _find_bracket_spans(text: str) -> List[Tuple[int, int]]:
    """Return start/end indices for non-nested bracket spans in *text*."""

    return [(m.start(), m.end()) for m in BRACKET_SPAN_PATTERN.finditer(text)]


def _build_replacement_patterns(keys: Iterable[str]):
    """Yield (key, compiled_regex) ordered by longest then lexicographic."""

    ordered_keys = sorted(keys, key=lambda k: (-len(k), k))
    boundary = rf"(?<![{TOKEN_CHARS_CLASS}]){{}}(?![{TOKEN_CHARS_CLASS}])"
    for key in ordered_keys:
        escaped_key = re.escape(key)
        pattern = re.compile(boundary.format(escaped_key))
        yield key, pattern


def _apply_to_segment(segment: str, mapping: Dict[str, str]) -> str:
    """Apply replacements to a non-protected text *segment*."""

    if not mapping:
        return segment

    for key, pattern in _build_replacement_patterns(mapping.keys()):
        segment = pattern.sub(mapping[key], segment)
    return segment


def _split_by_brackets(text: str) -> List[Tuple[bool, str]]:
    """Split *text* into (is_protected, chunk) pairs around brackets."""

    spans = _find_bracket_spans(text)
    if not spans:
        return [(False, text)]

    parts: List[Tuple[bool, str]] = []
    last_index = 0
    for start, end in spans:
        if last_index < start:
            parts.append((False, text[last_index:start]))
        parts.append((True, text[start:end]))
        last_index = end
    if last_index < len(text):
        parts.append((False, text[last_index:]))
    return parts


def apply_pronunciation_dict(text: str, mapping: Dict[str, str]) -> str:
    """Apply *mapping* to *text* using whole-word replacements.

    Args:
        text: The original text to process.
        mapping: Dictionary of exact words to their replacements.

    Returns:
        The text with replacements applied outside of bracketed spans.
    """

    if not text or not mapping:
        return text

    segments = _split_by_brackets(text)
    processed_segments: List[str] = []
    for is_protected, chunk in segments:
        if is_protected:
            processed_segments.append(chunk)
        else:
            processed_segments.append(_apply_to_segment(chunk, mapping))
    return "".join(processed_segments)

