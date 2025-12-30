"""Utilities for handling multi-voice synthesis requests.

This module is intentionally stdlib-only to keep parsing lightweight.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


_VOICE_DIRECTIVE_RE = re.compile(r"<voice\s*:\s*([A-Za-z0-9_-]+)\s*>")


def split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs separated by one or more blank lines.

    Internal newlines inside a paragraph are preserved. Blank paragraphs are
    ignored.
    """

    paragraphs: List[str] = []
    current_lines: List[str] = []

    for line in text.splitlines():
        if line.strip() == "":
            if current_lines:
                paragraphs.append("\n".join(current_lines).strip())
                current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        paragraphs.append("\n".join(current_lines).strip())

    return [p for p in paragraphs if p]


def _find_bracket_ranges(text: str) -> List[range]:
    """Return list of ranges covering bracket tokens [ ... ]."""

    ranges: List[range] = []
    start: Optional[int] = None
    depth = 0

    for idx, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "]" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                ranges.append(range(start, idx + 1))
                start = None

    return ranges


def _is_within_brackets(position: int, ranges: List[range]) -> bool:
    return any(position in r for r in ranges)


def parse_voice_directives(text: str) -> List[Dict[str, Optional[str]]]:
    """Parse `<voice:...>` directives into ordered voice segments.

    Directives contained inside bracket tokens `[...]` are ignored. Text before
    the first directive is returned with a ``None`` voice_id so callers can
    apply the default voice.
    """

    bracket_ranges = _find_bracket_ranges(text)
    matches = [
        m for m in _VOICE_DIRECTIVE_RE.finditer(text) if not _is_within_brackets(m.start(), bracket_ranges)
    ]

    segments: List[Dict[str, Optional[str]]] = []
    last_index = 0
    current_voice: Optional[str] = None

    for match in matches:
        segment_text = text[last_index:match.start()]
        if segment_text.strip():
            segments.append({"voice_id": current_voice, "text": segment_text})
        current_voice = match.group(1).strip()
        last_index = match.end()

    tail_text = text[last_index:]
    if tail_text.strip() or not segments:
        segments.append({"voice_id": current_voice, "text": tail_text})

    return segments


def build_segments_from_paragraphs(
    paragraphs: List[str], assignments: List[str], default_voice: str
) -> List[Dict[str, str]]:
    """Pair paragraphs with voice assignments or default voice."""

    segments: List[Dict[str, str]] = []
    for idx, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            continue
        voice_id = assignments[idx] if idx < len(assignments) and assignments[idx] else default_voice
        segments.append({"voice_id": voice_id, "text": paragraph})

    return segments

