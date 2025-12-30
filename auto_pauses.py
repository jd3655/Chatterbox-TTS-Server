"""
Auto pause insertion utilities.

This module adds synthetic pause tags to text based on punctuation and narration
style. It is intentionally dependency-free (stdlib only) so it can run early in
the text-processing pipeline.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple

MANUAL_PAUSE_CANONICAL_PATTERN = re.compile(
    r"\[pause:(\d+(?:\.\d+)?)s\]", re.IGNORECASE
)
MANUAL_PAUSE_SHORTHAND_PATTERN = re.compile(
    r"\[(\d+(?:\.\d+)?)s\]", re.IGNORECASE
)
BRACKET_TOKEN_PATTERN = re.compile(r"\[[^\[\]]+\]")
ELLIPSIS_PATTERN = re.compile(r"\.\.+")
SPACED_DASH_PATTERN = re.compile(r"\s-\s")

DISCOURSE_MARKERS = (
    "however",
    "therefore",
    "meanwhile",
    "in fact",
    "on the other hand",
    "so",
    "but",
)

BASE_PAUSES = {
    "audiobook": {
        "comma": 0.16,
        "semicolon": 0.32,
        "colon": 0.36,
        "emdash": 0.24,
        "sentence_end": 0.55,
        "paragraph": 1.15,
    },
    "youtube": {
        "comma": 0.12,
        "semicolon": 0.22,
        "colon": 0.24,
        "emdash": 0.18,
        "sentence_end": 0.38,
        "paragraph": 0.80,
    },
    "ad": {
        "comma": 0.07,
        "semicolon": 0.14,
        "colon": 0.16,
        "emdash": 0.12,
        "sentence_end": 0.26,
        "paragraph": 0.55,
    },
}


def _format_pause(seconds: float) -> str:
    formatted = f"{seconds:.3f}".rstrip("0").rstrip(".")
    if "." not in formatted:
        formatted = f"{formatted}.0"
    return f"[pause:{formatted}s]"


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _find_spans(pattern: re.Pattern, text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def _inside_spans(index: int, spans: Sequence[Tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in spans)


def _has_manual_pause_near(
    index: int, spans: Sequence[Tuple[int, int]], window: int = 3
) -> bool:
    return any(start <= index + window and end >= index - window for start, end in spans)


def _next_word(text: str, start_index: int, bracket_spans: Sequence[Tuple[int, int]]) -> str:
    idx = start_index
    length = len(text)
    while idx < length:
        if _inside_spans(idx, bracket_spans):
            # Skip entire bracket token
            span = next(span for span in bracket_spans if span[0] <= idx < span[1])
            idx = span[1]
            continue
        char = text[idx]
        if char.isalpha():
            end = idx
            while end < length and text[end].isalpha():
                end += 1
            return text[idx:end].lower()
        idx += 1
    return ""


def _words_in_sentence(text: str) -> int:
    cleaned = BRACKET_TOKEN_PATTERN.sub(" ", text)
    return len(re.findall(r"\b[\w']+\b", cleaned))


def _get_style_base(style: str) -> dict:
    style_lower = (style or "audiobook").lower()
    if style_lower == "dramatic":
        dramatic_base = {k: v * 1.35 for k, v in BASE_PAUSES["audiobook"].items()}
        return dramatic_base
    return BASE_PAUSES.get(style_lower, BASE_PAUSES["audiobook"])


def _should_skip_open_quote_boundary(text: str, index: int) -> bool:
    prev_non_space = index - 1
    while prev_non_space >= 0 and text[prev_non_space].isspace():
        prev_non_space -= 1
    if prev_non_space < 0:
        return False
    if text[prev_non_space] in {'"', "'"}:
        if prev_non_space == 0 or text[prev_non_space - 1].isspace():
            return True
    return False


def _compute_pause_seconds(
    boundary_type: str,
    punctuation_char: str,
    *,
    style_base: dict,
    speed_factor: float,
    strength: float,
    topup_only: bool,
    min_pause: float,
    max_pause: float,
    words_in_sentence: Optional[int],
    next_word: str,
) -> float:
    base_value = style_base.get(boundary_type)
    if base_value is None:
        return 0.0

    pause = base_value / max(speed_factor, 0.2)
    pause *= strength

    if boundary_type == "sentence_end" and words_in_sentence is not None:
        extra = _clamp((words_in_sentence - 18) / 40, 0.0, 0.35) * 0.25
        pause += extra
        if punctuation_char == "?":
            pause += 0.07

    if next_word:
        bump = 0.0
        markers = tuple(m.lower() for m in DISCOURSE_MARKERS)
        if next_word.startswith(markers):
            if boundary_type == "sentence_end":
                style_name = next(
                    (name for name, base in BASE_PAUSES.items() if base is style_base),
                    None,
                )
                if style_name == "youtube":
                    bump = 0.07
                elif style_name == "ad":
                    bump = 0.04
                else:
                    bump = 0.10
            elif boundary_type in {"comma", "semicolon", "colon", "emdash"}:
                # Apply a lighter bump for mid-sentence discourse markers
                bump = 0.04
        pause += bump

    pause = _clamp(pause, min_pause, max_pause)

    if topup_only and boundary_type != "paragraph":
        pause *= 0.65

    return pause


def _in_same_bracket_boundary(position: int, spans: Sequence[Tuple[int, int]]) -> bool:
    return any(start < position < end for start, end in spans)


def _collect_spaced_dash_boundaries(text: str) -> Iterable[int]:
    for match in SPACED_DASH_PATTERN.finditer(text):
        yield match.start() + 1


def insert_auto_pauses(
    text: str,
    style: str,
    speed_factor: float = 1.0,
    strength: float = 1.0,
    topup_only: bool = True,
    min_pause: float = 0.04,
    max_pause: float = 1.8,
) -> str:
    """
    Insert automatic pause tags into the provided text.
    """
    if not text:
        return text

    style_base = _get_style_base(style)
    bracket_spans = _find_spans(BRACKET_TOKEN_PATTERN, text)
    manual_pause_spans = sorted(
        _find_spans(MANUAL_PAUSE_CANONICAL_PATTERN, text)
        + _find_spans(MANUAL_PAUSE_SHORTHAND_PATTERN, text)
    )

    insertions: List[Tuple[int, str]] = []
    paragraph_pattern = re.compile(r"\n\s*\n")
    para_start = 0

    for para_match in paragraph_pattern.finditer(text):
        segment_end = para_match.start()
        insertions.extend(
            _collect_boundaries_for_segment(
                text,
                para_start,
                segment_end,
                bracket_spans,
                manual_pause_spans,
                style_base,
                speed_factor,
                strength,
                topup_only,
                min_pause,
                max_pause,
            )
        )

        boundary_pos = para_match.end()
        if not _inside_spans(boundary_pos, bracket_spans) and not _has_manual_pause_near(
            boundary_pos, manual_pause_spans
        ):
            pause_seconds = _compute_pause_seconds(
                "paragraph",
                "",
                style_base=style_base,
                speed_factor=speed_factor,
                strength=strength,
                topup_only=topup_only,
                min_pause=min_pause,
                max_pause=max_pause,
                words_in_sentence=None,
                next_word=_next_word(text, boundary_pos, bracket_spans),
            )
            if pause_seconds > 0:
                insertions.append((boundary_pos, _format_pause(pause_seconds)))
        para_start = para_match.end()

    insertions.extend(
        _collect_boundaries_for_segment(
            text,
            para_start,
            len(text),
            bracket_spans,
            manual_pause_spans,
            style_base,
            speed_factor,
            strength,
            topup_only,
            min_pause,
            max_pause,
        )
    )

    if not insertions:
        return text

    insertions = sorted(insertions, key=lambda x: x[0])
    new_parts: List[str] = []
    last_idx = 0
    for pos, tag in insertions:
        if pos < last_idx:
            continue
        new_parts.append(text[last_idx:pos])
        new_parts.append(tag)
        last_idx = pos
    new_parts.append(text[last_idx:])
    return "".join(new_parts)


def _collect_boundaries_for_segment(
    text: str,
    start: int,
    end: int,
    bracket_spans: Sequence[Tuple[int, int]],
    manual_pause_spans: Sequence[Tuple[int, int]],
    style_base: dict,
    speed_factor: float,
    strength: float,
    topup_only: bool,
    min_pause: float,
    max_pause: float,
) -> List[Tuple[int, str]]:
    insertions: List[Tuple[int, str]] = []
    sentence_start = start
    idx = start
    spaced_dash_positions = {
        start + pos for pos in _collect_spaced_dash_boundaries(text[start:end])
    }

    while idx < end:
        char = text[idx]
        if _inside_spans(idx, bracket_spans):
            idx += 1
            continue

        boundary_type = None
        punctuation_char = char
        insertion_pos = idx + 1
        words_in_sentence: Optional[int] = None

        if char in ".!?":
            if ELLIPSIS_PATTERN.match(text, idx):
                idx += 1
                continue
            if char == ".":
                if idx > start and idx + 1 < end and text[idx - 1].isdigit() and text[idx + 1].isdigit():
                    idx += 1
                    continue
            if _should_skip_open_quote_boundary(text, idx):
                idx += 1
                continue
            boundary_type = "sentence_end"
            words_in_sentence = _words_in_sentence(text[sentence_start : idx + 1])
            sentence_start = idx + 1
        elif char == ",":
            boundary_type = "comma"
        elif char == ";":
            boundary_type = "semicolon"
        elif char == ":":
            boundary_type = "colon"
        elif char in {"—", "–"} or (idx in spaced_dash_positions):
            boundary_type = "emdash"
        elif char in {"\n", "\r"}:
            sentence_start = idx + 1

        if boundary_type:
            while insertion_pos < end and text[insertion_pos] in {'"', "'"}:
                insertion_pos += 1

            if _in_same_bracket_boundary(idx, bracket_spans) or _in_same_bracket_boundary(
                insertion_pos, bracket_spans
            ):
                idx += 1
                continue

            if _has_manual_pause_near(insertion_pos, manual_pause_spans):
                idx += 1
                continue

            next_word = _next_word(text, insertion_pos, bracket_spans)
            pause_seconds = _compute_pause_seconds(
                boundary_type,
                punctuation_char,
                style_base=style_base,
                speed_factor=speed_factor,
                strength=strength,
                topup_only=topup_only,
                min_pause=min_pause,
                max_pause=max_pause,
                words_in_sentence=words_in_sentence,
                next_word=next_word,
            )
            if pause_seconds > 0 and not any(pos == insertion_pos for pos, _ in insertions):
                insertions.append((insertion_pos, _format_pause(pause_seconds)))

        idx += 1

    return insertions
