import pathlib
import re
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import auto_pauses


def _extract_first_pause_seconds(text: str) -> float:
    match = re.search(r"\[pause:(\d+(?:\.\d+)?)s\]", text)
    assert match, "No pause tag found"
    return float(match.group(1))


def test_preserves_paralinguistic_tags():
    text = "Hello [laugh], world! [cough] Then [chuckle] done."
    result = auto_pauses.insert_auto_pauses(
        text, "audiobook", topup_only=False, strength=1.0
    )
    assert "[laugh]" in result
    assert "[cough]" in result
    assert "[chuckle]" in result


def test_inserts_pauses_at_paragraph_breaks():
    text = "First line.\n\nSecond line."
    result = auto_pauses.insert_auto_pauses(
        text, "audiobook", topup_only=False, strength=1.0
    )
    assert "\n\n[pause:" in result


def test_inserts_pauses_at_sentence_boundaries():
    text = "Hello world. Next sentence."
    result = auto_pauses.insert_auto_pauses(
        text, "audiobook", topup_only=False, strength=1.0
    )
    assert "world.[pause" in result


def test_skips_when_manual_pause_present_near_boundary():
    text = "Hello.[pause:0.5s] Next sentence."
    result = auto_pauses.insert_auto_pauses(
        text, "audiobook", topup_only=False, strength=1.0
    )
    assert result.count("[pause:") == 2  # only manual plus final sentence end
    assert "Hello.[pause:0.5s][pause:" not in result


def test_does_not_insert_inside_bracket_tags():
    text = "Keep [note: inside, brackets] flowing smoothly."
    result = auto_pauses.insert_auto_pauses(
        text, "audiobook", topup_only=False, strength=1.0
    )
    assert not re.search(r"\[[^\]]*\[pause:[^\]]*\][^\]]*\]", result)


def test_speed_factor_scales_pauses():
    base_text = "Quick test."
    fast_result = auto_pauses.insert_auto_pauses(
        base_text, "audiobook", speed_factor=2.0, topup_only=False, strength=1.0
    )
    normal_result = auto_pauses.insert_auto_pauses(
        base_text, "audiobook", speed_factor=1.0, topup_only=False, strength=1.0
    )
    fast_pause = _extract_first_pause_seconds(fast_result)
    normal_pause = _extract_first_pause_seconds(normal_result)
    assert fast_pause == normal_pause / 2


def test_discourse_marker_bump_applied():
    text = "First thought. However, this continues."
    result = auto_pauses.insert_auto_pauses(
        text, "audiobook", topup_only=False, strength=1.0
    )
    assert "[pause:0.65" in result
