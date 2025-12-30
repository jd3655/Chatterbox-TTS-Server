import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pronunciation import apply_pronunciation_dict


def test_whole_word_replacements():
    mapping = {"Zilog": "ZY-log"}
    assert apply_pronunciation_dict("Zilog makes", mapping) == "ZY-log makes"
    assert apply_pronunciation_dict("Zilog,", mapping) == "ZY-log,"
    assert apply_pronunciation_dict("(Zilog)", mapping) == "(ZY-log)"
    assert apply_pronunciation_dict("Zilogic is different", mapping) == "Zilogic is different"


def test_case_sensitive_matches():
    mapping = {"Zilog": "ZY-log"}
    assert apply_pronunciation_dict("zilog", mapping) == "zilog"
    mapping_lower = {"zilog": "zee-log"}
    assert apply_pronunciation_dict("zilog", mapping_lower) == "zee-log"


def test_bracket_spans_are_protected():
    mapping = {"Zilog": "ZY-log"}
    assert apply_pronunciation_dict("Hello[laugh] Zilog", mapping) == "Hello[laugh] ZY-log"
    assert apply_pronunciation_dict("Hello[pause:0.3s]Zilog", mapping) == "Hello[pause:0.3s]ZY-log"
    assert apply_pronunciation_dict("Hello[laugh Zilog] world", mapping) == "Hello[laugh Zilog] world"


def test_overlap_and_stability():
    mapping = {"US": "you ess", "Zilog": "ZY-log"}
    text = "Zilog builds in the US, not US Zilogic."
    result = apply_pronunciation_dict(text, mapping)
    assert result == "ZY-log builds in the you ess, not you ess Zilogic."
