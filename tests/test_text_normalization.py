import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import text_normalization as tn


def _normalize(text: str) -> str:
    return tn.normalize_text(text, normalize_currency=True)


def test_currency_basic_cents_pair():
    assert (
        _normalize("$657.62")
        == "six hundred, fifty seven dollars and sixty two cents"
    )


def test_currency_with_commas_and_cents():
    assert (
        _normalize("$15,348.92")
        == "fifteen thousand, three hundred, forty eight dollars and ninety two cents"
    )


def test_single_dollar_and_cent():
    assert _normalize("$1.01") == "one dollar and one cent"
    assert _normalize("$1.10") == "one dollar and ten cents"


def test_whole_dollars_only():
    assert _normalize("$657") == "six hundred, fifty seven dollars"


def test_single_decimal_interpreted_as_cents():
    assert (
        _normalize("$657.6")
        == "six hundred, fifty seven dollars and sixty cents"
    )


def test_special_case_cents_only_and_zero():
    assert _normalize("$0.05") == "five cents"
    assert _normalize("$0.50") == "fifty cents"
    assert _normalize("$0.00") == "zero dollars"


def test_bracket_content_preserved():
    assert (
        _normalize("[laugh] $657.62")
        == "[laugh] six hundred, fifty seven dollars and sixty two cents"
    )
    assert _normalize("Hello[laugh $657.62]world") == "Hello[laugh $657.62]world"


def test_plain_numbers_unchanged_and_punctuation_preserved():
    assert _normalize("657.62") == "657.62"
    assert (
        _normalize("$12.")
        == "twelve dollars."
    )
    assert (
        _normalize("$0.05)")
        == "five cents)"
    )


def test_leaves_excessive_values_unchanged():
    assert tn.normalize_currency_usd("$1000000000", max_value=999_999_999) == "$1000000000"

