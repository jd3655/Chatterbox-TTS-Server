import re
from typing import Iterable, List, Tuple


ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]

TEENS = [
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]

TENS = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]


def int_to_words_us(n: int) -> str:
    """
    Convert an integer in the range 0..999,999,999 to its U.S. English words representation.

    Examples:
        0 -> "zero"
        657 -> "six hundred, fifty seven"
        15348 -> "fifteen thousand, three hundred, forty eight"
    """
    if n < 0 or n > 999_999_999:
        raise ValueError("int_to_words_us supports values from 0 to 999,999,999.")

    if n == 0:
        return "zero"

    def two_digit_words(num: int) -> str:
        if num < 10:
            return ONES[num]
        if num < 20:
            return TEENS[num - 10]
        tens_val, ones_val = divmod(num, 10)
        if ones_val == 0:
            return TENS[tens_val]
        return f"{TENS[tens_val]} {ONES[ones_val]}"

    def three_digit_words(num: int) -> str:
        hundreds, rem = divmod(num, 100)
        parts: List[str] = []
        if hundreds:
            parts.append(f"{ONES[hundreds]} hundred")
        if rem:
            if hundreds:
                parts.append(", ")
            parts.append(two_digit_words(rem))
        return "".join(parts) if parts else "zero"

    millions, remainder = divmod(n, 1_000_000)
    thousands, below_thousand = divmod(remainder, 1000)

    parts: List[str] = []
    if millions:
        parts.append(f"{three_digit_words(millions)} million")
    if thousands:
        if parts:
            parts.append(", ")
        parts.append(f"{three_digit_words(thousands)} thousand")
    if below_thousand or not parts:
        if parts and below_thousand:
            parts.append(", ")
        parts.append(three_digit_words(below_thousand))

    return "".join(parts)


def _find_protected_spans(text: str) -> List[Tuple[int, int]]:
    """
    Find non-nested bracket spans of the form [ ... ] and return start/end indices.
    """
    return [(m.start(), m.end()) for m in re.finditer(r"\[[^\]]*\]", text)]


def _apply_to_unprotected_segments(
    text: str, spans: Iterable[Tuple[int, int]], transform
) -> str:
    """
    Apply a transformation function to unprotected segments of text, leaving protected spans intact.
    """
    result_parts: List[str] = []
    last_index = 0
    for start, end in spans:
        if start > last_index:
            result_parts.append(transform(text[last_index:start]))
        result_parts.append(text[start:end])
        last_index = end
    if last_index < len(text):
        result_parts.append(transform(text[last_index:]))
    return "".join(result_parts)


def normalize_currency_usd(text: str, max_value: int = 999_999_999) -> str:
    """
    Normalize USD currency amounts to spoken words while preserving punctuation and bracketed tags.

    Examples:
        "$657.62" -> "six hundred, fifty seven dollars and sixty two cents"
        "$0.05" -> "five cents"
    """

    max_value = 999_999_999 if max_value is None else max_value
    currency_pattern = re.compile(
        r"\$((?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.(\d{1,2}))?"
    )

    def replace_match(match: re.Match) -> str:
        dollars_str = match.group(1)
        cents_str = match.group(2)

        dollars_int = int(dollars_str.replace(",", ""))
        if dollars_int > max_value:
            return match.group(0)

        if cents_str is None:
            cents_int = 0
        elif len(cents_str) == 1:
            cents_int = int(cents_str) * 10
        else:
            cents_int = int(cents_str)

        if dollars_int == 0 and cents_int > 0:
            cents_words = int_to_words_us(cents_int)
            cent_label = "cent" if cents_int == 1 else "cents"
            return f"{cents_words} {cent_label}"

        dollars_words = int_to_words_us(dollars_int)
        dollar_label = "dollar" if dollars_int == 1 else "dollars"

        if cents_int == 0:
            if dollars_int == 0:
                return "zero dollars"
            return f"{dollars_words} {dollar_label}"

        cents_words = int_to_words_us(cents_int)
        cent_label = "cent" if cents_int == 1 else "cents"
        return f"{dollars_words} {dollar_label} and {cents_words} {cent_label}"

    def transform(segment: str) -> str:
        return currency_pattern.sub(replace_match, segment)

    protected_spans = _find_protected_spans(text)
    if not protected_spans:
        return transform(text)

    return _apply_to_unprotected_segments(text, protected_spans, transform)


def normalize_text(
    text: str,
    *,
    normalize_currency: bool = False,
    currency_max_value: int = 999_999_999,
) -> str:
    """
    Orchestrate text normalization features. Currently supports optional USD currency normalization.
    """
    max_value = 999_999_999 if currency_max_value is None else currency_max_value
    normalized_text = text
    if normalize_currency:
        normalized_text = normalize_currency_usd(
            normalized_text, max_value=max_value
        )
    return normalized_text
