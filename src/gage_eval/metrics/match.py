from gage_eval.metrics.utils import (
    strip_numeric_punctuation,
    strip_punctuation,
)
from gage_eval.metrics.numeric import (
    str_to_float,
)
from gage_eval.metrics.unicode import (
    unicode_number_to_float  
)
from typing import Any, Dict, Optional, Literal

def match_str(
    value: str,
    target: str,
    location: Literal["begin", "end", "any", "exact"] = "end",
    ignore_case: bool = True,
    ignore_punctuation: bool = True,
    numeric: bool = False,
) -> tuple[str, bool]:
    # strip ws
    v = value.strip()
    t = target.strip()

    # baseline answer (will only change for numeric)
    answer = v

    # further cleanup
    if ignore_case:
        v = v.casefold()
        t = t.casefold()
    if numeric and t.isnumeric():
        # remove punctuation
        v = strip_numeric_punctuation(v)
        t = strip_numeric_punctuation(t)
        # normalize as required
        t = normalize_number(t)
        if location == "begin":
            words = re.split(r"\s+", v)
            v = first_number_normalized(words)
        elif location == "end":
            words = re.split(r"\s+", v)
            words.reverse()
            v = first_number_normalized(words)
        elif location == "exact":
            v = normalize_number(v)
        answer = v
    elif ignore_punctuation:
        v = strip_punctuation(v)
        t = strip_punctuation(t)

    # comparisons
    if location == "begin":
        return answer, v.startswith(t)
    elif location == "end":
        return answer, v.endswith(t)
    elif location == "exact":
        return answer, v == t
    else:
        return answer, t in v


def first_number_normalized(words: list[str]) -> str:
    number = next(
        (word for word in words if word.replace(".", "").isnumeric()), words[0]
    )
    return normalize_number(number)


def normalize_number(number: str, precision: int = 5) -> str:
    if number.replace(".", "").isnumeric():
        # first try parsing with our tried and true parser, if that fails
        # then there were unicode characters that are still .isnumeric()
        # for that case, parse with our new unicode parser
        try:
            num = str_to_float(number)
        except ValueError:
            num = unicode_number_to_float(number)
        return format(num, f".{precision}g")
    else:
        return number