"""Numeric utilities."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Optional, Callable, Literal, List, NamedTuple

def extract_numeric_answer(prediction: str, answer_type: str = "integer") -> Optional[str]:
    """Extract a numeric answer from free-form text (integer/float)."""
    if not prediction:
        return prediction

    # STEP 1: Try direct conversion (best effort).
    try:
        if answer_type == "integer":
            return str(int(float(prediction)))
        elif answer_type == "float":
            return str(float(prediction))
    except (ValueError, TypeError):
        pass

    # STEP 2: Extract the last numeric token with a simple regex.
    # Pattern: optional sign + digits + optional decimal part.
    pattern = pattern = r"-?(?:\d{1,3}(?:,\d{3})*|\d+)?\.\d+(?:[eE][+-]?\d+)?|-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d*)?(?:[eE][+-]?\d+)?"
    numbers = re.findall(pattern, prediction)

    if not numbers:
        return prediction

    last_num = numbers[-1]
    try:
        last_num = last_num.replace(",", "")
        if answer_type == "integer":
            return str(int(float(last_num)))
        elif answer_type == "float":
            return str(float(last_num))
    except (ValueError, TypeError):
        return prediction
    return prediction


def str_to_float(s: str) -> float:
    """Convert a str to float, handling exponent characters and Unicode fractions.

    The Python isnumeric() function returns True for strings that include exponents
    (e.g. 5²) and Unicode fractions (e.g. ½, ¾), however the float() function doesn't
    handle these characters. This function correctly handles both exponents and
    Unicode fractions when converting from str to float.

    Args:
       s (str): String to convert to float

    Returns:
       float: Converted value

    Raises:
       ValueError: If the string is not a valid numeric value.
    """
    # handle empty input
    if not s:
        raise ValueError("Input string is empty.")

    # Define common Unicode fractions and their float values
    fraction_map = {
        "½": 0.5,
        "⅓": 1 / 3,
        "⅔": 2 / 3,
        "¼": 0.25,
        "¾": 0.75,
        "⅕": 0.2,
        "⅖": 0.4,
        "⅗": 0.6,
        "⅘": 0.8,
        "⅙": 1 / 6,
        "⅚": 5 / 6,
        "⅐": 1 / 7,
        "⅛": 0.125,
        "⅜": 0.375,
        "⅝": 0.625,
        "⅞": 0.875,
        "⅑": 1 / 9,
        "⅒": 0.1,
    }

    superscript_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    superscript_chars = "⁰¹²³⁴⁵⁶⁷⁸⁹"

    # Special case: if string is a single fraction character
    if len(s) == 1 and s in fraction_map:
        return fraction_map[s]

    # Process the string character by character to handle mixed cases
    base_part = ""
    fraction_char = None
    exponent_part = ""

    i = 0
    while i < len(s):
        char = s[i]
 
        if char in fraction_map:
            # We found a fraction character - store it
            if fraction_char is not None:
                # If we already have a fraction character, that's invalid
                raise ValueError(f"Multiple fraction characters in '{s}'")
            fraction_char = char
        elif char in superscript_chars:
            # We found the start of an exponent - collect all superscript chars
            exponent_part = s[i:]
            break  # Stop processing - we've captured the exponent
        else:
            # Regular character - add to base part
            base_part += char

        i += 1

    # Calculate the base value (whole number + fraction if present)
    base_value = 0.0

    if base_part:
        # find the first valid float (LLMs may include additional spurious output)
        match = re.match(r"^([+-]?\d+(?:\.\d+)?)", base_part)
        if match is None:
            raise ValueError(f"Value could not be parsed as a float: {s}")
        base_part = match.group(1)

        try:
            base_value = float(base_part)
        except ValueError:
            raise ValueError(f"Invalid base part in '{s}'")

    if fraction_char:
        fraction_value = fraction_map[fraction_char]
        if base_value < 0:
            # For negative values, subtract the fraction (e.g., -2½ = -2.5)
            base_value -= fraction_value
        else:
            # For zero or positive values, add the fraction
            base_value += fraction_value
    elif not base_part:
        # If there's no base part and no fraction, default to 1.0
        base_value = 1.0

    # Handle exponent part if present
    if exponent_part:
        exponent_str = exponent_part.translate(superscript_map)
        try:
            # Interpret multiple superscript digits as a multi-digit exponent
            # e.g., "2⁴⁵" is 2^45, "½²³" is 0.5^23
            exponent = int(exponent_str)
            return base_value**exponent
        except ValueError:
            raise ValueError(f"Invalid exponent in '{s}'")
    else:
        return base_value
