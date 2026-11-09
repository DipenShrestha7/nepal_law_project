import sys
import unicodedata
from typing import Any


def setup_encoding() -> None:
    """Safely configures standard output streams to handle UTF-8 printing."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def is_english_chunk(value: Any) -> bool:
    """
    Determines if a text snippet is predominantly written in English script.

    Returns True if Latin characters dominate over Devanagari script letters.
    Ignores non-script characters (numbers, punctuation, whitespace) when evaluating language.
    """
    if value is None:
        return False

    text = unicodedata.normalize("NFC", str(value)).strip()
    if not text:
        return False

    latin_count = 0
    devanagari_count = 0

    # Iterates directly over characters without allocating regex lists in memory
    for char in text:
        code = ord(char)
        if (65 <= code <= 90) or (97 <= code <= 122):  # A-Z, a-z
            latin_count += 1
        elif 0x0900 <= code <= 0x097F:  # Devanagari Unicode Range
            devanagari_count += 1

    total_letters = latin_count + devanagari_count

    # If the text has no alphabetical script letters (e.g. only numbers or symbols)
    if total_letters == 0:
        return False

    # Must contain Latin letters AND Devanagari must be less than 25% of total script letters
    return latin_count > 0 and (devanagari_count / total_letters) < 0.25
