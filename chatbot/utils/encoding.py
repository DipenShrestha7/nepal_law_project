import re
import sys
import unicodedata
from typing import Any


def setup_encoding() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def is_english_chunk(value: Any) -> bool:
    if value is None:
        return False
    text = unicodedata.normalize("NFC", str(value)).strip()
    if not text:
        return False
    devanagari_chars = len(re.findall(r"[\u0900-\u097F]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    total_chars = len(text)
    if total_chars == 0:
        return False
    return latin_chars > 0 and (devanagari_chars / total_chars) < 0.25
