import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# Common legacy Nepali fonts found in government PDFs
LEGACY_FONTS = {"preeti", "kantipur", "himal", "pcs nepali", "font_preeti"}


def calculate_devanagari_health(text: str) -> float:
    """Computes a health score (0.0 to 1.0) for extracted Devanagari text.

    High scores indicate valid Unicode text. Low scores indicate corrupted or
    mismapped glyphs.
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    # 1. Ratio of characters in the Devanagari Unicode Block (U+0900 to U+097F)
    devanagari_chars = len(re.findall(r"[\u0900-\u097F]", text))
    total_non_whitespace = len(re.findall(r"\S", text))

    if total_non_whitespace == 0:
        return 0.0

    unicode_ratio = devanagari_chars / total_non_whitespace

    # 2. Count encoding artifacts common in broken PDF ToUnicode tables
    # - Halant followed by space or end of word (e.g., 'र् ')
    # - Isolated matras or double matras
    # - Displaced Reph (र्) standing alone
    anomaly_patterns = [
        r"्\s",  # Halant followed by space
        r"ि\s",  # Short-I matra followed by space
        r"[\u0900-\u097F]\s+्",  # Isolated halant
        r"\bर्\b",  # Isolated Reph
    ]

    anomalies = 0
    for pattern in anomaly_patterns:
        anomalies += len(re.findall(pattern, text))

    # Penalize score based on anomaly density
    anomaly_penalty = (anomalies * 5) / max(total_non_whitespace, 1)

    final_score = max(0.0, unicode_ratio - anomaly_penalty)
    return round(final_score, 3)


def has_legacy_fonts(page: fitz.Page) -> bool:
    """Inspects embedded PDF font metadata for legacy font names."""
    try:
        font_list = page.get_fonts()
        for font in font_list:
            font_name = font[3].lower()  # Font name index
            if any(legacy in font_name for legacy in LEGACY_FONTS):
                return True
    except Exception:
        pass
    return False
