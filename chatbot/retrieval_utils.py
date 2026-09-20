import re
import unicodedata
from typing import Any, Dict

NEPALI_LEGAL_TERMS = {
    "असभ्य आचरण": "अभद्र व्यवहार",
    "असभ्य व्यवहार": "अभद्र व्यवहार",
    "दुष्कर्म": "दुस्कृति",
    "दुष्कृति": "दुस्कृति",
    "गैर कानुनी थुना": "गैरकानुनी थुना",
    "गैरकानूनी थुना": "गैरकानुनी थुना",
    "स्थानीय न्यायिक समिति": "न्यायिक समिति",
}


def normalize_devanagari_terms(text: str) -> str:
    """Normalizes common Nepali legal variants without changing substantive text."""
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for variant, canonical in NEPALI_LEGAL_TERMS.items():
        normalized = normalized.replace(variant, canonical)
    return normalized


def metadata_header(payload: Dict[str, Any]) -> str:
    """Builds a compact structural prefix used for embedding and prompt citations."""
    fields = [
        (
            "Doc",
            payload.get("act_title")
            or payload.get("title_np")
            or payload.get("file_name"),
        ),
        ("Part", payload.get("part")),
        ("Chapter", payload.get("chapter")),
        ("Art", payload.get("article_number")),
        ("Sec", payload.get("section_number") or payload.get("rule_number")),
        ("Title", payload.get("title")),
        ("Keywords", payload.get("keywords")),
    ]
    return (
        "["
        + " | ".join(
            f"{label}: {value}" for label, value in fields if value not in (None, "")
        )
        + "]"
    )


def embedding_text(payload: Dict[str, Any], content: str) -> str:
    """Returns the text to embed while preserving clean content in the payload."""
    return f"{metadata_header(payload)}\n{normalize_devanagari_terms(content)}".strip()
