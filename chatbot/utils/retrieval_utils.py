import re
import unicodedata
from typing import Any, Dict

# Canonical mapping for common English legal variants and hyphenation inconsistencies
ENGLISH_LEGAL_TERMS = {
    "sub-section": "subsection",
    "sub section": "subsection",
    "bye-law": "byelaw",
    "bye law": "byelaw",
    "non-governmental": "nongovernmental",
    "ad-hoc": "adhoc",
}


def normalize_english_legal_text(text: str) -> str:
    """Normalizes English legal text: applies NFC Unicode, unifies dashes/hyphens,

    collapses redundant whitespace, and standardizes term variations.
    """
    if not text:
        return ""

    # 1. Unicode NFC Normalization
    normalized = unicodedata.normalize("NFC", text)

    # 2. Standardize non-standard Unicode dashes/hyphens to standard ASCII hyphen
    normalized = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", normalized)

    # 3. Collapse multiple spaces and newlines into single spaces
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # 4. Standardize common variant terms using case-insensitive substitution
    for variant, canonical in ENGLISH_LEGAL_TERMS.items():
        pattern = re.compile(r"\b" + re.escape(variant) + r"\b", re.IGNORECASE)
        normalized = pattern.sub(canonical, normalized)

    return normalized


def metadata_header(payload: Dict[str, Any]) -> str:
    """Builds a compact, structured metadata prefix used for embedding generation

    and LLM context citations.
    """
    doc_title = payload.get("act_title") or payload.get("source_file")

    part = payload.get("part")
    chapter = payload.get("chapter")
    article = payload.get("article_number")
    section = payload.get("section_number") or payload.get("rule_number")
    schedule = payload.get("schedule_number")  # Explicitly included for Schedules
    title = payload.get("title")
    keywords = payload.get("keywords")

    # Prevent duplicate header values if doc_title is identical to title
    if doc_title and title and doc_title.strip().lower() == title.strip().lower():
        title = None

    fields = [
        ("Doc", doc_title),
        ("Part", part),
        ("Chapter", chapter),
        ("Art", article),
        ("Sec", section),
        ("Sched", schedule),
        ("Title", title),
        ("Keywords", keywords),
    ]

    header_parts = [
        f"{label}: {value}"
        for label, value in fields
        if value not in (None, "", "null", "None")
    ]

    return f"[{' | '.join(header_parts)}]" if header_parts else ""


def embedding_text_en(payload: Dict[str, Any], content: str) -> str:
    """Combines structured metadata header and normalized text into a single string

    passed to the embedding model (BGE-M3).
    """
    header = metadata_header(payload)
    clean_content = normalize_english_legal_text(content)

    if header:
        return f"{header}\n{clean_content}".strip()
    return clean_content
