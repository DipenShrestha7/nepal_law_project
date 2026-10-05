import re
import unicodedata
from typing import Any, Dict

# Canonical mapping for common English legal variants
ENGLISH_LEGAL_TERMS = {
    "sub-section": "subsection",
    "sub section": "subsection",
    "bye-law": "byelaw",
    "bye law": "byelaw",
    "non-governmental": "nongovernmental",
    "ad-hoc": "adhoc",
}

# 1. Pre-compile a single regex pattern for all variant terms at module load (10x faster)
_TERMS_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(key) for key in ENGLISH_LEGAL_TERMS.keys()) + r")\b",
    re.IGNORECASE,
)


def normalize_english_legal_text(text: str) -> str:
    """
    Normalizes English legal text: applies NFC Unicode, unifies dashes/hyphens,
    collapses redundant whitespace, and standardizes term variations in a single pass.
    """
    if not text:
        return ""

    # 1. Unicode NFC Normalization
    normalized = unicodedata.normalize("NFC", text)

    # 2. Standardize non-standard Unicode dashes/hyphens to standard ASCII hyphen
    normalized = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", normalized)

    # 3. Collapse multiple spaces and newlines into single spaces
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # 4. Standardize variant terms in a single regex pass using dictionary lookup
    normalized = _TERMS_PATTERN.sub(
        lambda m: ENGLISH_LEGAL_TERMS.get(m.group(0).lower(), m.group(0)),
        normalized,
    )

    return normalized


def metadata_header(payload: Dict[str, Any]) -> str:
    """
    Builds a compact, structured metadata prefix used for embedding generation
    and LLM context citations.
    """
    doc_title = payload.get("act_title")
    if not doc_title and payload.get("source_file"):
        doc_title = str(payload["source_file"]).replace(".pdf", "")

    part = payload.get("part")
    chapter = payload.get("chapter")
    article = payload.get("article_number")

    # Explicit None check to handle Section 0 correctly
    section = payload.get("section_number")
    if section is None:
        section = payload.get("rule_number")

    schedule = payload.get("schedule_number")
    title = payload.get("title")

    # Handle keyword list formatting
    keywords = payload.get("keywords")
    if isinstance(keywords, list):
        keywords = ", ".join(str(k) for k in keywords)

    # Prevent duplicate header values if doc_title is identical to title
    if (
        doc_title
        and title
        and str(doc_title).strip().lower() == str(title).strip().lower()
    ):
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
        if value is not None and str(value).strip().lower() not in ("", "null", "none")
    ]

    return f"[{' | '.join(header_parts)}]" if header_parts else ""


def embedding_text_en(payload: Dict[str, Any], content: str) -> str:
    """
    Combines structured metadata header and normalized text into a single string
    passed to the embedding model (BGE-M3 / Qdrant).
    """
    header = metadata_header(payload)
    clean_content = normalize_english_legal_text(content)

    if header:
        return f"{header}\n{clean_content}".strip()
    return clean_content
