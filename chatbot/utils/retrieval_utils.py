import re
import unicodedata
from typing import Any, Dict, List
from rank_bm25 import BM25Okapi

# Canonical mapping for common English legal variants
ENGLISH_LEGAL_TERMS = {
    "sub-section": "subsection",
    "sub section": "subsection",
    "bye-law": "byelaw",
    "bye law": "byelaw",
    "non-governmental": "nongovernmental",
    "ad-hoc": "adhoc",
}

# Pre-compile regex pattern for variant terms
_TERMS_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(key) for key in ENGLISH_LEGAL_TERMS.keys()) + r")\b",
    re.IGNORECASE,
)


def normalize_english_legal_text(text: str) -> str:
    """
    Normalizes English legal text: applies NFC Unicode, unifies dashes/hyphens,
    collapses redundant whitespace, and standardizes term variations.
    """
    if not text:
        return ""

    # 1. Unicode NFC Normalization
    normalized = unicodedata.normalize("NFC", text)

    # 2. Standardize non-standard Unicode dashes/hyphens to ASCII hyphen
    normalized = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", normalized)

    # 3. Collapse multiple spaces and newlines
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # 4. Standardize variant terms
    normalized = _TERMS_PATTERN.sub(
        lambda m: ENGLISH_LEGAL_TERMS.get(m.group(0).lower(), m.group(0)),
        normalized,
    )

    return normalized


def tokenize_legal_text(text: str) -> List[str]:
    """
    Tokenizes text for BM25 ranking, preserving numbers, section IDs, and legal terms.
    """
    clean_text = normalize_english_legal_text(text).lower()
    return re.findall(r"\b\w+\b", clean_text)


def retrieve_top_sections_in_memory(
    sections: List[Dict[str, Any]], sub_queries: List[str], top_k_per_query: int = 3
) -> List[Dict[str, Any]]:
    """
    Executes in-memory BM25 keyword ranking across extracted PDF section blocks.
    Processes each sub-query independently to preserve multi-intent query coverage.
    """
    if not sections or not sub_queries:
        return []

    # Tokenize in-memory sections (Section title + content)
    corpus = [
        tokenize_legal_text(f"{sec.get('section_title', '')} {sec.get('content', '')}")
        for sec in sections
    ]

    bm25 = BM25Okapi(corpus)

    selected_sections = []
    seen_titles = set()

    for sub_q in sub_queries:
        q_tokens = tokenize_legal_text(sub_q)
        if not q_tokens:
            continue

        # Get top-k matches for this specific sub-intent
        top_docs = bm25.get_top_n(q_tokens, sections, n=top_k_per_query)

        for doc in top_docs:
            title = doc.get("section_title", "General Provision")
            if title not in seen_titles:
                seen_titles.add(title)
                selected_sections.append(doc)

    return selected_sections


def format_sections_for_context(act_title: str, sections: List[Dict[str, Any]]) -> str:
    """
    Formats the top BM25 retrieved section blocks into clean structured context for the LLM.
    """
    if not sections:
        return f"=== RETRIEVED DOCUMENTS IN CONTEXT ===\nAct: {act_title} | Provision: No matching sections found.\n======================================"

    formatted_parts = ["=== RETRIEVED DOCUMENTS IN CONTEXT ===", f"Act: {act_title}"]

    for sec in sections:
        title = sec.get("section_title", "General Provision")
        content = sec.get("content", "").strip()
        formatted_parts.append(f"\n--- {title} ---\n{content}")

    formatted_parts.append("\n======================================")
    return "\n".join(formatted_parts)
