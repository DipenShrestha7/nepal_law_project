import pymupdf
import re
import os
import unicodedata
from rank_bm25 import BM25Okapi
from langchain_openai import ChatOpenAI
from chatbot.tools.document.web_document_fetcher import (
    WebDocumentFetcher,
    DocumentFetchResult,
)

from chatbot.config import OPENROUTER_API_KEY, DEFAULT_MODEL

translator_llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model=DEFAULT_MODEL,
    temperature=0.1,
    default_headers={
        "HTTP-Referer": os.getenv("APP_URL", "http://localhost:3000"),
        "X-Title": "Nepal Legal Assistant - English",
    },
)


def extract_and_split_pdf_in_memory(pdf_bytes: bytes) -> list[str]:
    """
    Extracts text from PDF bytes in RAM and splits into logical sections.
    Supports both English and Devanagari text.
    """
    full_text = ""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text = page.get_text("text").strip()
            if text:
                full_text += text + "\n"

    full_text = unicodedata.normalize("NFC", full_text)
    full_text = re.sub(r"\s+", " ", full_text)

    # Split ONLY at actual section headings (e.g., "Section 14." or "दफा १४.")
    if re.search(r"दफा\s*[\d०-९]+\s*[\.:।\-]", full_text):
        raw_parts = re.split(r"(?=दफा\s*[\d०-९]+\s*[\.:।\-])", full_text)
        sections = [p.strip() for p in raw_parts if len(p.strip()) > 30]
    elif re.search(r"\bSection\s+\d+\s*[\.:\-]", full_text, flags=re.IGNORECASE):
        raw_parts = re.split(
            r"(?=\bSection\s+\d+\s*[\.:\-])", full_text, flags=re.IGNORECASE
        )
        sections = [p.strip() for p in raw_parts if len(p.strip()) > 30]
    else:
        # Larger 2,500-character fallback chunks (with 300-char overlap)
        sections = [full_text[i : i + 2500] for i in range(0, len(full_text), 2200)]

    return sections


def filter_top_sections_in_memory(
    sections: list[str], user_query: str, top_k: int = 3
) -> list[str]:
    """
    Runs lightning-fast BM25 keyword match over sections in memory.
    Execution time: < 0.01 seconds.
    """
    if not sections:
        return []

    tokenized_corpus = [re.findall(r"\w+", sec.lower()) for sec in sections]
    tokenized_query = re.findall(r"\w+", user_query.lower())

    bm25 = BM25Okapi(tokenized_corpus)
    top_sections = bm25.get_top_n(tokenized_query, sections, n=top_k)

    # A PDF may be split at sub-clause boundaries (for example, 29(1)(a)).
    # Expand each BM25 hit to every chunk belonging to its parent section so
    # penalties, exceptions, and appeal periods are returned together.
    section_patterns = (
        re.compile(r"^\s*Section\s+(\d+)", re.IGNORECASE),
        re.compile(r"^\s*दफा\s+(\d+)"),
    )

    def parent_section_key(section: str) -> tuple[str, str] | None:
        for pattern in section_patterns:
            match = pattern.match(section)
            if match:
                return (pattern.pattern, match.group(1))
        return None

    sections_by_parent: dict[tuple[str, str], list[str]] = {}
    for section in sections:
        key = parent_section_key(section)
        if key is not None:
            sections_by_parent.setdefault(key, []).append(section)

    expanded_sections: list[str] = []
    seen_sections: set[str] = set()
    for hit in top_sections:
        key = parent_section_key(hit)
        candidates = sections_by_parent.get(key, [hit]) if key is not None else [hit]
        for section in candidates:
            if section not in seen_sections:
                expanded_sections.append(section)
                seen_sections.add(section)

    return expanded_sections


async def translate_snippets_if_needed(snippets: list[str], is_english: bool) -> str:
    """
    If text is already English, returns as is.
    If Devanagari, translates ONLY the top filtered snippets using LLM API.
    """
    combined_text = "\n\n".join(snippets)

    if is_english or not combined_text.strip():
        return combined_text

    # Translate only the small required snippet
    prompt = f"""
    You are an expert Nepalese legal translator.
    Translate the following Nepalese legal provisions into clear, precise legal English.
    Preserve all section numbers, fines, terms of imprisonment, and legal conditions.

    Nepali Text:
    {combined_text}
    """

    response = await translator_llm.ainvoke(prompt)
    return response.content


async def live_web_search_and_extract(
    act_name: str,
    user_query: str,
    act_en: str | None = None,
    act_ne: str | None = None,
) -> dict:
    """
    Master Live Pipeline:
    1. Searches web (inspects up to 10 files, prefers English).
    2. Downloads best PDF to RAM.
    3. Splits and filters top sections in RAM using BM25.
    4. Translates required snippets via AI if Nepali.
    5. Returns clean English context ready for the answer node.
    """
    fetcher = WebDocumentFetcher()
    print(f"[Live Fetch] Searching web for latest version of '{act_name}'...")

    # 1. Inspect up to 10 candidate files (Prefers English, falls back to Nepali)
    fetch_result: DocumentFetchResult = await fetcher.fetch_best_document(
        act_name=act_name,
        act_en=act_en,
        act_ne=act_ne,
    )

    if not fetch_result.success:
        return {
            "success": False,
            "error": fetch_result.error_message
            or "Could not retrieve document from web.",
        }

    print(f"[Live Fetch] Retrieved document from: {fetch_result.source_url}")
    print(
        f"[Live Fetch] Language: {'English' if fetch_result.is_english else 'Nepali (Devanagari)'}"
    )

    # 2. Extract & split either a PDF or complete HTML statute in memory
    if fetch_result.file_bytes:
        sections = extract_and_split_pdf_in_memory(fetch_result.file_bytes)
    elif fetch_result.file_text:
        text = unicodedata.normalize("NFC", fetch_result.file_text)
        text = re.sub(r"\s+", " ", text).strip()
        sections = [text[i : i + 800] for i in range(0, len(text), 700)] if text else []
    else:
        return {
            "success": False,
            "error": "Document retrieval returned no usable PDF or HTML text.",
        }

    # 3. Filter top 3 most relevant sections using BM25
    top_snippets = filter_top_sections_in_memory(
        sections=sections, user_query=user_query, top_k=7
    )

    if not top_snippets:
        return {
            "success": False,
            "error": "No relevant sections found in downloaded document.",
        }

    # 4. Translate only the top 3 snippets if Devanagari
    clean_english_context = await translate_snippets_if_needed(
        snippets=top_snippets, is_english=fetch_result.is_english
    )

    # 5. Return live extracted context to Answer Generation Node
    return {
        "success": True,
        "source_url": fetch_result.source_url,
        "is_english_source": fetch_result.is_english,
        "extracted_context": clean_english_context,
    }
