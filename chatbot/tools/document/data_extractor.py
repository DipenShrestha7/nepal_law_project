import fitz
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
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""

    for page in doc:
        text = page.get_text("text").strip()
        if text:
            full_text += text + "\n"

    # Normalize Unicode (NFC)
    full_text = unicodedata.normalize("NFC", full_text)
    full_text = re.sub(r"\s+", " ", full_text)

    # Split by section markers: 'दफा' for Nepali, 'Section' for English
    if "दफा" in full_text:
        sections = [
            f"दफा {sec.strip()}"
            for sec in full_text.split("दफा")
            if len(sec.strip()) > 30
        ]
    elif "Section" in full_text:
        sections = [
            f"Section {sec.strip()}"
            for sec in full_text.split("Section")
            if len(sec.strip()) > 30
        ]
    else:
        # Fallback to 800-character paragraph chunks
        sections = [full_text[i : i + 800] for i in range(0, len(full_text), 700)]

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

    return top_sections


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


async def live_web_search_and_extract(act_name: str, user_query: str) -> dict:
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
    fetch_result: DocumentFetchResult = await fetcher.fetch_best_document(act_name)

    if not fetch_result.success or not fetch_result.file_bytes:
        return {
            "success": False,
            "error": fetch_result.error_message
            or "Could not retrieve document from web.",
        }

    print(f"[Live Fetch] Downloaded document from: {fetch_result.source_url}")
    print(
        f"[Live Fetch] Language: {'English' if fetch_result.is_english else 'Nepali (Devanagari)'}"
    )

    # 2. Extract & Split PDF in RAM
    sections = extract_and_split_pdf_in_memory(fetch_result.file_bytes)

    # 3. Filter top 3 most relevant sections using BM25
    top_snippets = filter_top_sections_in_memory(
        sections=sections, user_query=user_query, top_k=3
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
