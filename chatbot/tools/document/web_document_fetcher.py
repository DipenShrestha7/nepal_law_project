import asyncio
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import urljoin, unquote
import fitz  # PyMuPDF
import httpx
from bs4 import BeautifulSoup
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# 1. Output Data Structure
# ============================================================================


@dataclass
class DocumentFetchResult:
    success: bool
    source_url: str | None = None
    file_bytes: bytes | None = None
    file_text: str | None = None
    file_type: str | None = None  # "pdf", "embedded_pdf", "html_statute"
    is_english: bool = False
    target_collection: str | None = (
        None  # "core_english_statutes" or "nepali_legal_chunks"
    )
    error_message: str | None = None


# ============================================================================
# 2. Language & Completeness Evaluators
# ============================================================================


def is_english_pdf_bytes(pdf_bytes: bytes) -> bool:
    """
    Inspects page 1 of a PDF and calculates the ratio of Devanagari characters.
    Returns True if Devanagari ratio is under 10% (indicating English).
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) == 0:
            return False

        # Extract text from page 1 for rapid detection
        first_page_text = doc[0].get_text("text").strip()
        if not first_page_text:
            return False

        # Count Devanagari Unicode characters (\u0900 to \u097F)
        devanagari_chars = len(re.findall(r"[\u0900-\u097F]", first_page_text))
        total_chars = max(len(first_page_text), 1)

        devanagari_ratio = devanagari_chars / total_chars
        return devanagari_ratio < 0.10

    except Exception:
        return False


def is_complete_html_statute(html_str: str) -> bool:
    """
    Validates plain HTML text to ensure it represents a complete statute
    rather than a short blog post, summary, or incomplete snippet.
    """
    soup = BeautifulSoup(html_str, "html.parser")
    clean_text = soup.get_text(separator=" ")

    # Exclude short summaries (complete statutes are rarely under 6,000 chars)
    if len(clean_text) < 6000:
        return False

    # Verify structural statutory markers
    clean_text_lower = clean_text.lower()
    en_markers = ["section", "sub-section", "preamble", "chapter", "enacted by"]
    ne_markers = ["दफा", "उपदफा", "प्रस्तावना", "परिच्छेद", "ऐन"]

    has_en_structure = sum(1 for kw in en_markers if kw in clean_text_lower) >= 2
    has_ne_structure = sum(1 for kw in ne_markers if kw in clean_text_lower) >= 2

    return has_en_structure or has_ne_structure


# ============================================================================
# 3. HTML & Embedded PDF Resolver
# ============================================================================


async def resolve_and_download_url(client: httpx.AsyncClient, url: str) -> dict:
    """
    Fetches a URL. Direct PDFs are returned immediately.
    If the URL is an HTML landing page with an embedded PDF viewer (e.g. lawcommission.gov.np),
    parses the DOM to find and download the hidden PDF.
    """
    try:
        response = await client.get(url, timeout=12.0, follow_redirects=True)
        if response.status_code != 200:
            return {"valid": False, "url": url}

        content_type = response.headers.get("content-type", "").lower()

        # Case A: Direct PDF Link
        if "application/pdf" in content_type or response.content.startswith(b"%PDF"):
            return {
                "valid": True,
                "url": url,
                "file_bytes": response.content,
                "file_type": "pdf",
            }

        # Case B: HTML Landing Page (May contain embedded PDF viewer)
        if "text/html" in content_type:
            soup = BeautifulSoup(response.text, "html.parser")
            embedded_pdf_url = None

            # Look for <iframe>, <embed>, <object>, or <a> pointing to a PDF or PDF viewer
            for tag in soup.find_all(["iframe", "embed", "object", "a"]):
                src = tag.get("src") or tag.get("data") or tag.get("href")
                if not src:
                    continue

                # Unquote URL parameters (e.g., viewer.html?file=/pdf/law.pdf)
                src = unquote(src)

                if "file=" in src:
                    match = re.search(r"[?&]file=([^&]+)", src)
                    if match:
                        src = match.group(1)

                if ".pdf" in src.lower():
                    embedded_pdf_url = urljoin(url, src)
                    break

            # If an embedded PDF link was extracted, download it
            if embedded_pdf_url:
                pdf_res = await client.get(
                    embedded_pdf_url, timeout=12.0, follow_redirects=True
                )
                if pdf_res.status_code == 200 and pdf_res.content.startswith(b"%PDF"):
                    return {
                        "valid": True,
                        "url": embedded_pdf_url,
                        "file_bytes": pdf_res.content,
                        "file_type": "embedded_pdf",
                    }

            # If no embedded PDF exists, return HTML text for completeness verification
            return {
                "valid": True,
                "url": url,
                "file_text": response.text,
                "file_type": "html_text",
            }

    except Exception as e:
        print(f"[Resolver Warning] Failed inspecting {url}: {e}")

    return {"valid": False, "url": url}


# ============================================================================
# 4. Domain & Content Scoring Constants
# ============================================================================

# Blacklisted repositories that hold user uploads, student notes, or SEO blogs
BLACKLISTED_DOMAINS = [
    "slideshare.net",
    "coursehero.com",
    "studocu.com",
    "scribd.com",
    "academia.edu",
    "researchgate.net",
    "github.com",
    "drive.google.com",
    "wordpress.com",
    "blogspot.com",
    "medium.com",
]

# Trusted archives for official and legal texts
TRUSTED_ARCHIVES = [
    "lawcommission.gov.np",
    "natlex.ilo.org",
    "ilo.org",
    "molpa.gov.np",
    "nepalrajpatra.ddg.gov.np",
    "ag.gov.np",
    "supremecourt.gov.np",
    "fwld.org",
]

# Current acts should outrank historical/repealed editions when both are found.
CURRENT_EDITION_MARKERS = ["2019", "2076", "२०७६"]
HISTORICAL_EDITION_MARKERS = ["1993", "2049", "२०४९"]

# Official legal publishing and statutory structural markers
STATUTORY_MARKERS = [
    "act no.",
    "an act relating to",
    "preamble:",
    "whereas, it is expedient",
    "short title and commencement",
    "published in nepal gazette",
    "authentication date",
]

# Penalty phrases for user commentary or unofficial drafts
USER_SUMMARY_PENALTIES = [
    "submitted by",
    "assignment",
    "lecture notes",
    "summary of",
    "prepared by",
    "class notes",
    "presentation",
    "submitted to",
    "draft bill",
    "proposal",
]


def score_candidate_document(url: str, page1_text: str, is_english: bool) -> int:
    """
    Calculates a relevance & authority score based on legal structure, domain authority,
    and penalty keywords rather than strict domain filtering.
    """
    url_lower = url.lower()
    text_lower = page1_text.lower()

    # 1. Instant Disqualification for Blacklisted Repositories
    if any(junk_domain in url_lower for junk_domain in BLACKLISTED_DOMAINS):
        return -500

    score = 0

    # 2. Language Preference (+50 for English, +10 for Devanagari)
    if is_english:
        score += 50
    else:
        score += 10

    # 3. Known Legal Archive Domain Boost (+40)
    if any(domain in url_lower for domain in TRUSTED_ARCHIVES):
        score += 40
    if "lawcommission.gov.np" in url_lower:
        score += 60

    if any(
        marker in url_lower or marker in text_lower
        for marker in CURRENT_EDITION_MARKERS
    ):
        score += 100
    if any(
        marker in url_lower or marker in text_lower
        for marker in HISTORICAL_EDITION_MARKERS
    ):
        score -= 100

    # 4. Statutory Structural Headers (+30 points each)
    for marker in STATUTORY_MARKERS:
        if marker in text_lower:
            score += 30

    # 5. Penalties for Unofficial Disclaimers (-80) and Student Notes (-150)
    if "unofficial translation" in text_lower:
        score -= 80

    for penalty in USER_SUMMARY_PENALTIES:
        if penalty in text_lower or penalty in url_lower:
            score -= 150

    return score


# ============================================================================
# 5. Core Fetching & Selection Engine
# ============================================================================


class WebDocumentFetcher:
    def __init__(self, tavily_api_key: str | None = None):
        # Fetch up to 10 candidate links to inspect deeper results
        self.search_tool = TavilySearchResults(max_results=10)

    async def _search_web_for_links(self, act_name: str) -> list[str]:
        """
        Searches the open web using statutory anchor phrases without site restriction.
        """
        # Primary Query: Target official act anchor phrases across the web
        query = (
            f'"{act_name}" Nepal current prevailing law latest 2026 2083'
            f"official English PDF"
        )

        try:
            results = await self.search_tool.ainvoke({"query": query})
            urls = [res.get("url") for res in results if res.get("url")]

            if urls:
                return urls
        except Exception as e:
            print(f"[Search Warning] Primary query failed: {e}")

        # Fallback Query: Broader legal search if primary query yields no links
        fallback_query = (
            f'"{act_name}" Nepal current prevailing law official '
            f"lawcommission.gov.np PDF"
        )
        try:
            results = await self.search_tool.ainvoke({"query": fallback_query})
            return [res.get("url") for res in results if res.get("url")]
        except Exception as e:
            print(f"[Search Error] Fallback query failed: {e}")
            return []

    async def fetch_best_document(self, act_name: str) -> DocumentFetchResult:
        """
        Executes search, inspects top 10 candidate URLs concurrently, resolves HTML viewers,
        checks language, scores candidates, and selects the optimal document.
        """
        urls = await self._search_web_for_links(act_name)
        if not urls:
            return DocumentFetchResult(
                success=False,
                error_message=f"No search results found for '{act_name}'.",
            )

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            # Inspect top 10 candidate URLs concurrently
            tasks = [resolve_and_download_url(client, url) for url in urls[:10]]
            candidates = await asyncio.gather(*tasks)

        scored_candidates = []

        for candidate in candidates:
            if not candidate.get("valid"):
                continue

            # Case A: PDF Documents
            if candidate["file_type"] in ["pdf", "embedded_pdf"]:
                pdf_bytes = candidate["file_bytes"]
                is_english = is_english_pdf_bytes(pdf_bytes)

                # Extract first page text safely
                try:
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    page1_text = doc[0].get_text("text") if len(doc) > 0 else ""
                except Exception:
                    page1_text = ""

                # Calculate authority & relevance score
                score = score_candidate_document(
                    url=candidate["url"], page1_text=page1_text, is_english=is_english
                )

                scored_candidates.append(
                    {
                        "score": score,
                        "result": DocumentFetchResult(
                            success=True,
                            source_url=candidate["url"],
                            file_bytes=pdf_bytes,
                            file_type=candidate["file_type"],
                            is_english=is_english,
                            target_collection=(
                                "core_english_statutes"
                                if is_english
                                else "nepali_legal_chunks"
                            ),
                        ),
                    }
                )

            # Case B: Complete Plain HTML Text (If no embedded PDF was found)
            elif candidate["file_type"] == "html_text":
                html_str = candidate.get("file_text", "")
                if is_complete_html_statute(html_str):
                    devanagari_chars = len(re.findall(r"[\u0900-\u097F]", html_str))
                    is_english = (devanagari_chars / max(len(html_str), 1)) < 0.10

                    score = score_candidate_document(
                        url=candidate["url"],
                        page1_text=html_str[:2000],
                        is_english=is_english,
                    )

                    scored_candidates.append(
                        {
                            "score": score,
                            "result": DocumentFetchResult(
                                success=True,
                                source_url=candidate["url"],
                                file_text=html_str,
                                file_type="html_statute",
                                is_english=is_english,
                                target_collection=(
                                    "core_english_statutes"
                                    if is_english
                                    else "nepali_legal_chunks"
                                ),
                            ),
                        }
                    )

        # --- RANK AND RETURN HIGHEST SCORE ABOVE THRESHOLD ---
        MIN_ACCEPTABLE_SCORE = 30

        if scored_candidates:
            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
            best_candidate = scored_candidates[0]

            if best_candidate["score"] >= MIN_ACCEPTABLE_SCORE:
                print(
                    f"[Selection Success] Selected URL: {best_candidate['result'].source_url} "
                    f"(Score: {best_candidate['score']})"
                )
                return best_candidate["result"]
            else:
                print(
                    f"[Selection Rejected] Best candidate score ({best_candidate['score']}) "
                    f"did not meet minimum threshold ({MIN_ACCEPTABLE_SCORE})."
                )

        return DocumentFetchResult(
            success=False,
            error_message="No verified or complete legal documents found matching quality standards.",
        )
