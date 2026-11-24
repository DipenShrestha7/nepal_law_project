import gc
import re
from dataclasses import dataclass
from typing import List
from urllib.parse import unquote, urljoin

import httpx
import pymupdf
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024  # 15 MB safety cap for 512 MB RAM environments

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


def inspect_pdf_bytes(pdf_bytes: bytes) -> tuple[bool, str]:
    """
    Safely inspects page 1 of a PDF in a context manager (preventing C-heap leaks).
    Returns (is_english, page1_text).
    """
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            if len(doc) == 0:
                return False, ""

            first_page_text = doc[0].get_text("text").strip()
            if not first_page_text:
                return False, ""

            devanagari_chars = len(re.findall(r"[\u0900-\u097F]", first_page_text))
            total_chars = max(len(first_page_text), 1)
            is_english = (devanagari_chars / total_chars) < 0.10
            return is_english, first_page_text
    except Exception:
        return False, ""


def is_complete_html_statute(html_str: str) -> bool:
    """
    Validates plain HTML text to ensure it represents a complete statute
    rather than a short blog post, summary, or incomplete snippet.
    """
    soup = BeautifulSoup(html_str, "html.parser")
    clean_text = soup.get_text(separator=" ")

    if len(clean_text) < 6000:
        return False

    clean_text_lower = clean_text.lower()
    en_markers = ["section", "sub-section", "preamble", "chapter", "enacted by"]
    ne_markers = ["दफा", "उपदफा", "प्रस्तावना", "परिच्छेद", "ऐन"]

    has_en_structure = sum(1 for kw in en_markers if kw in clean_text_lower) >= 2
    has_ne_structure = sum(1 for kw in ne_markers if kw in clean_text_lower) >= 2

    return has_en_structure or has_ne_structure


# ============================================================================
# 3. Safe HTTP Streamer & HTML/Embedded PDF Resolver
# ============================================================================


async def _safe_get(
    client: httpx.AsyncClient, url: str, timeout: float = 15.0
) -> tuple[int, str, bytes]:
    """
    Streams response bytes up to MAX_DOWNLOAD_BYTES so oversized PDFs never trigger OOM.
    Returns (status_code, content_type, body_bytes).
    """
    async with client.stream(
        "GET", url, timeout=timeout, follow_redirects=True
    ) as response:
        content_type = response.headers.get("content-type", "").lower()
        if response.status_code != 200:
            return response.status_code, content_type, b""

        content_length = response.headers.get("content-length")
        if content_length and content_length.isdigit():
            if int(content_length) > MAX_DOWNLOAD_BYTES:
                raise ValueError(
                    f"File exceeds {MAX_DOWNLOAD_BYTES // (1024 * 1024)}MB limit"
                )

        chunks = bytearray()
        async for chunk in response.aiter_bytes(chunk_size=65536):
            chunks.extend(chunk)
            if len(chunks) > MAX_DOWNLOAD_BYTES:
                raise ValueError(
                    f"File exceeded {MAX_DOWNLOAD_BYTES // (1024 * 1024)}MB limit"
                )

        return 200, content_type, bytes(chunks)


def _extract_pdf_links_from_html(
    base_url: str, html_text: str
) -> tuple[list[str], list[str]]:
    """
    Extracts embedded PDF URLs (from tags, viewer query params, or raw JS/text URLs)
    and Law Commission detail URLs from an HTML page.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    pdf_urls: list[str] = []
    detail_urls: list[str] = []

    for tag in soup.find_all(["iframe", "embed", "object", "a"]):
        src = tag.get("src") or tag.get("data") or tag.get("href")
        if not src:
            continue

        src = unquote(src)
        if "file=" in src:
            match = re.search(r"[?&]file=([^&]+)", src)
            if match:
                src = match.group(1)

        full_url = urljoin(base_url, src)
        if ".pdf" in full_url.lower() and full_url not in pdf_urls:
            pdf_urls.append(full_url)
        elif (
            "lawcommission.gov.np/content/" in full_url.lower()
            and full_url not in detail_urls
        ):
            detail_urls.append(full_url)

    # Fallback: extract direct PDF URLs embedded in JS or viewer text (e.g. giwmscdnone.gov.np)
    for raw_pdf in re.findall(
        r'https?://[^\s"\'<>]+\.pdf', html_text, flags=re.IGNORECASE
    ):
        clean_pdf = unquote(raw_pdf)
        if clean_pdf not in pdf_urls:
            pdf_urls.append(clean_pdf)

    return pdf_urls, detail_urls


async def resolve_and_download_url(client: httpx.AsyncClient, url: str) -> dict:
    """
    Fetches a URL safely. Direct PDFs are returned immediately.
    If the URL is an HTML landing page with an embedded PDF viewer, extracts and downloads the PDF.
    """
    try:
        status, content_type, body = await _safe_get(client, url, timeout=15.0)
        if status != 200 or not body:
            return {"valid": False, "url": url}

        # Case A: Direct PDF Link
        if "application/pdf" in content_type or body.startswith(b"%PDF"):
            return {
                "valid": True,
                "url": url,
                "file_bytes": body,
                "file_type": "pdf",
            }

        # Case B: HTML Landing Page
        if "text/html" in content_type:
            html_text = body.decode("utf-8", errors="ignore")
            del body

            pdf_urls, detail_urls = _extract_pdf_links_from_html(url, html_text)

            for pdf_url in pdf_urls[:3]:
                try:
                    p_status, _, p_body = await _safe_get(client, pdf_url, timeout=15.0)
                    if p_status == 200 and p_body.startswith(b"%PDF"):
                        return {
                            "valid": True,
                            "url": pdf_url,
                            "file_bytes": p_body,
                            "file_type": "embedded_pdf",
                        }
                except Exception:
                    continue

            for detail_url in detail_urls[:3]:
                try:
                    d_status, _, d_body = await _safe_get(
                        client, detail_url, timeout=15.0
                    )
                    if d_status != 200 or not d_body:
                        continue
                    if d_body.startswith(b"%PDF"):
                        return {
                            "valid": True,
                            "url": detail_url,
                            "file_bytes": d_body,
                            "file_type": "pdf",
                        }

                    d_html = d_body.decode("utf-8", errors="ignore")
                    del d_body
                    sub_pdfs, _ = _extract_pdf_links_from_html(detail_url, d_html)
                    for sub_pdf in sub_pdfs[:2]:
                        p_status, _, p_body = await _safe_get(
                            client, sub_pdf, timeout=15.0
                        )
                        if p_status == 200 and p_body.startswith(b"%PDF"):
                            return {
                                "valid": True,
                                "url": sub_pdf,
                                "file_bytes": p_body,
                                "file_type": "embedded_pdf",
                            }

                    if is_complete_html_statute(d_html):
                        return {
                            "valid": True,
                            "url": detail_url,
                            "file_text": d_html,
                            "file_type": "html_text",
                        }
                except Exception:
                    continue

            return {
                "valid": True,
                "url": url,
                "file_text": html_text,
                "file_type": "html_text",
            }

    except Exception as e:
        print(f"[Resolver Warning] Failed inspecting {url}: {e}")

    return {"valid": False, "url": url}


# ============================================================================
# 4. Domain & Content Scoring Constants
# ============================================================================

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
    "apnilaw.com",
    "indiankanoon",
    "nepjol.info",
    "article/download",
    ".co.za",
    ".gov.in",
]

FOREIGN_STATUTE_DOMAINS = [
    ".co.za",
    ".gov.in",
    "indiankanoon",
    "legislation.gov.uk",
    "leginfo.legislature.ca.gov",
]

TRUSTED_ARCHIVES = [
    "lawcommission.gov.np",
    "giwmscdnone.gov.np",
    "ican.org.np",
    "bwcimplementation.org",
    "natlex.ilo.org",
    "ilo.org",
    "molpa.gov.np",
    "nepalrajpatra.ddg.gov.np",
    "ag.gov.np",
    "supremecourt.gov.np",
    "fwld.org",
]

CURRENT_EDITION_MARKERS = ["2017", "2074", "2076", "२०१७", "२०७४", "२०७६"]
HISTORICAL_EDITION_MARKERS = ["1993", "2049", "२०४९"]

STATUTORY_MARKERS = [
    "act no.",
    "an act relating to",
    "preamble:",
    "whereas, it is expedient",
    "short title and commencement",
    "published in nepal gazette",
    "authentication date",
]

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
    "abstract:",
    "keywords:",
    "law review",
    "nepal law review",
    "issn",
    "doi:",
    "journal of",
]


def build_search_queries(act_en: str, act_ne: str) -> List[str]:
    """Build ordered statutory PDF queries for English and Nepali document lookup."""
    raw_en = act_en.replace("(Code)", "Code").strip()
    title_variants = [raw_en]
    variant_without_code = re.sub(
        r"\s*\(Code\)", "", raw_en, flags=re.IGNORECASE
    ).strip()
    variant_without_national = re.sub(
        r"^National\s+", "", variant_without_code, flags=re.IGNORECASE
    ).strip()
    year_stripped = re.sub(r",\s*\d{4}(?:\s*\(\d{4}\))?$", "", raw_en).strip()

    for variant in (
        variant_without_code,
        variant_without_national,
        year_stripped,
    ):
        if variant and variant not in title_variants:
            title_variants.append(variant)

    act_ne = act_ne.strip()
    queries = []
    for title_variant in title_variants:
        queries.extend(
            [
                f'"{title_variant}" Nepal PDF',
                f'"{title_variant}" Nepal site:lawcommission.gov.np',
                f'"{title_variant}" Nepal site:gov.np',
                f'"{title_variant}" Nepal Act PDF English',
            ]
        )
    if act_ne:
        queries.append(f'"{act_ne}" नेपाल site:lawcommission.gov.np')
        queries.append(f'"{act_ne}" नेपाल PDF')

    return queries


def score_candidate_document(
    url: str,
    page1_text: str,
    is_english: bool,
    expected_title: str | None = None,
) -> int:
    """
    Calculates a relevance & authority score based on legal structure, domain authority,
    and penalty keywords.
    """
    url_lower = url.lower()
    text_lower = page1_text.lower()

    if any(junk in url_lower for junk in BLACKLISTED_DOMAINS):
        return -500
    if any(foreign in url_lower for foreign in FOREIGN_STATUTE_DOMAINS):
        return -500

    # Immediately disqualify academic journal papers (both PDF and HTML)
    if any(
        marker in text_lower[:1500]
        for marker in ("abstract:", "keywords:", "law review", "issn", "doi:")
    ):
        return -500

    if expected_title:
        expected = expected_title.lower()
        relevance_text = f"{url_lower} {text_lower}"
        if "penal" in expected or "criminal code" in expected:
            if not any(
                marker in relevance_text
                for marker in ("penal", "criminal code", "muluki", "फौजदारी")
            ):
                return -500
        elif "labour" in expected or "labor" in expected:
            if "labour" not in relevance_text and "labor" not in relevance_text:
                return -500
        else:
            significant_words = [
                word
                for word in re.findall(r"[a-z]{4,}", expected)
                if word not in {"act", "nepal", "code", "national"}
            ]
            if significant_words and not any(
                word in relevance_text for word in significant_words
            ):
                return -500

    score = 50 if is_english else 10

    if any(domain in url_lower for domain in TRUSTED_ARCHIVES):
        score += 40
    if "lawcommission.gov.np" in url_lower or "giwmscdnone.gov.np" in url_lower:
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

    for marker in STATUTORY_MARKERS:
        if marker in text_lower:
            score += 30

    if "unofficial translation" in text_lower:
        score -= 80

    for penalty in USER_SUMMARY_PENALTIES:
        if penalty in text_lower or penalty in url_lower:
            score -= 150

    if url_lower.endswith(".pdf"):
        score += 60
    if any(
        d in url_lower for d in (".gov.np", "faolex.fao.org", "lawcommission.gov.np")
    ):
        score += 100

    is_nepal_source = (
        ".gov.np" in url_lower
        or "nepal" in text_lower
        or "नेपाल" in page1_text
        or any(domain in url_lower for domain in TRUSTED_ARCHIVES)
    )
    if not is_nepal_source:
        return -500

    if (
        "short title and commencement" in text_lower
        or "संक्षिप्त नाम र प्रारम्भ" in page1_text
    ):
        score += 80

    return score


# ============================================================================
# 5. Core Fetching & Selection Engine
# ============================================================================


class WebDocumentFetcher:
    def __init__(self, tavily_api_key: str | None = None):
        self.search_tool = TavilySearch(max_results=10)

    async def _search_web_for_links(
        self, act_name: str, act_en: str | None = None, act_ne: str | None = None
    ) -> list[str]:
        """Searches the web using ordered statutory anchor phrases."""
        official_en = (act_en or act_name).strip()
        official_ne = (act_ne or "").strip()
        urls: list[str] = []
        seen_urls: set[str] = set()

        for query in build_search_queries(official_en, official_ne):
            try:
                results = await self.search_tool.ainvoke({"query": query})
                if isinstance(results, dict):
                    results = results.get("results", [])
                elif not isinstance(results, list):
                    results = []

                for res in results:
                    url = (
                        res.get("url")
                        if isinstance(res, dict)
                        else (
                            res
                            if isinstance(res, str) and res.startswith("http")
                            else None
                        )
                    )
                    if (
                        url
                        and url not in seen_urls
                        and not any(
                            blocked in url.lower() for blocked in BLACKLISTED_DOMAINS
                        )
                        and not any(
                            foreign in url.lower()
                            for foreign in FOREIGN_STATUTE_DOMAINS
                        )
                    ):
                        seen_urls.add(url)
                        urls.append(url)
            except Exception as e:
                print(f"[Search Warning] Query failed: {query}: {e}")

            if len(urls) >= 10:
                break

        return urls[:10]

    async def fetch_best_document(
        self, act_name: str, act_en: str | None = None, act_ne: str | None = None
    ) -> DocumentFetchResult:
        """
        Searches candidate URLs, checks top 4 sequentially in RAM, scores each document,
        and returns the highest-scoring verified DocumentFetchResult.
        """
        urls = await self._search_web_for_links(act_name, act_en, act_ne)
        if not urls:
            return DocumentFetchResult(
                success=False,
                error_message=f"No search results found for '{act_name}'.",
            )

        clean_urls = [
            u for u in urls if not any(bad in u.lower() for bad in BLACKLISTED_DOMAINS)
        ]
        clean_urls.sort(
            key=lambda u: (".gov.np" in u.lower(), u.lower().endswith(".pdf")),
            reverse=True,
        )

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/pdf,text/html,application/xhtml+xml,*/*;q=0.9",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "close",
        }

        MIN_ACCEPTABLE_SCORE = 30
        HIGH_CONFIDENCE_SCORE = 150
        evaluated_results: list[dict] = []

        async with httpx.AsyncClient(
            headers=headers,
            follow_redirects=True,
            timeout=15.0,
            verify=False,
        ) as client:
            # Inspect and score top 4 candidates ONE BY ONE
            for url in clean_urls[:4]:
                candidate = await resolve_and_download_url(client, url)
                if not candidate.get("valid"):
                    continue

                file_type = candidate.get("file_type")

                # Case A: PDF Documents
                if file_type in ("pdf", "embedded_pdf"):
                    pdf_bytes = candidate["file_bytes"]
                    is_english, page1_text = inspect_pdf_bytes(pdf_bytes)

                    score = score_candidate_document(
                        url=candidate["url"],
                        page1_text=page1_text,
                        is_english=is_english,
                        expected_title=act_name,
                    )

                    if score < MIN_ACCEPTABLE_SCORE:
                        del pdf_bytes, candidate
                        gc.collect()
                        continue

                    result_obj = DocumentFetchResult(
                        success=True,
                        source_url=candidate["url"],
                        file_bytes=pdf_bytes,
                        file_type=file_type,
                        is_english=is_english,
                        target_collection=(
                            "core_english_statutes"
                            if is_english
                            else "nepali_legal_chunks"
                        ),
                    )
                    evaluated_results.append({"score": score, "result": result_obj})

                    # Stop immediately if we found an authoritative statute PDF
                    if score >= HIGH_CONFIDENCE_SCORE:
                        break

                # Case B: Complete Plain HTML Text
                elif file_type == "html_text":
                    html_str = candidate.get("file_text", "")
                    if not is_complete_html_statute(html_str):
                        del html_str, candidate
                        gc.collect()
                        continue

                    devanagari_chars = len(re.findall(r"[\u0900-\u097F]", html_str))
                    is_english = (devanagari_chars / max(len(html_str), 1)) < 0.10

                    score = score_candidate_document(
                        url=candidate["url"],
                        page1_text=html_str[:2000],
                        is_english=is_english,
                        expected_title=act_name,
                    )

                    if score < MIN_ACCEPTABLE_SCORE:
                        del html_str, candidate
                        gc.collect()
                        continue

                    result_obj = DocumentFetchResult(
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
                    )
                    evaluated_results.append({"score": score, "result": result_obj})

                    if score >= HIGH_CONFIDENCE_SCORE:
                        break

        if evaluated_results:
            evaluated_results.sort(key=lambda x: x["score"], reverse=True)
            best = evaluated_results[0]

            # Free RAM from any lower-scoring candidates before returning
            for other in evaluated_results[1:]:
                other["result"].file_bytes = None
                other["result"].file_text = None
            gc.collect()

            print(
                f"[Selection Success] Selected URL: {best['result'].source_url} "
                f"(Score: {best['score']})"
            )
            return best["result"]

        return DocumentFetchResult(
            success=False,
            error_message="No verified or complete legal documents found matching quality standards.",
        )
