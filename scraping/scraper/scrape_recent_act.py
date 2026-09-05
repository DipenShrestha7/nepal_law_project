import os
import re
import time
import random
from typing import Any, cast
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from tqdm import tqdm

# ==========================================
# 1. SPECIFIC TARGET CONFIGURATION
# ==========================================
TARGET_PAGE_URL = "https://lawcommission.gov.np/category/2163/?page=2"
FALLBACK_PAGE_URL = "https://lawcommission.gov.np/category/2163/page/2/"
BASE_URL = "https://lawcommission.gov.np"
SAVE_DIR = "data/raw_pdfs/np/Recent_Act"

os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================================
# 2. HTTP SESSION WITH RETRIES (TYPE SAFE)
# ==========================================
def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Connection": "keep-alive",
        }
    )

    retries = Retry(
        total=5,
        backoff_factor=3,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )

    # Cast retries to Any to resolve HTTPAdapter type stub mismatches
    adapter = HTTPAdapter(max_retries=cast(Any, retries))
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


session = get_session()


def fetch_page_soup(url: str) -> BeautifulSoup | None:
    try:
        time.sleep(random.uniform(1.0, 2.0))
        res = session.get(url, timeout=30)
        if res.status_code == 200:
            return BeautifulSoup(res.content, "html.parser")
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return None


def resolve_pdf_link(detail_url: str) -> str | None:
    if detail_url.lower().endswith(".pdf"):
        return detail_url

    soup = fetch_page_soup(detail_url)
    if soup:
        pdf_anchor = soup.find("a", href=re.compile(r"\.pdf$", re.IGNORECASE))
        if pdf_anchor:
            href = pdf_anchor.get("href")
            # Explicit type narrowing on detail page anchor
            if isinstance(href, str) and href and href != "#":
                return urljoin(BASE_URL, href)
    return None


# ==========================================
# 3. PAGE 2 SCRAPER & DOWNLOADER
# ==========================================
def scrape_recent_act_page_2():
    print(f"Targeting Recent Act Page 2: {TARGET_PAGE_URL}")
    soup = fetch_page_soup(TARGET_PAGE_URL)

    if not soup or not soup.find_all("tr"):
        print("Query string format returned no rows, trying path format...")
        soup = fetch_page_soup(FALLBACK_PAGE_URL)

    if not soup:
        print("Failed to retrieve Page 2 content.")
        return

    table_rows = soup.find_all("tr")
    extracted_items = []

    for tr in table_rows:
        anchor = tr.find(
            "a", href=re.compile(r"\.pdf$|/detail/|/pages/|/content/", re.IGNORECASE)
        )
        if not anchor:
            continue

        href = anchor.get("href")
        # Explicit type narrowing to handle str | list[str] | None from BeautifulSoup
        if not isinstance(href, str) or not href or href == "#":
            continue

        full_url = urljoin(BASE_URL, href)
        cells = tr.find_all("td")

        title = (
            cells[1].get_text(strip=True)
            if len(cells) > 1
            else anchor.get_text(strip=True)
        )
        if not title:
            title = "Untitled_Law"

        extracted_items.append({"title": title, "url": full_url})

    print(f"Found {len(extracted_items)} entries on Recent Act Page 2.\n")

    for item in tqdm(extracted_items, desc="Downloading Recent Act Page 2"):
        direct_pdf_url = resolve_pdf_link(item["url"])
        if not direct_pdf_url:
            print(f"\nCould not find PDF link for: {item['title']}")
            continue

        clean_title = (
            re.sub(r'[\\/*?:"<>|]', "", item["title"]).strip().replace(" ", "_")
        )
        filename = f"{clean_title[:90]}.pdf"
        filepath = os.path.join(SAVE_DIR, filename)

        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            continue

        try:
            time.sleep(random.uniform(1.5, 3.0))
            res = session.get(direct_pdf_url, stream=True, timeout=45)

            if res.status_code == 200:
                with open(filepath, "wb") as f:
                    for chunk in res.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)
            elif res.status_code in [429, 503]:
                print(
                    f"\nServer rate limited (HTTP {res.status_code}). Waiting 15 seconds..."
                )
                time.sleep(15)
        except Exception as e:
            print(f"\nFailed downloading {filename}: {e}")


if __name__ == "__main__":
    scrape_recent_act_page_2()
