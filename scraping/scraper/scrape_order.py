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
# 1. FORMATION ORDER (PAGES 2-4) CONFIG
# ==========================================
CATEGORY_URL = (
    "https://lawcommission.gov.np/category/development-committee-formation-order/"
)
TARGET_PAGES = [2, 3, 4]
BASE_URL = "https://lawcommission.gov.np"
SAVE_DIR = "data/raw_pdfs/np/Formation_Order"

os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================================
# 2. TYPE-SAFE HTTP SESSION
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
            if isinstance(href, str) and href and href != "#":
                return urljoin(BASE_URL, href)
    return None


# ==========================================
# 3. SCRAPE FORMATION ORDER PAGES 2 TO 4
# ==========================================
def scrape_formation_order_pages():
    extracted_items = []

    for page_num in TARGET_PAGES:
        page_url = f"{CATEGORY_URL}?page={page_num}"
        print(f"Fetching Formation Order Page {page_num}: {page_url}")

        soup = fetch_page_soup(page_url)
        if not soup or not soup.find_all("tr"):
            fallback_url = f"{CATEGORY_URL}page/{page_num}/"
            print(
                f"Query string format returned no rows, trying path format: {fallback_url}"
            )
            soup = fetch_page_soup(fallback_url)

        if not soup:
            print(f"Skipping Page {page_num}: Could not fetch content.")
            continue

        table_rows = soup.find_all("tr")
        page_count = 0

        for tr in table_rows:
            anchor = tr.find(
                "a",
                href=re.compile(r"\.pdf$|/detail/|/pages/|/content/", re.IGNORECASE),
            )
            if not anchor:
                continue

            href = anchor.get("href")
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
                title = f"Formation_Order_Page{page_num}"

            extracted_items.append({"title": title, "url": full_url, "page": page_num})
            page_count += 1

        print(f"Found {page_count} entries on Page {page_num}.")

    print(f"\nTotal queued items across pages 2, 3, and 4: {len(extracted_items)}\n")

    for item in tqdm(extracted_items, desc="Downloading Formation Order PDFs"):
        direct_pdf_url = resolve_pdf_link(item["url"])
        if not direct_pdf_url:
            print(f"\nCould not find direct PDF link for: {item['title']}")
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
    scrape_formation_order_pages()
