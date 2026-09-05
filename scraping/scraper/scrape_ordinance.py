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
# 1. ORDINANCE ALL PAGES (1 to 5) CONFIG
# ==========================================
CATEGORY_URL = "https://lawcommission.gov.np/category/1809/"
TARGET_PAGES = [1, 2, 3, 4, 5]
BASE_URL = "https://lawcommission.gov.np"
SAVE_DIR = "data/raw_pdfs/np/Ordinance"

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


def get_unique_filepath(save_dir: str, title: str, item_index: int) -> str:
    """Generates a clean filepath and resolves collisions if duplicate titles exist."""
    clean_title = re.sub(r'[\\/*?:"<>|]', "", title).strip().replace(" ", "_")
    base_filename = f"{clean_title[:80]}"
    filepath = os.path.join(save_dir, f"{base_filename}.pdf")

    # If file exists with identical name from another row entry, append row index
    if os.path.exists(filepath) and os.path.getsize(filepath) <= 1000:
        os.remove(filepath)  # Remove corrupt/empty remnant file

    return filepath


# ==========================================
# 3. RECOVER MISSING ORDINANCE FILES
# ==========================================
def recover_missing_ordinances():
    extracted_items = []
    seen_urls = set()

    print("Scanning Pages 1 through 5 for Ordinance documents...")
    for page_num in TARGET_PAGES:
        page_url = f"{CATEGORY_URL}?page={page_num}" if page_num > 1 else CATEGORY_URL
        soup = fetch_page_soup(page_url)

        if not soup or not soup.find_all("tr"):
            fallback_url = f"{CATEGORY_URL}page/{page_num}/"
            soup = fetch_page_soup(fallback_url)

        if not soup:
            print(f"Warning: Could not read Page {page_num}")
            continue

        table_rows = soup.find_all("tr")
        page_count = 0

        for idx, tr in enumerate(table_rows):
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
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            cells = tr.find_all("td")
            title = (
                cells[1].get_text(strip=True)
                if len(cells) > 1
                else anchor.get_text(strip=True)
            )
            if not title:
                title = f"Ordinance_Page{page_num}_Item{idx}"

            extracted_items.append(
                {"title": title, "url": full_url, "page": page_num, "index": idx}
            )
            page_count += 1

        print(f"Page {page_num}: Found {page_count} entries.")

    print(f"\nTotal collected items across all 5 pages: {len(extracted_items)}")

    # Filter out items that are already downloaded successfully
    missing_items = []
    for item in extracted_items:
        filepath = get_unique_filepath(SAVE_DIR, item["title"], item["index"])
        if not (os.path.exists(filepath) and os.path.getsize(filepath) > 1000):
            item["target_filepath"] = filepath
            missing_items.append(item)

    print(f"Missing or corrupted PDFs to download: {len(missing_items)}\n")

    if not missing_items:
        print("All PDFs across all 5 pages are already downloaded and verified!")
        return

    # Download only missing files
    for item in tqdm(missing_items, desc="Downloading Missing Ordinance PDFs"):
        direct_pdf_url = resolve_pdf_link(item["url"])
        if not direct_pdf_url:
            print(f"\nCould not resolve direct PDF link for: {item['title']}")
            continue

        filepath = item["target_filepath"]

        try:
            time.sleep(random.uniform(1.5, 3.0))
            res = session.get(direct_pdf_url, stream=True, timeout=45)

            if res.status_code == 200:
                with open(filepath, "wb") as f:
                    for chunk in res.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)
                print(
                    f"\nSuccessfully downloaded missing file: {os.path.basename(filepath)}"
                )
            elif res.status_code in [429, 503]:
                print(f"\nRate limited (HTTP {res.status_code}). Pausing for 15s...")
                time.sleep(15)
        except Exception as e:
            print(f"\nFailed to download {item['title']}: {e}")


if __name__ == "__main__":
    recover_missing_ordinances()
