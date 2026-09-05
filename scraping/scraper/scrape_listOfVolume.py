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
# 1. 17 VOLUMES CONFIGURATION
# ==========================================
# Category IDs corresponding to Volumes 1 through 17
VOLUME_IDS = list(range(1762, 1770)) + list(range(1783, 1792))
BASE_URL = "https://lawcommission.gov.np"
BASE_STORAGE_DIR = "data/raw_pdfs/np/Volume_Wise_Act"

os.makedirs(BASE_STORAGE_DIR, exist_ok=True)


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

    # Cast retries to Any to pass Pyrefly/Pyright static checking
    adapter = HTTPAdapter(max_retries=cast(Any, retries))
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


session = get_session()


def fetch_page_soup(url: str) -> tuple[BeautifulSoup | None, int]:
    """Fetches a page and returns (BeautifulSoup object, http_status_code)."""
    try:
        time.sleep(random.uniform(1.0, 2.0))
        res = session.get(url, timeout=30)

        if res.status_code == 404:
            return None, 404

        if res.status_code == 200:
            soup = BeautifulSoup(res.content, "html.parser")
            # Check for soft 404 page render
            if "404" in soup.get_text() and "Not Found" in soup.get_text():
                return None, 404
            return soup, 200

    except Exception as e:
        print(f"Error fetching {url}: {e}")

    return None, 500


def resolve_pdf_link(detail_url: str) -> str | None:
    if detail_url.lower().endswith(".pdf"):
        return detail_url

    soup, status = fetch_page_soup(detail_url)
    if soup:
        pdf_anchor = soup.find("a", href=re.compile(r"\.pdf$", re.IGNORECASE))
        if pdf_anchor:
            href = pdf_anchor.get("href")
            # Explicit type narrowing for Pyrefly
            if isinstance(href, str) and href and href != "#":
                return urljoin(BASE_URL, href)
    return None


def extract_volume_heading(soup: BeautifulSoup, vol_id: int) -> str:
    """Extracts the official Volume title from page header (e.g. 'Volume 1: Constitutional Body...')."""
    heading = soup.find(
        re.compile(r"^h[1-4]$"), text=re.compile(r"Volume", re.IGNORECASE)
    )
    if not heading:
        heading = soup.find(
            lambda tag: tag.name in ["h1", "h2", "h3", "h4", "div"]
            and "Volume" in tag.text
        )

    if heading:
        raw_text = heading.get_text(strip=True)
        clean_text = re.sub(r'[\\/*?:"<>|]', "", raw_text).strip().replace(" ", "_")
        if clean_text:
            return clean_text[:80]

    return f"Volume_ID_{vol_id}"


# ==========================================
# 3. SCRAPE ALL VOLUMES UNTIL 404
# ==========================================
def scrape_all_volumes():
    print(f"Starting Volume Scraper across {len(VOLUME_IDS)} Category IDs...\n")

    for vol_id in VOLUME_IDS:
        category_url = f"{BASE_URL}/category/{vol_id}/"
        page = 1
        vol_folder_path = ""
        volume_items = []

        print(f"==========================================")
        print(f" Checking Category ID: {vol_id}")
        print(f"==========================================")

        while True:
            page_url = f"{category_url}?page={page}"
            soup, status_code = fetch_page_soup(page_url)

            # Stop condition: 404 Page Not Found or Server error
            if status_code == 404 or not soup:
                print(f"  Reached end of Volume (Page {page} returned {status_code}).")
                break

            # Set up volume directory based on title from Page 1
            if page == 1:
                vol_title = extract_volume_heading(soup, vol_id)
                vol_folder_path = os.path.join(BASE_STORAGE_DIR, vol_title)
                os.makedirs(vol_folder_path, exist_ok=True)
                print(f" Target Folder: {vol_folder_path}")

            table_rows = soup.find_all("tr")
            rows_found = 0

            for tr in table_rows:
                anchor = tr.find(
                    "a",
                    href=re.compile(
                        r"\.pdf$|/detail/|/pages/|/content/", re.IGNORECASE
                    ),
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
                    title = f"Volume_{vol_id}_Law"

                volume_items.append({"title": title, "url": full_url, "page": page})
                rows_found += 1

            if rows_found == 0:
                print(
                    f"  No entries on Page {page}. Ending pagination for Category {vol_id}."
                )
                break

            print(f"  Found {rows_found} items on Page {page}.")
            page += 1

        if not volume_items:
            print(f"No documents found for Category ID {vol_id}.\n")
            continue

        print(f"\nDownloading {len(volume_items)} PDFs for Category ID {vol_id}...")

        # Sequential file downloader for current volume
        for item in tqdm(volume_items, desc=f"Downloading Vol ID {vol_id}"):
            direct_pdf_url = resolve_pdf_link(item["url"])
            if not direct_pdf_url:
                continue

            clean_title = (
                re.sub(r'[\\/*?:"<>|]', "", item["title"]).strip().replace(" ", "_")
            )
            filename = f"{clean_title[:90]}.pdf"
            filepath = os.path.join(vol_folder_path, filename)

            if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
                continue

            try:
                time.sleep(random.uniform(1.2, 2.5))
                res = session.get(direct_pdf_url, stream=True, timeout=45)

                if res.status_code == 200:
                    with open(filepath, "wb") as f:
                        for chunk in res.iter_content(chunk_size=16384):
                            if chunk:
                                f.write(chunk)
                elif res.status_code in [429, 503]:
                    time.sleep(15)
            except Exception as e:
                print(f"\nFailed downloading {filename}: {e}")

        print(f"Completed Volume Category ID {vol_id}.\n")


if __name__ == "__main__":
    scrape_all_volumes()
