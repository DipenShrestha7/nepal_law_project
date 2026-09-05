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
# 1. RULES & REGULATIONS VOLUME CONFIG
# ==========================================
HUB_URL = "https://lawcommission.gov.np/pages/list-volume-regulation/"
BASE_URL = "https://lawcommission.gov.np"
BASE_STORAGE_DIR = "data/raw_pdfs/np/Rules_And_Regulations"

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

    # Cast retries to Any to resolve Pyrefly/Pyright static stub errors
    adapter = HTTPAdapter(max_retries=cast(Any, retries))
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


session = get_session()


def fetch_page_soup(url: str) -> tuple[BeautifulSoup | None, int]:
    """Fetches web page HTML and returns (soup, http_status_code)."""
    try:
        time.sleep(random.uniform(1.0, 2.0))
        res = session.get(url, timeout=30)

        if res.status_code == 404:
            return None, 404

        if res.status_code == 200:
            soup = BeautifulSoup(res.content, "html.parser")
            if "404" in soup.get_text() and "Not Found" in soup.get_text():
                return None, 404
            return soup, 200

    except Exception as e:
        print(f"Error fetching {url}: {e}")

    return None, 500


def resolve_pdf_link(detail_url: str) -> str | None:
    if detail_url.lower().endswith(".pdf"):
        return detail_url

    soup, _ = fetch_page_soup(detail_url)
    if soup:
        pdf_anchor = soup.find("a", href=re.compile(r"\.pdf$", re.IGNORECASE))
        if pdf_anchor:
            href = pdf_anchor.get("href")
            # Explicit type narrowing for Pyrefly
            if isinstance(href, str) and href and href != "#":
                return urljoin(BASE_URL, href)
    return None


# ==========================================
# 3. STAGE 1: HARVEST VOLUME LINKS FROM HUB
# ==========================================
def harvest_regulation_volumes() -> list[dict]:
    print(f"Fetching Regulation Volumes Hub: {HUB_URL}")
    soup, status = fetch_page_soup(HUB_URL)

    if not soup or status != 200:
        print("Failed to load Regulation Volumes Hub page.")
        return []

    table_rows = soup.find_all("tr")
    volume_targets = []

    # Skip row 0 (Header row containing 'Volume' and 'Name of Volume')
    for tr in table_rows[1:]:
        # Skip header rows containing <th>
        if tr.find("th"):
            continue

        cells = tr.find_all("td")
        anchor = tr.find(
            "a", href=re.compile(r"/category/|/pages/|/content/", re.IGNORECASE)
        )

        if not anchor:
            continue

        href = anchor.get("href")
        if not isinstance(href, str) or not href or href == "#":
            continue

        full_url = urljoin(BASE_URL, href)

        # Extract Volume Num (Col 1) and Volume Name (Col 2)
        vol_num = cells[0].get_text(strip=True) if len(cells) > 0 else ""
        vol_name = (
            cells[1].get_text(strip=True)
            if len(cells) > 1
            else anchor.get_text(strip=True)
        )

        full_title = f"{vol_num}_{vol_name}".strip("_ ")
        clean_folder = re.sub(r'[\\/*?:"<>|()]', "", full_title).replace(" ", "_")

        volume_targets.append({"folder_name": clean_folder, "url": full_url})

    print(f"Successfully harvested {len(volume_targets)} Regulation Volume links.\n")
    return volume_targets


# ==========================================
# 4. STAGE 2 & 3: CRAWL & DOWNLOAD VOLUMES
# ==========================================
def scrape_rules_and_regulations():
    volumes = harvest_regulation_volumes()

    for vol in volumes:
        vol_dir = os.path.join(BASE_STORAGE_DIR, vol["folder_name"])
        os.makedirs(vol_dir, exist_ok=True)

        print(f"==========================================")
        print(f" Processing Volume: {vol['folder_name']}")
        print(f" Storage Directory: {vol_dir}")
        print(f"==========================================")

        page = 1
        volume_items = []

        while True:
            page_url = f"{vol['url']}?page={page}"
            soup, status_code = fetch_page_soup(page_url)

            if status_code == 404 or not soup:
                print(f"  Page {page} returned {status_code}. Reached end of volume.")
                break

            table_rows = soup.find_all("tr")
            rows_found = 0

            for tr in table_rows:
                # Ignore table headers
                if tr.find("th"):
                    continue

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
                    title = f"Regulation_Page{page}"

                volume_items.append({"title": title, "url": full_url, "page": page})
                rows_found += 1

            if rows_found == 0:
                print(f"  No entries on Page {page}. Ending volume iteration.")
                break

            print(f"  Found {rows_found} items on Page {page}.")
            page += 1

        if not volume_items:
            print(f"No PDF entries found for {vol['folder_name']}.\n")
            continue

        print(f"Downloading {len(volume_items)} PDFs for {vol['folder_name']}...")

        for item in tqdm(volume_items, desc=f"Downloading {vol['folder_name'][:30]}"):
            direct_pdf_url = resolve_pdf_link(item["url"])
            if not direct_pdf_url:
                continue

            clean_title = (
                re.sub(r'[\\/*?:"<>|]', "", item["title"]).strip().replace(" ", "_")
            )
            filename = f"{clean_title[:90]}.pdf"
            filepath = os.path.join(vol_dir, filename)

            # Skip already downloaded files locally
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

        print(f"Completed {vol['folder_name']}.\n")


if __name__ == "__main__":
    scrape_rules_and_regulations()
