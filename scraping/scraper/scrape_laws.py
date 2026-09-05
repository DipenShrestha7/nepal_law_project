import os
import re
import time
import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import Any, cast
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
from tqdm import tqdm

BASE_URL = "https://lawcommission.gov.np"
BASE_STORAGE_DIR = "data/raw_pdfs/np"

MENU_SECTIONS = {
    "Recent_Act": {"url": "/category/2163/", "is_volume_hub": False},
    "Volume_Wise_Act": {"url": "/pages/list-volume-act/", "is_volume_hub": True},
    "Act_Not_In_Volume": {
        "url": "/category/2166/",
        "is_volume_hub": False,
    },
    "Ordinance": {"url": "/category/1809/", "is_volume_hub": False},
    "Rules_And_Regulations": {
        "url": "/pages/list-volume-regulation/",
        "is_volume_hub": False,
    },
    "Formation_Order": {
        "url": "/category/development-committee-formation-order/",
        "is_volume_hub": False,
    },
}


# ==========================================
# ROBUST HTTP SESSION WITH RETRY LOGIC
# ==========================================
def create_robust_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Connection": "keep-alive",
        }
    )

    # Auto-retry on connection resets, gateway errors, and rate limits
    retries = Retry(
        total=5,
        backoff_factor=3,  # Waits 3s, 6s, 12s, 24s between retries
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=cast(Any, retries))
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


session = create_robust_session()


def safe_fetch_page(url: str, max_attempts: int = 3) -> BeautifulSoup | None:
    """Fetches a web page with retry handling for connection resets."""
    for attempt in range(1, max_attempts + 1):
        try:
            time.sleep(random.uniform(1.5, 3.0))
            response = session.get(url, timeout=45)  # Increased timeout
            if response.status_code == 200:
                return BeautifulSoup(response.content, "html.parser")
        except Exception as e:
            print(
                f"\n[Attempt {attempt}/{max_attempts}] Network warning for {url}: {e}"
            )
            time.sleep(attempt * 5)
    return None


def resolve_final_pdf_link(initial_url: str) -> str | None:
    if initial_url.lower().endswith(".pdf"):
        return initial_url

    soup = safe_fetch_page(initial_url)
    if soup:
        pdf_anchor = soup.find("a", href=re.compile(r"\.pdf$", re.IGNORECASE))
        if pdf_anchor:
            href = pdf_anchor.get("href")
            if isinstance(href, str):
                return urljoin(BASE_URL, href)
    return None


def extract_pdfs_from_table_page(page_url: str, category_name: str) -> list[dict]:
    records = []
    soup = safe_fetch_page(page_url)
    if not soup:
        return records

    pdf_anchors = soup.find_all(
        "a", href=re.compile(r"\.pdf$|/detail/|/pages/", re.IGNORECASE)
    )

    for anchor in pdf_anchors:
        href = anchor.get("href")
        if not isinstance(href, str) or not href or href == "#":
            continue

        full_url = urljoin(BASE_URL, href)
        tr_parent = anchor.find_parent("tr")
        if tr_parent:
            cells = tr_parent.find_all("td")
            title = (
                cells[1].get_text(strip=True)
                if len(cells) > 1
                else anchor.get_text(strip=True)
            )
        else:
            title = anchor.get_text(strip=True) or "Untitled_Law"

        records.append(
            {"title": title, "initial_url": full_url, "category": category_name}
        )

    return records


def download_file_with_resume(url: str, filepath: str):
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        return  # Skip already downloaded valid files

    for attempt in range(1, 4):
        try:
            time.sleep(random.uniform(2.0, 4.0))  # Politeness delay
            res = session.get(url, stream=True, timeout=60)

            if res.status_code == 200:
                with open(filepath, "wb") as f:
                    for chunk in res.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)
                return
            elif res.status_code in [429, 503]:
                time.sleep(attempt * 10)
        except Exception as e:
            if attempt == 3:
                print(f"\nFailed to download {os.path.basename(filepath)}: {e}")
            time.sleep(5)


def run_resilient_scraper():
    for category_name, config in MENU_SECTIONS.items():
        category_dir = os.path.join(BASE_STORAGE_DIR, category_name)
        os.makedirs(category_dir, exist_ok=True)
        main_url = urljoin(BASE_URL, config["url"])

        print(f"\nProcessing Category: [{category_name}]")
        records_to_download = []

        if config["is_volume_hub"]:
            soup = safe_fetch_page(main_url)
            if soup:
                volume_anchors = soup.find_all(
                    "a", href=re.compile(r"/category/|/pages/")
                )
                sub_page_urls = set()
                for v_link in volume_anchors:
                    href = v_link.get("href")
                    if isinstance(href, str) and (
                        "volume" in href.lower() or "खण्ड" in v_link.text
                    ):
                        sub_page_urls.add(urljoin(BASE_URL, href))

                for sub_url in sub_page_urls:
                    found = extract_pdfs_from_table_page(sub_url, category_name)
                    records_to_download.extend(found)
        else:
            records_to_download = extract_pdfs_from_table_page(main_url, category_name)

        print(f"Queued {len(records_to_download)} items in [{category_name}]")

        for record in tqdm(records_to_download, desc=f"Downloading {category_name}"):
            # 1. Construct local filepath first
            clean_title = (
                re.sub(r'[\\/*?:"<>|]', "", record["title"]).strip().replace(" ", "_")
            )
            filename = f"{clean_title[:90]}.pdf"
            filepath = os.path.join(category_dir, filename)

            # 2. Check local disk space first (0 requests sent to web server)
            if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
                continue  # Moves instantly to the next item without web request

            # 3. Only resolve PDF link and download if file is missing locally
            direct_pdf_url = resolve_final_pdf_link(record["initial_url"])
            if not direct_pdf_url:
                continue

            download_file_with_resume(direct_pdf_url, filepath)


if __name__ == "__main__":
    run_resilient_scraper()
