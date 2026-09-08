import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List

import pdfplumber
import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
PDF_PATH = Path("data/raw_pdfs/np/Large_File/भन्सार_महसुल_ऐन,_२०८१.pdf")
LOG_FILE = Path("data/economic_act_checkpoint.json")
COLLECTION_NAME = "nepal_laws"

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

model = SentenceTransformer("BAAI/bge-m3")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=180)

BASE_ID = 9500000  # Reserved ID offset for Economic Act


def load_checkpoint() -> int:
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("last_page", 0)
    return 0


def save_checkpoint(page_num: int):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_page": page_num}, f, indent=2)


def process_tariff_table(
    table: List[List[str | None]], current_heading: str
) -> tuple[List[str], str]:
    """Processes tariff tables with forward-filling for Headings and hierarchical descriptions."""
    processed_rows = []

    for row in table:
        clean_row = [re.sub(r"\s+", " ", (cell or "")).strip() for cell in row]
        if not any(clean_row):
            continue

        # Skip table headers
        if (
            "शीर्षक" in clean_row[0]
            or "उपशीर्षक" in clean_row[1 if len(clean_row) > 1 else 0]
        ):
            continue

        # 4-Column Table (Export Tariff)
        if len(clean_row) == 4:
            heading, subheading, desc, rate = clean_row
            if heading:
                current_heading = heading

            if subheading or desc:
                row_str = f"[शीर्षक: {current_heading}] [उपशीर्षक: {subheading}] | विवरण: {desc} | निकासी दर: {rate}"
                processed_rows.append(row_str)

        # 5-Column Table (Import Tariff)
        elif len(clean_row) >= 5:
            heading, subheading, desc, saarc_rate, other_rate = clean_row[:5]
            if heading:
                current_heading = heading

            if subheading or desc:
                row_str = f"[शीर्षक: {current_heading}] [उपशीर्षक: {subheading}] | विवरण: {desc} | सार्क दर: {saarc_rate} | अन्य दर: {other_rate}"
                processed_rows.append(row_str)

        else:
            processed_rows.append(" | ".join(clean_row))

    return processed_rows, current_heading


def parse_economic_act(batch_pages: int = 10):
    start_page = load_checkpoint()
    file_name = PDF_PATH.name
    doc_title = PDF_PATH.stem.replace("_", " ")

    current_heading = ""  # Persist heading context across pages

    print(f"Starting Economic Act parsing: {file_name} from page {start_page + 1}")

    with pdfplumber.open(PDF_PATH) as pdf:
        total_pages = len(pdf.pages)

        for current_page_idx in range(start_page, total_pages, batch_pages):
            end_page_idx = min(current_page_idx + batch_pages, total_pages)
            page_chunks = []

            for page_num in range(current_page_idx, end_page_idx):
                page = pdf.pages[page_num]
                tables = page.extract_tables()

                page_text_blocks = []

                if tables:
                    for tbl in tables:
                        formatted_rows, current_heading = process_tariff_table(
                            tbl, current_heading
                        )
                        if formatted_rows:
                            page_text_blocks.append("\n".join(formatted_rows))
                else:
                    # Non-tabular legal narrative text
                    raw_text = page.extract_text() or ""
                    clean_txt = re.sub(
                        r"www\.lawcommission\.(?:com\.)?gov\.np",
                        "",
                        raw_text,
                        flags=re.IGNORECASE,
                    )
                    clean_txt = re.sub(r"नेपाल\s*कानून\s*आयोग", "", clean_txt).strip()
                    if clean_txt:
                        page_text_blocks.append(clean_txt)

                combined_content = "\n\n".join(page_text_blocks).strip()

                if len(combined_content) > 20:
                    page_chunks.append(
                        {"page_number": page_num + 1, "content_text": combined_content}
                    )

            if page_chunks:
                texts = [c["content_text"] for c in page_chunks]
                with torch.no_grad():
                    vectors = model.encode(texts, batch_size=8, show_progress_bar=False)

                points = []
                for idx, (chunk, vector) in enumerate(zip(page_chunks, vectors)):
                    point_id = BASE_ID + (chunk["page_number"] * 100) + idx
                    payload = {
                        "act_title": doc_title,
                        "file_name": file_name,
                        "category": "Economic Act / Tariff Schedule",
                        "page_number": chunk["page_number"],
                        "content_text": chunk["content_text"],
                        "doc_type": "tariff_schedule_page",
                    }
                    points.append(
                        PointStruct(
                            id=point_id, vector=vector.tolist(), payload=payload
                        )
                    )

                qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

            save_checkpoint(end_page_idx)
            print(
                f"Successfully processed and indexed pages {current_page_idx + 1} to {end_page_idx}/{total_pages}"
            )
            gc.collect()

    print(f"Indexing complete for {file_name}!")


if __name__ == "__main__":
    parse_economic_act(batch_pages=10)
