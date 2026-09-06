import os
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, List
import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from tqdm import tqdm

load_dotenv()

# ==========================================
# 1. CONFIGURATION
# ==========================================
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "nepal_laws"

ORDERS_DIR = Path("data/raw_pdfs/np/Formation_Order")
LOG_FILE = Path("data/formation_order_ingestion_log.json")

# Model and client initialization with 180s HTTP timeout
model = SentenceTransformer("BAAI/bge-m3")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=180)

NEPALI_DIGITS = {
    "०": "0",
    "१": "1",
    "२": "2",
    "३": "3",
    "४": "4",
    "५": "5",
    "६": "6",
    "७": "7",
    "८": "8",
    "९": "9",
}


def nepali_to_int(text: str) -> int | None:
    digits = "".join(
        [
            NEPALI_DIGITS.get(ch, ch)
            for ch in text
            if ch in NEPALI_DIGITS or ch.isdigit()
        ]
    )
    return int(digits) if digits else None


# ==========================================
# 2. STATE LOGGING & CLEANUP HELPERS
# ==========================================
def load_completed_files() -> set[str]:
    """Loads set of fully processed PDF file names."""
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def mark_file_completed(file_name: str, completed_set: set[str]):
    """Persists file completion to disk."""
    completed_set.add(file_name)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(list(completed_set), f, ensure_ascii=False, indent=2)


def purge_partial_file_chunks(file_name: str):
    """Deletes existing or partial points for a given file before ingesting."""
    try:
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(key="file_name", match=MatchValue(value=file_name))
                ]
            ),
        )
    except Exception as e:
        print(f"\nWarning during pre-purge of {file_name}: {e}")


def safe_upsert_with_retry(
    points: List[PointStruct], batch_size: int = 20, max_retries: int = 3
):
    """Uploads points in small batches with exponential backoff on network failures."""
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        for attempt in range(1, max_retries + 1):
            try:
                qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
                break
            except Exception as e:
                if attempt == max_retries:
                    raise e
                time.sleep(2 * attempt)


# ==========================================
# 3. TEXT PARSING & ORDER STRUCTURE CLEANING
# ==========================================
def clean_order_text(raw_text: str) -> str:
    """Strips Law Commission watermarks, noise symbols, and standalone page numbers."""
    text = re.sub(
        r"www\.lawcommission\.(?:com\.)?gov\.np", "", raw_text, flags=re.IGNORECASE
    )
    text = re.sub(r"नेपाल\s*कानून\s*आयोग", "", text)
    text = re.sub(r"\n\s*[०-९\d]+\s*\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_full_text(pdf_path: Path) -> str:
    pages_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    pages_text.append(txt)
    except Exception as e:
        print(f"\nError reading {pdf_path.name}: {e}")

    return clean_order_text("\n".join(pages_text))


def extract_order_header_metadata(text: str) -> Dict[str, Any]:
    """Extracts Gazette date, Enabling Act, and Amendment details."""
    gazette_match = re.search(r"राजपत्रमा प्रकाशित मिति\s*\n?\s*([०-९\d।\/\-]+)", text)
    enabling_act_match = re.search(
        r"([^\n]+ऐन,\s*[०-९\d]+)\s*को\s*दफा\s*[०-९\d]+\s*ले\s*दिएको\s*अधिकार", text
    )
    amendment_matches = re.findall(
        r"([^\n]+\(पहिलो|दोस्रो|तेस्रो|चौथो\s*संशोधन\)\s*आदेश,\s*[०-९\d]+)", text
    )

    return {
        "gazette_date": gazette_match.group(1).strip() if gazette_match else None,
        "enabling_act": (
            enabling_act_match.group(1).strip()
            if enabling_act_match
            else "विकास समिति ऐन, २०१३"
        ),
        "amendments": (
            [a[0].strip() for a in amendment_matches] if amendment_matches else []
        ),
    }


def parse_order_components(text: str) -> List[Dict[str, Any]]:
    chunks = []

    # Section marker regex for Formation Orders
    sec_pattern = re.compile(
        r"(?:^|\n)\s*([\*\#x\•]*\s*[०-९\d]+\.\s*[^:\n]+?\s*[:\n]|\bदफा\s+[०-९\d]+[\.\:\s]*)",
        re.MULTILINE,
    )

    sec_matches = list(sec_pattern.finditer(text))

    if sec_matches:
        # Preamble extraction
        preamble_text = text[: sec_matches[0].start()].strip()
        if len(preamble_text) > 20:
            chunks.append(
                {
                    "doc_type": "preamble",
                    "title": "प्रस्तावना / संक्षिप्त परिचय",
                    "content_text": preamble_text,
                }
            )

        for i, match in enumerate(sec_matches):
            start_idx = match.start()
            end_idx = (
                sec_matches[i + 1].start() if i + 1 < len(sec_matches) else len(text)
            )
            sec_text = text[start_idx:end_idx].strip()

            sec_first_line = match.group(1).strip()
            sec_num = nepali_to_int(sec_first_line)

            # Strip clean footnote symbols from section title
            title_clean = re.sub(r"^[\*\#x\•\s]+", "", sec_first_line)
            title_parts = title_clean.split(":", 1)
            sec_title = title_parts[0].strip()

            chunk_data: Dict[str, Any] = {
                "doc_type": "section",
                "title": sec_title,
                "content_text": sec_text,
            }

            if sec_num is not None:
                chunk_data["section_number"] = sec_num

            chunks.append(chunk_data)
    else:
        chunks.append({"doc_type": "full_text", "content_text": text})

    return chunks


# ==========================================
# 4. UNIVERSAL ORDER INGESTION PIPELINE
# ==========================================
def ingest_all_formation_orders():
    # Sort files by size so smaller files ingest first
    pdf_files = sorted(list(ORDERS_DIR.glob("*.pdf")), key=lambda p: p.stat().st_size)
    completed_files = load_completed_files()

    print(f"Total Formation Order PDFs found: {len(pdf_files)}")
    print(f"Already fully ingested: {len(completed_files)}")

    for file_idx, pdf_path in enumerate(
        tqdm(pdf_files, desc="Processing Formation Orders"), start=0
    ):
        file_name = pdf_path.name
        order_title = pdf_path.stem.replace("_", " ")

        if file_name in completed_files:
            continue

        purge_partial_file_chunks(file_name)

        raw_text = extract_full_text(pdf_path)
        if not raw_text or len(raw_text) < 50:
            mark_file_completed(file_name, completed_files)
            continue

        header_meta = extract_order_header_metadata(raw_text)
        doc_chunks = parse_order_components(raw_text)

        if not doc_chunks:
            continue

        print(f"\n[{file_name}] Extracted {len(doc_chunks)} chunks. Vectorizing...")

        # Fast Batch Encoding (32 items per tensor operation)
        chunk_texts = [c["content_text"] for c in doc_chunks]
        vectors = model.encode(chunk_texts, batch_size=32, show_progress_bar=True)

        file_points = []
        # Formula guarantees reserved ID range per file: 500,000 + (file_index * 1000) + chunk_index
        base_file_id = 500000 + (file_idx * 1000)

        for chunk_idx, (chunk, vector) in enumerate(zip(doc_chunks, vectors)):
            payload = {
                "act_title": order_title,
                "doc_type": chunk["doc_type"],
                "language": "np",
                "category": "Formation Order",
                "file_name": file_name,
                "content_text": chunk["content_text"],
                "enabling_act": header_meta["enabling_act"],
            }

            if header_meta["gazette_date"]:
                payload["gazette_date"] = header_meta["gazette_date"]
            if header_meta["amendments"]:
                payload["amendments"] = header_meta["amendments"]

            if "title" in chunk:
                payload["title"] = chunk["title"]
            if "section_number" in chunk:
                payload["section_number"] = chunk["section_number"]

            file_points.append(
                PointStruct(
                    id=base_file_id + chunk_idx, vector=vector.tolist(), payload=payload
                )
            )

        if file_points:
            safe_upsert_with_retry(file_points, batch_size=20)

        mark_file_completed(file_name, completed_files)

    print("\nIngestion completed successfully for all Formation Order files!")


if __name__ == "__main__":
    ingest_all_formation_orders()
