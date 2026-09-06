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

ORDINANCE_DIR = Path("data/raw_pdfs/np/Ordinance")
LOG_FILE = Path("data/ordinance_ingestion_log.json")

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
# 3. TEXT PARSING & CLEANING
# ==========================================
def clean_pdf_text(raw_text: str) -> str:
    """Strips watermarks, header URLs, and isolated page numbers."""
    text = re.sub(r"www\.lawcommission\.gov\.np", "", raw_text, flags=re.IGNORECASE)
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

    return clean_pdf_text("\n".join(pages_text))


def parse_ordinance_components(text: str) -> List[Dict[str, Any]]:
    chunks = []
    schedule_split = re.split(r"(?:^|\n)\s*(अनुसूची[\s\:\-]+)", text, maxsplit=1)
    main_body = schedule_split[0]
    schedule_body = (
        "".join(schedule_split[1:]).strip() if len(schedule_split) > 1 else None
    )

    sec_pattern = re.compile(
        r"(?:^|\n)\s*([०-९\d]+\.|\bदफा\s+[०-९\d]+[\.\:\s]*)", re.MULTILINE
    )
    matches = list(sec_pattern.finditer(main_body))

    if matches:
        preamble_text = main_body[: matches[0].start()].strip()
        if len(preamble_text) > 20:
            chunks.append(
                {
                    "doc_type": "preamble",
                    "title": "प्रस्तावना",
                    "content_text": preamble_text,
                }
            )

        for i, match in enumerate(matches):
            sec_label = match.group(1).strip()
            sec_num = nepali_to_int(sec_label)
            start_idx = match.start()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(main_body)

            chunk_data = {
                "doc_type": "section",
                "content_text": main_body[start_idx:end_idx].strip(),
            }
            if sec_num is not None:
                chunk_data["section_number"] = sec_num
                chunk_data["title"] = f"दफा {sec_num}"
            else:
                chunk_data["title"] = sec_label

            chunks.append(chunk_data)
    else:
        chunks.append({"doc_type": "full_text", "content_text": main_body})

    if schedule_body:
        chunks.append(
            {
                "doc_type": "schedule",
                "schedule_number": 1,
                "title": "अनुसूची",
                "content_text": schedule_body,
            }
        )

    return chunks


# ==========================================
# 4. UNIVERSAL INGESTION PIPELINE
# ==========================================
def ingest_all_ordinances():
    # Sort files deterministically so index mapping stays identical across runs
    pdf_files = sorted(list(ORDINANCE_DIR.glob("*.pdf")))[9:]
    completed_files = load_completed_files()

    print(f"Total PDFs found: {len(pdf_files)}")
    print(f"Already fully ingested: {len(completed_files)}")

    for file_idx, pdf_path in enumerate(
        tqdm(pdf_files, desc="Processing PDFs"), start=9
    ):
        file_name = pdf_path.name
        act_title = pdf_path.stem

        # Skip files marked 100% complete in state log
        if file_name in completed_files:
            continue

        # Purge partial points from previous interrupted attempts
        purge_partial_file_chunks(file_name)

        raw_text = extract_full_text(pdf_path)
        if not raw_text or len(raw_text) < 50:
            mark_file_completed(file_name, completed_files)
            continue

        doc_chunks = parse_ordinance_components(raw_text)
        file_points = []

        # Formula guarantees reserved ID range per file: 300,000 + (file_index * 1000) + chunk_index
        base_file_id = 300000 + (file_idx * 1000)

        for chunk_idx, chunk in enumerate(doc_chunks):
            payload = {
                "act_title": act_title,
                "doc_type": chunk["doc_type"],
                "language": "np",
                "category": "Ordinance",
                "file_name": file_name,
                "content_text": chunk["content_text"],
            }

            if "title" in chunk:
                payload["title"] = chunk["title"]
            if "section_number" in chunk:
                payload["section_number"] = chunk["section_number"]
            if "schedule_number" in chunk:
                payload["schedule_number"] = chunk["schedule_number"]

            vector = model.encode(
                chunk["content_text"], show_progress_bar=False
            ).tolist()

            file_points.append(
                PointStruct(id=base_file_id + chunk_idx, vector=vector, payload=payload)
            )

        # Upload with retries and small batch sizes (20 points/batch)
        if file_points:
            safe_upsert_with_retry(file_points, batch_size=20)

        # Mark 100% complete only after successful database commit
        mark_file_completed(file_name, completed_files)

    print("\nIngestion completed successfully for all files!")


if __name__ == "__main__":
    ingest_all_ordinances()
