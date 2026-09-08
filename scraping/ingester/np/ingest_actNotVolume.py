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

ACTS_DIR = Path("data/raw_pdfs/np/Act_Not_In_Volume")
LOG_FILE = Path("data/acts_ingestion_log.json")

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
# 3. PDF TEXT EXTRACTION & STRUCTURE PARSER
# ==========================================
def clean_act_text(raw_text: str) -> str:
    """Strips Law Commission watermarks, header noise, and page numbers."""
    text = re.sub(
        r"www\.lawcommission\.(?:com\.)?gov\.np", "", raw_text, flags=re.IGNORECASE
    )
    text = re.sub(r"नेपाल\s*कानून\s*आयोग", "", text)
    text = re.sub(r"\n\s*[०-९\d]+\s*\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_full_text(pdf_path: Path) -> str:
    """Scrapes page-by-page text using pdfplumber."""
    pages_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    pages_text.append(txt)
    except Exception as e:
        print(f"\nError reading {pdf_path.name}: {e}")

    return clean_act_text("\n".join(pages_text))


def extract_act_header_metadata(text: str) -> Dict[str, Any]:
    """Extracts Title and Act/Code number."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    title = lines[0] if lines else "अज्ञात ऐन/कानून"

    act_num_match = re.search(
        r"((?:संवत्\s*)?[०-९\d]+\s*सालको\s*(?:ऐन|नियमावली|अध्यादेश|संहिता)\s*नं\.\s*[०-९\d]+)",
        text,
    )

    return {
        "title_np": title,
        "act_number": act_num_match.group(1).strip() if act_num_match else None,
    }


def parse_act_components(text: str) -> List[Dict[str, Any]]:
    """Breaks text into flat sections and preamble while retaining chapter meta."""
    chunks = []

    # Extract Preamble
    preamble_match = re.search(
        r"प्रस्तावना\s*[:\n]\s*(.*?)(?=परिच्छेद|१\.)", text, re.DOTALL
    )
    if preamble_match:
        preamble_text = re.sub(r"\s+", " ", preamble_match.group(1).strip())
        if len(preamble_text) > 15:
            chunks.append(
                {
                    "doc_type": "preamble",
                    "title": "प्रस्तावना",
                    "chapter": None,
                    "content_text": preamble_text,
                }
            )

    # Split document by Chapters (परिच्छेद)
    chapters_raw = re.split(r"(\n\s*परिच्छेद-\s*[०-९\d]+[^\n]*)", text)
    section_pattern = (
        r"(?:^|\n)\s*([०-९\d]+)\.\s*([^\n:]+)\s*[:\n]\s*(.*?)(?=\n\s*[०-९\d]+\.|\Z)"
    )

    if len(chapters_raw) > 1:
        # Document contains explicit Chapters
        for idx in range(1, len(chapters_raw), 2):
            chap_title = re.sub(r"\s+", " ", chapters_raw[idx].strip())
            chap_content = chapters_raw[idx + 1] if idx + 1 < len(chapters_raw) else ""

            sections_found = re.findall(section_pattern, chap_content, re.DOTALL)
            for sec_num_str, sec_title, sec_body in sections_found:
                sec_num = nepali_to_int(sec_num_str)
                chunks.append(
                    {
                        "doc_type": "section",
                        "chapter": chap_title,
                        "section_number": sec_num,
                        "title": sec_title.strip(),
                        "content_text": f"{sec_num_str}. {sec_title.strip()}: "
                        + re.sub(r"\s+", " ", sec_body.strip()),
                    }
                )
    else:
        # Flat Document without explicit Chapters
        sections_found = re.findall(section_pattern, text, re.DOTALL)
        for sec_num_str, sec_title, sec_body in sections_found:
            sec_num = nepali_to_int(sec_num_str)
            chunks.append(
                {
                    "doc_type": "section",
                    "chapter": None,
                    "section_number": sec_num,
                    "title": sec_title.strip(),
                    "content_text": f"{sec_num_str}. {sec_title.strip()}: "
                    + re.sub(r"\s+", " ", sec_body.strip()),
                }
            )

    if not chunks:
        chunks.append({"doc_type": "full_text", "chapter": None, "content_text": text})

    return chunks


# ==========================================
# 4. ACTS & CODES INGESTION PIPELINE
# ==========================================
def ingest_all_acts():
    pdf_files = sorted(list(ACTS_DIR.glob("*.pdf")), key=lambda p: p.stat().st_size)
    completed_files = load_completed_files()

    print(f"Total Act/Code PDFs found: {len(pdf_files)}")
    print(f"Already fully ingested: {len(completed_files)}")

    for file_idx, pdf_path in enumerate(
        tqdm(pdf_files, desc="Processing Legal Documents"), start=0
    ):
        file_name = pdf_path.name
        doc_title = pdf_path.stem.replace("_", " ")

        if file_name in completed_files:
            continue

        purge_partial_file_chunks(file_name)

        raw_text = extract_full_text(pdf_path)
        if not raw_text or len(raw_text) < 50:
            mark_file_completed(file_name, completed_files)
            continue

        header_meta = extract_act_header_metadata(raw_text)
        doc_chunks = parse_act_components(raw_text)

        if not doc_chunks:
            continue

        print(f"\n[{file_name}] Extracted {len(doc_chunks)} chunks. Vectorizing...")

        chunk_texts = [c["content_text"] for c in doc_chunks]
        vectors = model.encode(chunk_texts, batch_size=32, show_progress_bar=True)

        file_points = []
        base_file_id = 600000 + (file_idx * 1000)  # Distinct reserved ID block

        for chunk_idx, (chunk, vector) in enumerate(zip(doc_chunks, vectors)):
            payload = {
                "act_title": doc_title,
                "title_np": header_meta["title_np"],
                "doc_type": chunk["doc_type"],
                "language": "np",
                "category": "Act",
                "file_name": file_name,
                "content_text": chunk["content_text"],
            }

            if header_meta["act_number"]:
                payload["act_number"] = header_meta["act_number"]
            if chunk.get("chapter"):
                payload["chapter"] = chunk["chapter"]
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

    print("\nIngestion completed successfully for all Act/Code files!")


if __name__ == "__main__":
    ingest_all_acts()
