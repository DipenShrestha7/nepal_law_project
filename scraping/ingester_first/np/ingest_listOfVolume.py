import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pdfplumber
import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Force UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Limit OpenMP/MKL thread pools to prevent virtual memory bloat
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONMALLOC"] = "malloc"

torch.set_num_threads(1)
load_dotenv()

# ==========================================
# 1. CONFIGURATION
# ==========================================
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "nepal_laws"

ACTS_DIR = Path("data/raw_pdfs/np/Volume_Wise_Act")
LOG_FILE = Path("data/volume_wise_act_ingestion_log.json")

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
    """Persists file completion state to disk."""
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
    """Uploads points in small batches with retry backoff."""
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
# 3. TEXT PARSING & ACT STRUCTURE CLEANING
# ==========================================
def clean_act_text(raw_text: str) -> str:
    """Strips Law Commission watermarks, noise symbols, and floating page numbers."""
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
    """Extracts Act Title, Authentication/Promulgation Date, Act Number, and Amendments."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    title = lines[0] if lines else "अज्ञात ऐन"

    # Capture Authentication or Lalmohar Publication Date
    auth_date_match = re.search(
        r"(?:प्रमाणीकरण(?:\s*र\s*प्रकाशन)?|लालमोहर\s*र\s*प्रकाशन)\s*मिति\s*\n?\s*([०-९\d।\/\-]+)",
        text,
    )

    # Capture Act Number (e.g., संवत् २०७४ सालको ऐन नं. १९)
    act_num_match = re.search(
        r"([०-९\d]+\s*सालको\s*ऐन\s*नं\.\s*[०-९\d]+)",
        text,
    )

    # Capture Amendment details
    amendment_matches = re.findall(
        r"([^\n]+(?:संशोधन गर्ने ऐन|संशोधन)\s*,?\s*[०-९\d]+(?:\s*[०-९\d।\/\-]+)?)",
        text,
    )

    return {
        "title_np": title,
        "authentication_date": (
            auth_date_match.group(1).strip() if auth_date_match else None
        ),
        "act_number": act_num_match.group(1).strip() if act_num_match else None,
        "amendments": (
            [a.strip() for a in amendment_matches] if amendment_matches else []
        ),
    }


def parse_act_components(text: str) -> List[Dict[str, Any]]:
    """Splits Act documents into Preamble, Chapters (परिच्छेद), Sections (दफा), and Schedules (अनुसूची)."""
    chunks = []

    # 1. Extract Preamble (प्रस्तावना)
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
                    "content_text": f"प्रस्तावना: {preamble_text}",
                }
            )

    # Separate Schedules (अनुसूची) if attached at the end of the Act
    main_text = text
    schedules_text = ""
    sched_split = re.split(
        r"(\n\s*\*?\s*अनुसूची\s*-\s*[०-९\d]+|\n\s*\*?\s*अनुसूची\s*[:\n])", text, maxsplit=1
    )
    if len(sched_split) > 1:
        main_text = sched_split[0]
        schedules_text = sched_split[1] + sched_split[2]

    # Section parsing regex: matches "1. Title: Content" or "१. संक्षिप्त नाम..."
    section_pattern = (
        r"(?:^|\n)\s*([०-९\d]+)\.\s*([^\n:]+)\s*[:\n]\s*(.*?)(?=\n\s*[०-९\d]+\.|\Z)"
    )

    # 2. Split Document by Chapters (परिच्छेद)
    chapters_raw = re.split(r"(\n\s*परिच्छेद\s*-\s*[०-९\d]+[^\n]*)", main_text)

    if len(chapters_raw) > 1:
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
        # Flat structure without explicit chapters
        sections_found = re.findall(section_pattern, main_text, re.DOTALL)
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

    # 3. Parse Schedules (अनुसूची) if extracted
    if schedules_text:
        schedules_raw = re.split(
            r"(\n\s*\*?\s*अनुसूची\s*(?:-\s*[०-९\d]+|[:\n]))", schedules_text
        )
        if len(schedules_raw) > 1:
            for idx in range(1, len(schedules_raw), 2):
                sched_header = re.sub(r"\s+", " ", schedules_raw[idx].strip())
                sched_body = (
                    schedules_raw[idx + 1] if idx + 1 < len(schedules_raw) else ""
                )
                chunks.append(
                    {
                        "doc_type": "schedule",
                        "chapter": None,
                        "title": sched_header,
                        "content_text": f"{sched_header}\n"
                        + re.sub(r"\s+", " ", sched_body.strip()),
                    }
                )

    if not chunks:
        chunks.append({"doc_type": "full_text", "chapter": None, "content_text": text})

    return chunks


# ==========================================
# 4. RECURSIVE VOLUME-WISE INGESTION PIPELINE
# ==========================================
def ingest_all_volume_wise_acts():
    # Traverses Volume_Wise_Act directory recursively (Volume_ID_1762, Volume_ID_1763, etc.)
    pdf_files = sorted(list(ACTS_DIR.rglob("*.pdf")), key=lambda p: p.stat().st_size)
    completed_files = load_completed_files()

    print(f"Total Volume-Wise Act PDFs found: {len(pdf_files)}")
    print(f"Already fully ingested: {len(completed_files)}")

    for file_idx, pdf_path in enumerate(
        tqdm(pdf_files, desc="Processing Volume-Wise Acts"), start=0
    ):
        file_name = pdf_path.name
        doc_title = pdf_path.stem.replace("_", " ")
        volume_folder = pdf_path.parent.name  # Captures folder like "Volume_ID_1762"

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

        print(
            f"\n[{volume_folder} -> {file_name}] Extracted {len(doc_chunks)} chunks. Vectorizing..."
        )

        chunk_texts = [c["content_text"] for c in doc_chunks]
        with torch.no_grad():
            vectors = model.encode(chunk_texts, batch_size=16, show_progress_bar=True)

        file_points = []
        # Distinct reserved ID block for Volume Wise Acts: 800,000 + (file_index * 1000)
        base_file_id = 800000 + (file_idx * 1000)

        for chunk_idx, (chunk, vector) in enumerate(zip(doc_chunks, vectors)):
            payload = {
                "act_title": doc_title,
                "title_np": header_meta["title_np"],
                "doc_type": chunk["doc_type"],
                "language": "np",
                "category": "Act",
                "volume_folder": volume_folder,
                "file_name": file_name,
                "content_text": chunk["content_text"],
            }

            if header_meta["authentication_date"]:
                payload["authentication_date"] = header_meta["authentication_date"]
            if header_meta["act_number"]:
                payload["act_number"] = header_meta["act_number"]
            if header_meta["amendments"]:
                payload["amendments"] = header_meta["amendments"]

            if chunk.get("chapter"):
                payload["chapter"] = chunk["chapter"]
            if "title" in chunk:
                payload["title"] = chunk["title"]
            if "section_number" in chunk:
                payload["section_number"] = chunk["section_number"]

            file_points.append(
                PointStruct(
                    id=base_file_id + chunk_idx,
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        if file_points:
            safe_upsert_with_retry(file_points, batch_size=20)

        mark_file_completed(file_name, completed_files)

        # Free allocations and trigger garbage collection per PDF file
        del raw_text, doc_chunks, chunk_texts, vectors, file_points
        gc.collect()

    print("\nIngestion completed successfully for all Volume-Wise Act files!")


if __name__ == "__main__":
    ingest_all_volume_wise_acts()
