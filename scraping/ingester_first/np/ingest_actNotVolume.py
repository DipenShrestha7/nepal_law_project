import gc
import json
import os
import re
import sys
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Dict, List

# Force UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 1. CRITICAL THREAD LOCKS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONMALLOC"] = "malloc"

import pdfplumber
import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

torch.set_num_threads(1)
load_dotenv()

# CONFIGURATION
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "nepal_laws"

CODES_DIR = Path("data/raw_pdfs/np/Act_Not_In_Volume")
LOG_FILE = Path("data/codes_ingestion_log.json")

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


def load_completed_files() -> set[str]:
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def mark_file_completed(file_name: str, completed_set: set[str]):
    completed_set.add(file_name)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(list(completed_set), f, ensure_ascii=False, indent=2)


def purge_partial_file_chunks(file_name: str):
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


def clean_code_text(raw_text: str) -> str:
    """Cleans page footers, watermarks, and normalizes Devanagari Unicode."""
    text = unicodedata.normalize("NFC", raw_text)
    text = re.sub(
        r"www\.lawcommission\.(?:com\.)?gov\.np", "", text, flags=re.IGNORECASE
    )
    text = re.sub(r"नेपाल\s*कानून\s*आयोग", "", text)
    # Remove standalone page numbers
    text = re.sub(r"\n\s*[०-९\d]+\s*\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_code_sections(text: str) -> List[Dict[str, Any]]:
    """Parses Muluki Codes by Parts (भाग), Chapters (परिच्छेद), and Sections (दफा)."""
    chunks = []

    # 1. Extract Preamble
    preamble_match = re.search(
        r"प्रस्तावना\s*[:\n]\s*(.*?)(?=भाग|परिच्छेद|१\.)", text, re.DOTALL
    )
    if preamble_match:
        preamble_text = re.sub(r"\s+", " ", preamble_match.group(1).strip())
        if len(preamble_text) > 15:
            chunks.append(
                {
                    "doc_type": "preamble",
                    "title": "प्रस्तावना",
                    "part": None,
                    "chapter": None,
                    "section_number": None,
                    "content_text": f"प्रस्तावना: {preamble_text}",
                }
            )

    # 2. Section Pattern: e.g. "४९. सार्वभौमसत्ता... :"
    section_pattern = (
        r"(?:^|\n)\s*([०-९\d]+)\.\s*([^\n:]+)\s*[:\n]\s*(.*?)(?=\n\s*[०-९\d]+\.|\Z)"
    )

    # Track active Part and Chapter context
    current_part = None
    current_chapter = None

    # Split by Parts/Chapters first to preserve hierarchy
    parts = re.split(r"(\n\s*भाग\s*-\s*[०-९\d]+[^\n]*)", text)

    for i in range(len(parts)):
        segment = parts[i]
        if re.match(r"^\n\s*भाग\s*-\s*[०-९\d]+", segment):
            current_part = re.sub(r"\s+", " ", segment.strip())
            continue

        # Split segment by Chapters
        chapters = re.split(r"(\n\s*परिच्छेद\s*-\s*[०-९\d]+[^\n]*)", segment)
        for j in range(len(chapters)):
            sub_segment = chapters[j]
            if re.match(r"^\n\s*परिच्छेद\s*-\s*[०-९\d]+", sub_segment):
                current_chapter = re.sub(r"\s+", " ", sub_segment.strip())
                continue

            # Extract individual Sections (दफा) inside current Part/Chapter context
            sections_found = re.findall(section_pattern, sub_segment, re.DOTALL)
            for sec_num_str, sec_title, sec_body in sections_found:
                sec_num = nepali_to_int(sec_num_str)
                clean_body = re.sub(r"\s+", " ", sec_body.strip())

                chunks.append(
                    {
                        "doc_type": "section",
                        "part": current_part,
                        "chapter": current_chapter,
                        "section_number": sec_num,
                        "title": sec_title.strip(),
                        "content_text": f"दफा {sec_num_str}. {sec_title.strip()}: {clean_body}",
                    }
                )

    # Fallback to sliding window if section extraction yields < 10 chunks on large PDFs
    if len(chunks) < 10:
        print("Fallback: Using sliding window chunker for unstructured pages...")
        full_clean = re.sub(r"\s+", " ", text)
        for i in range(0, len(full_clean), 1500):
            block = full_clean[i : i + 1500]
            if len(block.strip()) > 50:
                chunks.append(
                    {
                        "doc_type": "full_text_chunk",
                        "part": None,
                        "chapter": None,
                        "section_number": None,
                        "content_text": block.strip(),
                    }
                )

    return chunks


def ingest_all_large_codes():
    pdf_files = sorted(list(CODES_DIR.glob("*.pdf")), key=lambda p: p.stat().st_size)
    completed_files = load_completed_files()

    print(f"Total Code PDFs found in {CODES_DIR}: {len(pdf_files)}")
    print(f"Already completed: {len(completed_files)}")

    for pdf_path in tqdm(pdf_files, desc="Processing Muluki Codes"):
        file_name = pdf_path.name

        # Skip the 2-page act (handled by script 1)
        if "२०६३" in file_name and "जगाउने" in file_name:
            continue

        if file_name in completed_files:
            continue

        doc_title = pdf_path.stem.replace("_", " ")
        print(f"\nProcessing Large Code: {file_name}...")

        purge_partial_file_chunks(file_name)

        pages_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    pages_text.append(txt)

        raw_text = clean_code_text("\n".join(pages_text))
        if not raw_text or len(raw_text) < 100:
            mark_file_completed(file_name, completed_files)
            continue

        doc_chunks = parse_code_sections(raw_text)
        print(f"Extracted {len(doc_chunks)} section chunks. Vectorizing...")

        chunk_texts = [c["content_text"][:2500] for c in doc_chunks]

        with torch.no_grad():
            vectors = model.encode(chunk_texts, batch_size=8, show_progress_bar=False)

        file_points = []
        for chunk_idx, (chunk, vector) in enumerate(zip(doc_chunks, vectors)):
            payload = {
                "act_title": doc_title,
                "title_np": doc_title,
                "doc_type": chunk["doc_type"],
                "language": "np",
                "category": "Acts and Codes",
                "file_name": file_name,
                "content_text": chunk["content_text"],
            }

            if chunk.get("part"):
                payload["part"] = chunk["part"]
            if chunk.get("chapter"):
                payload["chapter"] = chunk["chapter"]
            if chunk.get("section_number"):
                payload["section_number"] = chunk["section_number"]
            if chunk.get("title"):
                payload["title"] = chunk["title"]

            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_name}_{chunk_idx}"))
            file_points.append(
                PointStruct(id=point_id, vector=vector.tolist(), payload=payload)
            )

        if file_points:
            safe_upsert_with_retry(file_points, batch_size=20)

        mark_file_completed(file_name, completed_files)

        del raw_text, doc_chunks, chunk_texts, vectors, file_points
        gc.collect()

    print("\nLarge Codes Ingestion Completed Successfully!")


if __name__ == "__main__":
    ingest_all_large_codes()
