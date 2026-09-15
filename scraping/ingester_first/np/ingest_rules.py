import os
import sys

# Force UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 1. CRITICAL THREAD LOCKS (Must be set BEFORE importing transformers/torch)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONMALLOC"] = "malloc"

import gc
import json
import re
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Dict, List

import pdfplumber
import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

torch.set_num_threads(1)

load_dotenv()

# ==========================================
# 1. CONFIGURATION
# ==========================================
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "nepal_laws"

RULES_DIR = Path("data/raw_pdfs/np/Rules_And_Regulations")
LOG_FILE = Path("data/rules_and_regulations_ingestion_log.json")

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
# 3. TEXT PARSING & RULES STRUCTURE CLEANING
# ==========================================
def clean_rules_text(raw_text: str) -> str:
    """Strips Law Commission watermarks, normalizes Unicode ligatures, and cleans noise."""
    # Apply NFC Unicode normalization to repair split Devanagari ligatures
    text = unicodedata.normalize("NFC", raw_text)

    text = re.sub(
        r"www\.lawcommission\.(?:com\.)?gov\.np", "", text, flags=re.IGNORECASE
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

    return clean_rules_text("\n".join(pages_text))


def extract_rules_header_metadata(text: str) -> Dict[str, Any]:
    """Extracts Title, Gazette publication date, Enabling Law, and Amendments."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    title = lines[0] if lines else "अज्ञात नियमावली"

    gazette_match = re.search(
        r"नेपाल राजपत्रमा प्रकाश?न मिति\s*\n?\s*([०-९\d।\/\-]+)", text
    )

    enabling_act_match = re.search(
        r"([^\n]+(?:ऐन|संहिता|कार्यविधि|ऐन,)[^\n]*को\s*दफा\s*[०-९\d]+\s*ले\s*दिएको\s*अधिकार|नेपालको\s*संविधानको\s*धारा\s*[०-९\d]+[^\n]*नियमावली\s*बनाएको\s*छ)",
        text,
    )

    amendment_matches = re.findall(
        r"([^\n]+\((?:पहिलो|दोस्रो|तेस्रो|चौथो|पाँचौँ)\s*संशोधन\)\s*नियमावली,\s*[०-९\d]+(?:\s*[०-९\d।\/\-]+)?)",
        text,
    )

    return {
        "title_np": title,
        "gazette_date": gazette_match.group(1).strip() if gazette_match else None,
        "enabling_act": (
            enabling_act_match.group(1).strip() if enabling_act_match else None
        ),
        "amendments": (
            [a.strip() for a in amendment_matches] if amendment_matches else []
        ),
    }


def parse_rules_components(text: str) -> List[Dict[str, Any]]:
    """Splits Rules and Regulations by Chapters (परिच्छेद) and individual Rules (नियम)."""
    chunks = []

    # Extract Preamble if present
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
    chapters_raw = re.split(r"(\n\s*परिच्छेद\s*-\s*[०-९\d]+[^\n]*)", text)
    rule_pattern = (
        r"(?:^|\n)\s*([०-९\d]+)\.\s*([^\n:]+)\s*[:\n]\s*(.*?)(?=\n\s*[०-९\d]+\.|\Z)"
    )

    if len(chapters_raw) > 1:
        for idx in range(1, len(chapters_raw), 2):
            chap_title = re.sub(r"\s+", " ", chapters_raw[idx].strip())
            chap_content = chapters_raw[idx + 1] if idx + 1 < len(chapters_raw) else ""

            rules_found = re.findall(rule_pattern, chap_content, re.DOTALL)
            for rule_num_str, rule_title, rule_body in rules_found:
                rule_num = nepali_to_int(rule_num_str)
                chunks.append(
                    {
                        "doc_type": "rule",
                        "chapter": chap_title,
                        "rule_number": rule_num,
                        "title": rule_title.strip(),
                        "content_text": f"{rule_num_str}. {rule_title.strip()}: "
                        + re.sub(r"\s+", " ", rule_body.strip()),
                    }
                )
    else:
        # Flat Structure without explicit chapters
        rules_found = re.findall(rule_pattern, text, re.DOTALL)
        for rule_num_str, rule_title, rule_body in rules_found:
            rule_num = nepali_to_int(rule_num_str)
            chunks.append(
                {
                    "doc_type": "rule",
                    "chapter": None,
                    "rule_number": rule_num,
                    "title": rule_title.strip(),
                    "content_text": f"{rule_num_str}. {rule_title.strip()}: "
                    + re.sub(r"\s+", " ", rule_body.strip()),
                }
            )

    if not chunks:
        # Fallback to sliding window if standard rule pattern matching fails
        # Truncate text into ~1500 char blocks to prevent memory explosion during vector encoding
        full_clean = re.sub(r"\s+", " ", text)
        for i in range(0, len(full_clean), 1500):
            block = full_clean[i : i + 1500]
            if len(block.strip()) > 30:
                chunks.append(
                    {
                        "doc_type": "full_text_chunk",
                        "chapter": None,
                        "content_text": block.strip(),
                    }
                )

    return chunks


# ==========================================
# 4. RECURSIVE SUBFOLDER INGESTION PIPELINE
# ==========================================
def ingest_all_rules_and_regulations():
    pdf_files = sorted(list(RULES_DIR.rglob("*.pdf")), key=lambda p: p.stat().st_size)
    completed_files = load_completed_files()

    print(f"Total Rules and Regulations PDFs found: {len(pdf_files)}")
    print(f"Already fully ingested: {len(completed_files)}")

    for pdf_path in tqdm(pdf_files, desc="Processing Rules & Regulations"):
        file_name = pdf_path.name
        doc_title = pdf_path.stem.replace("_", " ")
        khanda_category = pdf_path.parent.name

        if file_name in completed_files:
            continue

        purge_partial_file_chunks(file_name)

        raw_text = extract_full_text(pdf_path)
        if not raw_text or len(raw_text) < 50:
            mark_file_completed(file_name, completed_files)
            continue

        header_meta = extract_rules_header_metadata(raw_text)
        doc_chunks = parse_rules_components(raw_text)

        if not doc_chunks:
            continue

        print(
            f"\n[{khanda_category} -> {file_name}] Extracted {len(doc_chunks)} chunks. Vectorizing..."
        )

        # Cap chunk length at 2500 characters to prevent PyTorch memory spikes
        chunk_texts = [c["content_text"][:2500] for c in doc_chunks]

        # Safe batch inference with PyTorch gradient tracking completely turned off
        with torch.no_grad():
            vectors = model.encode(chunk_texts, batch_size=8, show_progress_bar=False)

        file_points = []
        for chunk_idx, (chunk, vector) in enumerate(zip(doc_chunks, vectors)):
            payload = {
                "act_title": doc_title,
                "title_np": header_meta["title_np"],
                "doc_type": chunk["doc_type"],
                "language": "np",
                "category": "Rules and Regulations",
                "subfolder_category": khanda_category,
                "file_name": file_name,
                "content_text": chunk["content_text"],
            }

            if header_meta["gazette_date"]:
                payload["gazette_date"] = header_meta["gazette_date"]
            if header_meta["enabling_act"]:
                payload["enabling_act"] = header_meta["enabling_act"]
            if header_meta["amendments"]:
                payload["amendments"] = header_meta["amendments"]

            if chunk.get("chapter"):
                payload["chapter"] = chunk["chapter"]
            if "title" in chunk:
                payload["title"] = chunk["title"]
            if "rule_number" in chunk:
                payload["rule_number"] = chunk["rule_number"]

            # Use deterministic UUIDs to avoid ID collisions in Qdrant
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_name}_{chunk_idx}"))

            file_points.append(
                PointStruct(id=point_id, vector=vector.tolist(), payload=payload)
            )

        if file_points:
            safe_upsert_with_retry(file_points, batch_size=20)

        mark_file_completed(file_name, completed_files)

        # Force aggressive memory cleanup per file
        del raw_text, doc_chunks, chunk_texts, vectors, file_points
        gc.collect()

    print("\nIngestion completed successfully for all Rules and Regulations files!")


if __name__ == "__main__":
    ingest_all_rules_and_regulations()
