import os
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Dict, Any, List
import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

load_dotenv()

# ==========================================
# 1. CONFIGURATION
# ==========================================
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "nepal_laws_np"
VECTOR_SIZE = 1024

ACTS_DIR = Path("data/raw_pdfs_small/np/Recent")
LOG_FILE = Path("data/raw_pdfs_small/recent.json")

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


def ensure_collection() -> bool:
    """Creates the collection and required filter index when missing."""
    created = False
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        created = True

    collection_info = qdrant.get_collection(COLLECTION_NAME)
    payload_schema = collection_info.payload_schema or {}
    if "file_name" not in payload_schema:
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="file_name",
            field_schema="keyword",
        )
    return created


def get_vector_name() -> str | None:
    """Returns the configured vector name, if the collection uses named vectors."""
    collection_info = qdrant.get_collection(COLLECTION_NAME)
    configured_vectors = collection_info.config.params.vectors
    if isinstance(configured_vectors, dict):
        return next(iter(configured_vectors))
    return None


def get_vector_for_point(vector: List[float], vector_name: str | None):
    """Supports both unnamed vectors and named vectors created in Qdrant Cloud."""
    return {vector_name: vector} if vector_name else vector


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
# 3. TEXT PARSING & ACT STRUCTURE CLEANING
# ==========================================
OCR_CORRECTIONS = {
    # Frequent PDF extraction substitutions found in government Nepali PDFs.
    "आर्थकि": "आर्थिक",
    "संशिप्त": "संक्षिप्त",
    "र्स": "यस",
    "र्ो": "यो",
    "िफा": "दफा",
    "उपिफा": "उपदफा",
    "अन्तःिल्ु क": "अन्तःशुल्क",
    "उत्पािन": "उत्पादन",
    "पिाथ": "पदार्थ",
    "असलु": "असुल",
    "बमोशजम": "बमोजिम",
    "नेपालर्भत्र": "नेपालभित्र",
    "तरूु न्त": "तुरुन्त",
    "अन्र्": "अन्य",
    "हनु ेछन": "हुनेछन्",
    "िेशख": "देखि",
}


def normalize_ocr_text(text: str) -> str:
    """Applies conservative corrections to recurring Unicode PDF OCR artifacts."""
    text = unicodedata.normalize("NFC", text)
    for incorrect, correct in OCR_CORRECTIONS.items():
        text = text.replace(incorrect, correct)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *([,:;।])", r"\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_act_text(raw_text: str) -> str:
    """Strips Nepal Law Commission watermarks, headers, and standalone page numbers."""
    text = re.sub(
        r"www\.lawcommission\.(?:com\.)?gov\.np", "", raw_text, flags=re.IGNORECASE
    )
    text = re.sub(r"नेपाल\s*कानून\s*आयोग", "", text)
    text = re.sub(r"\n\s*[०-९\d]+\s*\n", "\n", text)
    return normalize_ocr_text(text)


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

    return clean_act_text("\n".join(pages_text))


def extract_act_header_metadata(text: str) -> Dict[str, Any]:
    """Extracts Act Number, Gazette/Authentication Date, and Main Title."""
    gazette_match = re.search(
        r"(?:राजपत्रमा प्रकाशित मिति|प्रमाणीकरण र प्रकाशन मिति)\s*\n?\s*([०-९\d।\/\-]+)",
        text,
    )
    act_num_match = re.search(r"ऐन नं\.\s*([०-९\d]+)", text)

    lines = [normalize_ocr_text(line) for line in text.split("\n") if line.strip()]

    title_np = ""
    for line in lines:
        if re.search(r"(ऐन|नियमावली|अध्यादेश|संहिता)", line):
            title_np = line
            break

    return {
        "gazette_date": gazette_match.group(1).strip() if gazette_match else None,
        "act_number": nepali_to_int(act_num_match.group(1)) if act_num_match else None,
        "title_np": title_np or "अज्ञात ऐन",
    }


def parse_act_components(text: str) -> List[Dict[str, Any]]:
    chunks = []

    # Separate main act body from Schedules (अनुसूची)
    schedule_split = re.split(r"(?:^|\n)\s*(अनुसूची[\s\:\-\d]+)", text, maxsplit=1)
    main_body = schedule_split[0]
    schedule_body = (
        "".join(schedule_split[1:]).strip() if len(schedule_split) > 1 else None
    )

    # Match Chapter markers (e.g., "परिच्छेद-१ प्रारम्भिक")
    chapter_pattern = re.compile(
        r"(?:^|\n)\s*(परिच्छेद\s*[\-\–\:\s]*[०-९\d]+\s*[^\n]*)", re.MULTILINE
    )

    # Match Section markers (e.g., "१. संक्षिप्त नाम ... :" or "दफा १")
    sec_pattern = re.compile(
        r"(?:^|\n)\s*([०-९\d]+\.\s*[^:\n]+?\s*[:\n]|\bदफा\s+[०-९\d]+[\.\:\s]*)",
        re.MULTILINE,
    )

    sec_matches = list(sec_pattern.finditer(main_body))

    if sec_matches:
        # Preamble extraction
        preamble_text = main_body[: sec_matches[0].start()].strip()
        if len(preamble_text) > 20:
            chunks.append(
                {
                    "doc_type": "preamble",
                    "title": "प्रस्तावना",
                    "content_text": normalize_ocr_text(preamble_text),
                }
            )

        current_chapter = None

        for i, match in enumerate(sec_matches):
            start_idx = match.start()
            end_idx = (
                sec_matches[i + 1].start()
                if i + 1 < len(sec_matches)
                else len(main_body)
            )
            sec_text = normalize_ocr_text(main_body[start_idx:end_idx])

            # Check if a new Chapter header appeared before this Section
            chap_search = chapter_pattern.findall(
                main_body[sec_matches[i - 1].end() if i > 0 else 0 : start_idx]
            )
            if chap_search:
                current_chapter = chap_search[-1].strip()

            sec_first_line = normalize_ocr_text(match.group(1).strip())
            sec_num = nepali_to_int(sec_first_line.split(".")[0])

            # Separate section title from label (e.g., "१. संक्षिप्त नाम, प्रारम्भ र विस्तार :")
            title_parts = sec_first_line.split(":", 1)
            sec_title = title_parts[0].strip()

            chunk_data = {
                "doc_type": "section",
                "title": sec_title,
                "content_text": sec_text,
            }

            if sec_num is not None:
                chunk_data["section_number"] = sec_num
            if current_chapter:
                chunk_data["chapter"] = current_chapter

            chunks.append(chunk_data)
    else:
        chunks.append({"doc_type": "full_text", "content_text": main_body})

    # Process Schedules (Economic Act tariffs / tables)
    if schedule_body:
        schedules = re.split(
            r"(?:^|\n)\s*(?=अनुसूची\s*[\-\–\:\s]*[०-९\d]+)", schedule_body
        )
        for sched_idx, sched_text in enumerate(schedules):
            if not sched_text.strip():
                continue

            sched_match = re.match(
                r"^अनुसूची\s*[\-\–\:\s]*([०-९\d]+)", sched_text.strip()
            )
            sched_num = (
                nepali_to_int(sched_match.group(1)) if sched_match else sched_idx + 1
            )

            chunks.append(
                {
                    "doc_type": "schedule",
                    "schedule_number": sched_num,
                    "title": f"अनुसूची {sched_num}",
                    "content_text": normalize_ocr_text(sched_text),
                }
            )

    return chunks


# ==========================================
# 4. UNIVERSAL ACT INGESTION PIPELINE
# ==========================================
def ingest_all_acts():
    collection_created = ensure_collection()
    vector_name = get_vector_name()
    pdf_files = sorted(list(ACTS_DIR.glob("*.pdf")))
    # A recreated empty collection must not trust completion state from the old collection.
    completed_files = set() if collection_created else load_completed_files()

    print(f"Total Act PDFs found: {len(pdf_files)}")
    print(f"Already fully ingested: {len(completed_files)}")

    for file_idx, pdf_path in enumerate(
        tqdm(pdf_files, desc="Processing Act PDFs"), start=0
    ):
        file_name = pdf_path.name
        act_title = pdf_path.stem.replace("_", " ")

        if file_name in completed_files:
            continue

        purge_partial_file_chunks(file_name)

        raw_text = extract_full_text(pdf_path)
        if not raw_text or len(raw_text) < 50:
            mark_file_completed(file_name, completed_files)
            continue

        header_meta = extract_act_header_metadata(raw_text)
        doc_chunks = parse_act_components(raw_text)
        file_points = []

        # Formula guarantees reserved ID range per file: 400,000 + (file_index * 1000) + chunk_index
        base_file_id = 400000 + (file_idx * 1000)

        for chunk_idx, chunk in enumerate(doc_chunks):
            raw_content_text = chunk["content_text"].strip()
            content_text = re.sub(
                r"\s+", " ", normalize_ocr_text(raw_content_text)
            ).strip()
            title_np = normalize_ocr_text(chunk.get("title") or header_meta["title_np"])

            payload = {
                "act_title": header_meta["title_np"],
                "title_np": header_meta["title_np"],
                "title": title_np,
                "doc_type": chunk["doc_type"],
                "language": "np",
                "category": "Act",
                "file_name": file_name,
                "content_text": content_text,
                "raw_content_text": raw_content_text,
            }

            if header_meta["gazette_date"]:
                payload["gazette_date"] = header_meta["gazette_date"]
            if header_meta["act_number"]:
                payload["act_number"] = header_meta["act_number"]

            if "chapter" in chunk:
                payload["chapter"] = chunk["chapter"]
            if "section_number" in chunk:
                payload["section_number"] = chunk["section_number"]
            if "schedule_number" in chunk:
                payload["schedule_number"] = chunk["schedule_number"]

            vector = model.encode(content_text, show_progress_bar=False).tolist()

            file_points.append(
                PointStruct(
                    id=base_file_id + chunk_idx,
                    vector=get_vector_for_point(vector, vector_name),
                    payload=payload,
                )
            )

        if file_points:
            safe_upsert_with_retry(file_points, batch_size=20)

        mark_file_completed(file_name, completed_files)

    print("\nIngestion completed successfully for all Act files!")


if __name__ == "__main__":
    ingest_all_acts()
