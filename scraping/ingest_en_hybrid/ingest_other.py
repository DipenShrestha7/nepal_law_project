import os
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
import pdfplumber
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from chatbot.retrieval_utils import embedding_text

load_dotenv()

RAW_PDFS_DIR = "data/raw_pdfs/en"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "nepal_laws_en_hybrid"
LOG_FILE = Path("data/other_english_ingestion_log.json")
MAX_CHUNK_CHARS = 1800
CHUNK_OVERLAP_CHARS = 250
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "180"))
QDRANT_BATCH_SIZE = 10

# Initialize BGE-M3 Embedding Model (1024 dimensions)
print("Loading BAAI/bge-m3 embedding model...")
embedder = SentenceTransformer("BAAI/bge-m3")

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=QDRANT_TIMEOUT,
)


# 1. PAGE COUNT INSPECTION & PDF SORTING
def get_pdf_page_count(pdf_path: Path) -> int:
    """Returns the total number of pages in a PDF file."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception as e:
        print(f"Error opening {pdf_path.name}: {e}")
        return -1


def get_sorted_pdf_queue(directory_path: str) -> list[tuple[Path, int]]:
    """Scans directory and returns list of (pdf_path, page_count) sorted ascending by page count."""
    dir_path = Path(directory_path)
    pdf_files = list(dir_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{directory_path}'.")
        return []

    print(f"Inspecting page counts for {len(pdf_files)} PDF files...")
    pdf_metadata = []
    for pdf_path in pdf_files:
        page_count = get_pdf_page_count(pdf_path)
        if page_count > 0:
            pdf_metadata.append((pdf_path, page_count))

    # Sort ascending by page count
    pdf_metadata.sort(key=lambda item: item[1])

    print("\n--- Processing Queue (Ascending Order of Page Count) ---")
    for idx, (path, count) in enumerate(pdf_metadata, start=1):
        print(f"{idx}. {path.name} — {count} pages")
    print("------------------------------------------------------\n")

    return pdf_metadata


# 2. PDF TEXT EXTRACTION & CLEANING
def extract_pdf_pages(pdf_path: Path) -> str:
    """Extracts raw text page by page, stripping headers, footers, and aggregator watermarks."""
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines = []
                for line in text.splitlines():
                    # Filter isolated page numbers e.g. "(1)", "240"
                    if re.match(r"^\s*\(?\d+\)?\s*$", line):
                        continue
                    # Filter third-party / repository watermarks & repeated site headers
                    if re.search(
                        r"(Downloaded by|Scan to open|studocu|Nepal Law Commission|www\.lawcommission\.gov\.np)",
                        line,
                        re.IGNORECASE,
                    ):
                        continue
                    lines.append(line)
                full_text.append("\n".join(lines))

    return "\n\n".join(full_text)


def normalize_act_title(filename: str) -> str:
    """Build a stable act title directly from the file name without scanning body text."""
    if not filename:
        return "Unknown Act"

    name = os.path.splitext(filename)[0]
    name = name.replace("_", " ").replace("-", " ")
    if "constitution" in name.lower():
        return "Constitution of Nepal"
    return re.sub(r"\s+", " ", name).strip().title() or filename


# 3. STATUTORY STRUCTURE PARSER (Acts, Sections, Schedules)
def normalize_section_content(sec_num: str, sec_title: str, sec_content: str) -> str:
    """Normalizes whitespace and removes duplicated section headers from the body text."""
    sec_content = sec_content.strip()

    # Remove duplicated section header at the start of section body
    repeated_prefix = rf"^\s*{re.escape(sec_num)}\.\s*{re.escape(sec_title)}\s*:\s*"
    sec_content = re.sub(repeated_prefix, "", sec_content, count=1, flags=re.IGNORECASE)

    sec_content = re.sub(r"^\s*", "", sec_content)
    sec_content = re.sub(r"\s+", " ", sec_content).strip()

    if not sec_content:
        return f"Section {sec_num}. {sec_title}."

    return f"Section {sec_num}. {sec_title}: {sec_content}"


def parse_act_structure(raw_text: str, act_filename: str) -> list[dict]:
    """Parses statutory Act text into Preamble, Sections (1..N), and Schedules."""
    documents = []
    act_title = normalize_act_title(act_filename)

    # PREAMBLE EXTRACTION
    preamble_match = re.search(
        r"Preamble:\s*(.*?)(?=Chapter\s*-\s*1|Part\s*-\s*1|\n1\.\s+)",
        raw_text,
        re.DOTALL | re.IGNORECASE,
    )
    if preamble_match:
        preamble_text = preamble_match.group(1).strip()
        documents.append(
            {
                "doc_type": "preamble",
                "act_title": act_title,
                "title": f"Preamble of {act_title}",
                "section_number": None,
                "schedule_number": None,
                "content_text": f"Preamble: {preamble_text}",
                "language": "en",
            }
        )

    # SECTIONS: accept "1. Short title and commencement:", "SECTION 5",
    # "Section 12: ...", "Section-12 ..."
    section_pattern = r"(?is)(?=\n\s*(?:section|sections|खण्ड)\s*[-:.\s]*\s*(\d{1,3})\s*(?:[.:\-–—]\s*|\s+)\s*([A-Z][A-Za-z0-9\s,\-\(\)\'\"]+?)\s*[:.])"
    parts = re.split(section_pattern, raw_text)

    i = 1
    while i < len(parts):
        sec_num = parts[i].strip()
        sec_title = parts[i + 1].strip()
        sec_content = parts[i + 2].strip() if (i + 2) < len(parts) else ""

        next_boundary = re.search(
            r"(?is)(?=\n\s*(?:section|sections|खण्ड)\s*[-:.\s]*\s*\d{1,3}\b|\n\s*(?:schedule|schedules|सञ्चिका)\s*[-:.\s]*\s*\d{1,3}\b)",
            sec_content,
        )
        if next_boundary:
            sec_content = sec_content[: next_boundary.start()].strip()

        clean_content = normalize_section_content(sec_num, sec_title, sec_content)

        documents.append(
            {
                "doc_type": "section",
                "act_title": act_title,
                "title": sec_title,
                "section_number": int(sec_num) if sec_num.isdigit() else sec_num,
                "schedule_number": None,
                "content_text": clean_content,
                "language": "en",
            }
        )
        i += 3

    # SCHEDULES: accept "Schedule 8", "SCHEDULE 8", "Schedule 8:", "Schedule 8 (Relating to Article 57)"
    schedule_split_pattern = (
        r"(?is)(?=\n\s*(?:schedule|schedules|सञ्चिका)\s*[-:.\s]*\s*\d{1,3}\b)"
    )
    schedule_splits = re.split(schedule_split_pattern, raw_text)
    for sched_text in schedule_splits:
        match = re.search(
            r"(?is)(?:schedule|schedules|सञ्चिका)\s*[-:.\s]*\s*(\d{1,3})\b",
            sched_text,
        )
        if match:
            sched_num = int(match.group(1))
            lines = sched_text.strip().splitlines()
            sched_title = lines[0].strip() if lines else f"Schedule {sched_num}"
            clean_sched_content = re.sub(r"\s+", " ", sched_text).strip()

            documents.append(
                {
                    "doc_type": "schedule",
                    "act_title": act_title,
                    "title": sched_title,
                    "section_number": None,
                    "schedule_number": sched_num,
                    "content_text": clean_sched_content,
                    "language": "en",
                }
            )

    return documents


def _document_context(document: dict) -> str:
    """Returns a stable citation prefix for every generated chunk."""
    if document["doc_type"] == "section":
        return f"Section {document['section_number']}. {document['title']}"
    if document["doc_type"] == "schedule":
        return f"Schedule {document['schedule_number']}. {document['title']}"
    return document["title"]


def _split_text(text: str) -> list[str]:
    """Splits long legal text at natural boundaries with a small overlap."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        proposed_end = min(start + MAX_CHUNK_CHARS, len(text))
        end = proposed_end

        if proposed_end < len(text):
            boundary_candidates = [
                text.rfind(". ", start + MAX_CHUNK_CHARS // 2, proposed_end),
                text.rfind("; ", start + MAX_CHUNK_CHARS // 2, proposed_end),
                text.rfind(") ", start + MAX_CHUNK_CHARS // 2, proposed_end),
            ]
            boundary = max(boundary_candidates)
            if boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP_CHARS, start + 1)

    return chunks


def prepare_chunks(documents: list[dict], source_file: str) -> list[dict]:
    """Creates retrieval-sized chunks while retaining legal citation metadata."""
    chunks = []

    for document_index, document in enumerate(documents):
        context = _document_context(document)
        document_chunks = _split_text(document["content_text"])
        parent_id = f"{source_file}:{document_index}"

        for chunk_index, content in enumerate(document_chunks):
            if chunk_index > 0 and not content.startswith(context):
                content = f"{context}: {content}"

            chunk = {
                **document,
                "source_file": source_file,
                "parent_id": parent_id,
                "chunk_index": chunk_index,
                "chunk_count": len(document_chunks),
                "citation": (
                    f"{document['act_title']}, {context}"
                    if document["doc_type"] != "preamble"
                    else f"{document['act_title']}, Preamble"
                ),
                "content_text": content,
            }
            chunks.append(chunk)

    return chunks


def write_ingestion_log(
    file_name: str, total_chunks: int, status: str = "success"
) -> None:
    """Stores one current ingestion result per source file."""
    records = []
    if LOG_FILE.exists():
        with LOG_FILE.open("r", encoding="utf-8") as log_file:
            records = json.load(log_file)

    record = {
        "file_name": file_name,
        "total_chunks": total_chunks,
        "status": status,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    records = [existing for existing in records if existing["file_name"] != file_name]
    records.append(record)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("w", encoding="utf-8") as log_file:
        json.dump(records, log_file, ensure_ascii=False, indent=2)


def get_completed_files() -> set[str]:
    """Returns source files recorded as successfully ingested."""
    if not LOG_FILE.exists():
        return set()

    with LOG_FILE.open("r", encoding="utf-8") as log_file:
        records = json.load(log_file)

    return {
        record["file_name"] for record in records if record.get("status") == "success"
    }


# 4. QDRANT INGESTION
def ingest_to_qdrant(documents: list[dict]) -> int:
    """Generates embeddings and upserts parsed legal chunks into Qdrant."""
    if not documents:
        return 0

    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    collection_info = qdrant.get_collection(COLLECTION_NAME)
    payload_schema = collection_info.payload_schema or {}
    for field_name, field_schema in {
        "act_title": "keyword",
        "doc_type": "keyword",
        "source_file": "keyword",
        "language": "keyword",
        "article_number": "integer",
        "section_number": "integer",
        "schedule_number": "integer",
    }.items():
        if field_name not in payload_schema:
            qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_schema,
            )

    points = []
    print(f"Generating embeddings for {len(documents)} retrieval chunks...")

    for doc in tqdm(documents):
        vector = embedder.encode(embedding_text(doc, doc["content_text"])).tolist()
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{doc['source_file']}:{doc['parent_id']}:{doc['chunk_index']}",
            )
        )

        payload = {
            "act_title": doc["act_title"],
            "source_file": doc["source_file"],
            "doc_type": doc["doc_type"],
            "title": doc["title"],
            "article_number": doc.get("article_number"),
            "section_number": doc.get("section_number"),
            "schedule_number": doc.get("schedule_number"),
            "parent_id": doc["parent_id"],
            "chunk_index": doc["chunk_index"],
            "chunk_count": doc["chunk_count"],
            "citation": doc["citation"],
            "language": doc["language"],
            "content_text": doc["content_text"],
        }

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    print(
        f"Upserting {len(points)} points to Qdrant in batches of "
        f"{QDRANT_BATCH_SIZE}..."
    )
    for batch_start in range(0, len(points), QDRANT_BATCH_SIZE):
        batch = points[batch_start : batch_start + QDRANT_BATCH_SIZE]
        batch_number = batch_start // QDRANT_BATCH_SIZE + 1
        total_batches = (len(points) + QDRANT_BATCH_SIZE - 1) // QDRANT_BATCH_SIZE
        print(
            f"Uploading batch {batch_number}/{total_batches} "
            f"({len(batch)} points)..."
        )
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            timeout=QDRANT_TIMEOUT,
        )
    return len(points)


# 5. MAIN PIPELINE
if __name__ == "__main__":
    pdf_queue = get_sorted_pdf_queue(RAW_PDFS_DIR)
    completed_files = get_completed_files()

    if not pdf_queue:
        print("Pipeline aborted: No PDFs to ingest.")
        exit(0)

    for pdf_path, page_count in pdf_queue:
        if pdf_path.name in completed_files:
            print(f"Skipping already ingested file: {pdf_path.name}")
            continue

        print(f"\n==================================================")
        print(f"Processing: {pdf_path.name} ({page_count} pages)")
        print(f"==================================================")

        raw_text = extract_pdf_pages(pdf_path)
        parsed_docs = parse_act_structure(raw_text, pdf_path.name)
        chunks = prepare_chunks(parsed_docs, pdf_path.name)

        print(
            f"Parsed {len(parsed_docs)} legal entries into {len(chunks)} retrieval chunks "
            "(Preamble + Sections + Schedules)."
        )

        total_chunks = ingest_to_qdrant(chunks)
        write_ingestion_log(pdf_path.name, total_chunks)
        print(f"Logged {pdf_path.name}: {total_chunks} chunks")

    print("\nAll PDF documents ingested successfully in page-ascending order!")
