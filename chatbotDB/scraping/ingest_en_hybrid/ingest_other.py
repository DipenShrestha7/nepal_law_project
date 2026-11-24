import argparse
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from chatbot.utils.retrieval_utils import embedding_text_en as embedding_text_fn

load_dotenv()

DEFAULT_RAW_PDFS_DIR = "data/raw_pdfs/en"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "nepal_laws_en_hybrid"
LOG_FILE = Path("data/other_english_ingestion_log.json")
MAX_CHUNK_CHARS = 1800
CHUNK_OVERLAP_CHARS = 250
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "180"))
QDRANT_BATCH_SIZE = 10

# Initialize BGE-M3 Dense Model (1024 dims) & BM25 Sparse Model
print("Loading BAAI/bge-m3 dense embedding model...")
embedder = SentenceTransformer("BAAI/bge-m3")

print("Loading Qdrant/bm25 sparse embedding model...")
sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=QDRANT_TIMEOUT,
)


def ensure_payload_indexes(collection_name: str):
    """Create exact-match payload indexes used by hybrid Qdrant queries."""
    collection_info = qdrant.get_collection(collection_name)
    payload_schema = collection_info.payload_schema or {}
    index_definitions = {
        "act_title": "keyword",
        "doc_type": "keyword",
        "source_file": "keyword",
        "language": "keyword",
        "article_number": "integer",
        "section_number": "integer",
        "schedule_number": "integer",
    }

    for field_name, field_schema in index_definitions.items():
        if field_name not in payload_schema:
            qdrant.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )


# 1. PAGE COUNT INSPECTION & QUEUE BUILDING (Folder or Single File)
def get_pdf_page_count(pdf_path: Path) -> int:
    """Returns the total number of pages in a PDF file."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception as e:
        print(f"Error opening {pdf_path.name}: {e}")
        return -1


def build_pdf_queue(target_path_str: str) -> list[tuple[Path, int]]:
    """Builds a queue of (pdf_path, page_count).
    If target is a folder, returns all PDFs sorted ascending by page count.
    If target is a single file, returns just that file.
    """
    target_path = Path(target_path_str)

    if not target_path.exists():
        print(f"Error: Path '{target_path_str}' does not exist.")
        return []

    # Single File Case
    if target_path.is_file():
        if target_path.suffix.lower() != ".pdf":
            print(f"Error: Specified file '{target_path.name}' is not a PDF.")
            return []
        page_count = get_pdf_page_count(target_path)
        if page_count > 0:
            return [(target_path, page_count)]
        return []

    # Directory Case
    pdf_files = list(target_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in directory '{target_path_str}'.")
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
                    # Filter watermarks & repository headers
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
    """Build a stable act title directly from the file name."""
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

    repeated_prefix = rf"^\s*{re.escape(sec_num)}\.\s*{re.escape(sec_title)}\s*:\s*"
    sec_content = re.sub(repeated_prefix, "", sec_content, count=1, flags=re.IGNORECASE)

    sec_content = re.sub(r"^\s*", "", sec_content)
    sec_content = re.sub(r"\s+", " ", sec_content).strip()

    if not sec_content:
        return f"Section {sec_num}. {sec_title}."

    return f"Section {sec_num}. {sec_title}: {sec_content}"


def parse_act_structure(raw_text: str, act_filename: str) -> list[dict]:
    """Parses statutory Act text into Preamble, Sections, and Schedules."""
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

        # SECTIONS
        # Laws commonly use either "Section 1. Title:" or "1. Title:". Match
        # both forms at the start of a line so references inside section bodies
        # are not treated as new sections.
    numbered_section_pattern = re.compile(
        r"(?im)^\s*(\d{1,3})\.\s+([^\n]{2,120}?)(?::|$)"
    )
    explicit_section_pattern = re.compile(
        r"(?im)^\s*(?:section|sections|खण्ड)\s*[-:.\s]*"
        r"(\d{1,3})\s*[.\-–—:]\s*"
        r"([^\n]{2,120}?)(?::|$)"
    )
    numbered_matches = list(numbered_section_pattern.finditer(raw_text))
    explicit_matches = list(explicit_section_pattern.finditer(raw_text))
    section_matches = (
        numbered_matches
        if len(numbered_matches) >= len(explicit_matches)
        else explicit_matches
    )

    # A table of contents or repeated headers can contain the same section
    # number more than once. Keep the first occurrence of each number.
    unique_section_matches = []
    seen_section_numbers = set()
    for match in section_matches:
        section_number = int(match.group(1))
        if section_number in seen_section_numbers:
            continue
        seen_section_numbers.add(section_number)
        unique_section_matches.append(match)

    for index, match in enumerate(unique_section_matches):
        sec_num = match.group(1)
        sec_title = match.group(2).strip()
        next_start = (
            unique_section_matches[index + 1].start()
            if index + 1 < len(unique_section_matches)
            else len(raw_text)
        )
        sec_content = raw_text[match.end() : next_start].strip()

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

    # SCHEDULES
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
    """Splits long legal text at natural boundaries with overlap."""
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
    """Stores current ingestion result per source file."""
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


# 4. HYBRID QDRANT INGESTION
def ingest_to_qdrant(documents: list[dict]) -> int:
    """Generates Dense & Sparse embeddings and upserts hybrid points into Qdrant."""
    if not documents:
        return 0

    # Ensure collection exists configured for HYBRID SEARCH
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )
        ensure_payload_indexes(COLLECTION_NAME)
    else:
        ensure_payload_indexes(COLLECTION_NAME)

    points = []
    print(f"Generating dense & sparse vectors for {len(documents)} retrieval chunks...")

    for doc in tqdm(documents):
        text_to_embed = embedding_text_fn(doc, doc["content_text"])

        # Dense Vector (1024-dim BGE-M3)
        dense_vector = embedder.encode(text_to_embed).tolist()

        # Sparse Vector (BM25)
        sparse_vec_obj = list(sparse_embedder.embed([text_to_embed]))[0]

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

        points.append(
            models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vector,
                    "sparse": models.SparseVector(
                        indices=sparse_vec_obj.indices.tolist(),
                        values=sparse_vec_obj.values.tolist(),
                    ),
                },
                payload=payload,
            )
        )

    print(
        f"Upserting {len(points)} hybrid points to Qdrant Cloud in batches of {QDRANT_BATCH_SIZE}..."
    )
    for batch_start in range(0, len(points), QDRANT_BATCH_SIZE):
        batch = points[batch_start : batch_start + QDRANT_BATCH_SIZE]
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            timeout=QDRANT_TIMEOUT,
        )
    return len(points)


# 5. CLI & MAIN PIPELINE
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest statutory PDF acts into Qdrant using Hybrid Search."
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default=DEFAULT_RAW_PDFS_DIR,
        help="Path to a folder of PDFs or a single PDF file (default: data/raw_pdfs/en).",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-ingest files even if previously recorded as completed.",
    )

    args = parser.parse_args()

    pdf_queue = build_pdf_queue(args.path)
    completed_files = set() if args.force else get_completed_files()

    if not pdf_queue:
        print("Pipeline finished: No valid PDFs to ingest.")
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

    print("\nIngestion completed successfully with Hybrid Search setup!")
