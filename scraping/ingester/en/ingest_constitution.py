import re
import os
import uuid
import pdfplumber
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv

load_dotenv()
PDF_PATH = "data/raw_pdfs/en/constitution.pdf"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "nepal_laws"

# Initialize BGE-M3 Embedding Model (1024 dimensions)
print("Loading BAAI/bge-m3 embedding model...")
embedder = SentenceTransformer("BAAI/bge-m3")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# PARSING PDF TEXT


def extract_pdf_pages(pdf_path: str) -> str:
    """Extracts raw text page by page, removing page numbers."""
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                # Remove isolated footer/header page numbers (e.g. "(1)", "(240)")
                lines = [
                    line
                    for line in text.splitlines()
                    if not re.match(r"^\s*\(\d+\)\s*$", line)
                ]
                full_text.append("\n".join(lines))
    return "\n\n".join(full_text)


def parse_constitution_structure(raw_text: str) -> list[dict]:
    """Splits full text into Preamble, Articles, and Schedules 3-9."""
    documents = []

    # PREAMBLE EXTRACTION
    preamble_match = re.search(
        r"Preamble:\s*(.*?)(?=Part\s*-\s*1|\n1\.\s+)",
        raw_text,
        re.DOTALL | re.IGNORECASE,
    )
    if preamble_match:
        preamble_text = preamble_match.group(1).strip()
        documents.append(
            {
                "doc_type": "preamble",
                "title": "Preamble of the Constitution of Nepal",
                "article_number": None,
                "schedule_number": None,
                "content_text": f"Preamble: {preamble_text}",
                "language": "en",
            }
        )

    # ARTICLES EXTRACTION (Articles 1 through 308)
    # Matches numbered sections like "28. Right to Privacy:" or "29. Right against Exploitation:"
    article_pattern = r"(?=\n\s*(\d{1,3})\.\s+([A-Z][A-Za-z0-9\s,\-\(\)]+):)"
    parts = re.split(article_pattern, raw_text)

    # Process split groups (article_number, title, content)
    i = 1
    while i < len(parts):
        art_num = parts[i].strip()
        art_title = parts[i + 1].strip()
        art_content = parts[i + 2].strip() if (i + 2) < len(parts) else ""

        # Limit search window to start of Next Article or Schedule
        next_boundary = re.search(
            r"(?=\n\s*\d{1,3}\.\s+|Schedule\s*-\s*\d+)", art_content
        )
        if next_boundary:
            art_content = art_content[: next_boundary.start()].strip()

        # Clean excess spaces
        clean_content = re.sub(
            r"\s+", " ", f"Article {art_num}. {art_title}: {art_content}"
        )

        documents.append(
            {
                "doc_type": "article",
                "title": art_title,
                "article_number": int(art_num),
                "schedule_number": None,
                "content_text": clean_content,
                "language": "en",
            }
        )
        i += 3

    # SCHEDULES EXTRACTION (Only Schedules 3 through 9)
    schedule_splits = re.split(r"(?=Schedule\s*-\s*\d+)", raw_text, flags=re.IGNORECASE)
    for sched_text in schedule_splits:
        match = re.search(r"Schedule\s*-\s*(\d+)", sched_text, re.IGNORECASE)
        if match:
            sched_num = int(match.group(1))

            # Explicitly EXCLUDE Schedule 1 (National Flag) and Schedule 2 (Coat of Arms)
            if sched_num in [1, 2]:
                print(
                    f"Skipping Schedule-{sched_num} (Diagram / Image asset excluded)."
                )
                continue

            # Extract title line
            lines = sched_text.strip().splitlines()
            sched_title = lines[0] if lines else f"Schedule {sched_num}"
            clean_sched_content = re.sub(r"\s+", " ", sched_text).strip()

            documents.append(
                {
                    "doc_type": "schedule",
                    "title": sched_title,
                    "article_number": None,
                    "schedule_number": sched_num,
                    "content_text": clean_sched_content,
                    "language": "en",
                }
            )

    return documents


# VECTOR DATABASE INGESTION
def ingest_to_qdrant(documents: list[dict]):
    # Recreate collection with 1024 vector dimension for bge-m3
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    points = []
    print(f"Generating vectors for {len(documents)} parsed structural chunks...")

    for doc in tqdm(documents):
        # Generate 1024-dimension embedding
        vector = embedder.encode(doc["content_text"]).tolist()

        # Prepare Qdrant Point
        point_id = str(uuid.uuid4())
        payload = {
            "act_title": "Constitution of Nepal",
            "doc_type": doc["doc_type"],
            "title": doc["title"],
            "article_number": doc["article_number"],
            "schedule_number": doc["schedule_number"],
            "language": doc["language"],
            "content_text": doc["content_text"],
        }

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    # Batch upsert points to Qdrant Cloud
    print(f"Upserting {len(points)} points to Qdrant Cloud...")
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print("Ingestion complete successfully!")


# MAIN EXECUTION FLOW
if __name__ == "__main__":
    print("Extracting raw text from PDF...")
    raw_pdf_text = extract_pdf_pages(PDF_PATH)

    print("Parsing document hierarchy...")
    parsed_docs = parse_constitution_structure(raw_pdf_text)

    print(
        f"Parsed {len(parsed_docs)} legal entries (Preamble + Articles + Schedules 3-9)."
    )

    ingest_to_qdrant(parsed_docs)
