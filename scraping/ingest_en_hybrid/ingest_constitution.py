import re
import os
import uuid
import pdfplumber
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv
from chatbot.retrieval_utils_en import embedding_text_en
from fastembed import SparseTextEmbedding

load_dotenv()
PDF_PATH = "data/raw_pdfs/en/constitution.pdf"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "nepal_laws_en_hybrid"
CONSTITUTION_ACT_TITLE = "Constitution of Nepal"
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "120"))
QDRANT_BATCH_SIZE = 32

# Initialize BGE-M3 Embedding Model (1024 dimensions)
print("Loading BAAI/bge-m3 embedding model...")
embedder = SentenceTransformer("BAAI/bge-m3")

print("Loading BM25 sparse model...")
sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=QDRANT_TIMEOUT,
)


def ensure_payload_indexes(collection_name: str):
    """Create exact-match filter indexes used by retrieval queries."""
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


def normalize_article_content(art_num: str, art_title: str, art_content: str) -> str:
    """Remove duplicated article headers that appear at the start of the article body."""
    art_content = art_content.strip()

    repeated_prefix = rf"^\s*{re.escape(art_num)}\.\s*{re.escape(art_title)}\s*:\s*"
    art_content = re.sub(repeated_prefix, "", art_content, count=1, flags=re.IGNORECASE)

    art_content = re.sub(r"^\s*", "", art_content)
    art_content = re.sub(r"\s+", " ", art_content).strip()

    if not art_content:
        return f"Article {art_num}. {art_title}."

    return f"Article {art_num}. {art_title}: {art_content}"


def parse_constitution_structure(raw_text: str) -> list[dict]:
    """Robustly parses Preamble, all 308 Articles, and Schedules 3-9."""
    documents = []

    # 1. PREAMBLE
    preamble_match = re.search(
        r"Preamble\s*:\s*(.*?)(?=Part\s*[-–—]?\s*1|\n\s*1\.\s+)",
        raw_text,
        re.DOTALL | re.IGNORECASE,
    )
    if preamble_match:
        documents.append(
            {
                "doc_type": "preamble",
                "title": "Preamble of the Constitution of Nepal",
                "article_number": None,
                "section_number": None,
                "schedule_number": None,
                "content_text": f"Preamble: {preamble_match.group(1).strip()}",
                "language": "en",
            }
        )

    # 2. ARTICLES (Matches "1. Title:", "Article 1.", "1. Title -")
    # Finds start position of every article header
    article_matches = list(
        re.finditer(
            r"\n\s*(?:Article\s+)?(\d{1,3})\s*[\.\:\-–—]\s*([^\n\:\.\-\–—]+)[\.\:\-–—]?",
            raw_text,
            re.IGNORECASE,
        )
    )

    # Isolate text between article header positions, but keep only the first occurrence
    # of each real article number. The PDF text can repeat the same numbering in
    # table-of-contents / repeated headers, which creates inflated document counts.
    unique_articles: dict[int, dict] = {}
    for idx, match in enumerate(article_matches):
        art_num = int(match.group(1))
        if art_num in unique_articles:
            continue

        art_title = match.group(2).strip()

        # Text starts at current match and ends at the next article match (or Schedules start)
        start_pos = match.start()
        if idx + 1 < len(article_matches):
            end_pos = article_matches[idx + 1].start()
        else:
            # Stop if schedules begin
            sched_start = re.search(
                r"\n\s*Schedule\s*[-–—]?\s*1", raw_text[start_pos:], re.IGNORECASE
            )
            end_pos = start_pos + sched_start.start() if sched_start else len(raw_text)

        art_content = raw_text[start_pos:end_pos].strip()
        art_content = re.sub(r"\s+", " ", art_content)  # Clean whitespace

        unique_articles[art_num] = {
            "doc_type": "article",
            "title": art_title,
            "article_number": art_num,
            "section_number": None,
            "schedule_number": None,
            "content_text": art_content,
            "language": "en",
        }

    documents.extend(unique_articles.values())

    # 3. SCHEDULES (Schedules 3 to 9)
    schedule_matches = list(
        re.finditer(
            r"^[^\S\r\n]*[^\w\r\n]*Schedule\s*[-–—:]?\s*(\d{1,3})\s*$",
            raw_text,
            re.IGNORECASE | re.MULTILINE,
        )
    )

    unique_schedules: dict[int, dict] = {}
    for idx, match in enumerate(schedule_matches):
        sched_num = int(match.group(1))
        if sched_num in [1, 2] or sched_num in unique_schedules:
            continue  # Exclude diagram/image schedules and repeated matches

        start_pos = match.start()
        end_pos = (
            schedule_matches[idx + 1].start()
            if idx + 1 < len(schedule_matches)
            else len(raw_text)
        )

        sched_content = raw_text[start_pos:end_pos].strip()
        sched_content = re.sub(r"\s+", " ", sched_content)

        unique_schedules[sched_num] = {
            "doc_type": "schedule",
            "title": f"Schedule {sched_num}",
            "article_number": None,
            "section_number": None,
            "schedule_number": sched_num,
            "content_text": sched_content,
            "language": "en",
        }

    documents.extend(unique_schedules.values())

    return documents


# VECTOR DATABASE INGESTION
def ingest_to_qdrant(documents: list[dict]):
    if not documents:
        print(
            "No parsed documents to ingest; leaving existing Qdrant points unchanged."
        )
        return

    # 1. Recreate collection configured for HYBRID SEARCH
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
    print(f"Generating dense & sparse vectors for {len(documents)} parsed chunks...")

    source_file = os.path.basename(PDF_PATH)
    for doc_index, doc in enumerate(tqdm(documents)):
        text_to_embed = embedding_text_en(doc, doc["content_text"])

        # Generate 1024-dimension Dense Vector
        dense_vector = embedder.encode(text_to_embed).tolist()

        # Generate Sparse Vector (BM25)
        sparse_vec_obj = list(sparse_embedder.embed([text_to_embed]))[0]

        # Prepare Payload
        point_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{source_file}:constitution:{doc_index}")
        )
        payload = {
            "act_title": CONSTITUTION_ACT_TITLE,
            "source_file": source_file,
            "doc_type": doc["doc_type"],
            "title": doc["title"],
            "article_number": doc.get("article_number"),
            "section_number": doc.get("section_number"),
            "schedule_number": doc.get("schedule_number"),
            "language": doc["language"],
            "content_text": doc["content_text"],
        }

        # Save BOTH named vectors in PointStruct
        points.append(
            PointStruct(
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

    # Batch upsert points to Qdrant Cloud so large payloads do not hit the
    # write-timeout when a single upload is too large.
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

    point_ids = [point.id for point in points]
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="act_title",
                        match=models.MatchValue(value=CONSTITUTION_ACT_TITLE),
                    ),
                    models.FieldCondition(
                        key="source_file",
                        match=models.MatchValue(value=source_file),
                    ),
                ],
                must_not=[models.HasIdCondition(has_id=point_ids)],
            )
        ),
        wait=True,
        timeout=QDRANT_TIMEOUT,
    )
    print("Ingestion complete successfully with Hybrid Search setup!")


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
