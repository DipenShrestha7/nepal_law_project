import re
import sys
import unicodedata
from typing import Any, Dict, List, TypedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

try:
    from .retrieval_utils import metadata_header
except ImportError:
    from retrieval_utils import metadata_header

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    OPENROUTER_API_KEY,
    COLLECTION_NAME,
    FALLBACK_SCORE_THRESHOLD,
    SCORE_THRESHOLD,
)
from prompt.legal_system_prompt import LEGAL_SYSTEM_PROMPT

load_dotenv()


def get_collection_names() -> List[str]:
    """Return only English collections so this script ignores Nepali chunks."""
    raw = COLLECTION_NAME
    names = list(raw) if isinstance(raw, list) else [raw]
    english_names = [
        name
        for name in names
        if "en" in str(name).lower() or "english" in str(name).lower()
    ]
    return english_names or ["nepal_laws_en"]


def is_english_chunk(value: Any) -> bool:
    """Checks whether a payload chunk is predominantly English text."""
    if value is None:
        return False

    text = unicodedata.normalize("NFC", str(value)).strip()
    if not text:
        return False

    devanagari_chars = len(re.findall(r"[\u0900-\u097F]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    total_chars = len(text)

    if total_chars == 0:
        return False

    # Allow some legal references to Nepali terms, but reject content that is primarily Devanagari.
    return latin_chars > 0 and (devanagari_chars / total_chars) < 0.25


embedder = SentenceTransformer("BAAI/bge-m3")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="openrouter/free",
    temperature=0.1,
    default_headers={
        "HTTP-Referer": "https://localhost",
        "X-Title": "Nepal Legal Assistant - English",
    },
)

COLLECTIONS = get_collection_names()


class LegalGraphState(TypedDict):
    question: str
    search_query: str
    search_queries: List[str]
    documents: List[Dict[str, Any]]
    context_str: str
    answer: str


def extract_payload_metadata(point) -> Dict[str, Any]:
    """Safely reads metadata from English payload records."""
    payload = point.payload or {}

    act_title = (
        payload.get("act_title")
        or payload.get("title")
        or payload.get("file_name", "Unknown Law").replace(".pdf", "")
    )

    section_num = (
        payload.get("section_number")
        or payload.get("rule_number")
        or payload.get("clause_label")
        or payload.get("section_no")
        or "N/A"
    )

    chapter = (
        payload.get("chapter")
        or payload.get("part")
        or payload.get("category")
        or "N/A"
    )

    content_text = (
        payload.get("content_text")
        or payload.get("text")
        or payload.get("chunk_text")
        or ""
    )
    parent_content = payload.get("parent_content_text") or payload.get("parent_text")
    title = payload.get("title") or payload.get("doc_type") or ""

    return {
        "act_title": act_title,
        "chapter": chapter,
        "section_number": section_num,
        "article_number": payload.get("article_number"),
        "title": title,
        "content_text": content_text,
        "parent_content_text": parent_content,
        "metadata_header": metadata_header(payload),
        "score": getattr(point, "score", 1.0),
    }


def prepare_query_node(state: LegalGraphState) -> Dict[str, Any]:
    """Build the query using the original English user question."""
    user_query = (state["question"] or "").strip()
    search_queries = [user_query] if user_query else []
    return {
        "search_query": " | ".join(search_queries),
        "search_queries": search_queries,
    }


def retrieve_node(state: LegalGraphState) -> Dict[str, Any]:
    """Search only English collections and discard non-English chunks."""
    question = state["search_query"] or state["question"]
    search_queries = state.get("search_queries") or [question]

    if state["question"] not in search_queries:
        search_queries.insert(0, state["question"])

    search_results = []

    for collection_name in COLLECTIONS:
        if not qdrant.collection_exists(collection_name):
            continue

        collection_info = qdrant.get_collection(collection_name)
        configured_vectors = collection_info.config.params.vectors
        vector_name = None
        if isinstance(configured_vectors, dict):
            vector_name = next(iter(configured_vectors))

        for search_text in search_queries:
            if not search_text:
                continue
            query_vector = embedder.encode(search_text).tolist()
            points = qdrant.query_points(
                collection_name=collection_name,
                query=query_vector,
                using=vector_name,
                limit=20,
            ).points
            for point in points:
                payload = point.payload or {}
                text_field = (
                    payload.get("content_text")
                    or payload.get("text")
                    or payload.get("chunk_text")
                    or ""
                )
                if not is_english_chunk(text_field):
                    continue
                search_results.append(
                    (collection_name, point, getattr(point, "score", 1.0))
                )

    search_results.sort(key=lambda item: item[2], reverse=True)

    seen_documents = set()
    collected = []

    for collection_name, point, _ in search_results:
        payload = point.payload or {}
        chunk_data = extract_payload_metadata(point)
        content_text = (
            chunk_data["content_text"] or chunk_data["parent_content_text"] or ""
        )

        if not is_english_chunk(content_text):
            continue

        title_for_filter = " ".join(
            str(payload.get(key) or "")
            for key in ("act_title", "title_np", "title", "file_name")
        ).lower()

        if any(
            marker in title_for_filter for marker in ("लोप", "खारेज", "अध्यादेश", "ऐन")
        ):
            if "act" not in title_for_filter and "law" not in title_for_filter:
                continue

        document_key = (
            collection_name,
            payload.get("file_name"),
            payload.get("section_number")
            or payload.get("rule_number")
            or payload.get("article_number")
            or payload.get("schedule_number")
            or payload.get("title"),
            payload.get("doc_type"),
        )
        if document_key in seen_documents:
            continue
        seen_documents.add(document_key)

        chunk_data["collection_name"] = collection_name
        collected.append(chunk_data)

        if len(collected) >= 12:
            break

    if not collected:
        for collection_name in COLLECTIONS:
            if not qdrant.collection_exists(collection_name):
                continue
            offset = None
            while True:
                records, next_offset = qdrant.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for record in records:
                    payload = record.payload or {}
                    content_text = (
                        payload.get("content_text")
                        or payload.get("text")
                        or payload.get("chunk_text")
                        or ""
                    )
                    if not is_english_chunk(content_text):
                        continue
                    chunk_data = extract_payload_metadata(record)
                    chunk_data["collection_name"] = collection_name
                    collected.append(chunk_data)
                    if len(collected) >= 6:
                        break
                if len(collected) >= 6 or next_offset is None:
                    break
                offset = next_offset
            if collected:
                break

    if not collected:
        return {"documents": []}

    return {"documents": collected[:12]}


def format_context_node(state: LegalGraphState) -> Dict[str, Any]:
    """Format only the English chunks into a context block for the LLM."""
    docs = state["documents"]
    if not docs:
        return {"context_str": ""}

    blocks = []
    for idx, doc in enumerate(docs, start=1):
        clean_text = doc.get("parent_content_text") or doc.get("content_text") or ""
        clean_text = unicodedata.normalize("NFC", clean_text)
        block = (
            f"[{idx}] {doc['metadata_header']} | Collection: {doc['collection_name']} | Law: {doc['act_title']} | "
            f"Chapter: {doc['chapter']} | Article: {doc['article_number'] or 'N/A'} | "
            f"Section/Rule: {doc['section_number']}\n"
            f"Title: {doc['title']}\n"
            f"Content: {clean_text}\n"
        )
        blocks.append(block)

    return {"context_str": "\n\n".join(blocks)}


def generate_answer_node(state: LegalGraphState) -> Dict[str, Any]:
    """Generate an answer strictly from English legal chunks."""
    question = state["question"]
    context = state["context_str"]

    if not state["documents"]:
        return {
            "answer": "The provided legal context does not contain information regarding this query."
        }

    messages = [
        SystemMessage(content=LEGAL_SYSTEM_PROMPT),
        HumanMessage(
            content=f"<context>\n{context}\n</context>\n\nUSER QUESTION: {question}"
        ),
    ]

    response = llm.invoke(messages)
    raw_answer = str(response.content)
    cleaned_answer = re.sub(
        r"^User Safety:\s*\w+\s*", "", raw_answer, flags=re.IGNORECASE
    ).strip()
    return {"answer": cleaned_answer}


def build_english_rag_graph():
    workflow = StateGraph(LegalGraphState)
    workflow.add_node("prepare_query", prepare_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("format_context", format_context_node)
    workflow.add_node("generate", generate_answer_node)

    workflow.set_entry_point("prepare_query")
    workflow.add_edge("prepare_query", "retrieve")
    workflow.add_edge("retrieve", "format_context")
    workflow.add_edge("format_context", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


if __name__ == "__main__":
    app = build_english_rag_graph()

    print("=========================")
    print(" Nepal Legal RAG Chatbot")
    print("=========================\n")

    while True:
        user_input = input("User Query: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        if not user_input:
            continue

        initial_state = {
            "question": user_input,
            "search_query": "",
            "search_queries": [],
            "documents": [],
            "context_str": "",
            "answer": "",
        }

        result = app.invoke(initial_state)
        print("\n--- RESPONSE ---")
        print(result["answer"])
        print("-" * 50 + "\n")
