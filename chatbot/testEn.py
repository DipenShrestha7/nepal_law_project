import re
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

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
    # pyrefly: ignore [missing-import]
    from retrieval_utils import metadata_header  #
# pyrefly: ignore [missing-import]
from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    OPENROUTER_API_KEY,
    COLLECTION_NAME,
    FALLBACK_SCORE_THRESHOLD,
    SCORE_THRESHOLD,
)

# pyrefly: ignore [missing-import]
from prompt.legal_system_prompt import LEGAL_SYSTEM_PROMPT

load_dotenv()


def get_collection_names() -> List[str]:
    raw = COLLECTION_NAME
    names = list(raw) if isinstance(raw, list) else [raw]
    english_names = [
        name
        for name in names
        if "en" in str(name).lower() or "english" in str(name).lower()
    ]
    return english_names or ["nepal_laws_en"]


def is_english_chunk(value: Any) -> bool:
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


class StatutoryAnchors(TypedDict):
    article_number: Optional[int]
    section_number: Optional[int]
    schedule_number: Optional[int]
    target_act: Optional[str]
    is_constitutional_schedule_query: bool


class LegalGraphState(TypedDict):
    question: str
    search_query: str
    search_queries: List[str]
    anchors: StatutoryAnchors
    documents: List[Dict[str, Any]]
    context_str: str
    answer: str


# ==========================================
# 1. STATUTORY ANCHOR & STRUCTURE EXTRACTION
# ==========================================
def extract_statutory_anchors(query: str) -> StatutoryAnchors:
    text = unicodedata.normalize("NFC", query).strip()

    # Article Anchor (e.g. Article 27, Art. 27, 27th Article)
    art_match = re.search(
        r"\b(?:article|art\.?)\s*(?:no\.?|number)?\s*[-:]?\s*(\d{1,3})\b",
        text,
        re.IGNORECASE,
    )
    article_num = int(art_match.group(1)) if art_match else None

    # Section Anchor (e.g. Section 305, Sec 11, Section No. 11)
    sec_match = re.search(
        r"\b(?:section|sec\.?)\s*(?:no\.?|number)?\s*[-:]?\s*(\d{1,3})\b",
        text,
        re.IGNORECASE,
    )
    section_num = int(sec_match.group(1)) if sec_match else None

    # Schedule Anchor (Flexible regex: Schedule 8, SCHEDULE-8, Schedule 8:, Schedule No. 8)
    sched_match = re.search(
        r"\b(?:schedule|sched\.?)\s*(?:[-–—:]|no\.?|number)?\s*(\d{1,2})\b",
        text,
        re.IGNORECASE,
    )
    schedule_num = int(sched_match.group(1)) if sched_match else None

    # Detect if query asks how constitutional schedules (5, 6, 7, 8, 9) are operationalized in enabling acts
    is_const_sched = bool(
        schedule_num in [5, 6, 7, 8, 9]
        and re.search(
            r"\b(?:operational|implement|local government|act|enabl|power|function|duty)\b",
            text,
            re.IGNORECASE,
        )
    )

    # Detect Target Act
    target_act = None
    if re.search(r"\b(?:electronic transactions?|cyber|et act)\b", text, re.IGNORECASE):
        target_act = "Electronic Transactions Act"
    elif re.search(
        r"\b(?:criminal code|penal code|muluki criminal)\b", text, re.IGNORECASE
    ):
        target_act = "Penal Code"
    elif re.search(r"\b(?:civil code|muluki civil)\b", text, re.IGNORECASE):
        target_act = "Civil Code"
    elif re.search(
        r"\b(?:local government|local governance|lgoa)\b", text, re.IGNORECASE
    ):
        target_act = "Local Government Operation Act"
    elif re.search(r"\b(?:constitution)\b", text, re.IGNORECASE):
        target_act = "Constitution"

    return {
        "article_number": article_num,
        "section_number": section_num,
        "schedule_number": schedule_num,
        "target_act": target_act,
        "is_constitutional_schedule_query": is_const_sched,
    }


# ==========================================
# 2. CROSS-STATUTE TERMINOLOGY & QUERY NODE
# ==========================================
def prepare_query_node(state: LegalGraphState) -> Dict[str, Any]:
    user_query = (state["question"] or "").strip()
    anchors = extract_statutory_anchors(user_query)
    search_queries = [user_query]

    # Cross-statute legal concept expansion prompt
    expansion_prompt = f"""
You are a legal search query optimizer for Nepal Law (both Constitution, Specialized Acts like Electronic Transactions Act, Local Government Operation Act, and General Codes like National Penal Code / Muluki Criminal Code & Civil Code).

User Question: {user_query}

Generate 3-4 targeted English legal retrieval queries to locate relevant statutory provisions.
CRITICAL RULES:
1. If the query uses modern cyber/tech terms (e.g. cyber-defamation, online harassment, computer fraud, online financial scam, doxxing):
   - Query 1: Specialized terms in the Electronic Transactions Act, 2063 (e.g. computer source code, illegal access, data distortion).
   - Query 2: Equivalent traditional penal offenses in the National Penal (Code) Act, 2074 (e.g. libel, slander, defamation, character assassination, Section 305, 306, 307; outraging modesty, insult to modesty, intimidation; cheating, personation, fraud, forgery).
2. If the query asks about Constitutional Schedule powers (e.g. Schedule 8 local level powers) operationalized in an enabling Act (e.g. Local Government Operation Act):
   - Include substantive section concepts: "powers, functions and duties of municipality rural municipality local level Section 11 Local Government Operation Act"
3. Keep queries short, keyword-dense, and focused on statutory provisions. Output only the queries, one per line.
""".strip()

    try:
        response = llm.invoke([HumanMessage(content=expansion_prompt)])
        lines = [
            line.strip("- *").strip()
            for line in str(response.content).splitlines()
            if line.strip()
        ]
        for line in lines[:4]:
            if line and line.lower() != user_query.lower():
                search_queries.append(line)
    except Exception:
        # Fallback cross-statute term dictionary if LLM call is unavailable
        query_lower = user_query.lower()
        if any(
            w in query_lower
            for w in ["cyber defamation", "online defamation", "defamation online"]
        ):
            search_queries.append(
                "defamation libel slander reputation Section 305 Penal Code"
            )
            search_queries.append(
                "offenses against computer systems Electronic Transactions Act"
            )
        if any(
            w in query_lower
            for w in ["online harassment", "cyber harassment", "cyberstalking"]
        ):
            search_queries.append(
                "outraging modesty insult to modesty criminal intimidation Penal Code"
            )
        if any(
            w in query_lower for w in ["computer fraud", "online scam", "cyber fraud"]
        ):
            search_queries.append(
                "cheating by personation fraud criminal breach of trust Penal Code"
            )

    return {
        "search_query": " | ".join(search_queries),
        "search_queries": search_queries,
        "anchors": anchors,
    }


# ==========================================
# 3. ANCHOR-AWARE RETRIEVAL NODE
# ==========================================
def extract_payload_metadata(point) -> Dict[str, Any]:
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
    article_number = payload.get("article_number")
    schedule_number = payload.get("schedule_number")
    doc_type = payload.get("doc_type")

    # Dynamic Retrieval-Time Repair for Schedules (recovers schedules without re-ingesting)
    if schedule_number is None:
        header_sample = f"{title} {content_text[:350]}"
        sched_match = re.search(
            r"(?i)\b(?:SCHEDULE|Schedule|sched\.?)\s*(?:[-–—:]|no\.?|number)?\s*(\d{1,2})\b",
            header_sample,
        )
        if sched_match:
            schedule_number = int(sched_match.group(1))
            if not doc_type or doc_type == "section":
                doc_type = "schedule"

    return {
        "act_title": act_title,
        "chapter": chapter,
        "section_number": section_num,
        "article_number": article_number,
        "schedule_number": schedule_number,
        "doc_type": doc_type,
        "title": title,
        "content_text": content_text,
        "parent_content_text": parent_content,
        "metadata_header": metadata_header(payload),
        "score": getattr(point, "score", 1.0),
    }


def retrieve_node(state: LegalGraphState) -> Dict[str, Any]:
    anchors = state.get("anchors") or extract_statutory_anchors(state["question"])
    search_queries = state.get("search_queries") or [state["question"]]
    collected_results = []
    seen_keys: Set[Tuple[str, Any, Any, Any]] = set()

    # Step A: Exact Statutory Anchor Retrieval (solves Problem 4 & handles ingested variations)
    for collection_name in COLLECTIONS:
        if not qdrant.collection_exists(collection_name):
            continue

        if (
            anchors.get("article_number") is not None
            or anchors.get("section_number") is not None
            or anchors.get("schedule_number") is not None
        ):
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
                    matched_anchor = False

                    # Article exact match
                    if anchors.get("article_number") is not None:
                        if (
                            payload.get("doc_type") == "article"
                            and payload.get("article_number")
                            == anchors["article_number"]
                        ):
                            matched_anchor = True

                    # Section exact match
                    if anchors.get("section_number") is not None:
                        rec_sec = payload.get("section_number")
                        if str(rec_sec) == str(anchors["section_number"]):
                            matched_anchor = True

                    # Schedule exact match with dynamic retrieval-time text check
                    if anchors.get("schedule_number") is not None:
                        target_sched = anchors["schedule_number"]
                        rec_sched = payload.get("schedule_number")
                        if rec_sched is not None and str(rec_sched) == str(target_sched):
                            matched_anchor = True
                        else:
                            # Dynamic retrieval-time check on title and content_text
                            sched_regex = rf"(?i)\b(?:SCHEDULE|Schedule|sched\.?)\s*(?:[-–—:]|no\.?|number)?\s*{target_sched}\b"
                            candidate_sample = f"{payload.get('title') or ''} {str(payload.get('content_text') or '')[:350]}"
                            if re.search(sched_regex, candidate_sample):
                                matched_anchor = True

                    if matched_anchor:
                        collected_results.append(
                            (collection_name, record, 2.5)
                        )  # High priority boost

                if next_offset is None or len(collected_results) >= 10:
                    break
                offset = next_offset

    # Step B: Dense Vector Retrieval with Expanded Queries (solves Problem 1)
    for collection_name in COLLECTIONS:
        if not qdrant.collection_exists(collection_name):
            continue

        collection_info = qdrant.get_collection(collection_name)
        configured_vectors = collection_info.config.params.vectors
        vector_name = (
            next(iter(configured_vectors))
            if isinstance(configured_vectors, dict)
            else None
        )

        for search_text in search_queries:
            query_vector = embedder.encode(search_text).tolist()
            points = qdrant.query_points(
                collection_name=collection_name,
                query=query_vector,
                using=vector_name,
                limit=15,
            ).points

            for point in points:
                collected_results.append(
                    (collection_name, point, getattr(point, "score", 1.0))
                )

    # Step C: Filtering & Structure-Aware Ranking (solves Problem 3)
    collected_results.sort(key=lambda item: item[2], reverse=True)
    final_docs = []

    for collection_name, point, score in collected_results:
        payload = point.payload or {}
        chunk_data = extract_payload_metadata(point)
        content_text = (
            chunk_data["content_text"] or chunk_data["parent_content_text"] or ""
        )

        if not is_english_chunk(content_text):
            continue

        # Problem 3 fix: If user asks how Constitutional Schedule powers are operationalized in an enabling Act,
        # suppress tail administrative schedules/forms of the Act in favor of substantive sections.
        if anchors.get("is_constitutional_schedule_query"):
            doc_type = str(chunk_data.get("doc_type") or payload.get("doc_type") or "").lower()
            act_title = str(chunk_data.get("act_title") or payload.get("act_title") or "").lower()
            if "constitution" not in act_title and (doc_type == "schedule" or chunk_data.get("schedule_number") is not None):
                continue  # Suppress administrative forms / tail schedules of Acts

        doc_key = (
            collection_name,
            payload.get("file_name"),
            payload.get("section_number")
            or payload.get("article_number")
            or payload.get("schedule_number")
            or payload.get("title"),
            payload.get("doc_type"),
        )
        if doc_key in seen_keys:
            continue
        seen_keys.add(doc_key)

        chunk_data["collection_name"] = collection_name
        final_docs.append(chunk_data)
        if len(final_docs) >= 12:
            break

    return {"documents": final_docs}


def format_context_node(state: LegalGraphState) -> Dict[str, Any]:
    docs = state["documents"]
    if not docs:
        return {"context_str": ""}

    blocks = []
    for idx, doc in enumerate(docs, start=1):
        clean_text = unicodedata.normalize(
            "NFC", doc.get("parent_content_text") or doc.get("content_text") or ""
        )
        block = (
            f"[{idx}] {doc['metadata_header']} | Law: {doc['act_title']} | "
            f"Chapter/Part: {doc['chapter']} | Article: {doc['article_number'] or 'N/A'} | "
            f"Section/Rule: {doc['section_number']} | Schedule: {doc['schedule_number'] or 'N/A'}\n"
            f"Title: {doc['title']}\n"
            f"Content: {clean_text}\n"
        )
        blocks.append(block)

    return {"context_str": "\n\n".join(blocks)}


def generate_answer_node(state: LegalGraphState) -> Dict[str, Any]:
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
    cleaned_answer = re.sub(
        r"^User Safety:\s*\w+\s*", "", str(response.content), flags=re.IGNORECASE
    ).strip()
    return {"answer": cleaned_answer}


def build_english_rag_graph():
    workflow = StateGraph(LegalGraphState)  # pyrefly: ignore [bad-specialization]
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
            "anchors": {
                "article_number": None,
                "section_number": None,
                "schedule_number": None,
                "target_act": None,
                "is_constitutional_schedule_query": False,
            },
            "documents": [],
            "context_str": "",
            "answer": "",
        }
        result = app.invoke(initial_state)
        print("\n--- RESPONSE ---")
        print(result["answer"])
        print("-" * 50 + "\n")
