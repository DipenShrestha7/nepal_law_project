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
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

try:
    from chatbot.retrieval_utils_en import metadata_header
except ImportError:
    # pyrefly: ignore [missing-import]
    from chatbot.retrieval_utils_en import metadata_header  #
# pyrefly: ignore [missing-import]
from chatbot.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    OPENROUTER_API_KEY,
    FALLBACK_SCORE_THRESHOLD,
    SCORE_THRESHOLD,
)

# pyrefly: ignore [missing-import]
from chatbot.prompt.legal_system_prompt import LEGAL_SYSTEM_PROMPT_EN
from chatbot.models.model import StatutoryAnchors, LegalGraphState

load_dotenv()


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
sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
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

COLLECTIONS = ["nepal_laws_en_hybrid"]
MAX_SCHEDULE_CONTEXT_CHARS = 12000


# ==========================================
# 1. STATUTORY ANCHOR & STRUCTURE EXTRACTION
# ==========================================
def extract_statutory_anchors(query: str) -> StatutoryAnchors:
    text = unicodedata.normalize("NFC", query).strip()

    # Article Anchor (e.g. Article 27, Art. 27, 27th Article)
    article_numbers = [
        int(number)
        for match in re.finditer(
            r"\b(?:articles?|art\.?)\s*(?:no\.?|number)?\s*[-:]?\s*(\d{1,3}(?:\s*(?:/|,|and)\s*(?:articles?|art\.?)?\s*\d{1,3})*)",
            text,
            re.IGNORECASE,
        )
        for number in re.findall(r"\d{1,3}", match.group(1))
    ]
    article_num = article_numbers[0] if article_numbers else None

    # Section Anchor (e.g. Section 305, Sec 11, Section No. 11)
    section_numbers = [
        int(match.group(1))
        for match in re.finditer(
            r"\b(?:section|sec\.?)\s*(?:no\.?|number)?\s*[-:]?\s*(\d{1,3})\b",
            text,
            re.IGNORECASE,
        )
    ]
    section_num = section_numbers[0] if section_numbers else None

    # Schedule Anchor (Flexible regex: Schedule 8, SCHEDULE-8, Schedule 8:, Schedule No. 8)
    schedule_numbers = [
        int(match.group(1))
        for match in re.finditer(
            r"\b(?:schedule|sched\.?)\s*(?:[-–—:]|no\.?|number)?\s*(\d{1,2})\b",
            text,
            re.IGNORECASE,
        )
    ]
    schedule_num = schedule_numbers[0] if schedule_numbers else None

    # Detect if query asks how constitutional schedules (5, 6, 7, 8, 9) are operationalized in enabling acts
    is_const_sched = bool(
        any(number in [5, 6, 7, 8, 9] for number in schedule_numbers)
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
        "article_numbers": article_numbers,
        "section_numbers": section_numbers,
        "schedule_numbers": schedule_numbers,
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


def build_query_filter(anchors: StatutoryAnchors):
    required_conditions = [
        models.FieldCondition(key="language", match=models.MatchValue(value="en"))
    ]
    anchor_conditions = []

    if anchors.get("target_act") == "Constitution":
        required_conditions.append(
            models.FieldCondition(
                key="act_title",
                match=models.MatchValue(value="Constitution of Nepal"),
            )
        )

    for field_name, plural_name in (
        ("article_number", "article_numbers"),
        ("section_number", "section_numbers"),
        ("schedule_number", "schedule_numbers"),
    ):
        values = anchors.get(plural_name) or []
        if not values and anchors.get(field_name) is not None:
            values = [anchors[field_name]]
        if values:
            anchor_conditions.append(
                models.FieldCondition(key=field_name, match=models.MatchAny(any=values))
            )

    return models.Filter(
        must=required_conditions,
        should=anchor_conditions or None,
    )


def retrieve_node(state: LegalGraphState) -> Dict[str, Any]:
    anchors = state.get("anchors") or extract_statutory_anchors(state["question"])
    search_queries = state.get("search_queries") or [state["question"]]
    collected_results = []
    seen_keys: Set[Tuple[str, Any, Any, Any]] = set()

    # Filter in Qdrant so its payload indexes can narrow the hybrid search.
    query_filter = build_query_filter(anchors)

    for collection_name in COLLECTIONS:
        if not qdrant.collection_exists(collection_name):
            continue

        for search_text in search_queries:
            dense_vector = embedder.encode(search_text).tolist()
            sparse_vector = next(iter(sparse_embedder.embed([search_text])))
            points = qdrant.query_points(
                collection_name=collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector,
                        using="dense",  # Match your vector name ("dense" or "text-dense")
                        limit=15,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_vector.indices.tolist(),
                            values=sparse_vector.values.tolist(),
                        ),
                        using="sparse",  # Match your vector name ("sparse" or "text-sparse")
                        limit=15,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=15,
            ).points

            for point in points:
                collected_results.append(
                    (collection_name, point, getattr(point, "score", 1.0))
                )

    # Step C: Filtering & Structure-Aware Ranking
    collected_results.sort(key=lambda item: item[2], reverse=True)
    final_docs = []

    # CHANGE 1: Track counts per article/schedule to ensure context diversity
    seen_provisions: Dict[str, int] = {}

    for collection_name, point, score in collected_results:
        payload = point.payload or {}
        chunk_data = extract_payload_metadata(point)
        content_text = (
            chunk_data["content_text"] or chunk_data["parent_content_text"] or ""
        )

        if (
            chunk_data.get("doc_type") == "schedule"
            and len(content_text) > MAX_SCHEDULE_CONTEXT_CHARS
        ):
            print(
                f"Skipping oversized malformed schedule payload from {collection_name}; "
                "re-run constitution ingestion to refresh it."
            )
            continue

        if not is_english_chunk(content_text):
            continue

        # Suppress tail administrative schedules of Acts when querying Constitutional Schedules
        if anchors.get("is_constitutional_schedule_query"):
            doc_type = str(
                chunk_data.get("doc_type") or payload.get("doc_type") or ""
            ).lower()
            act_title = str(
                chunk_data.get("act_title") or payload.get("act_title") or ""
            ).lower()
            if "constitution" not in act_title and (
                doc_type == "schedule" or chunk_data.get("schedule_number") is not None
            ):
                continue

        # CHANGE 1 (Continued): Enforce Diversity Cap (Max 2 chunks per Article or Schedule)
        art_num = chunk_data.get("article_number")
        sched_num = chunk_data.get("schedule_number")
        prov_key = (
            f"art_{art_num}"
            if art_num is not None
            else (f"sched_{sched_num}" if sched_num is not None else "other")
        )

        if prov_key != "other" and seen_provisions.get(prov_key, 0) >= 2:
            continue

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
        seen_provisions[prov_key] = seen_provisions.get(prov_key, 0) + 1

        chunk_data["collection_name"] = collection_name
        final_docs.append(chunk_data)

        # CHANGE 2: Reduced maximum limit from 12 to 8 to avoid token context overflow
        if len(final_docs) >= 8:
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
        header_fields = [
            f"Authority: {doc['act_title']}",
            f"Document: {doc['title'] or doc['doc_type'] or 'Provision'}",
        ]
        if doc["article_number"] is not None:
            header_fields.append(f"Article: {doc['article_number']}")
        if doc["section_number"] != "N/A":
            header_fields.append(f"Section: {doc['section_number']}")
        if doc["schedule_number"] is not None:
            header_fields.append(f"Schedule: {doc['schedule_number']}")
        if doc["chapter"] != "N/A":
            header_fields.append(f"Chapter/Part: {doc['chapter']}")

        block = f"[{idx}] {' | '.join(header_fields)}\nContent: {clean_text}\n"
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
        SystemMessage(content=LEGAL_SYSTEM_PROMPT_EN),
        HumanMessage(
            content=f"<context>\n{context}\n</context>\n\nUSER QUESTION: {question}"
        ),
    ]
    response = llm.invoke(messages)
    content = str(response.content or "").strip()
    if not content and hasattr(response, "additional_kwargs"):
        content = str(
            response.additional_kwargs.get("reasoning", "")
            or response.additional_kwargs.get("thinking", "")
        ).strip()

    cleaned_answer = re.sub(
        r"^User Safety:\s*\w+\s*", "", content, flags=re.IGNORECASE
    ).strip()

    if not cleaned_answer:
        cleaned_answer = "Error: The model generated an empty response. Please check your API connection or select an explicit OpenRouter model."

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
                "article_numbers": [],
                "section_numbers": [],
                "schedule_numbers": [],
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
