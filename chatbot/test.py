import re
import sys
from typing import Any, Dict, List, TypedDict
import unicodedata

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

# pyrefly: ignore [missing-import]
from prompt.legal_system_prompt import LEGAL_SYSTEM_PROMPT

try:
    from .retrieval_utils import metadata_header
except ImportError:
    # pyrefly: ignore [missing-import]
    from retrieval_utils import metadata_header

# pyrefly: ignore [missing-import]
from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    OPENROUTER_API_KEY,
    COLLECTION_NAME,
    FALLBACK_SCORE_THRESHOLD,
    SCORE_THRESHOLD,
)

embedder = SentenceTransformer("BAAI/bge-m3")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="openrouter/free",
    temperature=0.1,
    default_headers={
        "HTTP-Referer": "https://localhost",
        "X-Title": "Nepal Legal Assistant",
    },
)

COLLECTIONS = (
    list(COLLECTION_NAME) if isinstance(COLLECTION_NAME, list) else [COLLECTION_NAME]
)

EXCLUDED_TITLE_MARKERS = (
    "repealed",
    "expired ordinance",
    "लोप",
    "खारेज",
    "पुरानो ऐन",
    "अध्यादेश (समाप्त)",
)
_reranker = None
_reranker_loaded = False


def get_reranker():
    global _reranker, _reranker_loaded
    if _reranker_loaded:
        return _reranker
    _reranker_loaded = True
    try:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        _reranker = None
    return _reranker


LEGAL_TERM_CORRECTIONS = {
    "जुर्माना": "जरिबाना",
    "कट-ऑफ": "अन्तिम सीमा",
    "फाइनान्सियल": "आर्थिक",
}


def normalize_legal_search_text(value: str) -> str:
    """Normalizes legal query text for title matching across English and Nepali."""
    if value is None:
        return ""

    text = unicodedata.normalize("NFC", value).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("&", " and ")
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[^\w\s\u0900-\u097f-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_act_title_query(question: str) -> bool:
    """Returns true when the user is asking for a specific act title, not a Constitution article."""
    text = normalize_legal_search_text(question)
    if not text:
        return False

    if re.search(
        r"\b(?:money bill|budget bill|article\s*110|अनुच्छेद\s*११०|धन\s*(?:सम्बन्ध|संबंध)\s*विधेयक|धन\s*(?:सम्बन्ध|संबंध)\s*संबंधी\s*विधेयक)\b",
        text,
    ):
        return False

    if re.search(r"\b(?:constitution|article|अनुच्छेद|schedule|अनुसूची)\b", text):
        return False

    return bool(re.search(r"\b(?:act|ऐन)\b", text))


def score_act_title_match(question: str, title: str) -> float:
    """Scores whether a retrieved title matches the user's act-title query."""
    query = normalize_legal_search_text(question)
    title_text = normalize_legal_search_text(title)
    if not query or not title_text:
        return 0.0

    if query in title_text or title_text in query:
        return 2.0

    if "financial" in query and ("financial" in title_text or "आर्थिक" in title_text):
        return 2.0
    if "economic" in query and ("economic" in title_text or "आर्थिक" in title_text):
        return 2.0

    query_tokens = set(token for token in query.split() if len(token) > 1)
    title_tokens = set(token for token in title_text.split() if len(token) > 1)
    if not query_tokens or not title_tokens:
        return 0.0

    overlap = len(query_tokens & title_tokens)
    if overlap:
        return overlap / max(len(query_tokens), len(title_tokens))
    return 0.0


def collect_act_title_matches(
    question: str, collection_names: List[str]
) -> List[tuple[str, Any, float]]:
    """Scans collections for explicit act titles matching the query before vector ranking."""
    if not detect_act_title_query(question):
        return []

    matches: List[tuple[str, Any, float]] = []
    for collection_name in collection_names:
        if not qdrant.collection_exists(collection_name):
            continue

        offset = None
        while True:
            records, next_offset = qdrant.scroll(
                collection_name=collection_name,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for record in records:
                payload = record.payload or {}
                title_text = (
                    payload.get("act_title")
                    or payload.get("title_np")
                    or payload.get("title")
                    or ""
                )
                if not title_text:
                    continue

                is_act_like = (
                    payload.get("category") == "Act"
                    or "ऐन" in str(title_text)
                    or "act" in str(title_text).lower()
                )
                if not is_act_like:
                    continue

                score = score_act_title_match(question, str(title_text))
                if score > 0:
                    matches.append((collection_name, record, score))

            if next_offset is None:
                break
            offset = next_offset

    matches.sort(key=lambda item: item[2], reverse=True)
    return matches[:5]


class LegalGraphState(TypedDict):
    question: str
    search_query: str
    search_queries: List[str]
    hyde_query: str
    article_number: int | None
    documents: List[Dict[str, Any]]
    context_str: str
    answer: str


# HELPER UTILITIES
def extract_article_number(question: str) -> int | None:
    """Extracts an explicit Constitution article number from English or Nepali text."""
    normalized = unicodedata.normalize("NFC", question)
    normalized = normalized.translate(
        str.maketrans(
            {
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
        )
    )

    article_match = re.search(
        r"(?<!\w)(?:article|art\.?)\s*(?:no\.?|number)?\s*[-:]?\s*(\d{1,3})(?!\w)",
        normalized,
        flags=re.IGNORECASE,
    ) or re.search(
        r"(?<!\w)अनुच्छेद\s*(?:नम्बर|संख्या)?\s*[-:]?\s*(\d{1,3})(?!\w)",
        normalized,
    )
    if article_match:
        return int(article_match.group(1))

    ordinal_match = re.search(
        r"(?<!\w)(\d{1,3})(?:st|nd|rd|th)\s+article(?!\w)",
        normalized,
        flags=re.IGNORECASE,
    )
    if ordinal_match:
        return int(ordinal_match.group(1))

    # Nepali commonly places the ordinal marker before the noun:
    # "२५औं अनुच्छेद", "२५ औं अनुच्छेद", or "२५औँ अनुच्छेद".
    nepali_ordinal_match = re.search(
        r"(?<!\w)(\d{1,3})\s*(?:औं|औँ)(?:को)?\s*अनुच्छेद",
        normalized,
    )
    return int(nepali_ordinal_match.group(1)) if nepali_ordinal_match else None


def preprocess_devanagari_text(text: str) -> str:
    """Repairs Devanagari Unicode ligatures, matra transpositions, and space glitches."""
    if not text:
        return ""

    # 1. Unicode NFC Normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Fix transposed short-I matra (ि) occurring before consonants/conjuncts
    text = re.sub(r"ि([\u0915-\u0939](?:्[\u0915-\u0939])*)", r"\1ि", text)

    # 3. Fix detached halants (्) followed by spaces
    text = re.sub(r"्\s+", "्", text)

    # 4. Remove artificial spaces inserted between letters and matras
    text = re.sub(r"(?<=\u0900-\u097F)\s+(?=[\u0902-\u094D])", "", text)

    return text.strip()


def clean_llm_output(text: str) -> str:
    """Strips OpenRouter guardrail metadata headers from completion strings."""
    if not text:
        return ""
    return re.sub(r"^User Safety:\s*\w+\s*", "", text, flags=re.IGNORECASE).strip()


def find_title_matches(
    question: str, records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Ranks in-memory title records for deterministic exact-title retrieval."""
    ranked = []
    for record in records:
        title = (
            record.get("act_title")
            or record.get("title_np")
            or record.get("title")
            or ""
        )
        score = score_act_title_match(question, str(title))
        if score > 0:
            ranked.append((score, record))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in ranked]


# GRAPH NODES
def prepare_query_node(state: LegalGraphState) -> Dict[str, Any]:
    """Uses one bounded HyDE call to turn facts into formal retrieval concepts."""
    user_query = state["question"].strip()
    article_number = extract_article_number(user_query)

    expansion_prompt = f"""
Analyze this user's Nepal-law question as a fact pattern. Generate retrieval queries,
not a legal answer. Infer the conduct, possible legal issues, remedies, and procedure
that could be relevant, without claiming that any offense has been established.

First classify the legal subject matter. Distinguish an ordinary dispute between
private persons (including nuisance, harassment, humiliation, offensive conduct,
injury, damage, compensation, or local dispute resolution) from a sector-regulation
question such as water-resource allocation, hydropower, irrigation licensing, or
environmental management. Do not let the name of a physical object alone determine
the legal domain.

    Return exactly 3 to 5 short formal legal concepts, one per line, with no numbering or explanation.
    These are hypothetical statutory concepts for retrieval, not conclusions about liability.
Include:
- the original facts in concise English;
- possible legal concepts in English;
- equivalent formal Nepali legal terms used in Nepal Acts, Codes, Rules, or Regulations;
- complaint, evidence, compensation, or procedure terms when relevant.
- the most likely governing legal domains, plus close statutory phrases for the conduct;
- do not generate sector-regulation queries unless the facts actually concern that sector.
- for public physical-conduct incidents, consider statutory concepts such as offensive
    substances, indecent behavior, nuisance, insult or humiliation, tort liability,
    emotional distress, compensation, and local judicial dispute resolution when the facts fit.
- when the facts describe one private person directing offensive conduct at another,
    generate targeted searches for the Muluki Criminal Code, 2074 and its indecent-
    behavior/offensive-substance provisions; the Muluki Civil Code, 2074 and its
    tort, emotional-distress, and compensation provisions; and the Local Government
    Operation Act, 2074 and Judicial Committee/local nuisance jurisdiction. Do not
    assume a specific section number; discover candidate provisions from the indexed
    legal data and verify them against the retrieved context.
- do not make constitutional writs, the Supreme Court, or extraordinary remedies a
    default search for a private neighborhood incident unless the facts involve state
    action or the user expressly asks about constitutional jurisdiction.

USER QUESTION:
{user_query}
""".strip()

    search_queries = [user_query]
    hyde_concepts = []
    try:
        expansion = str(llm.invoke(expansion_prompt).content)
        for line in expansion.splitlines():
            query = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if (
                query
                and query.lower() != user_query.lower()
                and query not in search_queries
            ):
                hyde_concepts.append(query)
            hyde_concepts = hyde_concepts[:5]
            search_queries.extend(hyde_concepts)
    except Exception:
        pass

    search_query = " | ".join(search_queries)

    # Keep the original question in state so answer-language detection sees the
    # user's language rather than the translated retrieval query.
    return {
        "search_query": search_query,
        "search_queries": search_queries,
        "hyde_query": " | ".join(hyde_concepts),
        "article_number": article_number,
    }


def extract_payload_metadata(point) -> Dict[str, Any]:
    """Safely retrieves chunk fields across varying document payload schemas."""
    payload = point.payload or {}

    # 1. Fallback chain for Document / Act Title
    act_title = (
        payload.get("act_title")
        or payload.get("title_np")
        or payload.get("file_name", "Unknown Law").replace(".pdf", "")
    )

    # 2. Fallback chain for Section / Rule / Clause Number
    section_num = (
        payload.get("section_number")
        or payload.get("rule_number")
        or payload.get("clause_label")
        or payload.get("section_no")
        or "N/A"
    )

    article_number = payload.get("article_number")

    # 3. Fallback chain for Chapter / Part
    chapter = (
        payload.get("chapter")
        or payload.get("part")
        or payload.get("category")
        or "N/A"
    )

    # 4. Fallback chain for Main Content Text
    content_text = (
        payload.get("content_text")
        or payload.get("text")
        or payload.get("chunk_text")
        or ""
    )
    parent_content = payload.get("parent_content_text") or payload.get("parent_text")

    # 5. Section / Chunk Title
    title = payload.get("title") or payload.get("doc_type") or ""

    return {
        "act_title": act_title,
        "chapter": chapter,
        "section_number": section_num,
        "article_number": article_number,
        "title": title,
        "content_text": content_text,
        "parent_content_text": parent_content,
        "metadata_header": metadata_header(payload),
        "score": getattr(point, "score", 1.0),
    }


def retrieve_node(state: LegalGraphState) -> Dict[str, Any]:
    question = state["search_query"]
    search_queries = state.get("search_queries") or [question]
    if state["question"] not in search_queries:
        search_queries.insert(0, state["question"])
    article_number = state.get("article_number")

    search_results = []
    if detect_act_title_query(state["question"]):
        title_matches = collect_act_title_matches(state["question"], COLLECTIONS)
        if title_matches:
            for collection_name, record, title_score in title_matches:
                search_results.append((collection_name, record, title_score))

    if not search_results:
        for collection_name in COLLECTIONS:
            if not qdrant.collection_exists(collection_name):
                continue

            if article_number is not None:
                # Do not depend on a Qdrant payload index here. Existing deployments
                # may contain the article field without an index, and exact article
                # lookup must still work across legacy and current collections.
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
                        if (
                            payload.get("doc_type") == "article"
                            and payload.get("article_number") == article_number
                        ):
                            search_results.append((collection_name, record, 1.0))

                    if next_offset is None:
                        break
                    offset = next_offset
                continue

            collection_info = qdrant.get_collection(collection_name)
            configured_vectors = collection_info.config.params.vectors
            vector_name = None
            if isinstance(configured_vectors, dict):
                vector_name = next(iter(configured_vectors))

            for search_text in search_queries:
                query_vector = embedder.encode(search_text).tolist()
                points = qdrant.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    using=vector_name,
                    limit=20,
                ).points
                for point in points:
                    search_results.append(
                        (collection_name, point, getattr(point, "score", 1.0))
                    )

    search_results.sort(key=lambda item: item[2], reverse=True)

    # Optional cross-encoder reranking keeps deployments without the model usable.
    try:
        reranker = get_reranker()
        if reranker is None:
            raise RuntimeError("Cross-encoder is unavailable")
        candidates = search_results[:20]
        pairs = [
            (state["question"], extract_payload_metadata(point)["content_text"])
            for _, point, _ in candidates
        ]
        scores = reranker.predict(pairs)
        search_results[: len(candidates)] = [
            item
            for item, _ in sorted(
                zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True
            )
        ]
    except Exception:
        pass

    def collect_documents(minimum_score: float) -> List[Dict[str, Any]]:
        documents = []
        seen_documents = set()
        for collection_name, point, point_score in search_results:
            if point_score < minimum_score:
                continue

            # Extract normalized metadata fields regardless of schema differences
            chunk_data = extract_payload_metadata(point)
            payload = point.payload or {}
            title_for_filter = " ".join(
                str(payload.get(key) or "")
                for key in ("act_title", "title_np", "title", "file_name")
            ).lower()
            if any(marker in title_for_filter for marker in EXCLUDED_TITLE_MARKERS):
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
            documents.append(chunk_data)
            if len(documents) >= 12:
                break
        return documents

    docs = collect_documents(SCORE_THRESHOLD)
    if not docs:
        # Broad fact patterns can have lower cosine scores than exact legal terms.
        # Keep a small candidate set so the grounded LLM can judge relevance instead
        # of converting a retrieval miss into a false "no data" response.
        docs = collect_documents(FALLBACK_SCORE_THRESHOLD)

    return {"documents": docs}


def format_context_node(state: LegalGraphState) -> Dict[str, Any]:
    """Node 2: Formats retrieved chunks into a clean prompt string with active Devanagari repair."""
    docs = state["documents"]

    if not docs:
        return {"context_str": ""}

    formatted_blocks = []
    for idx, c in enumerate(docs, start=1):
        # Run active character/matra cleaning pass on retrieved text
        clean_text = preprocess_devanagari_text(
            c.get("parent_content_text") or c["content_text"]
        )

        block = (
            f"[{idx}] {c['metadata_header']} | Collection: {c['collection_name']} | Law: {c['act_title']} | "
            f"Chapter: {c['chapter']} | Article: {c['article_number'] or 'N/A'} | "
            f"Section/Rule: {c['section_number']}\n"
            f"Title: {c['title']}\n"
            f"Content: {clean_text}\n"
        )
        formatted_blocks.append(block)

    return {"context_str": "\n\n".join(formatted_blocks)}


def generate_answer_node(state: LegalGraphState) -> Dict[str, Any]:
    """Node 3: Prompts OpenRouter LLM with strictly grounded context."""
    question = state["question"]
    context = state["context_str"]

    if not state["documents"]:
        return {
            "answer": "The requested legal provision was not found in the Nepali legal database."
        }

    messages = [
        SystemMessage(content=LEGAL_SYSTEM_PROMPT),
        HumanMessage(
            content=f"<context>\n{context}\n</context>\n\nUSER QUESTION: {question}"
        ),
    ]

    response = llm.invoke(messages)
    raw_answer = str(response.content)

    # Clean OpenRouter system headers immediately
    cleaned_answer = clean_llm_output(raw_answer)

    return {"answer": cleaned_answer}


def sanitize_legal_output_node(state: LegalGraphState) -> Dict[str, Any]:
    """Node 4: Post-processes output to enforce exact Nepali legal terminology."""
    text = state["answer"]

    # 1. Replace terminology drifts (e.g., 'जुर्माना' -> 'जरिबाना')
    for bad_term, good_term in LEGAL_TERM_CORRECTIONS.items():
        text = text.replace(bad_term, good_term)

    # 2. Convert 'धारा' to 'दफा' when referencing Acts (and not the Constitution)
    is_act = any("ऐन" in doc.get("act_title", "") for doc in state["documents"])
    if is_act:
        text = re.sub(r"धारा\s*([०-९\d]+)", r"दफा \1", text)

    return {"answer": text}


# WORKFLOW BUILDER
def build_legal_rag_graph():
    workflow = StateGraph(LegalGraphState)  # pyrefly: ignore [bad-specialization]

    # 1. Add nodes
    workflow.add_node("prepare_query", prepare_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("format_context", format_context_node)
    workflow.add_node("generate", generate_answer_node)
    workflow.add_node("sanitize_output", sanitize_legal_output_node)

    # 2. Set entry point
    workflow.set_entry_point("prepare_query")

    # 3. Connect sequential pipeline
    workflow.add_edge("prepare_query", "retrieve")
    workflow.add_edge("retrieve", "format_context")
    workflow.add_edge("format_context", "generate")
    workflow.add_edge("generate", "sanitize_output")
    workflow.add_edge("sanitize_output", END)

    return workflow.compile()


# EXECUTION LOOP
if __name__ == "__main__":
    app = build_legal_rag_graph()

    print("=========================")
    print(" Nepal Legal AI Chatbot")
    print("=========================\n")

    while True:
        user_input = input("User Query: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        if user_input:
            initial_state = {
                "question": user_input,
                "search_query": "",
                "search_queries": [],
                "article_number": None,
                "documents": [],
                "hyde_query": "",
                "context_str": "",
                "answer": "",
            }

            result = app.invoke(initial_state)

            print("\n--- RESPONSE ---")
            print(result["answer"])
            print("-" * 50 + "\n")
