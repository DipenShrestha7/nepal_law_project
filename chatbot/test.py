import re
from typing import Any, Dict, List, TypedDict
import unicodedata
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from prompt.legal_system_prompt import LEGAL_SYSTEM_PROMPT
from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    OPENROUTER_API_KEY,
    COLLECTION_NAME,
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

LEGAL_TERM_CORRECTIONS = {
    "जुर्माना": "जरिबाना",
    "कट-ऑफ": "अन्तिम सीमा",
    "फाइनान्सियल": "आर्थिक",
}


class LegalGraphState(TypedDict):
    question: str
    documents: List[Dict[str, Any]]
    context_str: str
    answer: str


# HELPER UTILITIES
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


# GRAPH NODES
def prepare_query_node(state: LegalGraphState) -> Dict[str, Any]:
    """Translates English queries to Nepali legal terminology to ensure high vector similarity."""
    user_query = state["question"].strip()

    # Detect if input contains Devanagari characters
    is_devanagari = any("\u0900" <= char <= "\u097f" for char in user_query)

    if not is_devanagari:
        # Use LLM to extract Nepali legal search keywords
        translation_prompt = (
            f"Translate this legal query into concise formal Nepali legal terms used in Nepal Acts. "
            f"Return ONLY the translated Nepali terms without explanation:\n{user_query}"
        )
        translated_query = str(llm.invoke(translation_prompt).content).strip()
        search_query = f"{translated_query} | {user_query}"
    else:
        search_query = user_query

    return {"question": search_query}


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

    # 5. Section / Chunk Title
    title = payload.get("title") or payload.get("doc_type") or ""

    return {
        "act_title": act_title,
        "chapter": chapter,
        "section_number": section_num,
        "title": title,
        "content_text": content_text,
        "score": point.score,
    }


def retrieve_node(state: LegalGraphState) -> Dict[str, Any]:
    question = state["question"]
    query_vector = embedder.encode(question).tolist()

    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=5
    ).points

    docs = []
    for point in search_results:
        if point.score < SCORE_THRESHOLD:
            continue

        # Extract normalized metadata fields regardless of schema differences
        chunk_data = extract_payload_metadata(point)
        docs.append(chunk_data)

    return {"documents": docs}


def format_context_node(state: LegalGraphState) -> Dict[str, Any]:
    """Node 2: Formats retrieved chunks into a clean prompt string with active Devanagari repair."""
    docs = state["documents"]

    if not docs:
        return {"context_str": ""}

    formatted_blocks = []
    for idx, c in enumerate(docs, start=1):
        # Run active character/matra cleaning pass on retrieved text
        clean_text = preprocess_devanagari_text(c["content_text"])

        block = (
            f"[{idx}] Law: {c['act_title']} | Chapter: {c['chapter']} | Section/Rule: {c['section_number']}\n"
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
    workflow = StateGraph(LegalGraphState)

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
                "documents": [],
                "context_str": "",
                "answer": "",
            }

            result = app.invoke(initial_state)

            print("\n--- RESPONSE ---")
            print(result["answer"])
            print("-" * 50 + "\n")
