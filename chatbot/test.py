import os
import re
from typing import Any, Dict, List, TypedDict
import unicodedata
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from chatbot.prompt.legal_system_prompt import LEGAL_SYSTEM_PROMPT

load_dotenv()

# CONFIGURATION & CLIENTS
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
COLLECTION_NAME = "nepal_laws"
SCORE_THRESHOLD = 0.32

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


# DEFINE GRAPH STATE
class LegalGraphState(TypedDict):
    question: str
    documents: List[Dict[str, Any]]
    context_str: str
    answer: str


def clean_llm_output(text: str) -> str:
    """Strips OpenRouter guardrail metadata headers from LLM completions."""
    if not text:
        return ""
    # Remove "User Safety: safe" or similar system prefixes
    cleaned = re.sub(r"^User Safety:\s*\w+\s*", "", text, flags=re.IGNORECASE).strip()
    return cleaned


# NEPALI PROMPT TO ENGLISH TRANSLATION
def prepare_query_node(state: LegalGraphState) -> Dict[str, Any]:
    """Translates English queries to formal Nepali to ensure 1:1 vector matching with Nepali chunks."""
    user_query = state["question"]

    # Detect if input is already in Devanagari
    is_devanagari = any("\u0900" <= char <= "\u097f" for char in user_query)

    if is_devanagari:
        # If user asked in Nepali, expand with English for dual-search capabilities
        prompt = f"Translate this legal query into concise English legal terms. Return ONLY the translation:\n{user_query}"
        en_translation = str(llm.invoke(prompt).content).strip()
        search_query = f"{user_query} | {en_translation}"
    else:
        # If user asked in English, translate to formal Nepali legal terms
        prompt = f"Translate this legal query into formal Nepali legal terminology used in Nepal Acts. Return ONLY the translated Nepali text:\n{user_query}"
        np_translation = str(llm.invoke(prompt).content).strip()
        search_query = f"{np_translation} | {user_query}"

    return {"question": search_query}


# FRAGMENTED TEXT REPARATION
def format_context_node(state: LegalGraphState) -> Dict[str, Any]:
    """Cleans and repairs Devanagari ligature corruptions before prompting the LLM."""
    docs = state["documents"]
    formatted_blocks = []

    for idx, c in enumerate(docs, start=1):
        raw_content = c["content_text"]

        # 1. Repair Unicode character assembly
        clean_content = unicodedata.normalize("NFC", raw_content)

        # 2. Fix common PDF extraction artifact spaces inside words
        clean_content = re.sub(
            r"(?<=\u0900-\u097F)\s+(?=[\u0902-\u094D])", "", clean_content
        )

        block = (
            f"[{idx}] Act/Law: {c['act_title']}\n"
            f"Chapter: {c['chapter']} | Section/Rule: {c['section_number']}\n"
            f"Text: {clean_content}\n"
        )
        formatted_blocks.append(block)

    return {"context_str": "\n".join(formatted_blocks)}


# DEFINE GRAPH NODES
def retrieve_node(state: LegalGraphState) -> Dict[str, Any]:
    """Node 1: Retrieve top matching legal clauses from Qdrant with score thresholding."""
    question = state["question"]
    query_vector = embedder.encode(question).tolist()

    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=5
    ).points

    docs = []
    for point in search_results:
        # Reject irrelevant chunks that fall below the quality threshold
        if point.score < SCORE_THRESHOLD:
            continue

        payload = point.payload or {}
        docs.append(
            {
                "act_title": payload.get(
                    "act_title", payload.get("title_np", "Unknown Law")
                ),
                "chapter": payload.get("chapter", "N/A"),
                "section_number": payload.get("section_number")
                or payload.get("rule_number", "N/A"),
                "title": payload.get("title", ""),
                "content_text": payload.get("content_text", ""),
                "score": point.score,
            }
        )

    return {"documents": docs}


def format_context_node(state: LegalGraphState) -> Dict[str, Any]:
    """Node 2: Format retrieved chunks into a clean prompt string with Devanagari repair."""
    docs = state["documents"]

    if not docs:
        return {"context_str": ""}

    formatted_blocks = []
    for idx, c in enumerate(docs, start=1):
        raw_text = c["content_text"]

        # 1. Fix broken Devanagari Unicode ligatures on the fly
        clean_text = unicodedata.normalize("NFC", raw_text)

        # 2. Remove illegal spaces inserted between Nepali letters and vowel signs (matras)
        clean_text = re.sub(r"(?<=\u0900-\u097F)\s+(?=[\u0902-\u094D])", "", clean_text)

        block = (
            f"[{idx}] Law: {c['act_title']} | Chapter: {c['chapter']} | Section/Rule: {c['section_number']}\n"
            f"Title: {c['title']}\n"
            f"Content: {clean_text}\n"
        )
        formatted_blocks.append(block)

    return {"context_str": "\n\n".join(formatted_blocks)}


def generate_answer_node(state: LegalGraphState) -> Dict[str, Any]:
    question = state["question"]
    context = state["context_str"]

    if not state["documents"]:
        return {
            "answer": "The requested legal provision was not found in the Nepali legal database."
        }

    system_prompt = LEGAL_SYSTEM_PROMPT

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"<context>\n{context}\n</context>\n\nUSER QUESTION: {question}"
        ),
    ]

    response = llm.invoke(messages)
    return {"answer": str(response.content)}


# BUILD LANGGRAPH WORKFLOW
def build_legal_rag_graph():
    workflow = StateGraph(LegalGraphState)  # type: ignore[bad-specialization]

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("format_context", format_context_node)
    workflow.add_node("generate", generate_answer_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "format_context")
    workflow.add_edge("format_context", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# EXECUTION LOOP
if __name__ == "__main__":
    app = build_legal_rag_graph()

    print("==================================================")
    print(" Nepal Legal AI Chatbot (LangGraph Engine Connected)")
    print("==================================================\n")

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
