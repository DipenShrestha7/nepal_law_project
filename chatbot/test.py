import os
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# ==========================================
# 1. CONFIGURATION & CLIENTS
# ==========================================
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
COLLECTION_NAME = "nepal_laws"

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


# ==========================================
# 2. DEFINE GRAPH STATE
# ==========================================
class LegalGraphState(TypedDict):
    question: str
    documents: List[Dict[str, Any]]
    context_str: str
    answer: str


# ==========================================
# 3. DEFINE GRAPH NODES
# ==========================================
def retrieve_node(state: LegalGraphState) -> Dict[str, Any]:
    """Node 1: Retrieve top matching legal clauses from Qdrant."""
    question = state["question"]
    query_vector = embedder.encode(question).tolist()

    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=5
    ).points

    docs = []
    for point in search_results:
        # Fix 2: Safely handle cases where point.payload is None
        payload = point.payload or {}
        docs.append(
            {
                "act_title": payload.get("act_title", "Unknown Law"),
                "chapter": payload.get("chapter", "N/A"),
                "section_number": payload.get("section_number")
                or payload.get("rule_number", "N/A"),
                "title": payload.get("title", ""),
                "content_text": payload.get("content_text", ""),
            }
        )

    return {"documents": docs}


def format_context_node(state: LegalGraphState) -> Dict[str, Any]:
    """Node 2: Format retrieved chunks into a clean prompt string."""
    docs = state["documents"]
    formatted_blocks = []

    for idx, c in enumerate(docs, start=1):
        block = (
            f"[{idx}] Law: {c['act_title']} | Chapter: {c['chapter']} | Section: {c['section_number']}\n"
            f"Title: {c['title']}\n"
            f"Content: {c['content_text']}\n"
        )
        formatted_blocks.append(block)

    return {"context_str": "\n".join(formatted_blocks)}


def generate_answer_node(state: LegalGraphState) -> Dict[str, Any]:
    """Node 3: Generate legal answer using OpenRouter LLM."""
    question = state["question"]
    context = state["context_str"]

    system_prompt = (
        "You are an expert legal assistant specializing in the legal system of Nepal.\n"
        "Answer the user's query strictly based on the provided retrieved legal context.\n\n"
        "RULES:\n"
        "1. Always cite specific Acts, Chapters, and Section/Rule numbers in your explanation.\n"
        "2. If asked in Devanagari/Nepali, reply in Nepali. If asked in English, reply in English.\n"
        "3. If the context does not contain sufficient legal proof, state clearly that the provision was not found."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"RETRIEVED LEGAL CONTEXT:\n{context}\n\nUSER QUESTION: {question}"
        ),
    ]

    response = llm.invoke(messages)
    return {"answer": str(response.content)}


# ==========================================
# 4. BUILD LANGGRAPH WORKFLOW
# ==========================================
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


# ==========================================
# 5. EXECUTION LOOP
# ==========================================
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
