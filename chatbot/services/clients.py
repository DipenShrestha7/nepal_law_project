from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

from chatbot.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    OPENROUTER_API_KEY,
)

load_dotenv()

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
