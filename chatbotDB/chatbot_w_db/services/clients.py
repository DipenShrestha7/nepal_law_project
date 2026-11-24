import os
from dotenv import load_dotenv

# 1. Load environment variables BEFORE importing internal config modules
load_dotenv()

import torch
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

from chatbot.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    OPENROUTER_API_KEY,
)

# 2. Automatically detect optimal hardware accelerator for Dense Embedder
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# 3. Initialize Shared Machine Learning Models
embedder = SentenceTransformer("BAAI/bge-m3", device=device)
sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")

# 4. Initialize Qdrant Vector DB Client
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

# 5. Initialize OpenRouter LLM Client with Explicit Model Target
DEFAULT_MODEL = "openrouter/free"

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model=DEFAULT_MODEL,
    temperature=0.1,
    default_headers={
        "HTTP-Referer": os.getenv("APP_URL", "http://localhost:3000"),
        "X-Title": "Nepal Legal Assistant - English",
    },
)

# 6. Global Configuration Constants
COLLECTIONS = ["nepal_laws_en_hybrid"]
MAX_SCHEDULE_CONTEXT_CHARS = 12000
