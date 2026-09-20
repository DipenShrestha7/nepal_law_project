import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
COLLECTION_NAME = ["nepal_laws_en", "nepal_laws_np"]
SCORE_THRESHOLD = 0.25
FALLBACK_SCORE_THRESHOLD = 0.10
