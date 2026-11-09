import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
SCORE_THRESHOLD = 0.25
FALLBACK_SCORE_THRESHOLD = 0.10
DEFAULT_MODEL = "openrouter/free"
