import os
from dotenv import load_dotenv

load_dotenv()

import torch
from langchain_openai import ChatOpenAI

from chatbot.config import OPENROUTER_API_KEY

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


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

MAX_SCHEDULE_CONTEXT_CHARS = 12000
