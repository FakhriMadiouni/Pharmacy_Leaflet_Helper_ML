"""All settings in one place. Values come from the .env file (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

MODE = os.getenv("INFERENCE_MODE", "mock").lower()          # mock | real
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")

EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "mock").lower()  # mock | fastembed
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mpnet").lower()  # minilm | mpnet

SEARCH_THRESHOLD = float(os.getenv("SEARCH_THRESHOLD", "0.20"))

# Order rules (used by the tools)
MIN_QTY = 1
MAX_QTY = 100        # more than this: always rejected
REVIEW_QTY = 50      # more than this (up to MAX_QTY): a human must review
MAX_TURNS = 5        # the agent may call the model at most 5 times
