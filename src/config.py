from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"

load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "bank-policy-rag")

COLLECTION_NAME = "bank_policy"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
RELEVANCE_THRESHOLD = 0.35

if not OPENAI_API_KEY:
    # Ingestion can still be inspected without the LLM, but answering/evaluation needs it.
    pass
