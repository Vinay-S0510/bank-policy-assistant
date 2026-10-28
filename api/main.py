import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import CHROMA_DIR
from src.ingest import build_vectorstore
from src.rag import answer_corrective

app = FastAPI(
    title="Bank Policy & Customer Compliance Assistant",
    description="RBI-document-grounded banking policy question answering.",
    version="1.0.0",
)

allowed_origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]

frontend_url = os.getenv("FRONTEND_URL")
if frontend_url:
    allowed_origins.append(frontend_url.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.on_event("startup")
def initialize_vectorstore():
    collection_marker = CHROMA_DIR / "chroma.sqlite3"

    if not collection_marker.exists():
        build_vectorstore()


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.get("/")
def root():
    return {
        "message": "Bank Policy & Customer Compliance Assistant",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: Question):
    try:
        return answer_corrective(payload.question)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to process the question. Check the server logs.",
        ) from exc