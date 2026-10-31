import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.ingest import load_policy_corpus
from src.rag import answer_corrective


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_INDEX = ROOT / "frontend" / "index.html"

app = FastAPI(
    title="Bank Policy & Customer Compliance Assistant",
    description="RBI-document-grounded banking policy question answering.",
    version="1.1.0",
)

allowed_origins = ["http://127.0.0.1:5500", "http://localhost:5500"]
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
def initialize_corpus() -> None:
    app.state.corpus = load_policy_corpus()


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(FRONTEND_INDEX)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "corpus_chunks": len(app.state.corpus)}


@app.post("/ask")
def ask(payload: Question) -> dict:
    try:
        return answer_corrective(payload.question, app.state.corpus)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to search the bundled RBI policy corpus.",
        ) from exc
