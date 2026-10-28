from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag import answer_corrective

app = FastAPI(
    title="Bank Policy & Customer Compliance Assistant",
    description="RBI-document-grounded banking policy question answering.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


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
        result = answer_corrective(payload.question)
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to process the question. Check the server logs.",
        ) from exc
