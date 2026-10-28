# Bank Policy & Customer Compliance Assistant

A RAG-based assistant over selected RBI banking-policy documents. The system is designed to answer policy questions with evidence and citations, refuse unsupported questions, and compare basic retrieval against corrective retrieval.

## Current corpus

- `rbi_credit_debit_cards.pdf`
- `rbi_kyc.pdf`
- `rbi_ombudsman.pdf`

`customer_service.pdf` is intentionally excluded for now because the uploaded document identifies itself as withdrawn. Do not treat it as a current production policy source.

## Architecture

PDFs → chunking → embeddings → Chroma → retrieval → corrective retrieval → Gemini → cited answer → FastAPI → evaluation/LangSmith

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your Google API key
```

## Put PDFs in `data/`

Copy the three active PDFs into `data/`.

## Build the vector store

```bash
python -m src.ingest
```

## Run evaluation

```bash
python -m tests.evaluate basic
python -m tests.evaluate corrective
```

The current evaluator is a baseline only. After the first working run, we will replace the simple lexical score with a proper groundedness/correctness evaluation and LangSmith experiment.
