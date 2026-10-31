from dataclasses import dataclass
from pathlib import Path
import re

from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR


ALLOWED_FILES = {
    "rbi_credit_debit_cards.pdf": {
        "title": "Master Direction - Credit Card and Debit Card – Issuance and Conduct Directions, 2022 — FAQs",
        "authority": "Reserve Bank of India",
        "status": "current-source-for-project",
    },
    "rbi_kyc.pdf": {
        "title": "Master Direction - Know Your Customer (KYC) Direction, 2016",
        "authority": "Reserve Bank of India",
        "status": "updated-as-on-2025-08-14",
    },
    "rbi_ombudsman.pdf": {
        "title": "Reserve Bank - Integrated Ombudsman Scheme, 2021",
        "authority": "Reserve Bank of India",
        "status": "scheme-document",
    },
}


@dataclass(frozen=True)
class PolicyChunk:
    page_content: str
    metadata: dict


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_text(text: str):
    """Split a page into small, overlapping excerpts without a model dependency."""
    text = _normalise_text(text)
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)


def load_policy_corpus(data_dir: Path = DATA_DIR) -> list[PolicyChunk]:
    """Read the bundled RBI PDFs into a compact in-memory retrieval corpus."""
    chunks: list[PolicyChunk] = []

    for filename, source_meta in ALLOWED_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing corpus file: {path}")

        with path.open("rb") as pdf_file:
            reader = PdfReader(pdf_file)
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                for excerpt in _split_text(text):
                    chunks.append(
                        PolicyChunk(
                            page_content=excerpt,
                            metadata={
                                "source": filename,
                                "title": source_meta["title"],
                                "authority": source_meta["authority"],
                                "corpus_status": source_meta["status"],
                                "page": page_number,
                            },
                        )
                    )

    if not chunks:
        raise RuntimeError("The RBI document corpus did not contain readable text.")

    return chunks
