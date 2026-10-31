import re
from typing import Iterable

from src.config import TOP_K
from src.ingest import PolicyChunk


STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "for", "how", "i", "in", "is",
    "my", "of", "on", "or", "the", "to", "what", "when", "with", "you", "your",
}
WORD_PATTERN = re.compile(r"[a-zA-Z]{3,}")


def _keywords(text: str) -> set[str]:
    return {
        word.lower()
        for word in WORD_PATTERN.findall(text)
        if word.lower() not in STOP_WORDS
    }


def _score(question: str, document: PolicyChunk) -> float:
    query_words = _keywords(question)
    if not query_words:
        return 0.0

    content = document.page_content.lower()
    overlap = sum(word in content for word in query_words)
    score = overlap / len(query_words)

    # Give useful policy phrases a small boost without relying on an embedding model.
    for phrase in (
        "officially valid document",
        "know your customer",
        "credit card",
        "debit card",
        "ombudsman",
        "complaint",
    ):
        if phrase in question.lower() and phrase in content:
            score += 0.25

    return score


def retrieve_documents(
    question: str, corpus: Iterable[PolicyChunk], k: int = TOP_K
) -> list[tuple[PolicyChunk, float]]:
    ranked = [
        (document, _score(question, document))
        for document in corpus
    ]
    ranked = [(document, score) for document, score in ranked if score > 0]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:k]


def _excerpt(text: str, question: str, limit: int = 700) -> str:
    """Prefer the sentence containing a question keyword for a concise answer."""
    keywords = _keywords(question)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    best = max(
        sentences,
        key=lambda sentence: sum(word in sentence.lower() for word in keywords),
        default=text,
    ).strip()
    return best[:limit].rstrip() + ("…" if len(best) > limit else "")


def _make_result(
    answer: str, documents: list[tuple[PolicyChunk, float]], mode: str
) -> dict:
    citations = []
    retrieved = []
    seen = set()

    for index, (document, score) in enumerate(documents, start=1):
        source = document.metadata["source"]
        page = document.metadata["page"]
        retrieved.append(
            {"source": source, "page": page, "score": round(score, 3)}
        )

        if (source, page) not in seen:
            citations.append(
                {"source_id": f"SOURCE {index}", "document": source, "page": page}
            )
            seen.add((source, page))

    return {
        "answer": answer,
        "citations": citations,
        "retrieved": retrieved,
        "mode": mode,
        "corrective_retry_used": False,
    }


def answer_corrective(question: str, corpus: Iterable[PolicyChunk]) -> dict:
    documents = retrieve_documents(question, corpus)
    if not documents:
        return _make_result(
            "The available RBI documents do not contain a matching passage for this question. "
            "Please try more specific policy terms or consult the relevant RBI publication.",
            [],
            "keyword-retrieval",
        )

    excerpts = []
    for index, (document, _) in enumerate(documents, start=1):
        source = document.metadata["source"]
        page = document.metadata["page"]
        excerpts.append(
            f"[SOURCE {index}] {source}, page {page}:\n{_excerpt(document.page_content, question)}"
        )

    answer = (
        "The following RBI policy excerpts are most relevant to your question. "
        "They are provided verbatim in summary form; verify important decisions against "
        "the cited source document.\n\n" + "\n\n".join(excerpts)
    )
    return _make_result(answer, documents, "keyword-retrieval")


def generate_answer(question: str, corpus: Iterable[PolicyChunk]) -> dict:
    """Compatibility wrapper for callers using the original function name."""
    return answer_corrective(question, corpus)
