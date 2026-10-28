from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TOP_K,
    RELEVANCE_THRESHOLD,
)


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


def retrieve_documents(question: str, k: int = TOP_K):
    db = get_vectorstore()

    # Retrieve semantically relevant chunks.
    semantic_results = db.similarity_search_with_relevance_scores(
        question,
        k=k,
    )

    # Exact keyword search across the indexed document chunks.
    exact_results = []
    question_lower = question.lower()

    keywords = []
    if "officially valid document" in question_lower or "ovd" in question_lower:
        keywords = ["officially valid document", "ovd means"]
    elif "kyc" in question_lower:
        keywords = ["know your customer", "kyc directions"]
    elif "credit card" in question_lower:
        keywords = ["credit card", "card issuer"]
    elif "ombudsman" in question_lower or "complaint" in question_lower:
        keywords = ["ombudsman", "complaint"]

    collection = db._collection
    records = collection.get(
        include=["documents", "metadatas"],
    )

    for keyword in keywords:
        for i, content in enumerate(records["documents"]):
            if keyword in (content or "").lower():
                metadata = records["metadatas"][i] or {}
                from langchain_core.documents import Document

                doc = Document(
                    page_content=content,
                    metadata=metadata,
                )
                exact_results.append((doc, 1.0))

    # Exact matches first, followed by semantic matches.
    combined = []
    seen = set()

    for doc, score in exact_results + semantic_results:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.page_content,
        )
        if key not in seen:
            seen.add(key)
            combined.append((doc, score))

    return combined[:k]

def _format_context(documents):
    blocks = []
    for i, (doc, score) in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        blocks.append(
            f"[SOURCE {i}] Document: {source}; Page: {page}; "
            f"Relevance: {score:.3f}\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def _make_result(answer, documents, mode, retry_used=False):
    citations = []
    retrieved = []
    seen = set()

    for i, (doc, score) in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        retrieved.append({
            "source": source,
            "page": page,
            "score": round(float(score), 3),
        })

        key = (source, page)
        if key not in seen:
            citations.append({
                "source_id": f"SOURCE {i}",
                "document": source,
                "page": page,
            })
            seen.add(key)

    return {
        "answer": answer,
        "citations": citations,
        "retrieved": retrieved,
        "mode": mode,
        "corrective_retry_used": retry_used,
    }


def _generate(question, documents, mode, retry_used=False):
    if not documents:
        return _make_result(
            "The available RBI documents do not provide enough information "
            "to answer this question.",
            [],
            mode,
            retry_used,
        )

    context = _format_context(documents)
    prompt = ChatPromptTemplate.from_template("""
You are a careful Indian banking-policy assistant.

Answer only from the supplied RBI document excerpts.
Do not use outside knowledge or invent facts.
If the excerpts do not support an answer, clearly say that the available
RBI documents do not provide enough information.
Cite each factual claim using the source labels shown in the excerpts,
such as [SOURCE 1]. Never invent a source label.
Be concise and distinguish what the document says from advice.

DOCUMENT EXCERPTS:
{context}

QUESTION:
{question}

ANSWER:
""")

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        max_retries=1,
    )
    response = (prompt | llm).invoke({
        "context": context,
        "question": question,
    })

    return _make_result(
        response.content,
        documents,
        mode,
        retry_used,
    )


def answer_basic(question: str):
    """Baseline: one retrieval pass, with no relevance-based retry."""
    docs = retrieve_documents(question, k=TOP_K)
    return _generate(question, docs, "basic")


def answer_corrective(question: str):
    """Retry retrieval with a broader query when initial relevance is weak."""
    docs = retrieve_documents(question, k=TOP_K)

    if docs and max(score for _, score in docs) >= RELEVANCE_THRESHOLD:
        return _generate(question, docs, "corrective")

    # Simple corrective retrieval: broaden the search using policy-related terms.
    retry_question = (
        f"{question} RBI banking regulation customer rights policy "
        "requirements complaint procedure"
    )
    retry_docs = retrieve_documents(retry_question, k=TOP_K)

    # Use whichever retrieval pass has the stronger top relevance score.
    first_score = max((score for _, score in docs), default=0.0)
    retry_score = max((score for _, score in retry_docs), default=0.0)

    if retry_score > first_score:
        selected = retry_docs
    else:
        selected = docs

    # Do not ask the LLM to answer when the evidence remains very weak.
    if not selected or max(score for _, score in selected) < RELEVANCE_THRESHOLD:
        return _make_result(
            "The available RBI documents do not provide enough information "
            "to answer this question. Please consult the relevant bank or "
            "an authoritative RBI source.",
            selected,
            "corrective",
            retry_used=True,
        )

    return _generate(
        question,
        selected,
        "corrective",
        retry_used=True,
    )


def generate_answer(question: str):
    """Compatibility wrapper for the earlier command-line test."""
    return answer_corrective(question)


if __name__ == "__main__":
    question = "Can a bank issue a credit card without my consent?"
    result = generate_answer(question)

    print("\nANSWER:\n")
    print(result["answer"])

    print("\nCITATIONS:")
    for citation in result["citations"]:
        print(
            f"[{citation['source_id']}] "
            f"{citation['document']} (page {citation['page']})"
        )
