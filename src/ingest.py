from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from .config import DATA_DIR, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

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


def load_documents():
    docs = []
    for filename, meta in ALLOWED_FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing corpus file: {path}")
        loaded = PyPDFLoader(str(path)).load()
        for doc in loaded:
            doc.metadata.update({
                "source": filename,
                "title": meta["title"],
                "authority": meta["authority"],
                "corpus_status": meta["status"],
                "page": int(doc.metadata.get("page", 0)) + 1,
            })
        docs.extend(loaded)
    return docs


def build_vectorstore():
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Loaded pages: {len(docs)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Vector store: {CHROMA_DIR}")
    return db


if __name__ == "__main__":
    build_vectorstore()
