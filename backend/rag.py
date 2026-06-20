"""
backend/rag.py
--------------
Handles all PDF ingestion (thread-scoped and global) and in-process FAISS
vector-store state.  Keeps retriever references in module-level dicts so they
survive across graph invocations within the same process.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.llm import get_embedding

# ── In-process RAG state ──────────────────────────────────────────────────────

_THREAD_RETRIEVERS: dict[str, object] = {}
_THREAD_METADATA:   dict[str, dict]   = {}
_GLOBAL_RETRIEVER:  Optional[object]  = None
_GLOBAL_METADATA:   dict              = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_thread_retriever(thread_id: Optional[str]):
    """Return the FAISS retriever for a given thread, or None."""
    return _THREAD_RETRIEVERS.get(thread_id) if thread_id else None


def get_global_retriever():
    """Return the global (shared) FAISS retriever, or None."""
    return _GLOBAL_RETRIEVER


def get_thread_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(thread_id, {})


def get_global_metadata() -> dict:
    return _GLOBAL_METADATA


def has_thread_pdf(thread_id: str) -> bool:
    return thread_id in _THREAD_RETRIEVERS


def has_global_pdf() -> bool:
    return _GLOBAL_RETRIEVER is not None


def remove_thread_rag(thread_id: str) -> None:
    """Clean up in-memory RAG state when a thread is deleted."""
    _THREAD_RETRIEVERS.pop(thread_id, None)
    _THREAD_METADATA.pop(thread_id, None)


def _sanitize_text(text) -> Optional[str]:
    if text is None:
        return None
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return None
    text = text.replace("\x00", " ").strip()
    return text if text else None


def _load_and_chunk_pdf(file_bytes: bytes) -> tuple[list, list]:
    """Load a PDF from bytes, return (raw_documents, text_chunks)."""
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(file_bytes)
            tmp_path = f.name

        loader    = PyPDFLoader(tmp_path)
        documents = loader.load()
        if not documents:
            raise ValueError("No pages could be extracted from the PDF.")

        for doc in documents:
            clean = _sanitize_text(doc.page_content)
            doc.page_content = clean if clean else ""

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        raw_chunks = splitter.split_documents(documents)
        chunks = []
        for chunk in raw_chunks:
            clean = _sanitize_text(chunk.page_content)
            if clean:
                chunk.page_content = clean
                chunks.append(chunk)

        if not chunks:
            raise ValueError(
                "All extracted text chunks are empty. "
                "The PDF may be scanned/image-based — please use an OCR-processed PDF."
            )
        return documents, chunks
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── Public ingestion API ──────────────────────────────────────────────────────

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """Ingest a PDF into a thread-scoped FAISS vector store."""
    if not file_bytes:
        raise ValueError("No file bytes provided.")

    documents, chunks = _load_and_chunk_pdf(file_bytes)
    vectorstore = FAISS.from_documents(chunks, get_embedding())
    retriever   = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    resolved_name = filename or f"pdf_{thread_id[:8]}.pdf"
    _THREAD_RETRIEVERS[thread_id] = retriever
    _THREAD_METADATA[thread_id]   = {
        "filename":      resolved_name,
        "num_chunks":    len(chunks),
        "num_documents": len(documents),
    }
    return {
        "filename":      resolved_name,
        "num_chunks":    len(chunks),
        "num_documents": len(documents),
    }


def ingest_global_pdf(file_bytes: bytes, filename: Optional[str] = None) -> dict:
    """Ingest a PDF into the global (shared) FAISS vector store."""
    global _GLOBAL_RETRIEVER, _GLOBAL_METADATA

    if not file_bytes:
        raise ValueError("No file bytes provided.")

    documents, chunks = _load_and_chunk_pdf(file_bytes)
    vectorstore       = FAISS.from_documents(chunks, get_embedding())
    _GLOBAL_RETRIEVER = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    resolved_name    = filename or "global_pdf.pdf"
    _GLOBAL_METADATA = {
        "filename":   resolved_name,
        "num_chunks": len(chunks),
        "documents":  len(documents),
    }
    return {
        "filename":  resolved_name,
        "num_chunks": len(chunks),
        "documents":  len(documents),
    }
