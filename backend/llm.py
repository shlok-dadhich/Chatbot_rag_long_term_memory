"""
backend/llm.py
--------------
Initialises the HuggingFace LLM (chat model) and the sentence-transformer
embedding model used across the application.
"""

from functools import lru_cache

from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint

from backend.config import (
    EMBEDDING_MODEL,
    HF_MAX_TOKENS,
    HF_REPO_ID,
    HF_TEMPERATURE,
)

# ── Chat model ────────────────────────────────────────────────────────────────

_endpoint = HuggingFaceEndpoint(
    repo_id=HF_REPO_ID,
    task="text_generation",
    max_new_tokens=HF_MAX_TOKENS,
    temperature=HF_TEMPERATURE,
)

model = ChatHuggingFace(llm=_endpoint)

# ── Embedding model ───────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embedding() -> HuggingFaceEmbeddings:
    """Load the embedding model lazily when a PDF is indexed."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
