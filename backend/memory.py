"""
backend/memory.py
-----------------
Long-term memory layer:
  - Pydantic schemas for memory extraction (MemoryItem, MemoryDecision)
  - Internal helpers: namespace, dedup, category inference, relevance scoring
  - Public CRUD API used by the graph nodes and the Streamlit frontend
"""

from __future__ import annotations

import re
import uuid
from difflib import SequenceMatcher
from typing import List, Literal, Optional

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from backend.config import MEMORY_PROMPT
from backend.database import postgres_store
from backend.llm import model
from langgraph.store.base import BaseStore

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class MemoryItem(BaseModel):
    text:         str            = Field(description="Short, atomic user memory sentence.")
    is_new:       bool           = Field(description="True if new info; False if duplicate of existing.")
    replaces_key: Optional[str]  = Field(
        default=None,
        description="ID of existing memory this updates/replaces.",
    )
    category: Optional[Literal["profile", "preferences", "projects", "goals"]] = Field(
        default=None,
        description="Memory category: profile, preferences, projects, or goals.",
    )


class MemoryDecision(BaseModel):
    should_write: bool
    memories:     List[MemoryItem] = Field(default_factory=list)


# ── Memory extractor chain ────────────────────────────────────────────────────

memory_parser    = PydanticOutputParser(pydantic_object=MemoryDecision)
memory_extractor = model | memory_parser


# ── Namespace helper ──────────────────────────────────────────────────────────

def user_ns(user_id: str) -> tuple:
    return ("user", user_id, "details")


# ── Internal store helpers ────────────────────────────────────────────────────

def get_user_memories_raw(user_id: str, store: BaseStore) -> list[dict]:
    """Return list of {key, data, category} dicts from the given store."""
    ns = user_ns(user_id)
    try:
        items = store.search(ns)
        return [
            {
                "key":      it.key,
                "data":     it.value.get("data", ""),
                "category": it.value.get("category", "profile"),
            }
            for it in items
            if it.value.get("data")
        ]
    except Exception:
        return []


def format_memories_for_prompt(memories_raw: list[dict]) -> str:
    """Format memories with IDs for injection into the memory-extraction prompt."""
    if not memories_raw:
        return "(empty)"
    return "\n".join(
        f"[{m['key']}][{m.get('category', 'profile')}] {m['data']}"
        for m in memories_raw
    )


def format_memories_plain(memories_raw: list[dict]) -> str:
    """Format memories as plain text grouped by category for the chat system prompt."""
    if not memories_raw:
        return "(empty)"
    groups: dict[str, list[str]] = {
        "profile":     [],
        "preferences": [],
        "projects":    [],
        "goals":       [],
    }
    for m in memories_raw:
        cat = m.get("category", "profile")
        if cat not in groups:
            cat = "profile"
        groups[cat].append(m["data"])

    lines: list[str] = []
    for cat in ("profile", "preferences", "projects", "goals"):
        if groups[cat]:
            lines.append(f"{cat.capitalize()}:")
            lines.extend([f"- {item}" for item in groups[cat]])
    return "\n".join(lines) if lines else "(empty)"


def normalize_memory_category(category: Optional[str]) -> str:
    if not category:
        return ""
    c = category.strip().lower()
    return c if c in {"profile", "preferences", "projects", "goals"} else ""


def infer_memory_category(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["prefer", "like", "dislike", "favorite", "favourite"]):
        return "preferences"
    if any(k in t for k in ["building", "project", "working on", "repo", "app", "tool"]):
        return "projects"
    if any(k in t for k in ["goal", "aim", "plan", "want to", "target", "roadmap"]):
        return "goals"
    return "profile"


def _normalize_memory_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    normalized = re.sub(r"\b(iiitn)\b", "iiit", normalized)
    normalized = re.sub(r"\bnagpur\b", "nagpur", normalized)
    normalized = re.sub(r"\bstudying\b", "student", normalized)
    normalized = re.sub(r"\bstudies\b", "student", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _memory_tokens(text: str) -> set[str]:
    stop_words = {
        "a", "an", "the", "at", "in", "on", "of", "and", "or", "to", "for",
        "is", "am", "are", "i", "im", "my", "me", "student", "currently",
    }
    return {
        tok
        for tok in _normalize_memory_text(text).split()
        if tok and tok not in stop_words
    }


def select_relevant_memories(
    memories_raw: list[dict],
    query_text: str,
    max_items: int = 8,
) -> list[dict]:
    """Return the top-N most relevant memories for the given query."""
    if not memories_raw:
        return []
    if len(memories_raw) <= max_items:
        return memories_raw

    q_tokens = _memory_tokens(query_text)
    if not q_tokens:
        return memories_raw[-max_items:]

    scored: list[tuple[float, dict]] = []
    for idx, mem in enumerate(memories_raw):
        mem_tokens = _memory_tokens(mem.get("data", ""))
        overlap = len(q_tokens & mem_tokens) / max(1, len(q_tokens | mem_tokens))
        ratio   = SequenceMatcher(
            None,
            _normalize_memory_text(query_text),
            _normalize_memory_text(mem.get("data", "")),
        ).ratio()
        recency_boost = idx / max(1, len(memories_raw)) * 0.05
        score = overlap * 0.75 + ratio * 0.2 + recency_boost
        scored.append((score, mem))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:max_items]]


def is_duplicate_memory(candidate: str, existing_memories: list[dict]) -> bool:
    """Return True when `candidate` is effectively a duplicate of an existing memory."""
    cand_norm = _normalize_memory_text(candidate)
    if not cand_norm:
        return True

    cand_tokens = _memory_tokens(candidate)

    for m in existing_memories:
        existing_text = m.get("data", "")
        exist_norm    = _normalize_memory_text(existing_text)
        if not exist_norm:
            continue

        # If candidate is a substring of an existing memory, it is a duplicate.
        # But if the existing memory is a substring of candidate, candidate has more info, so it's not a duplicate.
        if cand_norm == exist_norm or cand_norm in exist_norm:
            return True

        ratio = SequenceMatcher(None, cand_norm, exist_norm).ratio()
        if ratio >= 0.88:
            return True

        exist_tokens = _memory_tokens(existing_text)
        if cand_tokens and exist_tokens:
            # Check if the candidate's tokens are mostly already contained in the existing memory.
            overlap = len(cand_tokens & exist_tokens) / len(cand_tokens)
            if overlap >= 0.8:
                return True

    return False


# ── Public CRUD API (used by graph nodes and Streamlit frontend) ──────────────

def get_user_memories_list(user_id: str) -> list[dict]:
    """Return [{key, data, category}] for all stored memories of a user."""
    return get_user_memories_raw(user_id, postgres_store)


def delete_user_memory(user_id: str, memory_key: str) -> bool:
    """Delete a single memory by key. Returns True on success."""
    ns = user_ns(user_id)
    try:
        postgres_store.delete(ns, memory_key)
        return True
    except Exception:
        return False


def delete_all_user_memories(user_id: str) -> bool:
    """Delete ALL memories for a user. Returns True on success."""
    memories = get_user_memories_raw(user_id, postgres_store)
    try:
        for m in memories:
            postgres_store.delete(user_ns(user_id), m["key"])
        return True
    except Exception:
        return False


# ── Memory writing helper (called from graph remember_node) ──────────────────

def write_memories_from_message(user_id: str, last_text: str, store: BaseStore) -> None:
    """
    Extract memory-worthy facts from `last_text` and persist them to the store.
    Handles deduplication and replacement of stale memories automatically.
    This is best-effort — any exception is silently swallowed.
    """
    if not last_text.strip():
        return

    ns           = user_ns(user_id)
    existing_raw = get_user_memories_raw(user_id, store)
    memories_ctx = format_memories_for_prompt(existing_raw)

    try:
        decision: MemoryDecision = memory_extractor.invoke([
            SystemMessage(content=MEMORY_PROMPT.format(
                user_details_with_keys=memories_ctx,
                format_instructions=memory_parser.get_format_instructions(),
            )),
            {"role": "user", "content": last_text},
        ])

        if not decision.should_write:
            return

        current_memories = list(existing_raw)

        for mem in decision.memories:
            if not mem.is_new or not mem.text.strip():
                continue

            candidate = mem.text.strip()

            if mem.replaces_key:
                try:
                    store.delete(ns, mem.replaces_key)
                    current_memories = [
                        m for m in current_memories if m.get("key") != mem.replaces_key
                    ]
                except Exception:
                    pass

            if is_duplicate_memory(candidate, current_memories):
                continue

            category = (
                normalize_memory_category(mem.category)
                or infer_memory_category(candidate)
            )
            new_key = str(uuid.uuid4())
            store.put(ns, new_key, {"data": candidate, "category": category})
            current_memories.append({"key": new_key, "data": candidate, "category": category})

    except Exception:
        pass  # Memory is best-effort — never break the main chat flow
