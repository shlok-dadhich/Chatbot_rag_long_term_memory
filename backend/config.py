"""
backend/config.py
-----------------
Central place for all environment variables, constants, and prompt templates.
"""

import os
from typing import Any, Callable, TypeVar

from dotenv import load_dotenv

load_dotenv()

T = TypeVar("T")


def _read_streamlit_secret(key: str) -> Any:
  """Return a Streamlit secret when running on Streamlit Cloud."""
  try:
    import streamlit as st

    return st.secrets.get(key)
  except Exception:
    return None


def _get_setting(key: str, default: str = "") -> str:
  """Read config from env first, then Streamlit secrets, then default."""
  value = os.getenv(key)
  if value is None:
    value = _read_streamlit_secret(key)
  if value is None:
    return default
  return str(value).strip()


def _parse_setting(key: str, default: T, parser: Callable[[str], T]) -> T:
  raw = _get_setting(key, str(default))
  try:
    return parser(raw)
  except (TypeError, ValueError):
    return default


def _sync_secret_to_env(key: str) -> None:
  """Expose Streamlit secrets to libraries that only inspect os.environ."""
  if os.getenv(key):
    return
  value = _read_streamlit_secret(key)
  if value:
    os.environ[key] = str(value).strip()


def _bootstrap_hf_token_env() -> None:
  """Map numbered HF token vars to the standard env var if needed."""
  _sync_secret_to_env("HUGGINGFACEHUB_API_TOKEN")

  if os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip():
    return

  for key in (
    "HUGGINGFACEHUB_API_TOKEN1",
    "HUGGINGFACEHUB_API_TOKEN2",
    "HUGGINGFACEHUB_API_TOKEN3",
    "HUGGINGFACEHUB_API_TOKEN4",
  ):
    _sync_secret_to_env(key)
    token = os.getenv(key, "").strip()
    if token:
      os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
      return


_bootstrap_hf_token_env()

# ── Database ──────────────────────────────────────────────────────────────────

DB_URI: str = _get_setting(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/chatbot?sslmode=disable",
)

# ── HuggingFace ───────────────────────────────────────────────────────────────

HF_REPO_ID: str = _get_setting("HF_REPO_ID", "Qwen/Qwen2.5-7B-Instruct")
HF_MAX_TOKENS: int = _parse_setting("HF_MAX_TOKENS", 2048, int)
HF_TEMPERATURE: float = _parse_setting("HF_TEMPERATURE", 0.8, float)

EMBEDDING_MODEL: str = _get_setting(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ── External API keys ─────────────────────────────────────────────────────────

ALPHA_VINTAGE_KEY: str = _get_setting("ALPHA_VINTAGE_KEY", "")
WEATHER_API_KEY: str = _get_setting("WEATHER_API_KEY", "")

# ── App ───────────────────────────────────────────────────────────────────────

LTM_USER_ID: str = _get_setting("LTM_USER_ID", "u1")
COOKIE_SECRET: str = _get_setting("COOKIE_SECRET", "change-me-cookie-secret")

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful AI assistant with access to tools and persistent long-term memory.

If user-specific memory is available, use it to personalize responses by:
  - Addressing the user by name when appropriate
  - Referencing known projects, tools, or preferences
  - Keeping the tone friendly and tailored to the user

Only personalize based on KNOWN facts — never assume anything not in memory.

User's long-term memory:
{user_details_content}

Thread ID: {thread_id}
{pdf_context}
RULE: When calling rag_tool you MUST pass thread_id="{thread_id}".
"""

MEMORY_PROMPT = """\
You are a precise memory manager. Your job is to decide what should be stored or updated in the user's long-term memory.

EXISTING MEMORIES (with IDs):
{user_details_with_keys}

TASK — analyse the latest user message and:
1. Extract facts worth storing long-term (name, preferences, ongoing projects, tools used, goals).
2. For each extracted fact:
   - If it is genuinely NEW (not covered by any existing memory), set is_new=true.
   - If it is essentially the SAME as an existing memory, set is_new=false (skip it).
   - If it CONTRADICTS or UPDATES an existing memory (e.g. "I switched from Vue to React"),
     set is_new=true AND set replaces_key to the ID of the memory being replaced.
    - Assign one category: profile, preferences, projects, or goals.
3. Keep each memory as a single, short, atomic sentence.
4. Do NOT speculate — only store what the user explicitly stated.
5. If nothing is memory-worthy, return should_write=false with an empty memories list.

{format_instructions}
"""
