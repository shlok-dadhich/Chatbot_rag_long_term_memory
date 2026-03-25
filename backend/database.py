"""
backend/database.py
-------------------
Sets up the shared PostgreSQL connection pool, LangGraph checkpointer (conversation
persistence) and LangGraph store (long-term memory).
"""

import atexit
import os

from typing import Any

try:
    from psycopg_pool import ConnectionPool
except Exception:  # pragma: no cover - optional at runtime
    ConnectionPool = None  # type: ignore[assignment]

from langgraph.checkpoint.memory import MemorySaver

try:
    from langgraph.checkpoint.postgres import PostgresSaver
except Exception:  # pragma: no cover - optional at runtime
    PostgresSaver = None  # type: ignore[assignment]

try:
    from langgraph.store.postgres import PostgresStore
except Exception:  # pragma: no cover - optional at runtime
    PostgresStore = None  # type: ignore[assignment]

from langgraph.store.memory import InMemoryStore

from backend.config import DB_URI

# ── Shared connection pool ────────────────────────────────────────────────────

pool: Any = None
PERSISTENCE_MODE: str = "memory"
checkpointer: Any = MemorySaver()
postgres_store: Any = InMemoryStore()


def _can_use_postgres() -> bool:
    explicit_db_url = os.getenv("DATABASE_URL", "").strip()
    if not explicit_db_url:
        return False

    local_markers = ("localhost", "127.0.0.1")
    if any(marker in DB_URI for marker in local_markers):
        return False

    return bool(ConnectionPool and PostgresSaver and PostgresStore)


if _can_use_postgres():
    try:
        pool = ConnectionPool(
            conninfo=DB_URI,
            max_size=20,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )

        candidate_checkpointer = PostgresSaver(pool)
        candidate_checkpointer.setup()

        candidate_store = PostgresStore(pool)
        candidate_store.setup()

        checkpointer = candidate_checkpointer
        postgres_store = candidate_store
        PERSISTENCE_MODE = "postgres"
    except Exception:
        if pool is not None:
            try:
                pool.close()
            except Exception:
                pass
        pool = None
        PERSISTENCE_MODE = "memory"


def _close_pool() -> None:
    if pool is not None:
        try:
            pool.close()
        except Exception:
            # Avoid shutdown-time exceptions from bubbling during interpreter finalization.
            pass


atexit.register(_close_pool)

# ── Checkpointer/store are initialized above with a safe fallback ────────────
