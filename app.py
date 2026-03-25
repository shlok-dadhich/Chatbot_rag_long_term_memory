"""
app.py
------
LangGraph Long-Term Memory Chatbot — Streamlit entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import uuid
import warnings
from typing import Any

import streamlit as st

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
    module=r"langchain_core\._api\.deprecation",
)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="LTM Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Backend imports ───────────────────────────────────────────────────────────
from backend.config import LTM_USER_ID
from backend.graph import chatbot
from backend.llm import model
from backend.memory import delete_all_user_memories, delete_user_memory, get_user_memories_list
from backend.rag import ingest_global_pdf, ingest_pdf
from backend.threads import (
    delete_thread_conversation,
    get_thread_metadata,
    save_thread_title,
)

# ── Frontend helpers ──────────────────────────────────────────────────────────
from frontend.styles import CSS_STYLES
from frontend.utils import (
    content_to_text,
    generate_thread_id,
    get_view_mode,
    strip_memory_json,
)

# ── Inject global CSS ─────────────────────────────────────────────────────────
st.markdown(CSS_STYLES, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session-state helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_conversation(thread_id: str) -> list[dict]:
    """Fetch persisted conversation from the LangGraph checkpointer."""
    try:
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id":   st.session_state["user_id"],
            }
        }
        state = chatbot.get_state(config)
        msgs  = state.values.get("messages", []) if state.values else []

        history = []
        for m in msgs:
            if isinstance(m, (SystemMessage, ToolMessage)):
                continue
            if isinstance(m, HumanMessage):
                text = content_to_text(m.content).strip()
                if text:
                    history.append({"role": "user", "content": text})
            elif isinstance(m, AIMessage):
                text = strip_memory_json(content_to_text(m.content)).strip()
                if text:
                    history.append({"role": "assistant", "content": text})
        return history
    except Exception as exc:
        st.toast(f"⚠️ Could not load chat history: {exc}")
        return []


def switch_thread(thread_id: str) -> None:
    st.session_state["thread_id"]       = thread_id
    st.session_state["message_history"] = load_conversation(thread_id)
    st.session_state["last_loaded"]     = thread_id


def add_thread(thread_id: str, title: str = "New Chat") -> None:
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"][thread_id] = {"title": title, "titled": False}


def resolve_user_id() -> str:
    """Resolve active user id from query params/session/env, with fallback."""

    def _normalize(raw: Any) -> str:
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        if not isinstance(raw, str):
            return ""
        cleaned = raw.strip()
        return cleaned[:128] if cleaned else ""

    raw = _normalize(st.query_params.get("user_id") or st.query_params.get("user"))
    if raw:
        return raw

    session_user = _normalize(st.session_state.get("user_id"))
    if session_user:
        return session_user

    env_user = _normalize(LTM_USER_ID)
    if env_user and env_user.lower() not in {"u1", "default", "user"}:
        return env_user

    created = f"anon_{uuid.uuid4().hex}"
    try:
        # Persist the generated id in the URL for stable reload behavior.
        st.query_params["user_id"] = created
    except Exception:
        pass
    return created


# ─────────────────────────────────────────────────────────────────────────────
# Session-state bootstrap (runs once per browser session)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULTS: dict = {
    "chat_threads":    {},
    "ingested_pdfs":   {},
    "thread_titles":   {},
    "message_history": [],
    "last_loaded":     None,
    "user_id":         None,
    "confirm_delete":  None,
}

resolved_user_id = resolve_user_id()

if "initialized" not in st.session_state:
    st.session_state["initialized"] = True
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["user_id"] = resolved_user_id
    st.session_state["chat_threads"] = get_thread_metadata(st.session_state["user_id"]) or {}

    first_id = generate_thread_id()
    add_thread(first_id)
    st.session_state["thread_id"] = first_id

# Ensure all keys exist after hot-reload
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)

if st.session_state.get("user_id") != resolved_user_id:
    st.session_state["user_id"] = resolved_user_id
    st.session_state["chat_threads"] = get_thread_metadata(resolved_user_id) or {}
    st.session_state["thread_titles"] = {}
    st.session_state["message_history"] = []
    st.session_state["last_loaded"] = None
    st.session_state["confirm_delete"] = None

if "thread_id" not in st.session_state:
    first_id = generate_thread_id()
    st.session_state["thread_id"] = first_id

if st.session_state["thread_id"] not in st.session_state["chat_threads"]:
    add_thread(st.session_state["thread_id"])

add_thread(st.session_state["thread_id"])

if st.session_state["last_loaded"] != st.session_state["thread_id"]:
    switch_thread(st.session_state["thread_id"])


# ─────────────────────────────────────────────────────────────────────────────
# Convenience aliases
# ─────────────────────────────────────────────────────────────────────────────

THREAD_ID = st.session_state["thread_id"]
USER_ID   = st.session_state["user_id"]
CONFIG    = {"configurable": {"thread_id": THREAD_ID, "user_id": USER_ID}}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 LangGraph RAG")

    # ── Global PDF Knowledge Base ─────────────────────────────────────────
    with st.expander("🌍 Global Knowledge Base", expanded=False):
        st.caption("Accessible in ALL chats.")
        global_pdf = st.file_uploader(
            "Upload Global PDF",
            type=["pdf"],
            key="global_pdf",
            label_visibility="collapsed",
        )
        if global_pdf:
            _gkey = f"global_{global_pdf.name}_{global_pdf.size}"
            if _gkey not in st.session_state["ingested_pdfs"]:
                try:
                    with st.spinner("Indexing…"):
                        result = ingest_global_pdf(global_pdf.read(), filename=global_pdf.name)
                    st.session_state["ingested_pdfs"][_gkey] = True
                    st.success(f"✅ Indexed **{result['filename']}**")
                except Exception as exc:
                    st.error(f"Failed: {exc}")

    # ── New Chat button ───────────────────────────────────────────────────
    if st.button("➕  New Chat", use_container_width=True):
        new_id = generate_thread_id()
        add_thread(new_id)
        switch_thread(new_id)
        st.rerun()

    # ── Conversation list ─────────────────────────────────────────────────
    st.markdown("### Conversations")

    # Sync titles from DB
    db_meta = get_thread_metadata(USER_ID)
    for tid, info in db_meta.items():
        title = info.get("title") or "New conversation"
        st.session_state["thread_titles"][tid] = title
        st.session_state["chat_threads"].setdefault(tid, {"title": title, "titled": True})
        st.session_state["chat_threads"][tid].update({"title": title, "titled": True})

    all_ids = list(st.session_state["chat_threads"].keys())
    if THREAD_ID not in all_ids:
        all_ids.insert(0, THREAD_ID)

    for tid in reversed(all_ids):
        label = st.session_state["thread_titles"].get(
            tid,
            st.session_state["chat_threads"].get(tid, {}).get("title", "New conversation"),
        )
        is_active = tid == THREAD_ID
        icon      = "💬 " if is_active else "   "

        col_btn, col_del = st.columns([8, 1], gap="small")

        with col_btn:
            if st.button(
                f"{icon}{label}",
                key=f"thread_{tid}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if tid != THREAD_ID:
                    switch_thread(tid)
                    st.rerun()

        with col_del:
            if st.session_state["confirm_delete"] == tid:
                if st.button("✓", key=f"confirm_{tid}", help="Confirm delete"):
                    ok = delete_thread_conversation(tid, USER_ID)
                    st.session_state["chat_threads"].pop(tid, None)
                    st.session_state["thread_titles"].pop(tid, None)
                    st.session_state["ingested_pdfs"] = {
                        k: v
                        for k, v in st.session_state["ingested_pdfs"].items()
                        if not k.startswith(tid)
                    }
                    st.session_state["confirm_delete"] = None

                    if tid == THREAD_ID:
                        new_id = generate_thread_id()
                        add_thread(new_id)
                        switch_thread(new_id)

                    st.toast(
                        "🗑️ Conversation deleted." if ok else "⚠️ Delete failed.",
                        icon="✅" if ok else "⚠️",
                    )
                    st.rerun()
            else:
                if st.button("🗑️", key=f"del_{tid}", help="Delete conversation"):
                    st.session_state["confirm_delete"] = tid
                    st.rerun()

    # ── Long-Term Memory link ─────────────────────────────────────────────
    st.markdown("### Long-Term Memory")
    st.markdown(
        """
        <a href="?view=memories" target="_blank" style="text-decoration:none;color:inherit;">
            <div style="
                display:flex; align-items:center; justify-content:center;
                background:rgba(151,166,195,0.15);
                padding:8px 16px; border-radius:0.5rem;
                border:1px solid rgba(49,51,63,0.2);
                width:100%; cursor:pointer; text-align:center;
                margin-bottom:1rem; transition:background 0.2s;">
                🧠 View Long-Term Memory
            </div>
        </a>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Memory page  (?view=memories)
# ─────────────────────────────────────────────────────────────────────────────

if get_view_mode(st.query_params) == "memories":
    memories = get_user_memories_list(USER_ID)

    st.markdown("""
        <div class="memory-hero">
            <h1>Long-Term Memory</h1>
            <p>Persistent facts and preferences collected across all conversations.</p>
        </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([2, 2, 3])
    with col_a:
        st.metric("Total Memories", len(memories))
    with col_b:
        st.metric("User Profile", USER_ID)
    with col_c:
        if memories:
            if st.button("🗑️ Clear All Memories", type="secondary", use_container_width=True):
                if delete_all_user_memories(USER_ID):
                    st.success("All memories cleared.")
                    st.rerun()
                else:
                    st.error("Failed to clear memories.")

    st.markdown("### Stored Memories")

    if memories:
        for idx, mem in enumerate(memories, start=1):
            mem_col, del_col = st.columns([10, 1], gap="small")
            with mem_col:
                category = (mem.get("category") or "profile").strip().lower()
                st.markdown(f"""
                    <div class="memory-card">
                        <div class="memory-index">
                            Memory #{idx}
                            <span class="memory-category">{category}</span>
                        </div>
                        <div class="memory-text">{mem['data']}</div>
                    </div>
                """, unsafe_allow_html=True)
            with del_col:
                st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
                if st.button("✕", key=f"del_mem_{mem['key']}", help="Delete this memory"):
                    if delete_user_memory(USER_ID, mem["key"]):
                        st.toast("Memory deleted.", icon="✅")
                        st.rerun()
                    else:
                        st.toast("Failed to delete memory.", icon="⚠️")
    else:
        st.markdown("""
            <div class="memory-empty">
                No memories stored yet. Start chatting and share stable preferences,
                goals, or profile details — they'll appear here automatically.
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("[← Back to Chat](?view=chat)")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────────────────────────────────────

_current_title = st.session_state["chat_threads"].get(THREAD_ID, {}).get("title", "New Chat")
st.markdown(
    f'<div class="chat-header"><h1>{_current_title}</h1></div>',
    unsafe_allow_html=True,
)

# Render existing history
for msg in st.session_state.get("message_history", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input (Streamlit ≥ 1.37 — supports paperclip via accept_file) ───────
prompt = st.chat_input("Message LangGraph…", accept_file=True, file_type=["pdf"])

if prompt:
    # Normalise input to a consistent dict shape
    if isinstance(prompt, str):
        prompt_data = {"text": prompt, "files": []}
    elif isinstance(prompt, dict):
        prompt_data = {
            "text":  prompt.get("text", "") or "",
            "files": prompt.get("files", []) or [],
        }
    else:
        prompt_data = {
            "text":  getattr(prompt, "text",  "") or "",
            "files": getattr(prompt, "files", []) or [],
        }

    # 1️⃣  Handle PDF attachment
    if prompt_data.get("files"):
        uploaded_file = prompt_data["files"][0]
        uid = f"{THREAD_ID}_{uploaded_file.name}_{uploaded_file.size}"
        if uid not in st.session_state["ingested_pdfs"]:
            try:
                with st.spinner(f"Indexing {uploaded_file.name}…"):
                    result = ingest_pdf(
                        file_bytes=uploaded_file.read(),
                        thread_id=THREAD_ID,
                        filename=uploaded_file.name,
                    )
                st.session_state["ingested_pdfs"][uid] = True
                st.success(
                    f"✅ **{result['filename']}** indexed ({result['num_chunks']} chunks)."
                )
            except Exception as exc:
                st.error(f"PDF indexing failed: {exc}")

    # 2️⃣  Handle text message
    user_text = prompt_data.get("text", "").strip()
    if not user_text:
        st.stop()

    st.session_state["message_history"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    full_response = ""
    with st.chat_message("assistant"):
        placeholder  = st.empty()
        tools_shown: set[str] = set()
        raw_response = ""

        for msg, metadata in chatbot.stream(
            {
                "messages":  [HumanMessage(content=user_text)],
                "thread_id": THREAD_ID,
                "user_id":   USER_ID,
            },
            config=CONFIG,
            stream_mode="messages",
        ):
            # Only render tokens from chat_node
            if (
                isinstance(metadata, dict)
                and metadata.get("langgraph_node") not in (None, "chat_node")
            ):
                continue

            # Show tool-use indicators
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    name = tc.get("name")
                    if name and name not in tools_shown:
                        st.info(f"🔧 Using tool: **{name}**")
                        tools_shown.add(name)

            # Stream text tokens
            if isinstance(msg, AIMessage) and msg.content:
                chunk = content_to_text(msg.content)
                if chunk:
                    raw_response += chunk
                    placeholder.markdown(strip_memory_json(raw_response))

        full_response = strip_memory_json(raw_response)

    if full_response:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": full_response}
        )

    # 🏷️  Auto-generate thread title on the first real exchange
    thread_data = st.session_state["chat_threads"].get(THREAD_ID, {})
    if not thread_data.get("titled") and full_response:
        try:
            title_resp = model.invoke([HumanMessage(content=(
                "Generate a short, descriptive 3–4 word title for this conversation. "
                "Reply with ONLY the title — no quotes, no punctuation, no explanation.\n\n"
                f"User message: {user_text}"
            ))])
            new_title = (
                title_resp.content.strip().replace('"', "").replace("'", "") or "New Chat"
            )
            st.session_state["chat_threads"][THREAD_ID].update(
                {"title": new_title, "titled": True}
            )
            st.session_state["thread_titles"][THREAD_ID] = new_title
            save_thread_title(THREAD_ID, new_title, USER_ID)
            st.session_state["last_loaded"] = THREAD_ID
            st.rerun()
        except Exception:
            pass
