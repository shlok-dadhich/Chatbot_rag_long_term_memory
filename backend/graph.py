"""
backend/graph.py
----------------
Defines the LangGraph agent:
  - ChatState (TypedDict)
  - remember_node  — extracts & persists long-term memories
  - chat_node      — main LLM call with tool binding
  - tool_node      — custom tool executor (injects thread_id into rag_tool)
  - Compiled chatbot (checkpointed + store-backed)
"""

from __future__ import annotations

import json
import time
from typing import Annotated, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langgraph.store.base import BaseStore
from typing import TypedDict

from backend.config import SYSTEM_PROMPT_TEMPLATE
from backend.database import checkpointer, postgres_store
from backend.memory import (
    format_memories_plain,
    get_user_memories_raw,
    select_relevant_memories,
    write_memories_from_message,
)
from backend.rag import (
    get_global_metadata,
    get_global_retriever,
    get_thread_metadata,
    has_global_pdf,
    has_thread_pdf,
)
from backend.tools import llm_with_tools, tools


# ── Graph state ───────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    messages:  Annotated[list[BaseMessage], add_messages]
    thread_id: str
    user_id:   str


# ── System-message builder ────────────────────────────────────────────────────

def _build_system_message(
    thread_id: str,
    user_id: str,
    store: Optional[BaseStore],
    latest_user_text: str = "",
) -> SystemMessage:
    pdf_lines = []
    if has_thread_pdf(thread_id):
        pdf_lines.append(f'Thread PDF available — call rag_tool with thread_id="{thread_id}".')
    if has_global_pdf():
        pdf_lines.append("Global shared knowledge base is also available.")
    if not pdf_lines:
        pdf_lines.append("No documents uploaded yet.")

    pdf_context = "PDF Context:\n" + "\n".join(f"- {l}" for l in pdf_lines) + "\n"

    user_details = "(empty)"
    if store and user_id:
        raw      = get_user_memories_raw(user_id, store)
        selected = select_relevant_memories(raw, latest_user_text, max_items=8)
        user_details = format_memories_plain(selected)

    return SystemMessage(
        content=SYSTEM_PROMPT_TEMPLATE.format(
            user_details_content=user_details,
            thread_id=thread_id,
            pdf_context=pdf_context,
        )
    )


# ── Retry wrapper ─────────────────────────────────────────────────────────────

def _invoke_with_retry(
    messages: list[BaseMessage],
    retries: int = 3,
    delay_seconds: float = 0.5,
) -> BaseMessage:
    import logging

    logger = logging.getLogger(__name__)
    last_exc: Optional[Exception] = None
    backoff = delay_seconds
    for attempt in range(retries + 1):
        try:
            return llm_with_tools.invoke(messages)
        except Exception as exc:
            last_exc = exc
            # Log full exception for diagnostics (safe to log message, not secrets)
            logger.exception("LLM invocation failed on attempt %d/%d: %s", attempt + 1, retries + 1, exc)

            # If it's a permanent configuration/auth error, surface a helpful hint.
            err_text = str(exc).lower()
            if any(k in err_text for k in ("unauthorized", "401", "forbidden", "not found", "repo")):
                return AIMessage(
                    content=(
                        "Model error: authentication or model configuration issue detected. "
                        "Please verify your HuggingFace token and `HF_REPO_ID` setting and try again."
                    )
                )

            if attempt < retries:
                time.sleep(backoff)
                backoff *= 2

    # If all retries exhausted, return a friendly connectivity message.
    short = str(last_exc)[:400] if last_exc else "unknown error"
    return AIMessage(
        content=(
            "I am having a temporary model connectivity issue right now. "
            "Please retry your message in a moment.\n\n"
            f"(Error: {short})"
        )
    )


# ── Nodes ─────────────────────────────────────────────────────────────────────

def remember_node(state: ChatState, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Extract memories from the latest user message and persist them."""
    user_id = state.get("user_id") or config.get("configurable", {}).get("user_id", "")
    if not user_id:
        return {}

    last_text = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    write_memories_from_message(user_id, last_text, store)
    return {}


def chat_node(state: ChatState, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Invoke the LLM with the current conversation and injected system message."""
    thread_id = state.get("thread_id", "")
    user_id   = state.get("user_id") or config.get("configurable", {}).get("user_id", "")

    messages = list(state["messages"])
    latest_user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_user_text = (
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )
            break

    sys_msg = _build_system_message(thread_id, user_id, store, latest_user_text=latest_user_text)

    if messages and isinstance(messages[0], SystemMessage):
        messages[0] = sys_msg
    else:
        messages = [sys_msg] + messages

    response = _invoke_with_retry(messages, retries=1)
    return {"messages": [response]}


def tool_node(state: ChatState, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Custom tool executor that force-injects thread_id into every rag_tool call."""
    thread_id = state.get("thread_id", "")

    last_ai = next(
        (m for m in reversed(state["messages"]) if hasattr(m, "tool_calls") and m.tool_calls),
        None,
    )
    if last_ai is None:
        return {"messages": []}

    results = []
    for tc in last_ai.tool_calls:
        name    = tc["name"]
        args    = dict(tc.get("args", {}))
        tool_id = tc["id"]

        if name == "rag_tool":
            args["thread_id"] = thread_id

        matched = next((t for t in tools if t.name == name), None)
        if matched is None:
            output = {"error": f"Tool '{name}' not found."}
        else:
            try:
                output = matched.invoke(args)
            except Exception as exc:
                output = {"error": str(exc)}

        results.append(ToolMessage(
            content=json.dumps(output) if not isinstance(output, str) else output,
            tool_call_id=tool_id,
            name=name,
        ))

    return {"messages": results}


# ── Graph assembly ────────────────────────────────────────────────────────────

_graph = StateGraph(ChatState)

_graph.add_node("remember",  remember_node)
_graph.add_node("chat_node", chat_node)
_graph.add_node("tools",     tool_node)

_graph.add_edge(START,       "remember")
_graph.add_edge("remember",  "chat_node")
_graph.add_conditional_edges("chat_node", tools_condition)
_graph.add_edge("tools",     "chat_node")
_graph.add_edge("chat_node", END)

# Compile with persistence
chatbot = _graph.compile(checkpointer=checkpointer, store=postgres_store)
