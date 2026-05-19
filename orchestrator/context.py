from __future__ import annotations

from langchain_core.messages import BaseMessage


def build_context_window(messages: list[BaseMessage], window_size: int) -> list[BaseMessage]:
    """Return the last `window_size` messages for LLM prompt construction.

    Applied at node level so the full history is preserved in the checkpointer
    (for session replay / GET /sessions/{id}/messages) while keeping token
    usage bounded on long sessions.

    window_size=0 or window_size >= len(messages) → returns messages unchanged.
    """
    if window_size <= 0 or len(messages) <= window_size:
        return messages
    return messages[-window_size:]
