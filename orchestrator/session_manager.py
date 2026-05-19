from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import structlog
from supabase import AsyncClient

from orchestrator.errors import AuthError
from orchestrator.supabase_json import JsonDict, as_dict, as_dict_list

logger = structlog.get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


async def get_or_create_session(
    svc: AsyncClient,
    session_id: str,
    user_id: UUID,
    client_id: UUID,
    first_message: str | None = None,
) -> None:
    """Ensure a chat_sessions row exists for this session and user.

    - If the session doesn't exist yet: create it (auto-provision on first message).
    - If it already exists and belongs to this user: no-op.
    - If it already exists but belongs to a different user: raise AuthError.
    """
    response = (
        await svc.table("chat_sessions")
        .select("user_id")
        .eq("session_id", session_id)
        .maybe_single()
        .execute()
    )

    if response is None or response.data is None:
        title = (first_message or "").strip()[:100] or "New conversation"
        await (
            svc.table("chat_sessions")
            .insert(
                {
                    "session_id": session_id,
                    "user_id": str(user_id),
                    "client_id": str(client_id),
                    "title": title,
                    "message_count": 0,
                    "last_message_at": _now_iso(),
                }
            )
            .execute()
        )
        logger.info("session_created", session_id=session_id, user_id=str(user_id))
    else:
        row = as_dict(response.data)
        if row is None:
            raise RuntimeError(f"Malformed session row for {session_id!r}")
        existing_uid = str(row["user_id"])
        if existing_uid != str(user_id):
            raise AuthError(f"Session {session_id!r} belongs to a different user")


async def touch_session(svc: AsyncClient, session_id: str) -> None:
    """Increment message_count and bump last_message_at after a completed turn.

    Uses the increment_session_count SQL function for an atomic update.
    """
    await svc.rpc("increment_session_count", {"p_session_id": session_id}).execute()
    logger.debug("session_touched", session_id=session_id)


async def list_sessions(svc: AsyncClient, user_id: UUID) -> list[JsonDict]:
    """Return active sessions for a user, newest first (max 50)."""
    response = (
        await svc.table("chat_sessions")
        .select("session_id, title, message_count, last_message_at, status, created_at")
        .eq("user_id", str(user_id))
        .eq("status", "active")
        .order("last_message_at", desc=True)
        .limit(50)
        .execute()
    )
    return as_dict_list(response.data if response else None)


async def create_session(
    svc: AsyncClient,
    session_id: str,
    user_id: UUID,
    client_id: UUID,
    title: str | None = None,
) -> JsonDict:
    """Explicitly create a session before sending the first message.

    Returns the created row. Raises if session_id already exists.
    """
    response = (
        await svc.table("chat_sessions")
        .insert(
            {
                "session_id": session_id,
                "user_id": str(user_id),
                "client_id": str(client_id),
                "title": title or "New conversation",
                "message_count": 0,
                "last_message_at": _now_iso(),
            }
        )
        .execute()
    )
    rows = as_dict_list(response.data if response else None)
    if not rows:
        raise RuntimeError(f"Failed to create session {session_id!r}")
    logger.info("session_explicitly_created", session_id=session_id, user_id=str(user_id))
    return rows[0]
