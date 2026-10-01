from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import structlog
from supabase import AsyncClient

from orchestrator.errors import AuthError
from orchestrator.supabase_json import JsonDict, as_dict, as_dict_list

logger = structlog.get_logger(__name__)

# Title used when a session is pre-created (POST /sessions) with no explicit title,
# before the first real message arrives. Treated as "not yet titled".
_PLACEHOLDER_TITLE = "New conversation"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _derive_title(first_message: str | None) -> str:
    return (first_message or "").strip()[:100] or _PLACEHOLDER_TITLE


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
        .select("user_id, client_id, title, message_count")
        .eq("session_id", session_id)
        .maybe_single()
        .execute()
    )

    if response is None or response.data is None:
        title = _derive_title(first_message)
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
        return

    row = as_dict(response.data)
    if row is None:
        raise RuntimeError(f"Malformed session row for {session_id!r}")
    existing_uid = str(row["user_id"])
    if existing_uid != str(user_id):
        raise AuthError(f"Session {session_id!r} belongs to a different user")
    if str(row.get("client_id") or "") != str(client_id):
        raise AuthError(f"Session {session_id!r} belongs to a different client")

    # The session was pre-created (e.g. POST /sessions) before the first message.
    # If it's still untitled and no turns have happened, claim the title from the
    # first message so the truncated message shows up in the sidebar.
    new_title = _derive_title(first_message)
    if (
        new_title != _PLACEHOLDER_TITLE
        and str(row.get("title") or "") in ("", _PLACEHOLDER_TITLE)
        and int(row.get("message_count") or 0) == 0
    ):
        await (
            svc.table("chat_sessions")
            .update({"title": new_title})
            .eq("session_id", session_id)
            .execute()
        )
        logger.info("session_title_set", session_id=session_id, title=new_title)


async def touch_session(svc: AsyncClient, session_id: str) -> None:
    """Increment message_count and bump last_message_at after a completed turn.

    Uses the increment_session_count SQL function for an atomic update.
    """
    await svc.rpc("increment_session_count", {"p_session_id": session_id}).execute()
    logger.debug("session_touched", session_id=session_id)


async def list_sessions(
    svc: AsyncClient,
    user_id: UUID,
    client_id: UUID,
) -> list[JsonDict]:
    """Return active sessions for a user, newest first (max 50)."""
    response = (
        await svc.table("chat_sessions")
        .select("session_id, title, message_count, last_message_at, status, created_at")
        .eq("user_id", str(user_id))
        .eq("client_id", str(client_id))
        .eq("status", "active")
        .order("last_message_at", desc=True)
        .limit(50)
        .execute()
    )
    return as_dict_list(response.data if response else None)


async def archive_session(
    svc: AsyncClient,
    session_id: str,
    user_id: UUID,
    client_id: UUID,
) -> bool:
    """Soft-delete a session by flipping its status to 'archived'.

    Archived sessions are excluded from list_sessions, so they drop out of the
    sidebar while their LangGraph checkpoint history is preserved. Scoped to the
    owner inside the SQL function so this service-role call cannot touch another
    user's session.

    Returns True if a still-active session was archived, False if no matching
    active session existed (already archived, not found, or wrong owner).
    """
    response = await svc.rpc(
        "archive_session",
        {
            "p_session_id": session_id,
            "p_user_id": str(user_id),
            "p_client_id": str(client_id),
        },
    ).execute()
    archived = bool(response.data)
    logger.info(
        "session_archived" if archived else "session_archive_noop",
        session_id=session_id,
        user_id=str(user_id),
    )
    return archived


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
