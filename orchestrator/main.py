from __future__ import annotations

import datetime
import json
import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from supabase import AsyncClient, create_async_client

from orchestrator import session_documents, session_manager
from orchestrator.auth import (
    admin_create_auth_user,
    extract_bearer,
    get_client_id,
    resolve_runtime,
    verify_token,
)
from orchestrator.checkpointer import get_checkpointer
from orchestrator.config import settings
from orchestrator.confirm_intent import classify_confirmation_reply
from orchestrator.errors import AgentError, AuthError
from orchestrator.graph import build_graph
from orchestrator.observability import bind_request_context, configure_logging
from orchestrator.runtime import RuntimeContext
from orchestrator.sse_sanitize import compact_partial_state
from orchestrator.supabase_json import as_dict
from orchestrator.turn_stream import (
    Publisher,
    TurnAlreadyActiveError,
    TurnStreamRegistry,
    iter_turn_events,
)

logger = structlog.get_logger(__name__)

_VERSION = os.getenv("GIT_SHA", "dev")
_DOCUMENT_UPLOAD = File(...)


# ── Startup / shutdown ─────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    async with get_checkpointer() as checkpointer:
        app.state.checkpointer = checkpointer
        # Shared service-role Supabase client for session management and admin ops.
        app.state.svc = await create_async_client(
            settings.supabase_url,
            settings.supabase_service_key,
        )
        app.state.turn_streams = TurnStreamRegistry()
        logger.info("startup_complete", version=_VERSION)
        try:
            yield
        finally:
            await app.state.turn_streams.cancel_all()
    logger.info("shutdown_complete")


app = FastAPI(title="video-agent", version=_VERSION, lifespan=lifespan)

# CORS — required for a separately deployed frontend.
# Set CORS_ORIGINS="https://your-frontend.com" in production.
_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Accel-Buffering"],  # needed for SSE through nginx proxies
)


# ── Request / response models ──────────────────────────────────────────────────


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ConfirmRequest(BaseModel):
    session_id: str
    action: str  # "approved" | "rejected"


class CreateSessionRequest(BaseModel):
    session_id: str | None = None  # omit to get a server-generated UUID
    title: str | None = None


class AdminCreateUserRequest(BaseModel):
    email: str
    password: str
    client_id: str  # UUID string of the client to link the user to
    role: str = "member"
    display_name: str | None = None


class AdminProvisionClientRequest(BaseModel):
    slug: str
    display_name: str
    source_kind: str = "gdrive"
    plan_tier: str = "standard"
    b2_bucket: str | None = None
    b2_prefix: str | None = None


# ── Exception handlers ─────────────────────────────────────────────────────────


@app.exception_handler(AuthError)
async def auth_error_handler(_: Request, exc: AuthError) -> JSONResponse:
    return JSONResponse(status_code=401, content={"detail": str(exc)})


@app.exception_handler(AgentError)
async def agent_error_handler(_: Request, exc: AgentError) -> JSONResponse:
    logger.error("agent_error", error=str(exc))
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(session_documents.DocumentProcessingError)
async def document_processing_error_handler(
    _: Request,
    exc: session_documents.DocumentProcessingError,
) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# ── Helpers ────────────────────────────────────────────────────────────────────


def _json_default(obj: Any) -> str:
    """Fallback encoder for types json.dumps can't handle natively.

    Node updates can carry UUIDs (and datetimes) nested inside dicts/lists that
    survive compact_partial_state's top-level type filter.
    """
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _sse(event: str, data: dict[str, Any]) -> dict[str, str]:
    return {"event": event, "data": json.dumps(data, default=_json_default)}


def _thread_config(session_id: str, runtime: RuntimeContext | None = None) -> RunnableConfig:
    configurable: dict[str, Any] = {"thread_id": session_id}
    if runtime is not None:
        configurable["runtime"] = runtime
    return {"configurable": configurable}


def _serialize_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """Convert LangGraph BaseMessage objects to JSON-safe role/content dicts.

    HumanMessage → role "user"
    AIMessage     → role "assistant"
    Others (SystemMessage, tool messages, etc.) are skipped — they are
    internal agent plumbing, not part of the user-visible conversation.
    """
    out: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            out.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            out.append({"role": "assistant", "content": str(msg.content)})
    return out


def _require_admin(request: Request) -> None:
    """Guard admin endpoints: require the service role key as Bearer token."""
    auth = request.headers.get("Authorization", "").strip()
    expected = f"Bearer {settings.supabase_service_key}"
    if auth != expected:
        raise HTTPException(status_code=403, detail="Forbidden: admin key required")


async def _client_exists(svc: AsyncClient, client_id: uuid.UUID) -> bool:
    response = (
        await svc.table("clients_registry")
        .select("client_id")
        .eq("client_id", str(client_id))
        .maybe_single()
        .execute()
    )
    return response is not None and response.data is not None


def _turn_streams(request: Request) -> TurnStreamRegistry:
    registry = getattr(request.app.state, "turn_streams", None)
    if isinstance(registry, TurnStreamRegistry):
        return registry
    registry = TurnStreamRegistry()
    request.app.state.turn_streams = registry
    return registry


async def _run_chat_turn(
    *,
    session_id: str,
    message: str,
    svc: AsyncClient,
    runtime: RuntimeContext,
    checkpointer: Any,
    documents: list[dict[str, object]],
    publish: Publisher,
) -> None:
    graph = build_graph(checkpointer=checkpointer)
    thread_config = _thread_config(session_id, runtime)
    # Only these three keys. Any key present here overwrites the checkpointed value
    # for channels without a reducer, so everything the previous turn learned
    # (retrieved_segments, candidate_clips, final_recommendation, brand_doctrine, …)
    # must be left out and restored from the checkpoint. `messages` is safe because
    # it uses the add_messages reducer (appends); `session_documents` is re-read from
    # Postgres each turn and is authoritative.
    input_state: dict[str, Any] = {
        "messages": [HumanMessage(content=message)],
        "session_id": session_id,
        "session_documents": documents,
    }

    try:
        # If the previous turn left the graph parked at the post_stub confirmation
        # interrupt, interpret this message as the creator's response to it.
        prior_state = await graph.aget_state(thread_config)
        if prior_state.next and "post_stub" in prior_state.next:
            decision = await classify_confirmation_reply(runtime.llm, message)
            logger.info(
                "chat_confirmation_reply",
                decision=decision,
                session_id=session_id,
            )
            if decision == "approve":
                # Approve in-place: record the message, resume post_stub so the
                # OpusClip project is actually created, and finish this turn.
                await graph.aupdate_state(
                    thread_config,
                    {
                        "messages": [HumanMessage(content=message)],
                        "confirmation_response": "approved",
                        "awaiting_confirmation": False,
                    },
                )
                async for chunk in graph.astream(None, config=thread_config, stream_mode="updates"):
                    for node_name, node_update in chunk.items():
                        if not isinstance(node_update, dict):
                            continue
                        safe_update = compact_partial_state(node_name, node_update)
                        await publish(
                            _sse("node", {"node": node_name, "partial_state": safe_update})
                        )
                await session_manager.touch_session(svc, session_id)
                await publish(_sse("done", {"action": "opus_project_created"}))
                return

            # reject / other → clear the gate WITHOUT running post_stub (no project
            # created, no canned message), then handle this message as a fresh turn.
            await graph.aupdate_state(
                thread_config,
                {"confirmation_response": "rejected", "awaiting_confirmation": False},
                as_node="post_stub",
            )

        async for chunk in graph.astream(
            input_state,
            config=thread_config,
            stream_mode="updates",
        ):
            for node_name, node_update in chunk.items():
                if not isinstance(node_update, dict):
                    continue  # LangGraph emits __interrupt__ as a tuple; skip it
                safe_update = compact_partial_state(node_name, node_update)
                await publish(_sse("node", {"node": node_name, "partial_state": safe_update}))

        graph_state = await graph.aget_state(thread_config)

        # Bump session turn count after the graph completes.
        await session_manager.touch_session(svc, session_id)

        if graph_state.next and "post_stub" in graph_state.next:
            final_rec = graph_state.values.get("final_recommendation")
            await publish(_sse("awaiting_confirmation", {"recommendation": final_rec}))
        else:
            await publish(_sse("done", {}))

    except AgentError as exc:
        logger.error("chat_stream_failed", error=str(exc))
        await publish(_sse("error", {"detail": str(exc)}))
    except Exception:  # noqa: BLE001 — never leave subscribers without a terminal event
        logger.exception("chat_stream_crashed")
        await publish(_sse("error", {"detail": "Internal agent error"}))


async def _run_confirm_turn(
    *,
    session_id: str,
    action: str,
    svc: AsyncClient,
    runtime: RuntimeContext,
    checkpointer: Any,
    publish: Publisher,
) -> None:
    graph = build_graph(checkpointer=checkpointer)
    thread_config = _thread_config(session_id, runtime)

    try:
        await graph.aupdate_state(
            thread_config,
            {
                "confirmation_response": action,
                "awaiting_confirmation": False,
            },
        )

        async for chunk in graph.astream(
            None,
            config=thread_config,
            stream_mode="updates",
        ):
            for node_name, node_update in chunk.items():
                if not isinstance(node_update, dict):
                    continue
                await publish(_sse("node", {"node": node_name}))

        await session_manager.touch_session(svc, session_id)
        await publish(_sse("done", {}))

    except AgentError as exc:
        logger.error("confirm_stream_failed", error=str(exc))
        await publish(_sse("error", {"detail": str(exc)}))
    except Exception:  # noqa: BLE001 — never leave subscribers without a terminal event
        logger.exception("confirm_stream_crashed")
        await publish(_sse("error", {"detail": "Internal agent error"}))


# ── Chat endpoints ─────────────────────────────────────────────────────────────


@app.post("/chat")
async def chat(request: Request, body: ChatRequest) -> EventSourceResponse:
    svc: AsyncClient = request.app.state.svc
    runtime = await resolve_runtime(request, svc)

    bind_request_context(
        session_id=body.session_id,
        client_id=str(runtime.client_id),
    )

    # Auto-create session on first message; verify ownership on subsequent ones.
    await session_manager.get_or_create_session(
        svc,
        session_id=body.session_id,
        user_id=runtime.user_id,
        client_id=runtime.client_id,
        first_message=body.message,
    )
    documents = await session_documents.list_session_documents(
        svc,
        session_id=body.session_id,
        user_id=runtime.user_id,
    )

    registry = _turn_streams(request)
    try:
        turn = await registry.get_or_start(
            session_id=body.session_id,
            kind="chat",
            fingerprint=body.message,
            runner_factory=lambda publish: _run_chat_turn(
                session_id=body.session_id,
                message=body.message,
                svc=svc,
                runtime=runtime,
                checkpointer=request.app.state.checkpointer,
                documents=documents,
                publish=publish,
            ),
        )
    except TurnAlreadyActiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return EventSourceResponse(iter_turn_events(turn))


@app.post("/confirm")
async def confirm(request: Request, body: ConfirmRequest) -> EventSourceResponse:
    if body.action not in ("approved", "rejected"):
        raise HTTPException(status_code=422, detail="action must be 'approved' or 'rejected'")

    svc: AsyncClient = request.app.state.svc
    runtime = await resolve_runtime(request, svc)

    bind_request_context(
        session_id=body.session_id,
        client_id=str(runtime.client_id),
    )

    # Verify the session exists and belongs to this user before allowing confirmation.
    await session_manager.get_or_create_session(
        svc,
        session_id=body.session_id,
        user_id=runtime.user_id,
        client_id=runtime.client_id,
    )

    registry = _turn_streams(request)
    try:
        turn = await registry.get_or_start(
            session_id=body.session_id,
            kind="confirm",
            fingerprint=body.action,
            runner_factory=lambda publish: _run_confirm_turn(
                session_id=body.session_id,
                action=body.action,
                svc=svc,
                runtime=runtime,
                checkpointer=request.app.state.checkpointer,
                publish=publish,
            ),
        )
    except TurnAlreadyActiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return EventSourceResponse(iter_turn_events(turn))


# ── Session endpoints ──────────────────────────────────────────────────────────


@app.post("/sessions", status_code=201)
async def create_session(request: Request, body: CreateSessionRequest) -> dict[str, Any]:
    """Explicitly create a session before sending the first message.

    Useful for the frontend to pre-register a session and get its ID.
    If session_id is omitted, a server-generated UUID is returned.
    """
    svc: AsyncClient = request.app.state.svc
    token = extract_bearer(request)
    user_id = await verify_token(token)
    client_id = await get_client_id(user_id, svc)

    sid = body.session_id or str(uuid.uuid4())
    row = await session_manager.create_session(
        svc,
        session_id=sid,
        user_id=user_id,
        client_id=client_id,
        title=body.title,
    )
    return row


@app.post("/sessions/{session_id}/documents", status_code=201)
async def upload_session_document(
    session_id: str,
    request: Request,
    file: UploadFile = _DOCUMENT_UPLOAD,
) -> dict[str, Any]:
    """Upload a text-like document and attach it as session-scoped agent context."""
    svc: AsyncClient = request.app.state.svc
    token = extract_bearer(request)
    user_id = await verify_token(token)
    client_id = await get_client_id(user_id, svc)

    await session_manager.get_or_create_session(
        svc,
        session_id=session_id,
        user_id=user_id,
        client_id=client_id,
    )

    processed = await session_documents.process_upload_file(file)
    row = await session_documents.create_session_document(
        svc,
        session_id=session_id,
        user_id=user_id,
        client_id=client_id,
        document=processed,
    )

    logger.info(
        "session_document_uploaded",
        session_id=session_id,
        user_id=str(user_id),
        client_id=str(client_id),
        filename=processed.filename,
        byte_size=processed.byte_size,
    )

    return {
        "doc_id": str(row.get("doc_id")),
        "session_id": session_id,
        "filename": processed.filename,
        "content_type": processed.content_type,
        "byte_size": processed.byte_size,
        "char_count": processed.char_count,
        "summary": processed.summary,
        "status": row.get("status", "ready"),
        "created_at": row.get("created_at"),
    }


@app.get("/sessions")
async def list_sessions(request: Request) -> dict[str, Any]:
    """List the authenticated user's active sessions, newest first."""
    svc: AsyncClient = request.app.state.svc
    token = extract_bearer(request)
    user_id = await verify_token(token)

    sessions = await session_manager.list_sessions(svc, user_id)
    return {"sessions": sessions}


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, request: Request) -> dict[str, Any]:
    """Return the full message history for a session.

    Messages are reconstructed from the LangGraph checkpoint state.
    Only the authenticated session owner can retrieve messages.
    """
    svc: AsyncClient = request.app.state.svc
    token = extract_bearer(request)
    user_id = await verify_token(token)

    # Ownership check: verify the session belongs to this user.
    response = (
        await svc.table("chat_sessions")
        .select("user_id")
        .eq("session_id", session_id)
        .maybe_single()
        .execute()
    )
    row = as_dict(response.data if response else None)
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if str(row["user_id"]) != str(user_id):
        raise HTTPException(status_code=403, detail="Forbidden")

    # Load message history from LangGraph checkpoint state.
    runtime_client_id = await get_client_id(user_id, svc)
    graph = build_graph(checkpointer=request.app.state.checkpointer)
    thread_config = _thread_config(session_id)

    graph_state = await graph.aget_state(thread_config)
    raw_messages: list[BaseMessage] = graph_state.values.get("messages", [])
    messages = _serialize_messages(raw_messages)
    turn_status = await _turn_streams(request).get_status(session_id)

    return {
        "session_id": session_id,
        "client_id": str(runtime_client_id),
        "turn_status": turn_status,
        "message_count": len(messages),
        "messages": messages,
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict[str, Any]:
    """Archive (soft-delete) a session so it no longer appears in GET /sessions.

    This is a soft delete: the session's status is flipped to 'archived' and its
    LangGraph checkpoint history is left intact. Only the authenticated session
    owner can archive it. Idempotent — archiving an already-archived session
    returns the same 404 as a missing one (no active session to archive).
    """
    svc: AsyncClient = request.app.state.svc
    token = extract_bearer(request)
    user_id = await verify_token(token)

    archived = await session_manager.archive_session(svc, session_id, user_id)
    if not archived:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "status": "archived"}


# ── Admin endpoints ────────────────────────────────────────────────────────────


@app.post("/admin/clients", status_code=201)
async def admin_provision_client(
    request: Request,
    body: AdminProvisionClientRequest,
) -> dict[str, Any]:
    """Provision a tenant and seed its default doctrine.

    Requires the service role key as Bearer token (Authorization header).
    """
    _require_admin(request)

    svc: AsyncClient = request.app.state.svc
    response = await svc.rpc(
        "admin_provision_client",
        {
            "p_slug": body.slug,
            "p_display_name": body.display_name,
            "p_source_kind": body.source_kind,
            "p_b2_bucket": body.b2_bucket,
            "p_b2_prefix": body.b2_prefix,
            "p_plan_tier": body.plan_tier,
        },
    ).execute()

    client_id = str(response.data)
    logger.info(
        "admin_client_provisioned",
        client_id=client_id,
        slug=body.slug,
    )
    return {
        "client_id": client_id,
        "slug": body.slug,
        "display_name": body.display_name,
        "source_kind": body.source_kind,
        "b2_bucket": body.b2_bucket,
        "b2_prefix": body.b2_prefix,
        "plan_tier": body.plan_tier,
    }


@app.post("/admin/users", status_code=201)
async def admin_create_user(request: Request, body: AdminCreateUserRequest) -> dict[str, Any]:
    """Create a Supabase Auth user and link them to a client.

    Requires the service role key as Bearer token (Authorization header).

    Steps:
    1. Create the auth user in Supabase Auth (email_confirm=True so no email needed).
    2. Insert a user_profiles row linking the user to the given client_id.

    After this, the user can immediately authenticate via:
        POST <SUPABASE_URL>/auth/v1/token?grant_type=password
        body: {"email": "...", "password": "..."}
    and use the returned access_token as a Bearer token on all API calls.
    """
    _require_admin(request)

    try:
        client_uuid = uuid.UUID(body.client_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="client_id must be a valid UUID") from None

    svc: AsyncClient = request.app.state.svc

    if not await _client_exists(svc, client_uuid):
        raise HTTPException(
            status_code=404,
            detail=(
                "client_id was not found in clients_registry. "
                "Provision the client first with POST /admin/clients."
            ),
        )

    # 1. Create auth user.
    try:
        user_id = await admin_create_auth_user(body.email, body.password)
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 2. Link to client in user_profiles.
    await svc.rpc(
        "admin_link_user_to_client",
        {
            "p_user_id": str(user_id),
            "p_client_id": str(client_uuid),
            "p_role": body.role,
            "p_display_name": body.display_name,
        },
    ).execute()

    logger.info(
        "admin_user_created",
        user_id=str(user_id),
        client_id=str(client_uuid),
        email=body.email,
    )
    return {
        "user_id": str(user_id),
        "client_id": str(client_uuid),
        "email": body.email,
        "role": body.role,
    }


# ── Health ─────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": _VERSION}


@app.get("/readyz")
async def readyz() -> dict[str, str]:
    """Readiness probe: verifies the Postgres connection is reachable."""
    import psycopg

    try:
        async with await psycopg.AsyncConnection.connect(settings.supabase_db_url) as conn:
            await conn.execute("SELECT 1")
    except Exception as exc:
        logger.error("readyz_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Database not reachable") from exc
    return {"status": "ok"}
