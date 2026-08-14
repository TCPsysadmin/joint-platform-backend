from __future__ import annotations

import datetime
import json
import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import structlog
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from supabase import AsyncClient, create_async_client

from orchestrator import media_library, session_documents, session_manager
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
from orchestrator.supabase_json import as_dict, as_dict_list
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


class RegisterMediaStorageRequest(BaseModel):
    source_video_id: str
    title: str | None = None
    source_file: str | None = None
    b2_bucket: str
    b2_path: str
    thumbnail_b2_path: str | None = None


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
        client_id=runtime.client_id,
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
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key", max_length=128)] = None,
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
        upload_request_id=idempotency_key,
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
        "filename": row.get("filename") or processed.filename,
        "content_type": row.get("content_type") or processed.content_type,
        "byte_size": row.get("byte_size") or processed.byte_size,
        "char_count": row.get("char_count") or processed.char_count,
        "summary": row.get("summary") or processed.summary,
        "status": row.get("status", "ready"),
        "created_at": row.get("created_at"),
    }


@app.get("/sessions/{session_id}/documents")
async def get_session_documents(session_id: str, request: Request) -> dict[str, Any]:
    """List the documents currently attached to an authenticated user's session."""
    svc: AsyncClient = request.app.state.svc
    token = extract_bearer(request)
    user_id = await verify_token(token)
    client_id = await get_client_id(user_id, svc)

    response = (
        await svc.table("chat_sessions")
        .select("user_id,client_id")
        .eq("session_id", session_id)
        .maybe_single()
        .execute()
    )
    row = as_dict(response.data if response else None)
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if str(row["user_id"]) != str(user_id):
        raise HTTPException(status_code=403, detail="Forbidden")
    if str(row.get("client_id") or "") != str(client_id):
        raise HTTPException(status_code=403, detail="Forbidden")

    documents = await session_documents.list_session_documents(
        svc,
        session_id=session_id,
        user_id=user_id,
        client_id=client_id,
    )
    return {
        "session_id": session_id,
        "documents": [
            {
                "doc_id": str(document.get("doc_id")),
                "filename": document.get("filename"),
                "content_type": document.get("content_type"),
                "byte_size": document.get("byte_size"),
                "char_count": document.get("char_count"),
                "summary": document.get("summary"),
                "status": "ready",
                "created_at": document.get("created_at"),
            }
            for document in documents
        ],
    }


@app.get("/sessions")
async def list_sessions(request: Request) -> dict[str, Any]:
    """List the authenticated user's active sessions, newest first."""
    svc: AsyncClient = request.app.state.svc
    token = extract_bearer(request)
    user_id = await verify_token(token)
    client_id = await get_client_id(user_id, svc)

    sessions = await session_manager.list_sessions(svc, user_id, client_id)
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
    runtime_client_id = await get_client_id(user_id, svc)

    # Ownership check: verify the session belongs to this user and tenant.
    response = (
        await svc.table("chat_sessions")
        .select("user_id,client_id")
        .eq("session_id", session_id)
        .maybe_single()
        .execute()
    )
    row = as_dict(response.data if response else None)
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if str(row["user_id"]) != str(user_id):
        raise HTTPException(status_code=403, detail="Forbidden")
    if str(row.get("client_id") or "") != str(runtime_client_id):
        raise HTTPException(status_code=403, detail="Forbidden")

    # Load message history from LangGraph checkpoint state.
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
    client_id = await get_client_id(user_id, svc)

    archived = await session_manager.archive_session(svc, session_id, user_id, client_id)
    if not archived:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "status": "archived"}


# ── Media library endpoints ───────────────────────────────────────────────────


@app.get("/ingestion/config")
async def get_ingestion_config(request: Request) -> dict[str, Any]:
    """Return the signed-in tenant's configured ingestion destination."""
    svc: AsyncClient = request.app.state.svc
    token = extract_bearer(request)
    user_id = await verify_token(token)
    client_id = await get_client_id(user_id, svc)
    response = (
        await svc.table("clients_registry")
        .select(
            "client_id,display_name,drive_transcripts_intake_folder_id,"
            "drive_summaries_intake_folder_id,"
            "drive_transcripts_completed_folder_id,"
            "drive_summaries_completed_folder_id"
        )
        .eq("client_id", str(client_id))
        .maybe_single()
        .execute()
    )
    row = as_dict(response.data if response is not None else None)
    if not row:
        raise HTTPException(
            status_code=404,
            detail="Client ingestion configuration not found",
        )

    required = {
        "transcripts_folder_id": row.get("drive_transcripts_intake_folder_id"),
        "summaries_folder_id": row.get("drive_summaries_intake_folder_id"),
        "transcripts_completed_folder_id": row.get("drive_transcripts_completed_folder_id"),
        "summaries_completed_folder_id": row.get("drive_summaries_completed_folder_id"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise HTTPException(
            status_code=409,
            detail="Client ingestion folders are incomplete: " + ", ".join(missing),
        )

    return {
        "client_id": str(client_id),
        "name": str(row.get("display_name") or "Workspace"),
        **required,
    }


@app.get("/ingestion/destinations")
async def list_ingestion_destinations(request: Request) -> dict[str, Any]:
    """List only the authenticated user's configured Drive destination."""
    svc: AsyncClient = request.app.state.svc
    token = extract_bearer(request)
    user_id = await verify_token(token)
    client_id = await get_client_id(user_id, svc)

    response = (
        await svc.table("clients_registry")
        .select(
            "client_id,display_name,drive_transcripts_intake_folder_id,"
            "drive_summaries_intake_folder_id,"
            "drive_transcripts_completed_folder_id,"
            "drive_summaries_completed_folder_id"
        )
        .eq("client_id", str(client_id))
        .eq("status", "active")
        .order("display_name")
        .execute()
    )
    destinations: list[dict[str, str]] = []
    rows = as_dict_list(response.data if response is not None else None)
    for row in rows:
        # Defense in depth for this service-role query: never serialize a row
        # outside the authenticated tenant even if the upstream filter changes.
        if str(row.get("client_id") or "") != str(client_id):
            continue
        destination = {
            "client_id": str(row.get("client_id") or ""),
            "name": str(row.get("display_name") or "Workspace"),
            "transcripts_folder_id": str(row.get("drive_transcripts_intake_folder_id") or ""),
            "summaries_folder_id": str(row.get("drive_summaries_intake_folder_id") or ""),
            "transcripts_completed_folder_id": str(
                row.get("drive_transcripts_completed_folder_id") or ""
            ),
            "summaries_completed_folder_id": str(
                row.get("drive_summaries_completed_folder_id") or ""
            ),
        }
        if all(destination.values()):
            destinations.append(destination)

    return {"destinations": destinations}


@app.get("/media")
async def list_media(
    request: Request,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """List the authenticated tenant's videos as Drive-style folder cards."""
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    if offset < 0:
        raise HTTPException(status_code=422, detail="offset must be non-negative")

    svc: AsyncClient = request.app.state.svc
    runtime = await resolve_runtime(request, svc)
    items = await media_library.list_videos(
        svc,
        client_id=runtime.client_id,
        limit=limit,
        offset=offset,
    )
    await _attach_signed_thumbnail_urls(runtime, items)
    return {
        "items": items,
        "count": len(items),
        "limit": limit,
        "offset": offset,
    }


@app.post("/media/storage")
async def register_media_storage(
    body: RegisterMediaStorageRequest,
    request: Request,
) -> dict[str, Any]:
    """Attach archived B2 objects to a tenant-owned media-library record."""
    svc: AsyncClient = request.app.state.svc
    runtime = await resolve_runtime(request, svc)

    client_response = (
        await svc.table("clients_registry")
        .select("b2_bucket")
        .eq("client_id", str(runtime.client_id))
        .maybe_single()
        .execute()
    )
    client = as_dict(client_response.data if client_response else None)
    configured_bucket = str((client or {}).get("b2_bucket") or "").strip()
    if not configured_bucket:
        raise HTTPException(status_code=409, detail="Client B2 bucket is not configured")
    if body.b2_bucket.strip() != configured_bucket:
        raise HTTPException(status_code=409, detail="B2 bucket does not match this workspace")

    try:
        return await media_library.register_storage(
            svc,
            client_id=runtime.client_id,
            source_video_id=body.source_video_id,
            title=body.title,
            source_file=body.source_file,
            b2_path=body.b2_path,
            thumbnail_b2_path=body.thumbnail_b2_path,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/media/{source_video_id}")
async def get_media(source_video_id: str, request: Request) -> dict[str, Any]:
    """Open one video folder, including its summary and full transcript."""
    svc: AsyncClient = request.app.state.svc
    runtime = await resolve_runtime(request, svc)
    item = await media_library.get_video(
        svc,
        client_id=runtime.client_id,
        source_video_id=source_video_id,
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Video not found")
    await _attach_signed_thumbnail_urls(runtime, [item])
    return item


async def _attach_signed_thumbnail_urls(
    runtime: RuntimeContext,
    items: list[dict[str, Any]],
) -> None:
    """Fill private thumbnail URLs without exposing B2 credentials or bucket paths."""
    paths = [
        str(thumbnail["b2_path"])
        for item in items
        if isinstance((thumbnail := item.get("thumbnail")), dict)
        and not thumbnail.get("url")
        and thumbnail.get("b2_path")
    ]
    if not paths:
        return
    try:
        signed = await runtime.file_tool.get_path_download_urls(
            paths,
            valid_duration_seconds=settings.b2_download_url_ttl_seconds,
        )
    except Exception:
        logger.exception(
            "media_thumbnail_signing_failed",
            client_id=str(runtime.client_id),
        )
        return
    for item in items:
        thumbnail = item.get("thumbnail")
        if not isinstance(thumbnail, dict) or thumbnail.get("url"):
            continue
        path = str(thumbnail.get("b2_path") or "")
        if path in signed:
            thumbnail["url"] = signed[path]


@app.get("/media/{source_video_id}/video-url")
async def get_media_video_url(source_video_id: str, request: Request) -> dict[str, Any]:
    """Return a short-lived B2 URL for downloading one tenant-owned source video."""
    svc: AsyncClient = request.app.state.svc
    runtime = await resolve_runtime(request, svc)

    # Verify the media record belongs to this tenant before resolving storage.
    item = await media_library.get_video(
        svc,
        client_id=runtime.client_id,
        source_video_id=source_video_id,
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Video not found")

    url = await runtime.file_tool.get_download_url(
        source_video_id,
        valid_duration_seconds=settings.b2_download_url_ttl_seconds,
    )
    if not url:
        raise HTTPException(status_code=404, detail="Video file is not available")

    raw_video = item.get("video")
    video = raw_video if isinstance(raw_video, dict) else {}
    return {
        "url": url,
        "filename": video.get("source_file"),
        "expires_in": settings.b2_download_url_ttl_seconds,
    }


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
