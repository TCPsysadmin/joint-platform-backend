from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from orchestrator.auth import resolve_runtime
from orchestrator.checkpointer import get_checkpointer
from orchestrator.errors import AgentError, AuthError
from orchestrator.graph import build_graph
from orchestrator.observability import bind_request_context, configure_logging

logger = structlog.get_logger(__name__)

_VERSION = os.getenv("GIT_SHA", "dev")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    async with get_checkpointer() as checkpointer:
        app.state.checkpointer = checkpointer
        logger.info("startup_complete", version=_VERSION)
        yield
    logger.info("shutdown_complete")


app = FastAPI(title="video-agent", version=_VERSION, lifespan=lifespan)


# ── Request / response models ──────────────────────────────────────────────────


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ConfirmRequest(BaseModel):
    session_id: str
    action: str  # "approved" | "rejected"


# ── Exception handlers ─────────────────────────────────────────────────────────


@app.exception_handler(AuthError)
async def auth_error_handler(_: Request, exc: AuthError) -> JSONResponse:
    return JSONResponse(status_code=401, content={"detail": str(exc)})


@app.exception_handler(AgentError)
async def agent_error_handler(_: Request, exc: AgentError) -> JSONResponse:
    logger.error("agent_error", error=str(exc))
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ── Helpers ────────────────────────────────────────────────────────────────────


def _sse(event: str, data: dict[str, Any]) -> dict[str, str]:
    return {"event": event, "data": json.dumps(data)}


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.post("/chat")
async def chat(request: Request, body: ChatRequest) -> EventSourceResponse:
    runtime = await resolve_runtime(request)
    bind_request_context(
        session_id=body.session_id,
        client_id=str(runtime.client_id),
    )

    graph = build_graph(checkpointer=request.app.state.checkpointer)
    thread_config: dict[str, Any] = {
        "configurable": {
            "runtime": runtime,
            "thread_id": body.session_id,
        }
    }

    input_state: dict[str, Any] = {
        "messages": [HumanMessage(content=body.message)],
        "session_id": body.session_id,
    }

    async def generate() -> AsyncGenerator[dict[str, str], None]:
        try:
            async for chunk in graph.astream(
                input_state,
                config=thread_config,
                stream_mode="updates",
            ):
                for node_name, node_update in chunk.items():
                    safe_update = {
                        k: v
                        for k, v in node_update.items()
                        if k not in ("messages",)
                        and isinstance(v, (str, int, float, bool, list, dict, type(None)))
                    }
                    yield _sse("node", {"node": node_name, "partial_state": safe_update})

            graph_state = await graph.aget_state(thread_config)
            if graph_state.next and "post_stub" in graph_state.next:
                final_rec = graph_state.values.get("final_recommendation")
                yield _sse("awaiting_confirmation", {"recommendation": final_rec})
            else:
                yield _sse("done", {})
        except AgentError as exc:
            logger.error("chat_stream_failed", error=str(exc))
            yield _sse("error", {"detail": str(exc)})

    return EventSourceResponse(generate())


@app.post("/confirm")
async def confirm(request: Request, body: ConfirmRequest) -> EventSourceResponse:
    if body.action not in ("approved", "rejected"):
        raise HTTPException(status_code=422, detail="action must be 'approved' or 'rejected'")

    runtime = await resolve_runtime(request)
    bind_request_context(
        session_id=body.session_id,
        client_id=str(runtime.client_id),
    )

    graph = build_graph(checkpointer=request.app.state.checkpointer)
    thread_config: dict[str, Any] = {
        "configurable": {
            "runtime": runtime,
            "thread_id": body.session_id,
        }
    }

    async def generate() -> AsyncGenerator[dict[str, str], None]:
        await graph.aupdate_state(
            thread_config,
            {
                "confirmation_response": body.action,
                "awaiting_confirmation": False,
            },
        )

        async for chunk in graph.astream(
            None,
            config=thread_config,
            stream_mode="updates",
        ):
            for node_name, _ in chunk.items():
                yield _sse("node", {"node": node_name})

        yield _sse("done", {})

    return EventSourceResponse(generate())


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": _VERSION}


@app.get("/readyz")
async def readyz() -> dict[str, str]:
    """Readiness probe: verifies the Postgres connection string is reachable."""
    import psycopg

    from orchestrator.config import settings

    try:
        async with await psycopg.AsyncConnection.connect(settings.supabase_db_url) as conn:
            await conn.execute("SELECT 1")
    except Exception as exc:
        logger.error("readyz_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Database not reachable") from exc
    return {"status": "ok"}
