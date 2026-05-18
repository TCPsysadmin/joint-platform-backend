from __future__ import annotations

from uuid import UUID

import structlog
from fastapi import HTTPException, Request
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from pydantic import SecretStr
from supabase import AsyncClient, create_async_client

from orchestrator.config import settings
from orchestrator.errors import AuthError
from orchestrator.runtime import RuntimeContext
from orchestrator.tools import registry

logger = structlog.get_logger(__name__)


class _OpenAIEmbedder:
    """Thin async wrapper that satisfies the Embedder protocol."""

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(input=text, model=self._model)
        return response.data[0].embedding


async def resolve_runtime(request: Request) -> RuntimeContext:
    """Build a RuntimeContext from the incoming HTTP request.

    Auth strategy (v1): service-role key — bypasses RLS but passes client_id
    explicitly so queries are still scoped.

    TODO: when auth is wired, mint a per-client JWT with the client_id claim
    and use the anon key + that JWT. RLS becomes the enforcement layer.
    """
    client_id_str = request.headers.get("X-Client-ID", "").strip()
    if not client_id_str:
        raise HTTPException(status_code=400, detail="X-Client-ID header is required")

    try:
        client_id = UUID(client_id_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="X-Client-ID must be a valid UUID") from None

    try:
        supabase: AsyncClient = await create_async_client(
            settings.supabase_url,
            settings.supabase_service_key,
        )
    except Exception as exc:
        logger.error("supabase_client_init_failed", error=str(exc))
        raise AuthError("Could not initialise Supabase client") from exc

    tools = registry.get_tools(supabase, client_id)

    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=SecretStr(settings.openai_api_key),
    )
    embedder = _OpenAIEmbedder(client=openai_client, model=settings.embedding_model)

    return RuntimeContext(
        client_id=client_id,
        search_tool=tools.search,
        doctrine_tool=tools.doctrine,
        publish_tool=tools.publish,
        llm=llm,
        embedder=embedder,
    )
