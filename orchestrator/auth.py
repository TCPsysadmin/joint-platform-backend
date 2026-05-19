from __future__ import annotations

from uuid import UUID

import httpx
import structlog
from fastapi import Request
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from pydantic import SecretStr
from supabase import AsyncClient, create_async_client

from orchestrator.config import settings
from orchestrator.errors import AuthError
from orchestrator.runtime import RuntimeContext
from orchestrator.supabase_json import as_dict
from orchestrator.tools import registry

logger = structlog.get_logger(__name__)


class _OpenAIEmbedder:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(input=text, model=self._model)
        return response.data[0].embedding


def extract_bearer(request: Request) -> str:
    """Pull the Bearer token from the Authorization header."""
    auth = request.headers.get("Authorization", "").strip()
    if not auth.startswith("Bearer "):
        raise AuthError("Authorization header missing or not Bearer scheme")
    token = auth[len("Bearer ") :].strip()
    if not token:
        raise AuthError("Bearer token is empty")
    return token


async def verify_token(token: str) -> UUID:
    """Verify a Supabase Auth JWT and return the user's UUID.

    Calls the Supabase /auth/v1/user endpoint, which validates the JWT
    signature and expiry server-side.

    For testing without a frontend, obtain a token via:
        curl -X POST '<SUPABASE_URL>/auth/v1/token?grant_type=password' \\
          -H 'apikey: <SUPABASE_ANON_KEY>' \\
          -H 'Content-Type: application/json' \\
          -d '{"email": "user@example.com", "password": "password"}'
    Use the returned access_token as the Bearer value.
    """
    async with httpx.AsyncClient(timeout=10.0) as http:
        resp = await http.get(
            f"{settings.supabase_url}/auth/v1/user",
            headers={
                "apikey": settings.supabase_anon_key,
                "Authorization": f"Bearer {token}",
            },
        )

    if resp.status_code == 401:
        raise AuthError("Token is invalid or has expired")
    if resp.status_code != 200:
        raise AuthError(f"Auth verification failed (status {resp.status_code})")

    try:
        return UUID(resp.json()["id"])
    except (KeyError, ValueError) as exc:
        raise AuthError("Unexpected user payload from Supabase Auth") from exc


async def get_client_id(user_id: UUID, svc: AsyncClient) -> UUID:
    """Look up the client_id linked to a user via user_profiles."""
    response = (
        await svc.table("user_profiles")
        .select("client_id")
        .eq("user_id", str(user_id))
        .maybe_single()
        .execute()
    )
    if response is None or response.data is None:
        raise AuthError(
            f"No client profile found for user {user_id}. "
            "Ensure the user has been linked to a client via POST /admin/users."
        )
    row = as_dict(response.data)
    if row is None:
        raise AuthError("Malformed user profile row")
    try:
        return UUID(str(row["client_id"]))
    except (KeyError, ValueError) as exc:
        raise AuthError("Malformed client_id in user profile") from exc


async def resolve_runtime(request: Request, svc: AsyncClient) -> RuntimeContext:
    """Build a RuntimeContext from the incoming request.

    Auth flow:
    1. Extract Bearer JWT from Authorization header.
    2. Verify token with Supabase /auth/v1/user → get user_id.
    3. Look up client_id from user_profiles (service role).
    4. Build RuntimeContext with both IDs + tools/LLM/embedder.
    """
    token = extract_bearer(request)
    user_id = await verify_token(token)
    client_id = await get_client_id(user_id, svc)

    logger.info("auth_resolved", user_id=str(user_id), client_id=str(client_id))

    supabase_for_tools: AsyncClient = await create_async_client(
        settings.supabase_url,
        settings.supabase_service_key,
    )
    tools = registry.get_tools(supabase_for_tools, client_id)

    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=SecretStr(settings.openai_api_key),
    )
    embedder = _OpenAIEmbedder(client=openai_client, model=settings.embedding_model)

    return RuntimeContext(
        client_id=client_id,
        user_id=user_id,
        search_tool=tools.search,
        doctrine_tool=tools.doctrine,
        publish_tool=tools.publish,
        llm=llm,
        embedder=embedder,
    )


async def admin_create_auth_user(email: str, password: str) -> UUID:
    """Create a Supabase Auth user via the Admin API and return their UUID.

    Requires the service role key. Called by POST /admin/users.
    email_confirm=True skips the confirmation email so the account is
    immediately usable for testing.
    """
    async with httpx.AsyncClient(timeout=15.0) as http:
        resp = await http.post(
            f"{settings.supabase_url}/auth/v1/admin/users",
            headers={
                "apikey": settings.supabase_service_key,
                "Authorization": f"Bearer {settings.supabase_service_key}",
                "Content-Type": "application/json",
            },
            json={
                "email": email,
                "password": password,
                "email_confirm": True,
            },
        )

    if resp.status_code not in (200, 201):
        raise AuthError(f"Failed to create auth user: {resp.text}")

    try:
        return UUID(resp.json()["id"])
    except (KeyError, ValueError) as exc:
        raise AuthError("Unexpected response from Supabase Auth Admin API") from exc
