# services/langchain_memory.py
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import httpx


class LangChainMemoryService:
    """
    Memory service that stores conversation history in Supabase (via REST).
    This version expects the Supabase URL/key and a shared httpx.AsyncClient to be
    passed in at construction (no dotenv loading here).
    """

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        client: httpx.AsyncClient,
        logger: Optional[logging.Logger] = None,
        request_timeout: float = 10.0,
        retry_attempts: int = 3,
    ):
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase URL and key must be provided to LangChainMemoryService")

        self.supabase_url = supabase_url.rstrip("/")
        self.supabase_key = supabase_key
        self._client = client
        self.logger = logger or logging.getLogger("langchain_memory")
        self.request_timeout = request_timeout
        self.retry_attempts = max(1, int(retry_attempts))

        self.supabase_headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
        }

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    async def _request_with_retries(
        self,
        method: str,
        path_or_url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
    ) -> httpx.Response:
        """
        Generic HTTP request wrapper with retries for Supabase operations.
        method should be "get", "post", or "delete".
        path_or_url can be a full URL or a path (e.g. "/rest/v1/langchain_chat_history")
        """
        if path_or_url.lower().startswith("http"):
            url = path_or_url
        else:
            url = f"{self.supabase_url}/{path_or_url.lstrip('/')}"

        for attempt in range(self.retry_attempts):
            try:
                func = getattr(self._client, method.lower())
                # Only pass json parameter for methods that support request bodies
                if method.lower() in ['post', 'put', 'patch']:
                    resp = await func(
                        url,
                        headers=self.supabase_headers,
                        params=params,
                        json=json,
                        timeout=self.request_timeout,
                    )
                else:
                    resp = await func(
                        url,
                        headers=self.supabase_headers,
                        params=params,
                        timeout=self.request_timeout,
                    )
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as e:
                # 4xx errors are usually client errors — don't keep retrying
                status = getattr(e.response, "status_code", None)
                self.logger.exception(
                    "Supabase HTTP status error (attempt %s/%s) url=%s status=%s",
                    attempt + 1,
                    self.retry_attempts,
                    url,
                    status,
                )
                if status and 400 <= status < 500:
                    # Bubble it up — payload or authorization issue
                    raise
            except httpx.RequestError as e:
                self.logger.exception(
                    "Supabase request error (attempt %s/%s) url=%s error=%s",
                    attempt + 1,
                    self.retry_attempts,
                    url,
                    e,
                )

            # Backoff before next retry (exponential)
            if attempt < self.retry_attempts - 1:
                await asyncio.sleep(2 ** attempt)

        # After retries exhausted
        raise RuntimeError(f"Supabase request failed after {self.retry_attempts} attempts: {url}")

    async def add_user_message(self, session_id: str, message: str) -> None:
        """Add user message using Supabase function."""
        rpc_url = f"{self.supabase_url.rstrip('/')}/rest/v1/rpc/add_user_message"
        payload = {
            "session_id_param": session_id,
            "message_content": message
        }
        try:
            await self._request_with_retries(
                "post", rpc_url, json=payload
            )
        except Exception:
            self.logger.exception("Failed to add user message to Supabase")
            raise

    async def add_ai_message(self, session_id: str, message: str) -> None:
        """Add AI message using Supabase function."""
        rpc_url = f"{self.supabase_url.rstrip('/')}/rest/v1/rpc/add_ai_message"
        payload = {
            "session_id_param": session_id,
            "message_content": message
        }
        try:
            await self._request_with_retries(
                "post", rpc_url, json=payload
            )
        except Exception:
            self.logger.exception("Failed to add AI message to Supabase")
            raise

    async def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Get conversation history using Supabase function."""
        rpc_url = f"{self.supabase_url.rstrip('/')}/rest/v1/rpc/get_conversation_history"
        payload = {
            "session_id_param": session_id,
            "message_limit": limit
        }
        try:
            resp = await self._request_with_retries("post", rpc_url, json=payload)
            data = resp.json()
            if not data:
                return []
            # Format for OpenAI (role/content)
            return [
                {"role": "user" if msg["message_type"] == "human" else "assistant", "content": msg["content"]}
                for msg in data
            ]
        except Exception:
            self.logger.exception("Failed to fetch conversation history from Supabase")
            return []

    async def format_messages_for_openai(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        return await self.get_conversation_history(session_id, limit)

    async def clear_memory(self, session_id: str) -> None:
        """Clear conversation history using Supabase function."""
        rpc_url = f"{self.supabase_url.rstrip('/')}/rest/v1/rpc/clear_conversation_history"
        payload = {
            "session_id_param": session_id
        }
        try:
            await self._request_with_retries("post", rpc_url, json=payload)
        except Exception:
            self.logger.exception("Failed to clear memory for session %s", session_id)
            raise
