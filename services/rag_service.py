import os
import asyncio
import time
import logging
import re
from typing import List, Dict, Any, Optional

import httpx
import openai

from .langchain_memory import LangChainMemoryService
from .prompts import prompt_two

# ---- Constants / Config ----
MAX_MESSAGES = 20
TOP_K = 4  # retrieval top-k (keep small for latency/cost)
CONTEXT_CHAR_BUDGET = 8000  # ~6-8k chars ≈ ~2k tokens (roughly)
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.5"))
SUPABASE_MATCH_FN = os.getenv("SUPABASE_MATCH_FN", "match_documents_justin")  # make configurable
MATCH_THRESHOLD = float(os.getenv("SUPABASE_MATCH_THRESHOLD", "0.4"))

ACRONYM_MAP = {
    "TCP": "The Collaborative Process",
}

_acro_re_cache: Dict[str, re.Pattern] = {
    k: re.compile(rf"\b{re.escape(k)}\b", flags=re.IGNORECASE) for k in ACRONYM_MAP
}

def expand_acronyms(q: str) -> str:
    """Replace known acronyms in-place, once per occurrence."""
    for acro, full in ACRONYM_MAP.items():
        q = _acro_re_cache[acro].sub(f"{acro} ({full})", q)
    return q

def pack_context(docs: List[Dict[str, Any]], char_budget: int = CONTEXT_CHAR_BUDGET) -> str:
    """Concatenate doc contents up to a character budget (high-score docs first)."""
    parts, used = [], 0
    for d in docs:
        txt = (d.get("content") or "").strip()
        if not txt:
            continue
        remaining = char_budget - used
        if remaining <= 0:
            break
        if len(txt) > remaining:
            txt = txt[:remaining]
        parts.append(txt)
        used += len(txt)
    return "\n\n".join(parts)


# ---- Memory subclass using shared httpx client ----
class _PatchedLangChainMemoryService(LangChainMemoryService):
    """
    Overrides LangChainMemoryService I/O to use a shared httpx.AsyncClient
    without modifying the original module.
    """
    def __init__(self, client: httpx.AsyncClient):
        super().__init__()
        self._client = client

    async def add_user_message(self, session_id: str, message: str) -> None:
        payload = [{
            'session_id': session_id,
            'message_type': 'human',
            'content': message,
            'created_at': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }]
        try:
            await self._client.post(
                f"{self.supabase_url}/rest/v1/langchain_chat_history",
                headers=self.supabase_headers,
                json=payload,
            )
        except httpx.HTTPError:
            logging.getLogger("uvicorn.error").exception("Error adding user message")

    async def add_ai_message(self, session_id: str, message: str) -> None:
        payload = [{
            'session_id': session_id,
            'message_type': 'ai',
            'content': message,
            'created_at': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }]
        try:
            await self._client.post(
                f"{self.supabase_url}/rest/v1/langchain_chat_history",
                headers=self.supabase_headers,
                json=payload,
            )
        except httpx.HTTPError:
            logging.getLogger("uvicorn.error").exception("Error adding AI message")

    async def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        params = {
            "session_id": f"eq.{session_id}",
            "select": "message_type,content,created_at",
            "order": "created_at.asc",
            "limit": str(limit)
        }
        try:
            response = await self._client.get(
                f"{self.supabase_url}/rest/v1/langchain_chat_history",
                headers=self.supabase_headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            if not data:
                return []
            return [
                {"role": "user" if m["message_type"] == "human" else "assistant", "content": m["content"]}
                for m in data
            ]
        except httpx.HTTPError:
            logging.getLogger("uvicorn.error").exception("Error getting conversation history")
            return []

    async def clear_memory(self, session_id: str) -> None:
        try:
            await self._client.delete(
                f"{self.supabase_url}/rest/v1/langchain_chat_history",
                headers=self.supabase_headers,
                params={"session_id": f"eq.{session_id}"},
            )
        except httpx.HTTPError:
            logging.getLogger("uvicorn.error").exception("Error clearing memory")


# ---- RAG service ----
class RAGService:
    def __init__(self, client: httpx.AsyncClient, logger: Optional[logging.Logger] = None):
        self.client = client
        self.logger = logger or logging.getLogger("uvicorn.error")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set")
        self.openai_client = openai.AsyncOpenAI(api_key=api_key)

        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and key must be set in environment variables.")

        self.supabase_headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
        }

        # Use patched memory that shares the pooled httpx client
        self.memory_service = _PatchedLangChainMemoryService(client=self.client)

    async def aclose(self):
        # Close OpenAI client; shared httpx client is closed by app
        try:
            await self.openai_client.close()
        except Exception:
            self.logger.exception("Error closing OpenAI client")

    async def get_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text using OpenAI."""
        try:
            resp = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return resp.data[0].embedding
        except Exception as e:
            self.logger.exception("Error generating embedding")
            raise RuntimeError("Embedding generation failed") from e

    async def search_similar_documents(self, query_embedding: List[float], limit: int = TOP_K) -> List[Dict[str, Any]]:
        """Search for similar documents in Supabase via RPC."""
        rpc_url = f"{self.supabase_url}/rest/v1/rpc/{SUPABASE_MATCH_FN}"
        payload = {
            "query_embedding": query_embedding,
            "match_threshold": MATCH_THRESHOLD,
            "match_count": limit,
        }
        try:
            response = await self.client.post(
                rpc_url,
                headers=self.supabase_headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data if data else []
        except httpx.HTTPError:
            self.logger.exception("Vector search failed")
            return []  # No broken/incorrect fallback

    async def get_rag_response(self, user_message: str, session_id: Optional[str] = None, max_tokens: int = 500) -> Dict[str, Any]:
        """Generate a RAG-based AI response."""
        t0 = time.monotonic()
        try:
            # Ensure we have a session
            if not session_id:
                session_id = self.memory_service.create_session_id()

            # Store user message first so history includes it
            await self.memory_service.add_user_message(session_id, user_message)

            # Prepare retrieval
            expanded_message = expand_acronyms(user_message)

            # Embedding (timed)
            t_embed_start = time.monotonic()
            query_embedding = await self.get_embedding(expanded_message)
            t_embed = time.monotonic() - t_embed_start

            # Fetch history and docs concurrently (after embedding)
            t_io_start = time.monotonic()
            history_task = asyncio.create_task(
                self.memory_service.get_conversation_history(session_id, limit=MAX_MESSAGES)
            )
            docs_task = asyncio.create_task(
                self.search_similar_documents(query_embedding, limit=TOP_K)
            )
            conversation_history, similar_docs = await asyncio.gather(history_task, docs_task)
            t_io = time.monotonic() - t_io_start

            # Context packing
            sources: List[Dict[str, Any]] = []
            for d in similar_docs:
                sources.append({"id": d.get("id"), "metadata": d.get("metadata", {})})
            context = pack_context(similar_docs, char_budget=CONTEXT_CHAR_BUDGET)

            # Build messages
            system_prompt = prompt_two()  # keeps your existing prompt, expects {context}
            messages = [
                {"role": "system", "content": system_prompt.format(context=context)},
                *conversation_history,
                {"role": "user", "content": user_message},
            ]

            # Clamp tokens server-side (defense-in-depth; request model already validates)
            max_tokens = max(32, min(int(max_tokens), 1000))

            # LLM call (timed)
            t_llm_start = time.monotonic()
            response = await self.openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=max_tokens,
                temperature=OPENAI_TEMPERATURE,
            )
            t_llm = time.monotonic() - t_llm_start

            assistant_response = response.choices[0].message.content

            # Save assistant reply
            await self.memory_service.add_ai_message(session_id, assistant_response)

            # Log timings
            t_total = time.monotonic() - t0
            self.logger.info(
                "rag_response timings session=%s embed=%.3fs io=%.3fs llm=%.3fs total=%.3fs",
                session_id, t_embed, t_io, t_llm, t_total
            )

            return {
                "answer": assistant_response,
                "sources": sources,
                "session_id": session_id
            }

        except Exception as e:
            self.logger.exception("Error generating RAG response")
            # Let the API layer translate to generic 500
            raise RuntimeError("RAG response generation failed") from e

    async def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        return await self.memory_service.get_conversation_history(session_id, limit)

    async def create_new_session(self) -> str:
        return self.memory_service.create_session_id()

    async def clear_session(self, session_id: str) -> bool:
        try:
            await self.memory_service.clear_memory(session_id)
            return True
        except Exception:
            self.logger.exception("Error clearing session")
            return False
