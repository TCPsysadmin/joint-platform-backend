# services/rag_service.py
import os
import asyncio
import time
import logging
import re
import uuid
from typing import List, Dict, Any, Optional

import httpx
import openai

from .langchain_memory import LangChainMemoryService
from .prompts import prompt_two

# ---- Constants / Config ----
MAX_MESSAGES = 20
TOP_K = 4  # retrieval top-k (keep small for latency/cost)
CONTEXT_CHAR_BUDGET = 24000  # ~character budget; consider switching to token-based trimming
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


class RAGService:
    def __init__(self, client: httpx.AsyncClient, logger: Optional[logging.Logger] = None):
        self.client = client
        self.logger = logger or logging.getLogger("uvicorn.error")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set")
        # create OpenAI async client
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

        # Use the memory service that uses the shared httpx client
        self.memory_service = LangChainMemoryService(
            supabase_url=self.supabase_url,
            supabase_key=self.supabase_key,
            client=self.client,
            logger=self.logger,
        )

    async def aclose(self):
        # Close OpenAI client; shared httpx client is closed by app
        try:
            await self.openai_client.close()
        except Exception:
            self.logger.exception("Error closing OpenAI client")

    async def _retry_async_call(self, coro_factory, attempts: int = 3, base_delay: float = 1.0):
        """
        Helper to retry async calls (pass a zero-argument coroutine factory).
        Example: await self._retry_async_call(lambda: self.openai_client.embeddings.create(...))
        """
        last_exc = None
        for attempt in range(attempts):
            try:
                return await coro_factory()
            except Exception as e:
                last_exc = e
                self.logger.exception("Transient error on attempt %s/%s: %s", attempt + 1, attempts, e)
                if attempt < attempts - 1:
                    await asyncio.sleep(base_delay * (2 ** attempt))
        # re-raise last exception
        raise last_exc

    async def get_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text using OpenAI, with retries."""
        try:
            resp = await self._retry_async_call(
                lambda: self.openai_client.embeddings.create(model="text-embedding-3-small", input=text),
                attempts=3,
                base_delay=1.0,
            )
            return resp.data[0].embedding
        except Exception:
            self.logger.exception("Error generating embedding")
            raise RuntimeError("Embedding generation failed")

    async def search_similar_documents(self, query_embedding: List[float], limit: int = TOP_K) -> List[Dict[str, Any]]:
        """Search for similar documents in Supabase via RPC, with retries."""
        rpc_url = f"{self.supabase_url.rstrip('/')}/rest/v1/rpc/{SUPABASE_MATCH_FN}"
        payload = {
            "query_embedding": query_embedding,
            "match_threshold": MATCH_THRESHOLD,
            "match_count": limit,
        }

        # simple retry loop for the HTTP RPC call
        for attempt in range(3):
            try:
                response = await self.client.post(rpc_url, headers=self.supabase_headers, json=payload, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                return data if data else []
            except httpx.HTTPStatusError as e:
                self.logger.exception("Vector search returned HTTP error (attempt %s): %s", attempt + 1, e)
                # If 4xx, don't retry
                status = getattr(e.response, "status_code", None)
                if status and 400 <= status < 500:
                    raise
            except httpx.RequestError as e:
                self.logger.exception("Vector search request error (attempt %s): %s", attempt + 1, e)
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)

        self.logger.error("Vector search failed after retries")
        return []

    async def get_rag_response(self, user_message: str, session_id: Optional[str] = None, max_tokens: int = 500) -> Dict[str, Any]:
        """Generate a RAG-based AI response."""
        t0 = time.monotonic()
        try:
            # Ensure we have a session
            if not session_id:
                session_id = str(uuid.uuid4())

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
            history_task = asyncio.create_task(self.memory_service.get_conversation_history(session_id, limit=MAX_MESSAGES))
            docs_task = asyncio.create_task(self.search_similar_documents(query_embedding, limit=TOP_K))
            conversation_history, similar_docs = await asyncio.gather(history_task, docs_task)
            t_io = time.monotonic() - t_io_start

            # Context packing
            sources: List[Dict[str, Any]] = []
            for d in similar_docs:
                sources.append({"id": d.get("id"), "metadata": d.get("metadata", {})})
            context = pack_context(similar_docs, char_budget=CONTEXT_CHAR_BUDGET)

            # Build messages
            system_prompt = prompt_two()  # expects {context}
            messages = [
                {"role": "system", "content": system_prompt.format(context=context)},
                *conversation_history,
                {"role": "user", "content": user_message},
            ]

            # Clamp tokens server-side (defense-in-depth; request model already validates)
            max_tokens = max(32, min(int(max_tokens), 1000))

            # LLM call (timed) with retries
            t_llm_start = time.monotonic()
            try:
                response = await self._retry_async_call(
                    lambda: self.openai_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=OPENAI_TEMPERATURE,
                    ),
                    attempts=3,
                    base_delay=1.0,
                )
            except Exception:
                self.logger.exception("LLM call failed after retries")
                raise

            t_llm = time.monotonic() - t_llm_start

            # Safely extract assistant text
            assistant_response = ""
            try:
                # library-specific shapes: try the common pattern used previously
                assistant_response = response.choices[0].message.content
            except Exception:
                # last-resort attempt for alternate response shapes
                try:
                    assistant_response = getattr(response.choices[0], "text", "") or ""
                except Exception:
                    assistant_response = ""

            if assistant_response is None:
                assistant_response = ""

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

        except Exception:
            self.logger.exception("Error generating RAG response")
            raise RuntimeError("RAG response generation failed")

    async def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        return await self.memory_service.get_conversation_history(session_id, limit)

    async def create_new_session(self) -> str:
        return str(uuid.uuid4())

    async def clear_session(self, session_id: str) -> bool:
        try:
            await self.memory_service.clear_memory(session_id)
            return True
        except Exception:
            self.logger.exception("Error clearing session")
            return False

    async def health_check_openai(self) -> Dict[str, Any]:
        """Check OpenAI API connectivity and response time."""
        start_time = time.monotonic()
        try:
            # Simple embedding call to test connectivity
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input="health check"
            )
            response_time = time.monotonic() - start_time
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time * 1000, 2),
                "model": "text-embedding-3-small"
            }
        except Exception as e:
            response_time = time.monotonic() - start_time
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": round(response_time * 1000, 2)
            }

    async def health_check_supabase(self) -> Dict[str, Any]:
        """Check Supabase connectivity and response time."""
        start_time = time.monotonic()
        try:
            # Simple query to test connectivity - just check if we can reach the API
            health_url = f"{self.supabase_url.rstrip('/')}/rest/v1/"
            response = await self.client.get(
                health_url,
                headers=self.supabase_headers,
                timeout=5.0
            )
            response_time = time.monotonic() - start_time
            
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time_ms": round(response_time * 1000, 2),
                    "url": self.supabase_url
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}",
                    "response_time_ms": round(response_time * 1000, 2)
                }
        except Exception as e:
            response_time = time.monotonic() - start_time
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": round(response_time * 1000, 2)
            }

    async def health_check_vector_search(self) -> Dict[str, Any]:
        """Check vector search functionality with a simple test query."""
        start_time = time.monotonic()
        try:
            # Create a simple test embedding
            test_embedding = await self.get_embedding("test query")
            
            # Test the vector search function
            results = await self.search_similar_documents(test_embedding, limit=1)
            response_time = time.monotonic() - start_time
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time * 1000, 2),
                "results_count": len(results),
                "function": SUPABASE_MATCH_FN
            }
        except Exception as e:
            response_time = time.monotonic() - start_time
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": round(response_time * 1000, 2)
            }
