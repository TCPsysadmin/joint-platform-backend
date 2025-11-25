# services/rag_service.py
import os
import asyncio
import time
import logging
import re
from typing import List, Dict, Any, Optional, AsyncGenerator

import httpx
import openai
from xai_sdk import AsyncClient
from xai_sdk.chat import user, system, assistant
from .langchain_memory import LangChainMemoryService
from .prompts import *
from .encryption import AESEncryptor

# ---- Constants / Config ----
MAX_MESSAGES = 20
TOP_K = 3  # retrieval top-k (keep small for latency/cost)
CONTEXT_CHAR_BUDGET = 15000  # Optimized for 3 documents - allows ~5000 chars per doc (~1250 tokens per doc)
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.5"))
SUPABASE_MATCH_FN = os.getenv("SUPABASE_MATCH_FN", "tcpdb_v2_search")  # make configurable
MAX_GROK_REQUESTS = 40

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
            # Truncate if needed, but always include at least something from first doc
            txt = txt[:remaining]
        parts.append(txt)
        used += len(txt)
    return "\n\n".join(parts) if parts else ""  # Return empty string if no docs, not None


class RAGService:
    def __init__(self, client: httpx.AsyncClient, logger: Optional[logging.Logger] = None, encryptor: AESEncryptor = None):
        self.client = client
        self.logger = logger or logging.getLogger("uvicorn.error")

        openai_api_key = os.getenv("OPENAI_API_KEY")
        x_api_key = os.getenv("GROK_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set")
        if not x_api_key:
            raise ValueError("OPENAI_API_KEY must be set")
        # create OpenAI async client
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.grok_client = AsyncClient(api_key=x_api_key, timeout=4800)
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
            encryptor=encryptor
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
                attempts=2,  # Reduced retries for faster failure
                base_delay=0.5,  # Reduced delay for faster retries
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
            "match_count": limit,
            "filter": {}
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

    async def get_rag_response(
        self, 
        user_message: str, 
        contact_id: str,
        session_id: Optional[str] = None, 
        max_tokens: int = 500, 
        model: Optional[str] = None,
        history_token_limit: Optional[int] = 1000
    ) -> Dict[str, Any]:
        """Generate a RAG-based AI response."""
        t0 = time.monotonic()
        try:
            # Ensure we have a session
            if not session_id:
                session_id = await self.memory_service.create_session(contact_id)

            # Check if this is the first message and set title if so
            await self.handle_first_message_title(session_id, contact_id, user_message)
            
            # Store user message first so history includes it
            await self.memory_service.add_user_message(session_id, contact_id, user_message)

            # Prepare retrieval
            expanded_message = expand_acronyms(user_message)

            # Embedding (timed)
            t_embed_start = time.monotonic()
            query_embedding = await self.get_embedding(expanded_message)
            t_embed = time.monotonic() - t_embed_start

            # Fetch history and docs concurrently (after embedding)
            # Use token-based trimming instead of message count
            t_io_start = time.monotonic()
            history_task = asyncio.create_task(
                self.memory_service.get_conversation_history(
                    session_id, 
                    contact_id, 
                    max_tokens=history_token_limit,
                    include_summary=True
                )
            )
            docs_task = asyncio.create_task(self.search_similar_documents(query_embedding, limit=TOP_K))
            conversation_history, similar_docs = await asyncio.gather(history_task, docs_task)
            t_io = time.monotonic() - t_io_start

            # Context packing
            sources: List[Dict[str, Any]] = []
            for d in similar_docs:
                sources.append({"id": d.get("id"), "metadata": d.get("metadata", {})})
            context = pack_context(similar_docs, char_budget=CONTEXT_CHAR_BUDGET)

            # Build messages
            system_prompt = prompt_seven()  # expects {context}
            messages = [
                {"role": "system", "content": system_prompt.format(context=context)},
                *conversation_history,
                {"role": "user", "content": user_message},
            ]

            # Clamp tokens server-side (defense-in-depth; request model already validates)
            max_tokens = max(32, min(int(max_tokens), 1000))

            # Determine which model to use
            selected_model = model if model else MODEL_NAME
            
            # Validate model selection
            valid_models = ["gpt-5", "gpt-4.1", "gpt-5-nano", "gpt-5-mini", "gpt-4o-mini"]
            if model and model not in valid_models:
                raise ValueError(f"Invalid model '{model}'. Valid options: {', '.join(valid_models)}")

            # Get model-specific parameters
            model_params = {"max_tokens": max_tokens}

            # LLM call (timed) with retries
            t_llm_start = time.monotonic()
            try:
                response = await self._retry_async_call(
                    lambda: self.openai_client.chat.completions.create(
                        model=selected_model,
                        messages=messages,
                        **model_params,
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
            await self.memory_service.add_ai_message(session_id, contact_id, assistant_response)
            
            # Check and update summary if needed (async, non-blocking)
            asyncio.create_task(
                self.check_and_update_summary(session_id, contact_id)
            )

            # Log timings
            t_total = time.monotonic() - t0
            self.logger.info(
                "rag_response timings session=%s embed=%.3fs io=%.3fs llm=%.3fs total=%.3fs",
                session_id, t_embed, t_io, t_llm, t_total
            )

            return {
                "answer": assistant_response,
                "sources": sources,
                "session_id": session_id,
                "model": selected_model
            }

        except Exception:
            self.logger.exception("Error generating RAG response")
            raise RuntimeError("RAG response generation failed")

    async def get_conversation_history(
        self, 
        session_id: str, 
        contact_id: str,
        max_tokens: Optional[int] = None,
        include_summary: bool = True
    ) -> List[Dict[str, Any]]:
        return await self.memory_service.get_conversation_history(
            session_id, 
            contact_id, 
            max_tokens=max_tokens,
            include_summary=include_summary
        )

    async def get_recent_sessions(self, contact_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the N most recent sessions for a user."""
        return await self.memory_service.get_recent_sessions(contact_id, limit)

    async def create_new_session(self, contact_id: str) -> str:
        """Create a new session for a user and return the session_id."""
        return await self.memory_service.create_session(contact_id)

    async def clear_session(self, session_id: str, contact_id: str) -> bool:
        """
        Archive a session (sets is_archived = true).
        Messages remain linked but session won't appear in active sessions.
        """
        try:
            return await self.memory_service.delete_session(session_id, contact_id)
        except Exception:
            self.logger.exception("Error deleting session")
            return False

    async def generate_session_summary(
        self,
        session_id: str,
        contact_id: str,
        existing_summary: Optional[str],
        new_messages: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Generate a summary of the conversation using the LLM.
        
        Args:
            session_id: Session ID
            contact_id: User's contact ID
            existing_summary: Previous summary if it exists
            new_messages: New messages to summarize (newest first)
        
        Returns:
            Generated summary string, or None if generation failed
        """
        try:
            # Format messages for summarization (reverse to chronological order)
            messages_for_summary = []
            for msg in reversed(new_messages):
                role = "user" if msg.get("role") == "human" else "assistant"
                content = msg.get("content", "")
                messages_for_summary.append(f"{role.capitalize()}: {content}")
            
            conversation_text = "\n\n".join(messages_for_summary)
            
            # Build summary prompt
            if existing_summary:
                summary_prompt = f"""You are summarizing a conversation. There is already a summary of earlier messages:

Previous Summary:
{existing_summary}

New Messages Since Last Summary:
{conversation_text}

Please create a concise, comprehensive summary that:
1. Incorporates the previous summary
2. Adds key information from the new messages
3. Maintains important context and decisions
4. Is no more than 3-4 sentences

Summary:"""
            else:
                summary_prompt = f"""You are summarizing a conversation. Please create a concise summary that captures:
1. The main topics discussed
2. Key decisions or conclusions
3. Important context for future conversations
4. Is no more than 3-4 sentences

Conversation:
{conversation_text}

Summary:"""

            # Generate summary using OpenAI (use a cheaper/faster model for summaries)
            response = await self._retry_async_call(
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",  # Use cheaper model for summaries
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates concise conversation summaries."},
                        {"role": "user", "content": summary_prompt}
                    ],
                    max_tokens=200,  # Keep summaries short
                    temperature=0.3  # Lower temperature for more consistent summaries
                ),
                attempts=2,
                base_delay=1.0,
            )

            summary = response.choices[0].message.content.strip()
            self.logger.info("Generated summary for session %s: %s", session_id, summary[:100])
            return summary

        except Exception as e:
            self.logger.exception("Failed to generate session summary for session %s", session_id)
            return None

    async def generate_session_title(self, summary: str) -> Optional[str]:
        """
        Generate a concise title (max 60 characters) from the session summary.
        
        Args:
            summary: The session summary text
        
        Returns:
            A short title string, or None if generation failed
        """
        try:
            # Use LLM to generate a concise title from the summary
            title_prompt = f"""Based on this conversation summary, generate a short, descriptive title (maximum 60 characters, no quotes):

Summary:
{summary}

Title:"""

            response = await self._retry_async_call(
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates concise, descriptive titles for conversations."},
                        {"role": "user", "content": title_prompt}
                    ],
                    max_tokens=20,  # Titles should be short
                    temperature=0.5
                ),
                attempts=2,
                base_delay=1.0,
            )

            title = response.choices[0].message.content.strip()
            # Remove quotes if present
            title = title.strip('"\'')
            # Truncate to 60 characters if needed
            if len(title) > 60:
                title = title[:57] + "..."
            
            self.logger.info("Generated title: %s", title)
            return title

        except Exception as e:
            self.logger.exception("Failed to generate session title")
            # Fallback: use first sentence of summary, truncated
            if summary:
                first_sentence = summary.split('.')[0].strip()
                if len(first_sentence) > 60:
                    return first_sentence[:57] + "..."
                return first_sentence
            return None

    async def handle_first_message_title(self, session_id: str, contact_id: str, user_message: str) -> None:
        """
        Check if this is the first message in a session and set the title if so.
        
        Args:
            session_id: Session ID
            contact_id: User's contact ID
            user_message: The user message that might be the first one
        """
        try:
            # Check if this session has any existing messages
            history = await self.memory_service.get_conversation_history(
                session_id, 
                contact_id, 
                max_tokens=1,  # Just need to check if any messages exist
                include_summary=False
            )
            
            # If no history exists, this is the first message - generate and set title
            if not history:
                title = user_message if len(user_message) < 30 else user_message[:30]
                await self.memory_service.update_session_title(session_id, contact_id, title)
                self.logger.info("Set title for new session %s: %s", session_id, title)
                
        except Exception as e:
            # Don't fail the main request if title generation fails
            self.logger.exception("Error setting title for first message in session %s", session_id)

    async def check_and_update_summary(
        self,
        session_id: str,
        contact_id: str,
        message_count_threshold: int = 12,
        token_count_threshold: int = 1500
    ) -> None:
        """
        Check if summary should be updated and generate/update it if needed.
        This should be called after adding messages to a session.
        """
        try:
            should_update, metadata, new_messages = await self.memory_service.should_update_summary(
                session_id,
                contact_id,
                message_count_threshold=message_count_threshold,
                token_count_threshold=token_count_threshold
            )

            if not should_update or not new_messages:
                return

            # Get existing summary
            existing_summary = metadata.get("summary") if metadata else None

            # Generate new summary
            summary = await self.generate_session_summary(
                session_id,
                contact_id,
                existing_summary,
                new_messages
            )

            if summary:
                # Find the most recent message ID (newest message has highest ID)
                # Messages are newest first, so first message has the highest ID
                last_message_id = None
                for msg in new_messages:
                    msg_id = msg.get("id")
                    if msg_id:
                        if last_message_id is None or msg_id > last_message_id:
                            last_message_id = msg_id

                if last_message_id:
                    # Generate a title from the summary
                    title = await self.generate_session_title(summary)
                    
                    await self.memory_service.update_session_summary(
                        session_id,
                        contact_id,
                        summary,
                        last_message_id,
                        title=title
                    )
                    self.logger.info("Updated summary for session %s (last_message_id: %s)", session_id, last_message_id)
                else:
                    self.logger.warning("Could not determine last_message_id for session %s, skipping summary update", session_id)

        except Exception as e:
            # Don't fail the main request if summary generation fails
            self.logger.exception("Error in check_and_update_summary for session %s", session_id)

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

    async def get_gpt_response_stream(
            self,
            user_message: str,
            contact_id: str,
            session_id: Optional[str] = None,
            max_tokens: int = 700,
            model: Optional[str] = None,
            history_token_limit: Optional[int] = 1000
        ) -> AsyncGenerator[Dict[str, Any], None]:
            """Generate a streaming RAG-based AI response using Server-Sent Events."""
            t0 = time.monotonic()

            try:
                # Ensure we have a session
                if not session_id:
                    session_id = await self.memory_service.create_session(contact_id)

                yield {
                    "type": "session",
                    "session_id": session_id,
                    "timestamp": time.time()
                }

                # Check if this is the first message and set title if so
                await self.handle_first_message_title(session_id, contact_id, user_message)
                
                await self.memory_service.add_user_message(session_id, contact_id, user_message)

                yield {
                    "type": "status",
                    "message": "Processing query...",
                    "timestamp": time.time()
                }

                # Expand acronyms and get embedding
                expanded_message = expand_acronyms(user_message)

                t_embed_start = time.monotonic()
                query_embedding = await self.get_embedding(expanded_message)
                t_embed = time.monotonic() - t_embed_start

                yield {
                    "type": "status",
                    "message": "Searching knowledge base...",
                    "timestamp": time.time()
                }

                # Retrieve history and docs concurrently
                # Use token-based trimming instead of message count
                t_io_start = time.monotonic()
                history_task = asyncio.create_task(
                    self.memory_service.get_conversation_history(
                        session_id, 
                        contact_id,
                        max_tokens=history_token_limit,
                        include_summary=True
                    )
                )
                docs_task = asyncio.create_task(
                    self.search_similar_documents(query_embedding, limit=TOP_K)
                )
                conversation_history, similar_docs = await asyncio.gather(history_task, docs_task)
                t_io = time.monotonic() - t_io_start

                # Prepare context & sources
                sources = [{"id": d.get("id"), "metadata": d.get("metadata", {})} for d in similar_docs]
                context = pack_context(similar_docs, char_budget=CONTEXT_CHAR_BUDGET)

                yield {
                    "type": "sources",
                    "sources": sources,
                    "timestamp": time.time()
                }

                # Build messages for the LLM
                system_prompt = prompt_seven()
                messages = [
                    {"role": "system", "content": system_prompt.format(context=context)},
                    *conversation_history,
                    {"role": "user", "content": user_message},
                ]

                max_tokens = max(32, min(int(max_tokens), 1000))
                selected_model = model if model else MODEL_NAME

                valid_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4o-mini"]
                if model != "gpt-4o-mini":
                    yield {
                        "type": "error",
                        "error": f"Invalid model '{model}'. Valid options: {', '.join(valid_models)}",
                        "timestamp": time.time()
                    }
                    return

                yield {
                    "type": "status",
                    "message": f"Generating response using {selected_model}...",
                    "timestamp": time.time()
                }

                # Fixed parameter handling — always use `max_tokens`
                model_params = get_model_params(selected_model, max_tokens)

                t_llm_start = time.monotonic()
                full_response = ""

                try:
                    stream = await self._retry_async_call(
                        lambda: self.openai_client.chat.completions.create(
                            model=selected_model,
                            messages=messages,
                            stream=True,
                            **model_params,
                        ),
                        attempts=3,
                        base_delay=1.0,
                    )

                    async for chunk in stream:
                        try:
                            delta = getattr(chunk.choices[0], "delta", None)
                            content = getattr(delta, "content", None) if delta else None
                        except Exception:
                            content = None

                        if content:
                            full_response += content
                            yield {
                                "type": "content",
                                "content": content,
                                "timestamp": time.time()
                            }

                except Exception as e:
                    self.logger.exception("LLM streaming call failed after retries")
                    yield {
                        "type": "error",
                        "error": f"Failed to generate response: {str(e)}",
                        "timestamp": time.time()
                    }
                    return

                t_llm = time.monotonic() - t_llm_start

                # Save the full AI response
                if full_response:
                    await self.memory_service.add_ai_message(session_id, contact_id, full_response)
                    
                    # Check and update summary if needed (async, non-blocking)
                    asyncio.create_task(
                        self.check_and_update_summary(session_id, contact_id)
                    )

                t_total = time.monotonic() - t0
                self.logger.info(
                    "rag_response_stream timings session=%s embed=%.3fs io=%.3fs llm=%.3fs total=%.3fs",
                    session_id, t_embed, t_io, t_llm, t_total
                )

                yield {
                    "type": "done",
                    "session_id": session_id,
                    "model": selected_model,
                    "total_time": t_total,
                    "timestamp": time.time()
                }

            except Exception as e:
                self.logger.exception("Error generating streaming RAG response")
                yield {
                    "type": "error",
                    "error": f"RAG response generation failed: {str(e)}",
                    "timestamp": time.time()
                }
                    
    async def get_grok_response_stream(
                self,
                user_message: str,
                contact_id: str,
                session_id: Optional[str] = None,
                max_tokens: int = 1500,
                model: Optional[str] = None,
                history_token_limit: Optional[int] = 1000
            ) -> AsyncGenerator[Dict[str, Any], None]:
                """Generate a streaming RAG-based AI response using Server-Sent Events."""
                t0 = time.monotonic()

                try:
                    # Ensure we have a session
                    if not session_id:
                        session_id = await self.memory_service.create_session(contact_id)

                    yield {
                        "type": "session",
                        "session_id": session_id,
                        "timestamp": time.time()
                    }

                    # Start message storage and embedding generation in parallel
                    expanded_message = expand_acronyms(user_message)
                    
                    # Check if this is the first message and set title, then store message and generate embedding concurrently
                    title_task = asyncio.create_task(
                        self.handle_first_message_title(session_id, contact_id, user_message)
                    )
                    store_message_task = asyncio.create_task(
                        self.memory_service.add_user_message(session_id, contact_id, user_message)
                    )
                    embedding_task = asyncio.create_task(
                        self.get_embedding(expanded_message)
                    )
                    
                    yield {
                        "type": "status",
                        "message": "Processing query...",
                        "timestamp": time.time()
                    }
                    
                    # Wait for all to complete
                    t_embed_start = time.monotonic()
                    await title_task  # Ensure title is set if needed
                    await store_message_task  # Ensure message is stored
                    query_embedding = await embedding_task
                    t_embed = time.monotonic() - t_embed_start

                    yield {
                        "type": "status",
                        "message": "Searching knowledge base...",
                        "timestamp": time.time()
                    }

                    # Retrieve history and docs concurrently
                    # Use token-based trimming instead of message count
                    t_io_start = time.monotonic()
                    history_task = asyncio.create_task(
                        self.memory_service.get_conversation_history(
                            session_id, 
                            contact_id,
                            max_tokens=history_token_limit,
                            include_summary=True
                        )
                    )
                    docs_task = asyncio.create_task(
                        self.search_similar_documents(query_embedding, limit=TOP_K)
                    )
                    conversation_history, similar_docs = await asyncio.gather(history_task, docs_task)
                    t_io = time.monotonic() - t_io_start

                    # Prepare context & sources
                    sources = [{"id": d.get("id"), "metadata": d.get("metadata", {})} for d in similar_docs]
                    context_budget = CONTEXT_CHAR_BUDGET  # Already optimized, no need for min cap
                    context = pack_context(similar_docs, char_budget=context_budget)

                    yield {
                        "type": "sources",
                        "sources": sources,
                        "timestamp": time.time()
                    }

                    # Build messages for the LLM
                    system_prompt = prompt_seven()
                    messages = [
                        system(system_prompt.format(context=context))
                    ]

                    for msg in conversation_history:
                        if msg["role"] == "assistant":
                            messages.append(assistant(msg["content"]))
                        else:
                            messages.append(user(msg["content"]))
                    messages.append(user(user_message))

                    max_tokens = 1200
                    selected_model = "grok-4-1-fast-reasoning"

                    t_llm_start = time.monotonic()
                    full_response = ""

                    try:
                        chat = await self._retry_async_call(
                                lambda: asyncio.to_thread(
                                    self.grok_client.chat.create,
                                    model=selected_model,
                                    max_tokens=max_tokens,
                                    messages=messages,
                                ),
                                attempts=2,  # Reduced retries for faster failure
                                base_delay=0.5,  # Reduced delay for faster retries
                            )

                        async for response, chunk in chat.stream():
                            content = getattr(chunk, "content", None)
                            if content:
                                full_response += content
                                yield {
                                    "type": "content",
                                    "content": content,
                                    "timestamp": time.time()
                                }

                    except Exception as e:
                        self.logger.exception("LLM streaming call failed after retries")
                        yield {
                            "type": "error",
                            "error": f"Failed to generate response: {str(e)}",
                            "timestamp": time.time()
                        }
                        return
                    finally:
                        t_llm = time.monotonic() - t_llm_start

                    # Save the full AI response (non-blocking - don't wait for it)
                    if full_response:
                        # Save message in background to not block response
                        asyncio.create_task(
                            self.memory_service.add_ai_message(session_id, contact_id, full_response)
                        )
                        
                        # Check and update summary if needed (async, non-blocking)
                        asyncio.create_task(
                            self.check_and_update_summary(session_id, contact_id)
                        )

                    t_total = time.monotonic() - t0
                    self.logger.info(
                        "rag_response_stream timings session=%s embed=%.3fs io=%.3fs llm=%.3fs total=%.3fs",
                        session_id, t_embed, t_io, t_llm, t_total
                    )

                    yield {
                        "type": "done",
                        "session_id": session_id,
                        "model": selected_model,
                        "total_time": t_total,
                        "timestamp": time.time()
                    }

                except Exception as e:
                    self.logger.exception("Error generating streaming RAG response")
                    yield {
                        "type": "error",
                        "error": f"RAG response generation failed: {str(e)}",
                        "timestamp": time.time()
                    }


def get_model_params(model: str, max_tokens: int) -> Dict[str, Any]:
    params = {}
    new_token_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1"]

    if model in new_token_models:
        params["max_completion_tokens"] = 2000
    else:
        params["max_tokens"] = max_tokens

    return params