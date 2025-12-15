# services/langchain_memory.py
import asyncio
import logging
import uuid
import tiktoken
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import httpx
from .encryption import AESEncryptor

class LangChainMemoryService:
    """
    Memory service that stores conversation history in Supabase (via REST).
    Updated to work with new schema: chat_sessions + chat_messages tables.
    Supports contact_id, UUID session_id, token-based trimming, and session summaries.
    """

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        client: httpx.AsyncClient,
        encryptor: AESEncryptor, 
        logger: Optional[logging.Logger] = None,
        request_timeout: float = 5.0,  # Reduced timeout for faster failures
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
        self.encryptor = encryptor
        
        # Initialize tiktoken encoder for token counting (using cl100k_base for GPT-4/Grok)
        try:
            self.token_encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.logger.warning("Failed to load tiktoken, falling back to character-based estimation")
            self.token_encoder = None

        self.supabase_headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
        }

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text. Uses fast estimation for better performance."""
        # Use fast character-based estimation (much faster than tiktoken)
        # 1 token ≈ 4 characters is a good approximation for English text
        return len(text) // 4

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
        path_or_url can be a full URL or a path (e.g. "/rest/v1/rpc/create_session")
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
                # Log the actual error response body for debugging
                try:
                    error_body = e.response.text
                    self.logger.error(
                        "Supabase HTTP status error (attempt %s/%s) url=%s status=%s body=%s",
                        attempt + 1,
                        self.retry_attempts,
                        url,
                        status,
                        error_body,
                    )
                except:
                    self.logger.error(
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

    # ============================================================================
    # SESSION MANAGEMENT
    # ============================================================================

    async def create_session(self, contact_id: str) -> str:
        """
        Create a new chat session for a user and return the session_id (UUID as string).
        The Supabase function automatically handles the 10-session limit by deleting the oldest session if needed.
        """
        rpc_url = f"{self.supabase_url}/rest/v1/rpc/create_encrypted_session"
        payload = {"p_contact_id": contact_id}
        try:
            resp = await self._request_with_retries("post", rpc_url, json=payload)
            session_id = resp.json()
            # Supabase returns UUID as string, ensure it's a string
            return str(session_id) if session_id else str(uuid.uuid4())
        except Exception:
            self.logger.exception("Failed to create session in Supabase")
            raise


    async def get_recent_sessions(self, contact_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the N most recent sessions for a user, ordered by last_message_at DESC.
        Returns list of session dicts with: id, contact_id, created_at, last_message_at, title, summary, is_archived
        """
        # url = f"{self.supabase_url}/rest/v1/chat_sessions"
        url = f"{self.supabase_url}/rest/v1/encrypted_sessions"
        params = {
            "contact_id": f"eq.{contact_id}",
            "is_archived": "eq.false",  # Only get active (non-archived) sessions
            "order": "last_message_at.desc",
            "limit": str(limit),
            "select": (
                "id,"
                "contact_id,"
                "created_at,"
                "last_message_at,"
                "title,"
                "ciphertext,"
                "nonce,"
                "is_archived"
            )
        }

        try:
            # Perform request
            resp = await self._request_with_retries("get", url, params=params)
            data = resp.json()
            # Always returns list (possibly empty)
            return data if data else []

        except httpx.HTTPStatusError as e:
            # Log detailed Supabase error body
            error_body = e.response.text
            try:
                if e.response.headers.get("content-type", "").startswith("application/json"):
                    self.logger.error("Supabase JSON error: %s", e.response.json())
            except Exception:
                pass

            self.logger.exception(
                "Failed to fetch recent sessions from Supabase (status=%s): %s",
                e.response.status_code,
                error_body,
            )
            return []

        except Exception as e:
            self.logger.exception("Unexpected error fetching recent sessions: %s", str(e))
            return []

    async def delete_session(self, session_id: str, contact_id: str) -> bool:
        """
        Archive a session (sets is_archived = true).
        Messages remain linked but session won't appear in active sessions.
        This calls a Supabase RPC function that archives the session.
        Returns True if successful, False otherwise.
        """
        rpc_url = f"{self.supabase_url}/rest/v1/rpc/delete_encrypted_session"
        payload = {
            "p_session_id": session_id,
            "p_contact_id": contact_id
        }
        try:
            await self._request_with_retries("post", rpc_url, json=payload)
            self.logger.info("Deleted session %s for contact_id: %s", session_id, contact_id)
            return True
        except httpx.HTTPStatusError as e:
            # Log the actual error from Supabase for debugging
            error_detail = ""
            try:
                if e.response:
                    error_detail = e.response.text
                    self.logger.error("Supabase error deleting session %s: %s (status: %s)", 
                                    session_id, error_detail, e.response.status_code)
            except Exception:
                pass
            self.logger.exception("Failed to delete session %s for contact_id: %s", session_id, contact_id)
            return False
        except Exception:
            self.logger.exception("Failed to delete session %s for contact_id: %s", session_id, contact_id)
            return False

    # ============================================================================
    # ENCRYPTION VERSIONS (COMMENTED - READY FOR WHEN DB SCHEMA SUPPORTS IT)
    # ============================================================================

    """
    ENCRYPTED VERSION - READY FOR WHEN DB SCHEMA IS UPDATED
    These methods will encrypt/decrypt messages when the database schema
    is updated to support ciphertext and nonce columns.
    
    To enable:
    1. Update chat_messages table to have ciphertext and nonce columns
    2. Update add_message RPC to accept ciphertext/nonce params
    3. Update get_recent_messages to return ciphertext/nonce
    4. Uncomment these methods and comment out the plaintext versions above
    """

    async def add_user_message(self, session_id: str, contact_id: str, message: str) -> None:
        """
        Encrypts and stores a user message in Supabase.
        """
        encrypted = self.encryptor.encrypt(message)
        rpc_url = f"{self.supabase_url}/rest/v1/rpc/add_encrypted_message"
        payload = {
            "p_session_id": session_id,
            "p_contact_id": contact_id,
            "p_role": "human",
            "p_ciphertext": encrypted["ciphertext"],
            "p_nonce": encrypted["nonce"]
        }
        try:
            await self._request_with_retries("post", rpc_url, json=payload)
        except Exception:
            self.logger.exception("Failed to add encrypted user message to Supabase")
            raise

    async def add_ai_message(self, session_id: str, contact_id: str, message: str) -> None:
        """
        Encrypts and stores an AI-generated message in Supabase.
        """
        encrypted = self.encryptor.encrypt(message)
        rpc_url = f"{self.supabase_url}/rest/v1/rpc/add_encrypted_message"
        payload = {
            "p_session_id": session_id,
            "p_contact_id": contact_id,
            "p_role": "ai",
            "p_ciphertext": encrypted["ciphertext"],
            "p_nonce": encrypted["nonce"]
        }
        try:
            await self._request_with_retries("post", rpc_url, json=payload)
        except Exception:
            self.logger.exception("Failed to add encrypted AI message to Supabase")
            raise

    async def get_recent_messages(
        self, 
        session_id: str, 
        contact_id: str, 
        limit: int = 14
    ) -> List[Dict[str, Any]]:
        """
        Get recent messages and decrypt them.
        Returns messages with decrypted content.
        """
        rpc_url = f"{self.supabase_url}/rest/v1/rpc/get_recent_encrypted_messages"
        payload = {
            "p_session_id": session_id,
            "p_contact_id": contact_id,
            "p_limit": limit
        }
        try:
            resp = await self._request_with_retries("post", rpc_url, json=payload)
            data = resp.json()
            if not data:
                return []
            
            # Decrypt all messages in parallel using thread pool (decryption is CPU-bound)
            def decrypt_message(msg):
                try:
                    plaintext = self.encryptor.decrypt(
                        msg["ciphertext"],
                        msg["nonce"]
                    )
                    return {
                        "role": msg["role"],
                        "content": plaintext,
                        "created_at": msg["created_at"]
                    }
                except Exception as e:
                    self.logger.warning("Failed to decrypt message: %s", e)
                    return None
            
            # Decrypt all messages concurrently in thread pool
            loop = asyncio.get_event_loop()
            decryption_tasks = [
                loop.run_in_executor(None, decrypt_message, msg) 
                for msg in data
            ]
            decrypted_results = await asyncio.gather(*decryption_tasks)
            
            # Filter out None results (failed decryptions)
            decrypted_messages = [msg for msg in decrypted_results if msg is not None]
            
            return decrypted_messages
        except Exception:
            self.logger.exception("Failed to fetch or decrypt messages from Supabase")
            return []

    async def get_session_metadata(
        self, 
        session_id: str, 
        contact_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get session metadata including encrypted summary info and title.
        Returns dict with: summary_ciphertext, summary_nonce, summary_updated_at, summary_last_message_id, title
        Note: This returns raw encrypted data for summary, but title is plaintext
        """
        url = f"{self.supabase_url}/rest/v1/encrypted_sessions"
        params = {
            "id": f"eq.{session_id}",
            "contact_id": f"eq.{contact_id}",
            "is_archived": "eq.false",  # Only get metadata from active sessions
            "select": "ciphertext,nonce,summary_updated_at,summary_last_message_id,title"
        }
        try:
            resp = await self._request_with_retries("get", url, params=params)
            data = resp.json()
            if data and len(data) > 0:
                return data[0]
            return None
        except Exception:
            self.logger.exception("Failed to fetch session metadata from Supabase")
            return None

    async def get_session_summary(self, session_id: str, contact_id: str) -> Optional[str]:
        """
        Get and decrypt the summary for a specific session if it exists.
        """
        metadata = await self.get_session_metadata(session_id, contact_id)
        if not metadata:
            return None
        
        summary_ciphertext = metadata.get("ciphertext")
        summary_nonce = metadata.get("nonce")
        
        if not summary_ciphertext or not summary_nonce:
            return None
        
        try:
            decrypted_summary = self.encryptor.decrypt(summary_ciphertext, summary_nonce)
            return decrypted_summary
        except Exception as e:
            self.logger.warning("Failed to decrypt session summary: %s", e)
            return None

    async def update_session_summary(
        self,
        session_id: str,
        contact_id: str,
        summary: str,
        last_message_id: int,
        title: Optional[str] = None
    ) -> None:
        """
        Encrypt and update the session summary (and optionally title) in the database.
        
        Args:
            session_id: Session ID
            contact_id: User's contact ID
            summary: The session summary text (will be encrypted)
            last_message_id: ID of the last message included in the summary
            title: Optional title to update (stored in plaintext)
        """
        # Encrypt the summary
        encrypted_summary = self.encryptor.encrypt(summary)
        
        url = f"{self.supabase_url}/rest/v1/encrypted_sessions"
        params = {
            "id": f"eq.{session_id}",
            "contact_id": f"eq.{contact_id}",
            "is_archived": "eq.false"  # Only update summary for active sessions
        }
        payload = {
            "ciphertext": encrypted_summary["ciphertext"],
            "nonce": encrypted_summary["nonce"],
            "summary_updated_at": self._now_iso(),
            "summary_last_message_id": last_message_id
        }
        # Only update title if provided (title remains plaintext for UI display)
        if title is not None:
            payload["title"] = title
        
        try:
            # Using PATCH to update existing session
            headers = dict(self.supabase_headers)
            headers["Prefer"] = "return=minimal"  # Don't return the updated row
            
            resp = await self._client.patch(
                url,
                headers=headers,
                params=params,
                json=payload,
                timeout=self.request_timeout
            )
            resp.raise_for_status()
            
            if title:
                self.logger.info("Updated encrypted session summary and title for session %s", session_id)
            else:
                self.logger.info("Updated encrypted session summary for session %s", session_id)
        except Exception:
            self.logger.exception("Failed to update encrypted session summary in Supabase")
            raise
    
    async def get_title(self, 
        session_id: str, 
        contact_id: str) -> str:
        """
        Retrieve the session title.
        If the session has no title, generate one from the earliest decrypted user message.
        Titles are stored in plaintext in encrypted_sessions.title.
        """

        url = f"{self.supabase_url}/rest/v1/encrypted_sessions"
        params = {
            "id": f"eq.{session_id}",
            "contact_id": f"eq.{contact_id}",
            "select": "title",
        }
        try:
            resp = await self._request_with_retries("get", url, params=params)
            data = resp.json()

            if data and data[0].get("title"):
                return data[0]["title"]
        except Exception:
            self.logger.exception("Failed to fetch title from session metadata")
            return None

    async def update_session_title(
        self,
        session_id: str,
        contact_id: str,
        title: str
    ) -> None:
        """
        Update just the session title in the database.
        
        Args:
            session_id: Session ID
            contact_id: User's contact ID  
            title: The new title for the session (stored in plaintext)
        """
        url = f"{self.supabase_url}/rest/v1/encrypted_sessions"
        params = {
            "id": f"eq.{session_id}",
            "contact_id": f"eq.{contact_id}",
            "is_archived": "eq.false"  # Only update title for active sessions
        }
        payload = {
            "title": title
        }
        
        try:
            # Using PATCH to update existing session
            headers = dict(self.supabase_headers)
            headers["Prefer"] = "return=minimal"  # Don't return the updated row
            
            resp = await self._client.patch(
                url,
                headers=headers,
                params=params,
                json=payload,
                timeout=self.request_timeout
            )
            resp.raise_for_status()
            
            self.logger.info("Updated session title for session %s: %s", session_id, title)
        except Exception:
            self.logger.exception("Failed to update session title in Supabase")
            raise

    async def get_messages_since_summary(
        self,
        session_id: str,
        contact_id: str,
        summary_last_message_id: Optional[int]
    ) -> List[Dict[str, Any]]:
        """
        Get and decrypt messages that have been added since the last summary.
        If summary_last_message_id is None, returns all messages.
        Returns decrypted messages.
        """
        # First, get encrypted messages from database
        url = f"{self.supabase_url}/rest/v1/encrypted_messages"
        
        # Build query based on whether we have a last message ID
        if summary_last_message_id is None:
            # Get all messages for this session
            params = {
                "session_id": f"eq.{session_id}",
                "contact_id": f"eq.{contact_id}",
                "order": "created_at.desc",
                "limit": "200",
                "select": "id,role,ciphertext,nonce,created_at"
            }
        else:
            # Get only messages after the last summarized one
            params = {
                "session_id": f"eq.{session_id}",
                "contact_id": f"eq.{contact_id}",
                "id": f"gt.{summary_last_message_id}",  # Greater than last summarized ID
                "order": "created_at.desc",
                "limit": "200",
                "select": "id,role,ciphertext,nonce,created_at"
            }
        
        try:
            resp = await self._request_with_retries("get", url, params=params)
            encrypted_messages = resp.json()
            
            if not encrypted_messages:
                return []
            
            # Decrypt each message
            decrypted_messages = []
            for msg in encrypted_messages:
                try:
                    if msg.get("ciphertext") and msg.get("nonce"):
                        plaintext = self.encryptor.decrypt(
                            msg["ciphertext"],
                            msg["nonce"]
                        )
                        decrypted_messages.append({
                            "id": msg["id"],
                            "role": msg["role"],
                            "content": plaintext,
                            "created_at": msg["created_at"]
                        })
                except Exception as e:
                    self.logger.warning("Failed to decrypt message ID %s: %s", msg.get("id"), e)
                    continue
            
            return decrypted_messages
            
        except Exception:
            self.logger.exception("Failed to fetch messages since summary from Supabase")
            return []

    async def get_conversation_history(
        self, 
        session_id: str, 
        contact_id: str,
        max_tokens: Optional[int] = None,
        include_summary: bool = True
    ) -> List[Dict[str, str]]:
        """
        Get conversation history with decryption and token-based trimming.
        
        Args:
            session_id: UUID of the session
            contact_id: User's contact ID
            max_tokens: Maximum tokens to include (if None, returns all messages)
            include_summary: Whether to include decrypted session summary as a system message
        
        Returns:
            List of message dicts with role/content, ordered chronologically (oldest first).
            If include_summary=True and summary exists, includes it as first message with role="system".
        """
        # Fetch session summary and messages in parallel for better performance
        # Fetch only 10 most recent messages for faster decryption
        fetch_limit = 14
        
        if include_summary:
            summary_task = asyncio.create_task(
                self.get_session_summary(session_id, contact_id)  # This will decrypt
            )
            messages_task = asyncio.create_task(
                self.get_recent_messages(session_id, contact_id, limit=fetch_limit)  # This will decrypt
            )
            summary, messages_raw = await asyncio.gather(summary_task, messages_task)
        else:
            summary = None
            messages_raw = await self.get_recent_messages(session_id, contact_id, limit=fetch_limit)
        
        if not messages_raw:
            # If no messages but we have a summary, return just the summary
            if summary:
                return [{"role": "system", "content": f"Previous conversation summary: {summary}"}]
            return []
        
        # Format messages and build result list
        messages = []
        total_tokens = 0
        
        # Add summary as first message if available (before recent messages)
        if summary:
            summary_msg = f"Previous conversation summary: {summary}"
            summary_tokens = self._count_tokens(summary_msg)
            if max_tokens is None or summary_tokens <= max_tokens:
                messages.append({"role": "system", "content": summary_msg})
                total_tokens += summary_tokens
        
        # Collect messages starting from most recent and working backwards
        # messages_raw is already decrypted and ordered newest first
        messages_to_include = []
        for msg in messages_raw:
            role = "user" if msg["role"] == "human" else "assistant"
            content = msg["content"]  # Already decrypted
            
            if max_tokens is not None:
                msg_tokens = self._count_tokens(content)
                if total_tokens + msg_tokens > max_tokens:
                    # Stop if adding this message would exceed limit
                    break
                total_tokens += msg_tokens
            else:
                # If no token limit, just estimate
                total_tokens += len(content) // 4
            
            # Append to list (will be in reverse chronological order: newest first)
            messages_to_include.append({"role": role, "content": content})
        
        # Reverse to get chronological order (oldest first) for LLM context
        messages_to_include.reverse()
        messages.extend(messages_to_include)
        
        return messages

    async def format_messages_for_openai(
        self, 
        session_id: str, 
        contact_id: str,
        max_tokens: Optional[int] = None,
        include_summary: bool = True
    ) -> List[Dict[str, str]]:
        """Alias for get_conversation_history for backward compatibility."""
        return await self.get_conversation_history(session_id, contact_id, max_tokens, include_summary)

    async def should_update_summary(
        self,
        session_id: str,
        contact_id: str,
        message_count_threshold: int = 12,
        token_count_threshold: int = 1500
    ) -> tuple:
        """
        Check if session summary should be updated.
        Works with encrypted messages by decrypting them for token counting.
        
        Returns:
            (should_update, session_metadata, new_messages)
            Note: new_messages are returned decrypted
        """
        metadata = await self.get_session_metadata(session_id, contact_id)
        if not metadata:
            return False, None, []
        
        summary_last_message_id = metadata.get("summary_last_message_id")
        # This will return decrypted messages
        new_messages = await self.get_messages_since_summary(
            session_id, contact_id, summary_last_message_id
        )
        
        if not new_messages:
            return False, metadata, []
        
        # Check message count threshold
        if len(new_messages) >= message_count_threshold:
            return True, metadata, new_messages
        
        # Check token count threshold (messages are already decrypted)
        total_tokens = sum(self._count_tokens(msg.get("content", "")) for msg in new_messages)
        if total_tokens >= token_count_threshold:
            return True, metadata, new_messages
        
        return False, metadata, new_messages

    async def clear_memory(self, session_id: str, contact_id: str) -> None:
        """
        Clear all messages in a session.
        Note: This doesn't delete the session itself, just the messages.
        For full session deletion, use delete_session() instead.
        Works the same for encrypted messages since we're just deleting records.
        """
        url = f"{self.supabase_url}/rest/v1/encrypted_messages"
        params = {
            "session_id": f"eq.{session_id}",
            "contact_id": f"eq.{contact_id}"
        }
        try:
            await self._request_with_retries("delete", url, params=params)
        except Exception:
            self.logger.exception("Failed to clear memory for session %s", session_id)
            raise