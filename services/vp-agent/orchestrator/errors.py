from __future__ import annotations


class AgentError(Exception):
    """Base class for all agent errors."""


class RetrievalError(AgentError):
    """Raised when transcript or asset retrieval fails."""


class AmbiguousSourceError(RetrievalError):
    """Raised when a filename identifies multiple videos equally well."""


class LLMError(AgentError):
    """Raised when an LLM call fails or returns unparseable output."""


class ToolError(AgentError):
    """Raised when a tool call (search, doctrine, publish) fails."""


class AuthError(AgentError):
    """Raised when client identity cannot be resolved."""
