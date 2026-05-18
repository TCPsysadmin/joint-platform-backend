from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from langchain_core.language_models.chat_models import BaseChatModel

from orchestrator.tools.protocols import DoctrineTool, Embedder, PublishTool, SearchTool


@dataclass
class RuntimeContext:
    """Per-request context injected via config["configurable"]["runtime"].

    Never stored in AgentState — not serialised to the checkpointer.
    """

    client_id: UUID
    search_tool: SearchTool
    doctrine_tool: DoctrineTool
    publish_tool: PublishTool
    llm: BaseChatModel
    embedder: Embedder
