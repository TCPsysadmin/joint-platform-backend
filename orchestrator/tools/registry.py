from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from supabase import AsyncClient

from orchestrator.tools.protocols import DoctrineTool, PublishTool, SearchTool


@dataclass
class ToolSet:
    search: SearchTool
    doctrine: DoctrineTool
    publish: PublishTool


def get_tools(supabase: AsyncClient, client_id: UUID) -> ToolSet:
    """Return the active tool implementations based on USE_MCP_TOOLS config.

    MCP integration points (3 files change when MCP comes in):
      1. This file: add the `if settings.use_mcp_tools` branch importing from tools/mcp/
      2. orchestrator/tools/mcp/__init__.py  (new file)
      3. orchestrator/tools/mcp/<tool>.py    (new file per tool)
    Nodes never change — they only touch the protocol.
    """
    from orchestrator.config import settings  # local import avoids circular dep at module level

    if settings.use_mcp_tools:
        raise NotImplementedError(
            "MCP tools are not yet implemented. Set USE_MCP_TOOLS=false."
        )

    from orchestrator.tools.local.doctrine import SupabaseDoctrineTool
    from orchestrator.tools.local.opusclip_stub import OpusClipStub
    from orchestrator.tools.local.supabase_search import SupabaseSearchTool

    return ToolSet(
        search=SupabaseSearchTool(supabase=supabase, client_id=client_id),
        doctrine=SupabaseDoctrineTool(supabase=supabase, client_id=client_id),
        publish=OpusClipStub(),
    )
