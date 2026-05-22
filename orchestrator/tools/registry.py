from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import structlog
from supabase import AsyncClient

from orchestrator.tools.protocols import DoctrineTool, FileTool, PublishTool, SearchTool

logger = structlog.get_logger(__name__)


@dataclass
class ToolSet:
    search: SearchTool
    doctrine: DoctrineTool
    publish: PublishTool
    file: FileTool


def get_tools(supabase: AsyncClient, client_id: UUID) -> ToolSet:
    from orchestrator.config import settings
    from orchestrator.tools.local.b2_file import B2FileTool
    from orchestrator.tools.local.doctrine import SupabaseDoctrineTool
    from orchestrator.tools.local.opusclip import OpusClipTool
    from orchestrator.tools.local.opusclip_stub import OpusClipStub
    from orchestrator.tools.local.supabase_search import SupabaseSearchTool

    publish: PublishTool
    if settings.opusclip_api_key:
        publish = OpusClipTool(
            api_key=settings.opusclip_api_key,
            base_url=settings.opusclip_api_url,
        )
    else:
        logger.warning("opusclip_using_stub", reason="OPUSCLIP_API_KEY not configured")
        publish = OpusClipStub()

    return ToolSet(
        search=SupabaseSearchTool(supabase=supabase, client_id=client_id),
        doctrine=SupabaseDoctrineTool(supabase=supabase, client_id=client_id),
        publish=publish,
        file=B2FileTool(
            supabase=supabase,
            client_id=client_id,
            key_id=settings.b2_key_id,
            application_key=settings.b2_application_key,
        ),
    )
