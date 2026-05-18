from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from orchestrator.config import settings


@asynccontextmanager
async def get_checkpointer() -> AsyncGenerator[AsyncPostgresSaver, None]:
    """Async context manager that yields a ready-to-use Postgres checkpointer."""
    async with AsyncPostgresSaver.from_conn_string(settings.supabase_db_url) as checkpointer:
        await checkpointer.setup()
        yield checkpointer
