from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from orchestrator.config import settings


async def _setup_checkpointer(checkpointer: AsyncPostgresSaver) -> None:
    """Run LangGraph migrations; tolerate pooler prepared-statement races on reload."""
    try:
        await checkpointer.setup()
    except psycopg.errors.DuplicatePreparedStatement:
        # Supabase pooler + uvicorn --reload can hit this if setup runs twice on a
        # pooled connection. Safe to continue when checkpoint tables already exist.
        async with checkpointer._cursor() as cur:
            await cur.execute("SELECT 1 FROM checkpoint_migrations LIMIT 1")


@asynccontextmanager
async def get_checkpointer() -> AsyncGenerator[AsyncPostgresSaver, None]:
    """Async context manager that yields a ready-to-use Postgres checkpointer."""
    # prepare_threshold=None: required for PgBouncer / Supabase pooler URLs.
    async with await AsyncConnection.connect(
        settings.supabase_db_url,
        autocommit=True,
        prepare_threshold=None,
        row_factory=dict_row,
    ) as conn:
        checkpointer = AsyncPostgresSaver(conn)
        await _setup_checkpointer(checkpointer)
        yield checkpointer
