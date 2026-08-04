from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

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


async def _configure_connection(conn: psycopg.AsyncConnection[DictRow]) -> None:
    await conn.set_autocommit(True)
    conn.row_factory = dict_row


@asynccontextmanager
async def get_checkpointer() -> AsyncGenerator[AsyncPostgresSaver, None]:
    """Async context manager that yields a ready-to-use Postgres checkpointer."""
    # A pool (rather than a single connection) so that connections dropped by the
    # Supabase pooler are detected via `check` and replaced instead of raising
    # "the connection is closed" on the next request.
    # prepare_threshold=None: required for PgBouncer / Supabase pooler URLs.
    async with AsyncConnectionPool(
        settings.supabase_db_url,
        min_size=1,
        max_size=4,
        kwargs={"prepare_threshold": None},
        configure=_configure_connection,
        check=AsyncConnectionPool.check_connection,
        open=False,
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await _setup_checkpointer(checkpointer)
        yield checkpointer
