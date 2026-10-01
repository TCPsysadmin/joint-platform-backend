"""Uvicorn loop factory for Windows + psycopg async compatibility."""

from __future__ import annotations

import asyncio


def loop_factory() -> asyncio.AbstractEventLoop:
    """Selector event loop — required for psycopg async (uvicorn uses Proactor on Windows)."""
    return asyncio.SelectorEventLoop()
