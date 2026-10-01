"""Local dev server entrypoint (Windows-safe).

Uvicorn picks ProactorEventLoop on Windows, which breaks psycopg async. This script
passes a SelectorEventLoop factory instead. Prefer:

    python run.py

On Windows, avoid plain `uvicorn orchestrator.main:app` unless you pass:

    --loop orchestrator.eventloop:loop_factory
"""

from __future__ import annotations

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "orchestrator.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        loop="orchestrator.eventloop:loop_factory",
    )
