from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)

type SSEPayload = dict[str, str]
type QueueItem = SSEPayload | None
type Publisher = Callable[[SSEPayload], Awaitable[None]]
type RunnerFactory = Callable[[Publisher], Awaitable[None]]


class TurnAlreadyActiveError(RuntimeError):
    """Raised when a session already has a different active turn running."""


@dataclass
class ActiveTurn:
    """In-memory fanout for one detached graph run.

    The graph task owns progress production. SSE clients can subscribe and
    unsubscribe without controlling the lifetime of the graph task.
    """

    session_id: str
    kind: str
    fingerprint: str
    task: asyncio.Task[None] | None = None
    created_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    _history: list[SSEPayload] = field(default_factory=list)
    _subscribers: set[asyncio.Queue[QueueItem]] = field(default_factory=set)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def done(self) -> bool:
        return self.finished_at is not None

    def matches(self, *, kind: str, fingerprint: str) -> bool:
        return self.kind == kind and self.fingerprint == fingerprint

    async def publish(self, payload: SSEPayload) -> None:
        async with self._lock:
            self._history.append(payload)
            subscribers = list(self._subscribers)

        for subscriber in subscribers:
            subscriber.put_nowait(payload)

    async def finish(self) -> None:
        async with self._lock:
            if self.finished_at is None:
                self.finished_at = time.monotonic()
            subscribers = list(self._subscribers)
            self._subscribers.clear()

        for subscriber in subscribers:
            subscriber.put_nowait(None)

    async def subscribe(self) -> asyncio.Queue[QueueItem]:
        queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        async with self._lock:
            for payload in self._history:
                queue.put_nowait(payload)
            if self.done:
                queue.put_nowait(None)
            else:
                self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[QueueItem]) -> None:
        async with self._lock:
            self._subscribers.discard(queue)


class TurnStreamRegistry:
    """Tracks detached session turns and lets SSE requests attach to them."""

    def __init__(self, *, completed_ttl_seconds: float = 300.0) -> None:
        self._completed_ttl_seconds = completed_ttl_seconds
        self._turns: dict[str, ActiveTurn] = {}
        self._lock = asyncio.Lock()

    async def get_or_start(
        self,
        *,
        session_id: str,
        kind: str,
        fingerprint: str,
        runner_factory: RunnerFactory,
    ) -> ActiveTurn:
        async with self._lock:
            self._prune_completed_locked()
            existing = self._turns.get(session_id)
            if existing is not None:
                if existing.done:
                    if existing.matches(kind=kind, fingerprint=fingerprint):
                        return existing
                    self._turns.pop(session_id, None)
                elif existing.matches(kind=kind, fingerprint=fingerprint):
                    return existing
                else:
                    raise TurnAlreadyActiveError(
                        f"Session {session_id!r} already has an active turn"
                    )

            turn = ActiveTurn(session_id=session_id, kind=kind, fingerprint=fingerprint)
            turn.task = asyncio.create_task(
                self._execute(session_id=session_id, turn=turn, runner_factory=runner_factory),
                name=f"{kind}-turn:{session_id}",
            )
            self._turns[session_id] = turn
            return turn

    async def get_status(self, session_id: str) -> str:
        async with self._lock:
            self._prune_completed_locked()
            turn = self._turns.get(session_id)
            if turn is None:
                return "idle"
            if turn.done:
                return "completed"
            return "running"

    async def cancel_all(self) -> None:
        async with self._lock:
            turns = list(self._turns.values())
            self._turns.clear()

        for turn in turns:
            if turn.task is not None and not turn.task.done():
                turn.task.cancel()

        for turn in turns:
            if turn.task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await turn.task
            await turn.finish()

    async def _execute(
        self,
        *,
        session_id: str,
        turn: ActiveTurn,
        runner_factory: RunnerFactory,
    ) -> None:
        try:
            await runner_factory(turn.publish)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("detached_turn_crashed", session_id=session_id, kind=turn.kind)
            await turn.publish(
                {
                    "event": "error",
                    "data": json.dumps({"detail": "Internal agent error"}),
                }
            )
        finally:
            await turn.finish()

    def _prune_completed_locked(self) -> None:
        now = time.monotonic()
        expired = [
            session_id
            for session_id, turn in self._turns.items()
            if turn.finished_at is not None and now - turn.finished_at > self._completed_ttl_seconds
        ]
        for session_id in expired:
            self._turns.pop(session_id, None)


async def iter_turn_events(turn: ActiveTurn) -> AsyncGenerator[SSEPayload, None]:
    queue = await turn.subscribe()
    try:
        while True:
            payload = await queue.get()
            if payload is None:
                break
            yield payload
    finally:
        await turn.unsubscribe(queue)
