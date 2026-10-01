from __future__ import annotations

import asyncio
import json

import pytest

from orchestrator.turn_stream import (
    Publisher,
    TurnAlreadyActiveError,
    TurnStreamRegistry,
    iter_turn_events,
)


def _event(name: str) -> dict[str, str]:
    return {"event": name, "data": json.dumps({})}


@pytest.mark.asyncio
async def test_turn_continues_after_subscriber_disconnect() -> None:
    release = asyncio.Event()

    async def runner(publish: Publisher) -> None:
        await publish(_event("node"))
        await release.wait()
        await publish(_event("done"))

    registry = TurnStreamRegistry()
    turn = await registry.get_or_start(
        session_id="sess-1",
        kind="chat",
        fingerprint="hello",
        runner_factory=runner,
    )

    stream = iter_turn_events(turn)
    first = await stream.__anext__()
    assert first["event"] == "node"
    await stream.aclose()

    release.set()
    assert turn.task is not None
    await turn.task

    replay = [event async for event in iter_turn_events(turn)]
    assert [event["event"] for event in replay] == ["node", "done"]
    assert await registry.get_status("sess-1") == "completed"


@pytest.mark.asyncio
async def test_same_turn_request_attaches_to_existing_runner() -> None:
    started = 0
    release = asyncio.Event()

    async def runner(publish: Publisher) -> None:
        nonlocal started
        started += 1
        await publish(_event("node"))
        await release.wait()
        await publish(_event("done"))

    registry = TurnStreamRegistry()
    first = await registry.get_or_start(
        session_id="sess-2",
        kind="chat",
        fingerprint="same message",
        runner_factory=runner,
    )
    second = await registry.get_or_start(
        session_id="sess-2",
        kind="chat",
        fingerprint="same message",
        runner_factory=runner,
    )
    await asyncio.sleep(0)

    assert first is second
    assert started == 1

    release.set()
    assert first.task is not None
    await first.task


@pytest.mark.asyncio
async def test_different_active_turn_in_same_session_conflicts() -> None:
    async def runner(_: Publisher) -> None:
        await asyncio.Event().wait()

    registry = TurnStreamRegistry()
    turn = await registry.get_or_start(
        session_id="sess-3",
        kind="chat",
        fingerprint="first message",
        runner_factory=runner,
    )

    with pytest.raises(TurnAlreadyActiveError):
        await registry.get_or_start(
            session_id="sess-3",
            kind="chat",
            fingerprint="second message",
            runner_factory=runner,
        )

    await registry.cancel_all()
    assert turn.task is not None
    assert turn.task.cancelled()
