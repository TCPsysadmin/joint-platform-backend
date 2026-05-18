from __future__ import annotations

import structlog
from uuid import uuid4

from orchestrator.tools.protocols import ClipPayload, PublishResult

logger = structlog.get_logger(__name__)


class OpusClipStub:
    """Stub PublishTool — logs the payload and returns a fake clip ID.

    Replace with the real OpusClip client when the API contract is finalised.
    """

    async def create_and_post_clip(self, payload: ClipPayload) -> PublishResult:
        clip_id = f"stub-{uuid4()}"
        logger.info(
            "opusclip_stub_called",
            clip_id=clip_id,
            video_id=payload.video_id,
            start_seconds=payload.start_seconds,
            end_seconds=payload.end_seconds,
        )
        return PublishResult(
            clip_id=clip_id,
            status="stub_posted",
            url=None,
            metadata={"stub": True},
        )
