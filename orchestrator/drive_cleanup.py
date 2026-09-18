from __future__ import annotations

import httpx

from orchestrator.config import settings


class DriveCleanupError(RuntimeError):
    """Raised when the private Drive cleanup workflow cannot finish."""


async def trash_drive_files(file_ids: list[str]) -> int:
    """Move unique Google Drive files to Trash through the private n8n workflow."""
    unique_ids = list(dict.fromkeys(file_id.strip() for file_id in file_ids if file_id.strip()))
    if not unique_ids:
        return 0
    if not settings.media_deletion_webhook_url:
        raise DriveCleanupError("Media deletion webhook is not configured")

    headers: dict[str, str] = {}
    if settings.media_deletion_webhook_secret:
        headers["X-VP-Media-Delete-Secret"] = settings.media_deletion_webhook_secret

    try:
        async with httpx.AsyncClient(timeout=60.0) as http:
            response = await http.post(
                settings.media_deletion_webhook_url,
                headers=headers,
                json={"file_ids": unique_ids},
            )
    except httpx.HTTPError as exc:
        raise DriveCleanupError("Could not reach the Drive cleanup workflow") from exc

    if response.status_code != 200:
        raise DriveCleanupError(
            f"Drive cleanup workflow returned status {response.status_code}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise DriveCleanupError("Drive cleanup workflow returned invalid JSON") from exc
    if payload.get("ok") is not True or payload.get("trashed") != len(unique_ids):
        raise DriveCleanupError("Drive cleanup workflow returned an incomplete result")
    return len(unique_ids)
