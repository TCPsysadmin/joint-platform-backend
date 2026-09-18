from __future__ import annotations

import asyncio
import json
from typing import Any, cast
from urllib.parse import quote

import httpx
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account

from orchestrator.config import settings

_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
_DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"


class DriveCleanupError(RuntimeError):
    """Raised when Google Drive cleanup cannot finish."""


def _credentials() -> service_account.Credentials:
    raw_credentials = settings.google_drive_service_account_json.strip()
    if not raw_credentials:
        raise DriveCleanupError("Google Drive service-account credentials are not configured")

    try:
        service_account_info: Any = json.loads(raw_credentials)
        if not isinstance(service_account_info, dict):
            raise ValueError("credentials must be a JSON object")
        credentials = service_account.Credentials.from_service_account_info(  # type: ignore[no-untyped-call]
            service_account_info,
            scopes=[_DRIVE_SCOPE],
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise DriveCleanupError("Google Drive service-account credentials are invalid") from exc

    delegated_user = settings.google_drive_delegated_user.strip()
    if delegated_user:
        credentials = cast(
            service_account.Credentials,
            credentials.with_subject(delegated_user),
        )
    return cast(service_account.Credentials, credentials)


async def _access_token() -> str:
    credentials = _credentials()
    try:
        await asyncio.to_thread(credentials.refresh, GoogleAuthRequest())
    except Exception as exc:
        raise DriveCleanupError("Could not authenticate with Google Drive") from exc
    token = credentials.token
    if not isinstance(token, str) or not token:
        raise DriveCleanupError("Google Drive authentication returned no access token")
    return token


async def trash_drive_files(file_ids: list[str]) -> int:
    """Move unique Google Drive files to Trash directly through Drive API v3."""
    unique_ids = list(dict.fromkeys(file_id.strip() for file_id in file_ids if file_id.strip()))
    if not unique_ids:
        return 0

    token = await _access_token()
    try:
        async with httpx.AsyncClient(timeout=60.0) as http:
            for file_id in unique_ids:
                response = await http.patch(
                    f"{_DRIVE_FILES_URL}/{quote(file_id, safe='')}",
                    params={"supportsAllDrives": "true", "fields": "id,trashed"},
                    headers={"Authorization": f"Bearer {token}"},
                    json={"trashed": True},
                )
                # A missing file already satisfies the requested end state and
                # keeps retries safe after a partially completed cleanup.
                if response.status_code == 404:
                    continue
                if response.status_code != 200:
                    raise DriveCleanupError(
                        f"Google Drive rejected cleanup with status {response.status_code}"
                    )
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise DriveCleanupError("Google Drive returned invalid JSON") from exc
                if payload.get("trashed") is not True:
                    raise DriveCleanupError("Google Drive did not confirm that the file was trashed")
    except httpx.HTTPError as exc:
        raise DriveCleanupError("Could not reach Google Drive") from exc
    return len(unique_ids)
