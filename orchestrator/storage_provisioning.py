from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import httpx

from orchestrator.config import settings

_B2_AUTH_URL = "https://api.backblazeb2.com/b2api/v4/b2_authorize_account"
_DRIVE_FOLDER_KEYS = (
    "drive_transcripts_intake_folder_id",
    "drive_summaries_intake_folder_id",
    "drive_transcripts_completed_folder_id",
    "drive_summaries_completed_folder_id",
)


class StorageProvisioningError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProvisionedStorage:
    b2_bucket: str
    drive_transcripts_intake_folder_id: str
    drive_summaries_intake_folder_id: str
    drive_transcripts_completed_folder_id: str
    drive_summaries_completed_folder_id: str


def workspace_bucket_name(client_id: UUID) -> str:
    prefix = "".join(
        character
        for character in settings.b2_workspace_bucket_prefix.lower()
        if character.isalnum() or character == "-"
    ).strip("-")
    prefix = (prefix or "vpstorage")[:24].rstrip("-")
    return f"{prefix}-{client_id.hex}"


async def ensure_private_b2_bucket(client_id: UUID) -> str:
    if not settings.b2_key_id or not settings.b2_application_key:
        raise StorageProvisioningError("B2 workspace provisioning is not configured")

    bucket_name = workspace_bucket_name(client_id)
    async with httpx.AsyncClient(timeout=20.0) as http:
        auth_response = await http.get(
            _B2_AUTH_URL,
            auth=(settings.b2_key_id, settings.b2_application_key),
        )
        if auth_response.status_code != 200:
            raise StorageProvisioningError("Could not authorize the B2 provisioning key")

        auth = auth_response.json()
        storage_api = auth.get("apiInfo", {}).get("storageApi", {})
        capabilities = set(storage_api.get("allowed", {}).get("capabilities") or [])
        missing = {"listBuckets", "writeBuckets"} - capabilities
        if missing:
            raise StorageProvisioningError(
                "B2 provisioning key is missing capabilities: " + ", ".join(sorted(missing))
            )

        api_url = str(storage_api.get("apiUrl") or "")
        account_id = str(auth.get("accountId") or "")
        token = str(auth.get("authorizationToken") or "")
        if not api_url or not account_id or not token:
            raise StorageProvisioningError("B2 authorization response was incomplete")

        headers = {"Authorization": token}
        list_response = await http.post(
            f"{api_url}/b2api/v4/b2_list_buckets",
            headers=headers,
            json={"accountId": account_id, "bucketName": bucket_name},
        )
        if list_response.status_code != 200:
            raise StorageProvisioningError("Could not check for the workspace B2 bucket")
        buckets = list_response.json().get("buckets") or []
        if buckets:
            return bucket_name

        create_response = await http.post(
            f"{api_url}/b2api/v4/b2_create_bucket",
            headers=headers,
            json={
                "accountId": account_id,
                "bucketName": bucket_name,
                "bucketType": "allPrivate",
            },
        )
        if create_response.status_code != 200:
            detail = create_response.json().get("code", "unknown_error")
            if detail == "duplicate_bucket_name":
                retry_list = await http.post(
                    f"{api_url}/b2api/v4/b2_list_buckets",
                    headers=headers,
                    json={"accountId": account_id, "bucketName": bucket_name},
                )
                if retry_list.status_code == 200 and retry_list.json().get("buckets"):
                    return bucket_name
            raise StorageProvisioningError(f"Could not create the workspace B2 bucket ({detail})")

    return bucket_name


async def provision_drive_folders(client_id: UUID, display_name: str) -> dict[str, str]:
    if not settings.storage_provisioning_webhook_url:
        raise StorageProvisioningError("Google Drive provisioning webhook is not configured")
    if not settings.storage_provisioning_webhook_secret:
        raise StorageProvisioningError("Google Drive provisioning webhook secret is not configured")

    async with httpx.AsyncClient(timeout=45.0) as http:
        response = await http.post(
            settings.storage_provisioning_webhook_url,
            headers={"X-VP-Provisioning-Secret": settings.storage_provisioning_webhook_secret},
            json={
                "client_id": str(client_id),
                "display_name": display_name,
                "parent_folder_id": settings.google_drive_provisioning_parent_id,
            },
        )
    if response.status_code != 200:
        raise StorageProvisioningError(
            f"Google Drive provisioning failed (status {response.status_code})"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise StorageProvisioningError("Google Drive provisioning returned invalid JSON") from exc

    result: dict[str, str] = {}
    for key in _DRIVE_FOLDER_KEYS:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise StorageProvisioningError(f"Google Drive provisioning did not return {key}")
        result[key] = value.strip()
    return result


async def provision_workspace_storage(client_id: UUID, display_name: str) -> ProvisionedStorage:
    bucket = await ensure_private_b2_bucket(client_id)
    folders = await provision_drive_folders(client_id, display_name)
    return ProvisionedStorage(
        b2_bucket=bucket,
        drive_transcripts_intake_folder_id=folders["drive_transcripts_intake_folder_id"],
        drive_summaries_intake_folder_id=folders["drive_summaries_intake_folder_id"],
        drive_transcripts_completed_folder_id=folders["drive_transcripts_completed_folder_id"],
        drive_summaries_completed_folder_id=folders["drive_summaries_completed_folder_id"],
    )
