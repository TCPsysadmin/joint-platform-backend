from __future__ import annotations

from orchestrator import storage_provisioning
from tests.conftest import FAKE_CLIENT_ID


def test_workspace_bucket_name_is_private_workspace_identifier(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        storage_provisioning.settings,
        "b2_workspace_bucket_prefix",
        "VP Storage!",
    )

    assert storage_provisioning.workspace_bucket_name(FAKE_CLIENT_ID) == (
        "vpstorage-00000000000000000000000000000001"
    )
