"""Tests for session-owned artifact metadata and transactions."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from conferllm.artifacts import (
    Artifact,
    ArtifactError,
    ArtifactTransaction,
    resolve_relative_path,
    verify_artifact_file,
)

SESSION_ID = "20260904-0123456789abcdef0123456789abcdef"


def test_artifact_record_and_public_serialization(tmp_path: Path) -> None:
    data = b"\x89PNG\r\n\x1a\nimage"
    artifact = Artifact(
        id="t0002-output-001",
        direction="output",
        index=0,
        mime_type="image/png",
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        relative_path="turn-0002/output-001.png",
    )

    assert Artifact.from_record(artifact.to_record()) == artifact
    assert artifact.to_public(SESSION_ID, tmp_path) == {
        "id": "t0002-output-001",
        "kind": "image",
        "direction": "output",
        "index": 0,
        "mime_type": "image/png",
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "uri": (f"conferllm://sessions/{SESSION_ID}/artifacts/t0002-output-001"),
        "local_path": str(tmp_path / "turn-0002" / "output-001.png"),
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": "../output-001"},
        {"id": "t0000-output-001"},
        {"id": "t0001-input-001", "direction": "output"},
        {"id": "t0001-output-002", "index": 0},
        {"mime_type": "text/plain"},
        {"sha256": "ABC"},
        {"relative_path": "../output-001.png"},
        {"relative_path": "turn-0001/../output-001.png"},
        {"relative_path": "turn-0001/output-999.png"},
    ],
)
def test_artifact_rejects_inconsistent_or_unsafe_fields(
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "id": "t0001-output-001",
        "direction": "output",
        "index": 0,
        "mime_type": "image/png",
        "size_bytes": 5,
        "sha256": hashlib.sha256(b"image").hexdigest(),
        "relative_path": "turn-0001/output-001.png",
    }
    values.update(overrides)

    with pytest.raises(ArtifactError):
        Artifact(**values)  # type: ignore[arg-type]


def test_artifact_record_rejects_missing_and_extra_fields() -> None:
    record = {
        "id": "t0001-output-001",
        "kind": "image",
        "direction": "output",
        "index": 0,
        "mime_type": "image/png",
        "size_bytes": 5,
        "sha256": hashlib.sha256(b"image").hexdigest(),
        "relative_path": "turn-0001/output-001.png",
        "source_path": "/private/input.png",
    }

    with pytest.raises(ArtifactError, match="unexpected source_path"):
        Artifact.from_record(record)


def test_transaction_stages_installs_and_finalizes_ordered_artifacts(
    tmp_path: Path,
) -> None:
    date_dir = tmp_path / "sessions" / "2026" / "09" / "04"
    asset_root = date_dir / f"{SESSION_ID}.assets"
    transaction = ArtifactTransaction(
        session_id=SESSION_ID,
        turn=3,
        asset_root=asset_root,
        staging_parent=date_dir,
    )

    first = transaction.stage_bytes(
        b"first",
        direction="input",
        mime_type="image/png",
    )
    second = transaction.stage_bytes(
        b"second",
        direction="input",
        mime_type="image/jpeg",
    )
    third = transaction.stage_bytes(
        b"third",
        direction="output",
        mime_type="image/webp",
    )
    transaction.install()
    transaction.finalize()

    assert [item.id for item in transaction.artifacts] == [
        "t0003-input-001",
        "t0003-input-002",
        "t0003-output-001",
    ]
    assert [first.index, second.index, third.index] == [0, 1, 0]
    assert stat.S_IMODE(asset_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(transaction.final_path.stat().st_mode) == 0o700
    for artifact, expected in zip(
        transaction.artifacts,
        (b"first", b"second", b"third"),
        strict=True,
    ):
        path = resolve_relative_path(asset_root, artifact.relative_path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert verify_artifact_file(path, artifact) == expected
    assert not transaction.staging_path.exists()


def test_stage_failure_rolls_back_the_entire_staging_batch(tmp_path: Path) -> None:
    date_dir = tmp_path / "sessions" / "2026" / "09" / "04"
    transaction = ArtifactTransaction(
        session_id=SESSION_ID,
        turn=1,
        asset_root=date_dir / f"{SESSION_ID}.assets",
        staging_parent=date_dir,
    )
    transaction.stage_bytes(
        b"first",
        direction="output",
        mime_type="image/png",
    )

    with (
        patch("conferllm.artifacts.os.write", side_effect=OSError("disk full")),
        pytest.raises(ArtifactError, match="disk full"),
    ):
        transaction.stage_bytes(
            b"second",
            direction="output",
            mime_type="image/png",
        )

    assert not transaction.staging_path.exists()
    assert not transaction.final_path.exists()


def test_installed_transaction_can_be_rolled_back_before_finalize(
    tmp_path: Path,
) -> None:
    date_dir = tmp_path / "sessions" / "2026" / "09" / "04"
    transaction = ArtifactTransaction(
        session_id=SESSION_ID,
        turn=1,
        asset_root=date_dir / f"{SESSION_ID}.assets",
        staging_parent=date_dir,
    )
    transaction.stage_bytes(
        b"image",
        direction="output",
        mime_type="image/png",
    )
    transaction.install()

    transaction.rollback()

    assert not transaction.final_path.exists()


def test_install_collision_preserves_existing_committed_turn(
    tmp_path: Path,
) -> None:
    date_dir = tmp_path / "sessions" / "2026" / "09" / "04"
    asset_root = date_dir / f"{SESSION_ID}.assets"
    committed = asset_root / "turn-0001"
    committed.mkdir(mode=0o700, parents=True)
    committed_file = committed / "output-001.png"
    committed_file.write_bytes(b"committed")
    transaction = ArtifactTransaction(
        session_id=SESSION_ID,
        turn=1,
        asset_root=asset_root,
        staging_parent=date_dir,
    )
    transaction.stage_bytes(
        b"replacement",
        direction="output",
        mime_type="image/png",
    )

    with pytest.raises(ArtifactError, match="already exists"):
        transaction.install()

    assert committed_file.read_bytes() == b"committed"
    assert not transaction.staging_path.exists()


def test_verify_artifact_detects_size_and_hash_changes(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"expected")
    artifact = Artifact(
        id="t0001-input-001",
        direction="input",
        index=0,
        mime_type="image/png",
        size_bytes=len(b"expected"),
        sha256=hashlib.sha256(b"expected").hexdigest(),
        relative_path="turn-0001/input-001.png",
    )

    assert verify_artifact_file(path, artifact) == b"expected"
    path.write_bytes(b"tampered")

    with pytest.raises(ArtifactError, match="hash does not match"):
        verify_artifact_file(path, artifact)


def test_resolve_relative_path_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "turn-0001").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactError, match="outside"):
        resolve_relative_path(root, "turn-0001/input-001.png")


def test_transaction_context_rolls_back_when_not_finalized(tmp_path: Path) -> None:
    date_dir = tmp_path / "sessions" / "2026" / "09" / "04"
    with ArtifactTransaction(
        session_id=SESSION_ID,
        turn=1,
        asset_root=date_dir / f"{SESSION_ID}.assets",
        staging_parent=date_dir,
    ) as transaction:
        transaction.stage_bytes(
            b"image",
            direction="input",
            mime_type="image/png",
        )
        staging_path = transaction.staging_path

    assert not staging_path.exists()
    assert not transaction.final_path.exists()
