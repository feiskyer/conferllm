"""Tests for persistent ConferLLM sessions."""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import stat
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from conferllm.session import (
    LoadedSession,
    SessionError,
    SessionStore,
    derive_session_name,
    generate_session_id,
)

SESSION_ID = "20260904-0123456789abcdef0123456789abcdef"
OTHER_SESSION_ID = "20260904-fedcba9876543210fedcba9876543210"
CREATED_AT = datetime(2026, 9, 4, 12, 34, 56, tzinfo=timezone.utc)
USER_MESSAGE = {"role": "user", "content": "你好，世界"}
ASSISTANT_MESSAGE = {"role": "assistant", "content": "你好！"}


def _lock_in_child(
    root: str,
    session_id: str,
    events: multiprocessing.Queue[Any],
) -> None:
    store = SessionStore(Path(root))
    events.put("ready")
    with store.lock(session_id):
        events.put("acquired")


def _create(
    store: SessionStore,
    session_id: str = SESSION_ID,
    *,
    name: str = "Test session",
    model: str = "reasoning",
    created_at: datetime = CREATED_AT,
) -> None:
    store.create_session(
        session_id,
        name,
        model,
        USER_MESSAGE,
        ASSISTANT_MESSAGE,
        response_id="response-1",
        usage={"total_tokens": 12},
        created_at=created_at,
    )


def test_generate_session_id_uses_date_and_uuid4_hex() -> None:
    with patch("conferllm.session.uuid.uuid4") as mock_uuid4:
        mock_uuid4.return_value = uuid_value = type(
            "UUIDValue",
            (),
            {"hex": "a" * 32},
        )()
        session_id = generate_session_id(CREATED_AT)

    assert session_id == f"20260904-{uuid_value.hex}"
    mock_uuid4.assert_called_once_with()


def test_session_store_exposes_id_and_name_helpers(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")

    with patch("conferllm.session.uuid.uuid4") as mock_uuid4:
        mock_uuid4.return_value = type("UUIDValue", (), {"hex": "b" * 32})()
        assert store.generate_session_id(CREATED_AT) == f"20260904-{'b' * 32}"
    assert store.derive_session_name("  hello\nworld ") == "hello world"


@pytest.mark.parametrize(
    ("message", "explicit_name", "expected"),
    [
        ("  explain\n raft\tleader election  ", None, "explain raft leader election"),
        (" \n\t ", None, "Untitled session"),
        ("ignored", "  Raft\n notes ", "Raft notes"),
        ("界" * 61, None, f"{'界' * 60}…"),
        ("界" * 60, None, "界" * 60),
    ],
)
def test_derive_session_name(
    message: str,
    explicit_name: str | None,
    expected: str,
) -> None:
    assert derive_session_name(message, explicit_name) == expected


def test_derive_session_name_rejects_empty_explicit_name() -> None:
    with pytest.raises(SessionError, match="must not be empty"):
        derive_session_name("fallback", " \n ")


def test_session_paths_are_deterministic_and_do_not_create_assets(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions")

    assert store.session_path(SESSION_ID) == (
        tmp_path / "sessions" / "2026" / "09" / "04" / f"{SESSION_ID}.jsonl"
    )
    assert store.assets_path(SESSION_ID).name == f"{SESSION_ID}.assets"
    assert not store.assets_path(SESSION_ID).exists()


@pytest.mark.parametrize(
    "session_id",
    [
        "../20260904-0123456789abcdef0123456789abcdef",
        "20260904-0123456789ABCDEF0123456789abcdef",
        "20260904-short",
        "20260230-0123456789abcdef0123456789abcdef",
        "20260904-0123456789abcdef0123456789abcdef.jsonl",
    ],
)
def test_session_paths_reject_invalid_or_traversal_ids(
    tmp_path: Path,
    session_id: str,
) -> None:
    store = SessionStore(tmp_path / "sessions")

    with pytest.raises(SessionError, match="Invalid session ID"):
        store.session_path(session_id)


def test_session_path_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    outside = tmp_path / "outside"
    store = SessionStore(root)
    outside.mkdir()
    (root / "2026").symlink_to(outside, target_is_directory=True)

    with pytest.raises(SessionError, match="symlink"):
        store.session_path(SESSION_ID)


def test_session_store_rejects_symlink_root_without_chmodding_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "actual-sessions"
    target.mkdir()
    target.chmod(0o755)
    root = tmp_path / "sessions"
    root.symlink_to(target, target_is_directory=True)

    with pytest.raises(SessionError, match="must not be a symlink"):
        SessionStore(root)

    assert stat.S_IMODE(target.stat().st_mode) == 0o755


def test_create_load_and_append_round_trip(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    metadata = store.create_session(
        SESSION_ID,
        "  Test\n session ",
        "reasoning",
        USER_MESSAGE,
        ASSISTANT_MESSAGE,
        response_id="response-1",
        usage={"total_tokens": 12},
        created_at=CREATED_AT,
    )

    assert metadata.name == "Test session"
    assert metadata.to_dict() == {
        "session_id": SESSION_ID,
        "name": "Test session",
        "model": "reasoning",
        "created_at": CREATED_AT.isoformat(),
    }
    loaded = store.load_session(SESSION_ID)
    assert loaded == LoadedSession(
        metadata=metadata,
        messages=[USER_MESSAGE, ASSISTANT_MESSAGE],
        next_turn=2,
    )

    second_user = {"role": "user", "content": "Continue"}
    second_assistant = {"role": "assistant", "content": "Continued"}
    with store.lock(SESSION_ID):
        store.append_turn(
            SESSION_ID,
            2,
            second_user,
            second_assistant,
            response_id="response-2",
            usage={"total_tokens": 20},
            created_at=CREATED_AT + timedelta(minutes=1),
        )

    loaded = store.load_session(SESSION_ID)
    assert loaded.messages == [
        USER_MESSAGE,
        ASSISTANT_MESSAGE,
        second_user,
        second_assistant,
    ]
    assert loaded.next_turn == 3

    records = [
        json.loads(line)
        for line in store.session_path(SESSION_ID)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["type"] for record in records] == ["session", "turn", "turn"]
    assert [record.get("turn") for record in records] == [None, 1, 2]
    assert records[0]["schema_version"] == 2
    assert records[1]["artifacts"] == []
    assert records[2]["artifacts"] == []
    assert records[1]["response_id"] == "response-1"
    assert records[2]["usage"] == {"total_tokens": 20}


def test_schema_v1_load_and_append_preserves_legacy_format(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session_file = store.session_path(SESSION_ID)
    session_file.parent.mkdir(mode=0o700, parents=True)
    records = [
        {
            "type": "session",
            "schema_version": 1,
            "session_id": SESSION_ID,
            "name": "Legacy",
            "model": "reasoning",
            "created_at": CREATED_AT.isoformat(),
        },
        {
            "type": "turn",
            "turn": 1,
            "created_at": CREATED_AT.isoformat(),
            "user": USER_MESSAGE,
            "assistant": ASSISTANT_MESSAGE,
            "response_id": None,
            "usage": None,
        },
    ]
    session_file.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )
    original_header = session_file.read_text(encoding="utf-8").splitlines()[0]

    loaded = store.load_session(SESSION_ID)
    assert loaded.schema_version == 1
    assert loaded.artifacts == []
    with store.lock(SESSION_ID):
        store.append_turn(
            SESSION_ID,
            2,
            {"role": "user", "content": "Legacy continuation"},
            {"role": "assistant", "content": "Still legacy"},
        )

    persisted = session_file.read_text(encoding="utf-8").splitlines()
    assert persisted[0] == original_header
    assert json.loads(persisted[-1]).get("artifacts") is None
    assert store.load_session(SESSION_ID).schema_version == 1


def test_schema_v1_canonical_artifact_path_remains_resource_readable(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    session_file = store.session_path(SESSION_ID)
    session_file.parent.mkdir(mode=0o700, parents=True)
    records = [
        {
            "type": "session",
            "schema_version": 1,
            "session_id": SESSION_ID,
            "name": "Legacy",
            "model": "vision",
            "created_at": CREATED_AT.isoformat(),
        },
        {
            "type": "turn",
            "turn": 1,
            "created_at": CREATED_AT.isoformat(),
            "user": USER_MESSAGE,
            "assistant": ASSISTANT_MESSAGE,
            "response_id": None,
            "usage": None,
        },
    ]
    session_file.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )
    transaction = store.artifact_transaction(SESSION_ID, 2)
    artifact = transaction.stage_bytes(
        b"legacy output",
        direction="output",
        mime_type="image/png",
    )
    canonical_path = store.assets_path(SESSION_ID) / artifact.relative_path
    assistant = {
        "role": "assistant",
        "content": f"Generated: {canonical_path}",
    }

    with store.lock(SESSION_ID):
        store.append_turn(
            SESSION_ID,
            2,
            {"role": "user", "content": "Generate"},
            assistant,
            artifact_transaction=transaction,
        )

    loaded = store.load_session(SESSION_ID)
    assert loaded.schema_version == 1
    assert loaded.artifacts == []
    assert store.read_artifact(SESSION_ID, artifact.id) == b"legacy output"
    assert store.resolve_artifact(SESSION_ID, artifact.id) == canonical_path


def test_v2_artifacts_round_trip_and_resolve_with_integrity_check(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    transaction = store.artifact_transaction(SESSION_ID, 1)
    input_artifact = transaction.stage_bytes(
        b"input image",
        direction="input",
        mime_type="image/png",
    )
    output_artifact = transaction.stage_bytes(
        b"output image",
        direction="output",
        mime_type="image/jpeg",
    )
    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe these."},
            {"type": "image_ref", "artifact_id": input_artifact.id},
        ],
    }
    assistant = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Done."},
            {"type": "image_ref", "artifact_id": output_artifact.id},
        ],
    }

    store.create_session(
        SESSION_ID,
        "Artifacts",
        "vision",
        user,
        assistant,
        artifact_transaction=transaction,
        created_at=CREATED_AT,
    )

    loaded = store.load_session(SESSION_ID)
    assert loaded.schema_version == 2
    assert loaded.artifacts == [input_artifact, output_artifact]
    assert loaded.turns[0]["artifacts"] == [
        input_artifact.to_record(),
        output_artifact.to_record(),
    ]
    assert store.read_artifact(SESSION_ID, input_artifact.id) == b"input image"
    assert store.resolve_artifact(SESSION_ID, output_artifact.id).read_bytes() == (
        b"output image"
    )

    original_path = store.resolve_artifact(SESSION_ID, input_artifact.id)
    original_path.write_bytes(b"modified")
    with pytest.raises(SessionError, match="size does not match"):
        store.read_artifact(SESSION_ID, input_artifact.id)


def test_resolve_artifact_requires_a_session_reference(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    _create(store)
    unreferenced = store.assets_path(SESSION_ID) / "turn-0001" / "output-001.png"
    unreferenced.parent.mkdir(mode=0o700, parents=True)
    unreferenced.write_bytes(b"not referenced")

    with pytest.raises(SessionError, match="not referenced"):
        store.resolve_artifact(SESSION_ID, "t0001-output-001")


def test_image_ref_must_reference_an_artifact_in_the_same_turn(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    user = {
        "role": "user",
        "content": [
            {"type": "image_ref", "artifact_id": "t0001-input-001"},
        ],
    }

    with pytest.raises(SessionError, match="does not reference"):
        store.create_session(
            SESSION_ID,
            "Missing artifact",
            "vision",
            user,
            ASSISTANT_MESSAGE,
            created_at=CREATED_AT,
        )


def test_append_jsonl_failure_rolls_back_installed_turn_directory(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    _create(store)
    session_file = store.session_path(SESSION_ID)
    original = session_file.read_bytes()
    transaction = store.artifact_transaction(SESSION_ID, 2)
    artifact = transaction.stage_bytes(
        b"generated",
        direction="output",
        mime_type="image/png",
    )
    assistant = {
        "role": "assistant",
        "content": [{"type": "image_ref", "artifact_id": artifact.id}],
    }

    with (
        store.lock(SESSION_ID),
        patch(
            "conferllm.session.os.replace",
            side_effect=OSError("simulated replace failure"),
        ),
        pytest.raises(SessionError, match="simulated replace failure"),
    ):
        store.append_turn(
            SESSION_ID,
            2,
            {"role": "user", "content": "Generate"},
            assistant,
            artifact_transaction=transaction,
        )

    assert session_file.read_bytes() == original
    assert not transaction.final_path.exists()
    assert not transaction.staging_path.exists()


def test_create_is_atomic_and_does_not_overwrite_existing_session(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    _create(store)
    original = store.session_path(SESSION_ID).read_bytes()

    with pytest.raises(SessionError, match="already exists"):
        _create(store, name="Replacement")

    assert store.session_path(SESSION_ID).read_bytes() == original
    assert not list(store.session_path(SESSION_ID).parent.glob("*.tmp"))


def test_permissions_are_restrictive(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    store = SessionStore(root)
    _create(store)
    with store.lock(SESSION_ID):
        store.append_turn(
            SESSION_ID,
            2,
            {"role": "user", "content": "Continue"},
            {"role": "assistant", "content": "Continued"},
        )

    session_file = store.session_path(SESSION_ID)
    lock_file = session_file.with_name(f"{SESSION_ID}.lock")
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(session_file.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(session_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(lock_file.stat().st_mode) == 0o600


def test_append_rejects_nonsequential_turn_without_modifying_file(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    _create(store)
    original = store.session_path(SESSION_ID).read_bytes()

    with (
        store.lock(SESSION_ID),
        pytest.raises(SessionError, match="expected turn 2"),
    ):
        store.append_turn(
            SESSION_ID,
            3,
            {"role": "user", "content": "Skipped"},
            {"role": "assistant", "content": "No"},
        )

    assert store.session_path(SESSION_ID).read_bytes() == original


def test_append_write_failure_leaves_original_unchanged_and_removes_temp_file(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    _create(store)
    session_file = store.session_path(SESSION_ID)
    original = session_file.read_bytes()
    real_write = os.write
    write_calls = 0

    def fail_after_partial_write(descriptor: int, data: bytes) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return real_write(descriptor, data[: max(1, len(data) // 2)])
        raise OSError("simulated write failure")

    with (
        store.lock(SESSION_ID),
        patch("conferllm.session.os.write", side_effect=fail_after_partial_write),
        pytest.raises(SessionError, match="simulated write failure"),
    ):
        store.append_turn(
            SESSION_ID,
            2,
            {"role": "user", "content": "Unstored"},
            {"role": "assistant", "content": "Unstored"},
        )

    assert write_calls == 2
    assert session_file.read_bytes() == original
    assert not list(session_file.parent.glob("*.tmp"))


def test_append_replace_failure_leaves_original_unchanged_and_removes_temp_file(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    _create(store)
    session_file = store.session_path(SESSION_ID)
    original = session_file.read_bytes()

    with (
        store.lock(SESSION_ID),
        patch(
            "conferllm.session.os.replace",
            side_effect=OSError("simulated replace failure"),
        ),
        pytest.raises(SessionError, match="simulated replace failure"),
    ):
        store.append_turn(
            SESSION_ID,
            2,
            {"role": "user", "content": "Unstored"},
            {"role": "assistant", "content": "Unstored"},
        )

    assert session_file.read_bytes() == original
    assert not list(session_file.parent.glob("*.tmp"))


@pytest.mark.parametrize(
    ("lines", "expected_line"),
    [
        (["not json"], 1),
        (
            [
                json.dumps(
                    {
                        "type": "session",
                        "schema_version": 999,
                        "session_id": SESSION_ID,
                        "name": "Bad",
                        "model": "reasoning",
                        "created_at": CREATED_AT.isoformat(),
                    }
                )
            ],
            1,
        ),
        (
            [
                json.dumps(
                    {
                        "type": "session",
                        "schema_version": 1,
                        "session_id": SESSION_ID,
                        "name": "Bad",
                        "model": "reasoning",
                        "created_at": CREATED_AT.isoformat(),
                    }
                )
            ],
            2,
        ),
        (
            [
                json.dumps(
                    {
                        "type": "session",
                        "schema_version": 1,
                        "session_id": SESSION_ID,
                        "name": "Bad",
                        "model": "reasoning",
                        "created_at": CREATED_AT.isoformat(),
                    }
                ),
                json.dumps(
                    {
                        "type": "turn",
                        "turn": 2,
                        "created_at": CREATED_AT.isoformat(),
                        "user": USER_MESSAGE,
                        "assistant": ASSISTANT_MESSAGE,
                    }
                ),
            ],
            2,
        ),
        (
            [
                json.dumps(
                    {
                        "type": "session",
                        "schema_version": 1,
                        "session_id": SESSION_ID,
                        "name": "Bad",
                        "model": "reasoning",
                        "created_at": CREATED_AT.isoformat(),
                    }
                ),
                '{"type":"turn"',
            ],
            2,
        ),
    ],
)
def test_corruption_errors_include_session_id_and_line(
    tmp_path: Path,
    lines: list[str],
    expected_line: int,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    session_file = store.session_path(SESSION_ID)
    session_file.parent.mkdir(parents=True)
    session_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(SessionError) as error:
        store.load_session(SESSION_ID)

    assert SESSION_ID in str(error.value)
    assert f"line {expected_line}" in str(error.value)


def test_list_sessions_filters_orders_and_limits(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    sessions = [
        (
            "20260903-11111111111111111111111111111111",
            "Raft overview",
            "reasoning",
            datetime(2026, 9, 3, 20, tzinfo=timezone.utc),
        ),
        (
            "20260904-22222222222222222222222222222222",
            "Paxos notes",
            "reasoning",
            datetime(2026, 9, 4, 9, tzinfo=timezone.utc),
        ),
        (
            "20260904-33333333333333333333333333333333",
            "RAFT deployment",
            "fast",
            datetime(2026, 9, 4, 10, tzinfo=timezone.utc),
        ),
    ]
    for session_id, name, model, created_at in sessions:
        _create(
            store,
            session_id,
            name=name,
            model=model,
            created_at=created_at,
        )

    assert [item.session_id for item in store.list_sessions()] == [
        sessions[2][0],
        sessions[1][0],
        sessions[0][0],
    ]
    assert [item.session_id for item in store.list_sessions(query="raft")] == [
        sessions[2][0],
        sessions[0][0],
    ]
    assert [item.session_id for item in store.list_sessions(model="reasoning")] == [
        sessions[1][0],
        sessions[0][0],
    ]
    assert [
        item.session_id
        for item in store.list_sessions(
            query="raft",
            model="reasoning",
            since=date(2026, 9, 3),
            until=date(2026, 9, 3),
        )
    ] == [sessions[0][0]]
    assert len(store.list_sessions(limit=2)) == 2
    assert len(store.list_sessions(limit=0)) == 3

    with pytest.raises(SessionError, match="non-negative"):
        store.list_sessions(limit=-1)


def test_list_sessions_reads_only_headers(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    _create(store)
    with store.session_path(SESSION_ID).open("a", encoding="utf-8") as session_file:
        session_file.write("corrupt body\n")

    assert [item.session_id for item in store.list_sessions()] == [SESSION_ID]
    with pytest.raises(SessionError, match="line 3"):
        store.load_session(SESSION_ID)


def test_list_sessions_detailed_keeps_valid_sessions_and_warns_on_bad_header(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    _create(store)
    corrupt_file = store.session_path(OTHER_SESSION_ID)
    corrupt_file.write_text("not json\n", encoding="utf-8")

    result = store.list_sessions_detailed(limit=0)

    assert [item.session_id for item in result.sessions] == [SESSION_ID]
    assert len(result.warnings) == 1
    assert result.warnings[0].session_id == OTHER_SESSION_ID
    assert "line 1" in result.warnings[0].message
    assert store.list_sessions(limit=0) == result.sessions


def test_same_session_locks_serialize_across_processes(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    context = multiprocessing.get_context("spawn")
    events = context.Queue()

    with store.lock(SESSION_ID):
        process = context.Process(
            target=_lock_in_child,
            args=(str(store.root), SESSION_ID, events),
        )
        process.start()
        assert events.get(timeout=5) == "ready"
        with pytest.raises(queue.Empty):
            events.get(timeout=0.2)

    assert events.get(timeout=5) == "acquired"
    process.join(timeout=5)
    assert process.exitcode == 0


def test_different_session_locks_are_independent(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    context = multiprocessing.get_context("spawn")
    events = context.Queue()

    with store.lock(SESSION_ID):
        process = context.Process(
            target=_lock_in_child,
            args=(str(store.root), OTHER_SESSION_ID, events),
        )
        process.start()
        assert events.get(timeout=5) == "ready"
        assert events.get(timeout=5) == "acquired"

    process.join(timeout=5)
    assert process.exitcode == 0


def test_lock_is_released_when_process_exits(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    context = multiprocessing.get_context("spawn")
    events = context.Queue()
    process = context.Process(
        target=_lock_in_child,
        args=(str(store.root), SESSION_ID, events),
    )
    process.start()
    assert events.get(timeout=5) == "ready"
    assert events.get(timeout=5) == "acquired"
    process.join(timeout=5)
    assert process.exitcode == 0

    with store.lock(SESSION_ID):
        assert os.path.exists(store.session_path(SESSION_ID).parent)
