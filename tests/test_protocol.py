"""Exercise the MCP wire contract and legacy replay, not only tool callbacks."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import patch

import anyio
import pytest
from mcp import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

from conferllm.chat import ChatService
from conferllm.server import create_mcp_server
from conferllm.session import SessionError
from tests.chat_fixtures import PNG, configured_client, response

DATA_URL = "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")


async def test_mcp_client_server_round_trip(tmp_path: Path) -> None:
    """Initialize, create, read images, continue, list, and report a tool error."""
    client = configured_client(tmp_path / "sessions")
    server = create_mcp_server(client)
    with (
        anyio.fail_after(10),
        patch(
            "conferllm.client.litellm.completion",
            side_effect=[
                response("image", images=[{"image_url": {"url": DATA_URL}}]),
                response("follow-up"),
            ],
        ) as provider,
    ):
        async with (
            create_client_server_memory_streams() as (client_streams, server_streams),
            anyio.create_task_group() as tasks,
        ):
            tasks.start_soon(
                server._lowlevel_server.run,
                *server_streams,
                server._lowlevel_server.create_initialization_options(),
            )
            async with ClientSession(*client_streams) as session:
                initialized = await session.initialize()
                assert initialized.server_info.name == "conferllm"
                tools = await session.list_tools()
                assert len(tools.tools) == 6

                created = await session.call_tool(
                    "create_chat",
                    {"model": "vision", "message": "draw", "images": [DATA_URL]},
                )
                assert not created.is_error
                envelope = created.structured_content
                assert envelope is not None
                assert json.loads(created.content[0].text) == envelope
                assert "local_path" not in json.dumps(envelope)
                session_id = envelope["session"]["id"]
                artifact = next(
                    item
                    for item in envelope["artifacts"]
                    if item["direction"] == "output"
                )
                retrieved = await session.read_resource(artifact["uri"])
                assert base64.b64decode(retrieved.contents[0].blob) == PNG

                continued = await session.call_tool(
                    "continue_chat",
                    {"session_id": session_id, "message": "describe"},
                )
                assert not continued.is_error
                assert continued.structured_content["session"]["turn"] == 2
                listed = await session.call_tool("list_sessions", {})
                assert (
                    listed.structured_content["sessions"][0]["session_id"] == session_id
                )
                invalid = await session.call_tool("chat", {"message": "no target"})
                assert invalid.is_error
                assert "[invalid_request]" in invalid.content[0].text
            tasks.cancel_scope.cancel()
    assert provider.call_count == 2
    history = provider.call_args_list[1].kwargs["messages"]
    assert history[0]["content"][1]["image_url"]["url"] == DATA_URL
    assert history[1]["content"][1]["image_url"]["url"] == DATA_URL


def test_schema_one_continuation_keeps_images_replayable(tmp_path: Path) -> None:
    client = configured_client(tmp_path / "sessions")
    service = ChatService(client)
    store = service.session_store
    with patch("conferllm.client.litellm.completion", return_value=response()):
        first = service.chat("hello", model="vision")
    path = store.session_path(first.session_id)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["schema_version"] = 1
    records[1].pop("artifacts")
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    with patch(
        "conferllm.client.litellm.completion",
        side_effect=[
            response(None, images=[{"image_url": {"url": DATA_URL}}]),
            response(),
        ],
    ) as provider:
        second = service.chat("draw", session_id=first.session_id, images=[DATA_URL])
        service.chat("describe", session_id=first.session_id)

    loaded = store.load_session(first.session_id)
    assert loaded.schema_version == 1
    assert all("artifacts" not in turn for turn in loaded.turns)
    assert store.read_artifact(first.session_id, "t0002-output-001") == PNG
    assert second.content[0]["artifact_id"] == "t0002-output-001"
    replay = provider.call_args_list[1].kwargs["messages"]
    assert replay[2]["content"][1]["image_url"]["url"] == DATA_URL
    assert replay[3]["content"][0]["image_url"]["url"] == DATA_URL


def test_recovery_never_discards_a_committed_turn(tmp_path: Path) -> None:
    client = configured_client(tmp_path / "sessions")
    service = ChatService(client)
    store = service.session_store
    with patch.object(client, "chat", return_value=response(DATA_URL)):
        first = service.chat("draw", model="vision")
        stale_snapshot = store.load_session(first.session_id)
        service.chat("draw more", session_id=first.session_id)
    with store.lock(first.session_id), pytest.raises(SessionError, match="stale"):
        store.discard_uncommitted_artifacts(stale_snapshot)
    assert store.load_session(first.session_id).next_turn == 3
    assert store.read_artifact(first.session_id, "t0002-output-001") == PNG
