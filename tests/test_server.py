"""Integration tests for the ConferLLM MCP server."""

import asyncio
import base64
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ImageContent, ResourceLink, TextContent

from conferllm.config import ConferLLMConfig, ModelConfig
from conferllm.server import (
    create_mcp_server,
    initialize_client,
    process_response_for_images,
    run_server,
)

SESSION_ID = "20260904-0123456789abcdef0123456789abcdef"


@dataclass
class ContractChatResult:
    """Minimal final ChatResult contract used to isolate MCP adapter tests."""

    session_id: str = SESSION_ID
    name: str = "Greeting"
    model: str = "gpt-4"
    answer: Any = "Test response"
    text: str = "Test response"
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, *, include_local_paths: bool = True) -> dict[str, Any]:
        artifacts = []
        for artifact in self.artifacts:
            serialized = dict(artifact)
            if not include_local_paths:
                serialized.pop("local_path", None)
            artifacts.append(serialized)
        return {
            "schema_version": "conferllm.chat.response.v1",
            "ok": True,
            "session": {
                "id": self.session_id,
                "name": self.name,
                "model": self.model,
                "turn": 1,
            },
            "message": {"text": self.text, "content": []},
            "artifacts": artifacts,
            "warnings": [],
            "session_id": self.session_id,
            "name": self.name,
            "model": self.model,
            "answer": self.answer,
        }


def make_response(
    content: object = "Test response",
    *,
    response_id: str = "test-123",
) -> MagicMock:
    """Create a LiteLLM-like response mock."""
    response = MagicMock()
    response.model_dump.return_value = {
        "id": response_id,
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"total_tokens": 10},
    }
    return response


class TestMCPIntegration:
    """Test MCP server integration."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.config = ConferLLMConfig(
            model_list=[
                ModelConfig(
                    model_name="gpt-4",
                    litellm_params={
                        "model": "openai/gpt-4",
                        "api_key": "test-key",
                        "max_tokens": 2048,
                        "temperature": 0.7,
                    },
                ),
                ModelConfig(
                    model_name="claude-sonnet",
                    litellm_params={
                        "model": "anthropic/claude-3-5-sonnet-20241022",
                        "api_key": "test-key",
                    },
                ),
            ]
        )

    @pytest.mark.asyncio
    async def test_create_mcp_server(self) -> None:
        """Register tools without creating the default session directory."""
        with patch("conferllm.server.SessionStore") as session_store:
            server = create_mcp_server()

            assert server.name == "conferllm"
            tools = await server.list_tools()

        session_store.assert_not_called()
        assert {tool.name for tool in tools} == {
            "chat",
            "create_chat",
            "continue_chat",
            "list_sessions",
            "list_models",
            "get_model_info",
        }

    @pytest.mark.asyncio
    async def test_create_chat_tool_returns_contract(self) -> None:
        """Create a chat through the typed MCP tool."""
        service = MagicMock()
        service.chat.return_value = ContractChatResult()
        store = MagicMock()
        with patch("conferllm.server.ChatService", return_value=service):
            server = create_mcp_server(MagicMock(), session_store=store)
            result = await server.call_tool(
                "create_chat",
                {
                    "model": "gpt-4",
                    "message": "Hello!",
                    "name": "Greeting",
                    "include_raw_response": True,
                },
            )

        assert result.structured_content is not None
        structured = result.structured_content
        assert structured["schema_version"] == "conferllm.chat.response.v1"
        assert structured["name"] == "Greeting"
        assert structured["model"] == "gpt-4"
        assert structured["answer"] == "Test response"
        assert isinstance(result.content[0], TextContent)
        service.chat.assert_called_once_with(
            "Hello!",
            model="gpt-4",
            session_id=None,
            name="Greeting",
            images=None,
            image_output_dir=None,
            include_raw_response=True,
        )

    @pytest.mark.asyncio
    async def test_continue_chat_tool_uses_session_id(self) -> None:
        """Continue a chat through the typed MCP tool."""
        service = MagicMock()
        service.chat.return_value = ContractChatResult(answer="Second answer")
        with patch("conferllm.server.ChatService", return_value=service):
            server = create_mcp_server(MagicMock(), session_store=MagicMock())
            result = await server.call_tool(
                "continue_chat",
                {"session_id": SESSION_ID, "message": "Follow-up"},
            )

        assert result.structured_content["session_id"] == SESSION_ID
        assert result.structured_content["answer"] == "Second answer"
        service.chat.assert_called_once_with(
            "Follow-up",
            model=None,
            session_id=SESSION_ID,
            name=None,
            images=None,
            image_output_dir=None,
            include_raw_response=False,
        )

    @pytest.mark.asyncio
    async def test_chat_tools_run_blocking_service_calls_concurrently(self) -> None:
        """Run independent synchronous chat calls in worker threads."""
        provider_barrier = threading.Barrier(2)
        provider_threads: set[int] = set()
        provider_threads_lock = threading.Lock()

        def service_chat(message: str, **kwargs: object) -> ContractChatResult:
            with provider_threads_lock:
                provider_threads.add(threading.get_ident())
            provider_barrier.wait(timeout=5)
            return ContractChatResult(
                session_id=str(kwargs["session_id"]),
                answer=f"Answer to {message}",
            )

        service = MagicMock()
        service.chat.side_effect = service_chat
        session_ids = [
            "20260904-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "20260904-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        ]
        with patch("conferllm.server.ChatService", return_value=service):
            server = create_mcp_server(MagicMock(), session_store=MagicMock())
            results = await asyncio.gather(
                server.call_tool(
                    "continue_chat",
                    {"session_id": session_ids[0], "message": "Follow-up 0"},
                ),
                server.call_tool(
                    "continue_chat",
                    {"session_id": session_ids[1], "message": "Follow-up 1"},
                ),
            )

        assert {result.structured_content["session_id"] for result in results} == set(
            session_ids
        )
        assert len(provider_threads) == 2

    @pytest.mark.asyncio
    async def test_chat_tool_accepts_deprecated_image_path_alias(
        self, tmp_path: Path
    ) -> None:
        """Forward the compatibility alias as image_output_dir."""
        service = MagicMock()
        service.chat.return_value = ContractChatResult()
        target_dir = str(tmp_path / "saved_images")
        with patch("conferllm.server.ChatService", return_value=service):
            server = create_mcp_server(MagicMock(), session_store=MagicMock())
            result = await server.call_tool(
                "chat",
                {
                    "model": "gpt-4",
                    "message": "Hello!",
                    "image_path": target_dir,
                },
            )

        assert result.structured_content["answer"] == "Test response"
        assert service.chat.call_args.kwargs["image_output_dir"] == target_dir

    @pytest.mark.asyncio
    async def test_chat_tool_rejects_both_image_output_arguments(self) -> None:
        """Reject ambiguous old and new output-directory arguments."""
        server = create_mcp_server(MagicMock(), session_store=MagicMock())

        with pytest.raises(ToolError, match=r"\[invalid_request\]"):
            await server.call_tool(
                "chat",
                {
                    "model": "gpt-4",
                    "message": "Hello!",
                    "image_path": "/tmp/old",
                    "image_output_dir": "/tmp/new",
                },
            )

    @pytest.mark.asyncio
    async def test_chat_tool_rejects_relative_mcp_image_paths(self) -> None:
        """Keep remote image paths unambiguous on the server filesystem."""
        service = MagicMock()
        with patch("conferllm.server.ChatService", return_value=service):
            server = create_mcp_server(MagicMock(), session_store=MagicMock())

            with pytest.raises(ToolError, match=r"\[invalid_request\]"):
                await server.call_tool(
                    "create_chat",
                    {
                        "model": "gpt-4",
                        "message": "Describe it.",
                        "images": ["relative/image.png"],
                    },
                )

        service.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_tool_not_initialized(self) -> None:
        """Return a stable configuration error for a missing client."""
        server = create_mcp_server()

        with pytest.raises(ToolError, match=r"\[configuration_error\]"):
            await server.call_tool(
                "chat",
                {"model": "gpt-4", "message": "Hello!"},
            )

    @pytest.mark.asyncio
    async def test_list_sessions_tool_filters(self) -> None:
        """Expose the versioned session-list envelope through MCP."""
        metadata = MagicMock()
        metadata.to_dict.return_value = {
            "session_id": SESSION_ID,
            "name": "Raft notes",
            "model": "gpt-4",
            "created_at": "2026-09-04T10:00:00+08:00",
        }
        warning = MagicMock()
        warning.to_dict.return_value = {
            "session_id": "20260903-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "message": "corrupt header",
        }
        detailed = MagicMock()
        detailed.sessions = [metadata]
        detailed.warnings = [warning]
        store = MagicMock()
        store.list_sessions_detailed.return_value = detailed
        server = create_mcp_server(session_store=store)

        result = await server.call_tool(
            "list_sessions",
            {"query": "raft", "model": "gpt-4", "limit": 10},
        )

        assert result.structured_content == {
            "schema_version": "conferllm.sessions.response.v1",
            "sessions": [
                {
                    "session_id": SESSION_ID,
                    "name": "Raft notes",
                    "model": "gpt-4",
                    "created_at": "2026-09-04T10:00:00+08:00",
                }
            ],
            "warnings": [
                {
                    "session_id": "20260903-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    "message": "corrupt header",
                }
            ],
        }

    @pytest.mark.asyncio
    async def test_chat_returns_ordered_inline_and_linked_images(self) -> None:
        """Inline small images and link large images without local paths."""
        small = b"\x89PNG\r\n\x1a\nsmall"
        artifacts = [
            {
                "id": "t0001-output-001",
                "direction": "output",
                "mime_type": "image/png",
                "size_bytes": len(small),
                "uri": f"conferllm://sessions/{SESSION_ID}/artifacts/t0001-output-001",
                "local_path": "/private/server/small.png",
            },
            {
                "id": "t0001-output-002",
                "direction": "output",
                "mime_type": "image/jpeg",
                "size_bytes": 1024 * 1024 + 1,
                "uri": f"conferllm://sessions/{SESSION_ID}/artifacts/t0001-output-002",
                "local_path": "/private/server/large.jpg",
            },
        ]
        service = MagicMock()
        service.chat.return_value = ContractChatResult(artifacts=artifacts)
        store = MagicMock()
        store.read_artifact.return_value = small

        with patch("conferllm.server.ChatService", return_value=service):
            server = create_mcp_server(MagicMock(), session_store=store)
            result = await server.call_tool(
                "create_chat",
                {"model": "gpt-4", "message": "Generate two images"},
            )

        assert isinstance(result.content[0], TextContent)
        assert isinstance(result.content[1], ImageContent)
        assert base64.b64decode(result.content[1].data) == small
        assert result.content[1].mime_type == "image/png"
        assert isinstance(result.content[2], ResourceLink)
        assert result.content[2].uri == artifacts[1]["uri"]
        assert "local_path" not in result.structured_content["artifacts"][0]
        assert "local_path" not in result.structured_content["artifacts"][1]
        store.read_artifact.assert_called_once_with(
            SESSION_ID,
            "t0001-output-001",
        )

    @pytest.mark.asyncio
    async def test_artifact_resource_reads_session_owned_bytes(self) -> None:
        """Expose generated images through the stable conferllm resource URI."""
        store = MagicMock()
        store.read_artifact.return_value = b"artifact bytes"
        server = create_mcp_server(session_store=store)
        uri = f"conferllm://sessions/{SESSION_ID}/artifacts/t0001-output-001"

        contents = await server.read_resource(uri)

        assert list(contents)[0].content == b"artifact bytes"
        store.read_artifact.assert_called_once_with(
            SESSION_ID,
            "t0001-output-001",
        )

    @pytest.mark.asyncio
    async def test_list_models_tool(self) -> None:
        """Return configured model names as MCP content."""
        mock_client = MagicMock()
        mock_client.list_models.return_value = ["gpt-4", "claude-sonnet"]
        server = create_mcp_server(mock_client)

        result = await server.call_tool("list_models", {})

        assert [item.text for item in result.content] == [
            "gpt-4",
            "claude-sonnet",
        ]
        assert result.structured_content == {"result": ["gpt-4", "claude-sonnet"]}
        mock_client.list_models.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_get_model_info_tool(self) -> None:
        """Return non-secret model metadata as structured content."""
        model_info = {
            "model_name": "gpt-4",
            "provider_model": "openai/gpt-4",
            "configured_params": ["api_key", "max_tokens"],
        }
        mock_client = MagicMock()
        mock_client.get_model_info.return_value = model_info
        server = create_mcp_server(mock_client)

        result = await server.call_tool("get_model_info", {"model": "gpt-4"})

        assert result.structured_content == model_info
        assert json.loads(result.content[0].text) == model_info
        mock_client.get_model_info.assert_called_once_with("gpt-4")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_name", ["list_models", "get_model_info"])
    async def test_model_tools_not_initialized(self, tool_name: str) -> None:
        """Reject model operations when no client was injected."""
        server = create_mcp_server()
        arguments = {"model": "gpt-4"} if tool_name == "get_model_info" else {}

        with pytest.raises(ToolError, match=r"\[configuration_error\]"):
            await server.call_tool(tool_name, arguments)

    def test_initialize_client_success(self) -> None:
        """Load configuration and construct a client synchronously."""
        with (
            patch(
                "conferllm.server.ConferLLMConfig.load_config",
                return_value=self.config,
            ) as mock_load,
            patch("conferllm.server.LLMClient") as mock_client_class,
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            result = initialize_client()

        assert result is mock_client
        mock_load.assert_called_once_with(None)
        mock_client_class.assert_called_once_with(self.config)

    def test_initialize_client_with_config_path(self, tmp_path: Path) -> None:
        """Pass a custom configuration path to the loader."""
        custom_path = tmp_path / "config.yaml"
        with (
            patch(
                "conferllm.server.ConferLLMConfig.load_config",
                return_value=self.config,
            ) as mock_load,
            patch("conferllm.server.LLMClient") as mock_client_class,
        ):
            result = initialize_client(custom_path)

        assert result is mock_client_class.return_value
        mock_load.assert_called_once_with(custom_path)

    def test_initialize_client_failure(self) -> None:
        """Propagate configuration errors."""
        with (
            patch(
                "conferllm.server.ConferLLMConfig.load_config",
                side_effect=ValueError("Config error"),
            ),
            pytest.raises(ValueError, match="Config error"),
        ):
            initialize_client()

    @pytest.mark.asyncio
    async def test_server_tools_metadata(self) -> None:
        """Expose descriptions that explain each model tool."""
        tools = await create_mcp_server().list_tools()
        descriptions = {tool.name: tool.description for tool in tools}

        assert "configured model" in descriptions["chat"]
        assert "persistent chat" in descriptions["create_chat"]
        assert "stored history" in descriptions["continue_chat"]
        assert "stored sessions" in descriptions["list_sessions"]
        assert "configured model names" in descriptions["list_models"]
        assert "configuration metadata" in descriptions["get_model_info"]

    @pytest.mark.parametrize(
        ("transport", "expected_transport", "expected_kwargs"),
        [
            ("stdio", "stdio", {}),
            ("sse", "sse", {"host": "0.0.0.0", "port": 8080}),
            (
                "http",
                "streamable-http",
                {"host": "0.0.0.0", "port": 8080},
            ),
        ],
    )
    def test_run_server_transport(
        self,
        transport: str,
        expected_transport: str,
        expected_kwargs: dict[str, object],
    ) -> None:
        """Map CLI transport names to MCP 2.x run modes."""
        mock_client = MagicMock()
        mock_server = MagicMock()
        mock_store = MagicMock()
        with (
            patch("conferllm.server.initialize_client", return_value=mock_client),
            patch(
                "conferllm.server.SessionStore",
                return_value=mock_store,
            ) as mock_store_class,
            patch(
                "conferllm.server.create_mcp_server",
                return_value=mock_server,
            ) as mock_create,
        ):
            run_server(
                transport=transport,  # type: ignore[arg-type]
                host="0.0.0.0",
                port=8080,
                log_level="DEBUG",
            )

        mock_store_class.assert_called_once_with(
            mock_client.config.get_session_root.return_value
        )
        mock_create.assert_called_once_with(
            mock_client,
            session_store=mock_store,
            log_level="DEBUG",
        )
        mock_server.run.assert_called_once_with(expected_transport, **expected_kwargs)

    def test_run_server_rejects_unknown_transport(self) -> None:
        """Reject transport names outside the public CLI choices."""
        with (
            patch(
                "conferllm.server.initialize_client",
                return_value=MagicMock(),
            ) as mock_initialize,
            patch(
                "conferllm.server.create_mcp_server",
                return_value=MagicMock(),
            ),
            pytest.raises(ValueError, match="Unsupported transport"),
        ):
            run_server(transport="websocket")  # type: ignore[arg-type]
        mock_initialize.assert_not_called()


class TestImageResponseProcessing:
    """Test image extraction from model responses."""

    def test_process_multimodal_list_without_mutating_input(self) -> None:
        """Save image blocks while preserving the original response."""
        image_bytes = b"\x89PNG\r\n\x1a\nfake image data"
        data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode(
            "ascii"
        )
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Here is the result."},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    }
                }
            ]
        }

        processed = process_response_for_images(response)

        assert (
            response["choices"][0]["message"]["content"][1]["image_url"]["url"]
            == data_url
        )
        image_path = Path(
            processed["choices"][0]["message"]["content"][1]["image_url"]["url"]
        )
        try:
            assert image_path.read_bytes() == image_bytes
            assert (
                processed["choices"][0]["message"]["content"][0]["text"]
                == "Here is the result."
            )
        finally:
            image_path.unlink(missing_ok=True)

    def test_process_string_content_in_custom_directory(self, tmp_path: Path) -> None:
        """Save embedded image URLs found in string content."""
        image_bytes = b"\xff\xd8\xfffake image data"
        data_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode(
            "ascii"
        )
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"Generated image: {data_url}",
                    }
                }
            ]
        }
        target_dir = tmp_path / "images"

        processed = process_response_for_images(response, target_dir)

        saved_path = Path(
            processed["choices"][0]["message"]["content"].removeprefix(
                "Generated image: "
            )
        )
        assert saved_path.parent == target_dir
        assert saved_path.read_bytes() == image_bytes


class TestToolErrors:
    """Test exceptions raised by injected clients."""

    @pytest.mark.asyncio
    async def test_chat_tool_error_handling(self) -> None:
        """Expose provider failures with a stable MCP-visible code."""
        service = MagicMock()
        service.chat.side_effect = RuntimeError("API Error")
        with patch("conferllm.server.ChatService", return_value=service):
            server = create_mcp_server(MagicMock(), session_store=MagicMock())
            with pytest.raises(ToolError, match=r"\[provider_error\] API Error"):
                await server.call_tool(
                    "chat",
                    {"model": "gpt-4", "message": "Hello!"},
                )

    @pytest.mark.asyncio
    async def test_get_model_info_tool_error_handling(self) -> None:
        """Wrap model lookup failures as unexpected MCP tool errors."""
        mock_client = MagicMock()
        mock_client.get_model_info.side_effect = ValueError("Model Error")
        server = create_mcp_server(mock_client)

        with pytest.raises(ToolError, match=r"\[invalid_request\] Model Error"):
            await server.call_tool("get_model_info", {"model": "gpt-4"})
