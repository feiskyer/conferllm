"""MCP server for ConferLLM."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Literal, NoReturn

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp.types import CallToolResult, ImageContent, ResourceLink, TextContent

from . import __version__
from .chat import ChatResult, ChatService
from .client import LLMClient
from .config import ConferLLMConfig
from .errors import ConferLLMError, normalize_error
from .images import (
    extract_and_save_base64_images,
    prepare_image_output_dir,
    process_response_for_images,
)
from .session import SessionStore

logger = logging.getLogger(__name__)

Transport = Literal["stdio", "sse", "http"]
INLINE_IMAGE_LIMIT_BYTES = 1024 * 1024

__all__ = [
    "Transport",
    "INLINE_IMAGE_LIMIT_BYTES",
    "create_mcp_server",
    "extract_and_save_base64_images",
    "initialize_client",
    "prepare_image_output_dir",
    "process_response_for_images",
    "run_server",
]


def initialize_client(config_path: Path | None = None) -> LLMClient:
    """Load configuration and construct a ConferLLM client."""
    config = ConferLLMConfig.load_config(config_path)
    client = LLMClient(config)
    logger.info("Loaded configuration with %d models", len(config.model_list))
    for model in config.list_available_models():
        logger.info("Available model: %s", model)
    return client


def _optional_date(value: str | None, argument: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{argument} must use YYYY-MM-DD format") from error


def _without_server_paths(value: Any) -> Any:
    if isinstance(value, list):
        return [_without_server_paths(item) for item in value]
    if not isinstance(value, dict):
        return value

    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"local_path", "exported_path"}:
            continue
        if key == "path" and isinstance(item, str) and Path(item).is_absolute():
            continue
        sanitized[key] = _without_server_paths(item)
    return sanitized


def _output_artifacts(result: ChatResult) -> list[dict[str, Any]]:
    return [
        artifact for artifact in result.artifacts if artifact["direction"] == "output"
    ]


def _validate_mcp_images(images: list[str] | None) -> None:
    """Require remote callers to use data URLs or absolute server-local paths."""
    for index, image in enumerate(images or []):
        if image.lower().startswith("data:"):
            continue
        if not Path(image).is_absolute():
            raise ConferLLMError(
                "invalid_request",
                "MCP image inputs must be data URLs or absolute server-local paths.",
                details={"index": index},
            )


def _chat_tool_result(
    result: ChatResult,
    store: SessionStore,
) -> CallToolResult:
    structured = _without_server_paths(result.to_dict(include_local_paths=False))
    content: list[Any] = []

    session_id = result.session_id
    for artifact in _output_artifacts(result):
        artifact_id = artifact["id"]
        mime_type = artifact["mime_type"]
        size = artifact["size_bytes"]
        data: bytes | None = None
        if size <= INLINE_IMAGE_LIMIT_BYTES:
            try:
                data = store.read_artifact(session_id, artifact_id)
            except (OSError, ValueError):
                # The chat has committed. An optional inline rendering failure
                # must not make a caller retry a successful, billable turn.
                structured["warnings"].append(
                    f"Unable to inline artifact '{artifact_id}'; "
                    "use its resource URI to retry reading it."
                )

        if data is not None and len(data) <= INLINE_IMAGE_LIMIT_BYTES:
            content.append(
                ImageContent(
                    data=base64.b64encode(data).decode("ascii"),
                    mime_type=mime_type,
                )
            )
        else:
            content.append(
                ResourceLink(
                    name=artifact_id,
                    uri=artifact["uri"],
                    mime_type=mime_type,
                    size=size,
                )
            )

    # Text-only MCP clients must also receive the ID needed for continuation.
    content.insert(
        0, TextContent(text=json.dumps(structured, ensure_ascii=False, default=str))
    )
    return CallToolResult(content=content, structured_content=structured)


def _raise_public_error(error: Exception, *, resource: bool = False) -> NoReturn:
    public_error = normalize_error(error)
    exception_type = ResourceError if resource else ToolError
    raise exception_type(str(public_error)) from error


def create_mcp_server(
    client: LLMClient | None = None,
    *,
    session_store: SessionStore | None = None,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
) -> MCPServer[None]:
    """Create the ConferLLM MCP server."""
    store = session_store
    chat_service: ChatService | None = None

    server: MCPServer[None] = MCPServer(
        "conferllm",
        title="ConferLLM",
        description="Use locally configured AI models through persistent chats.",
        version=__version__,
        log_level=log_level,
    )

    def require_client() -> LLMClient:
        if client is None:
            raise ConferLLMError(
                "configuration_error",
                "ConferLLM client not initialized.",
            )
        return client

    def require_store() -> SessionStore:
        nonlocal store
        if store is None:
            store = SessionStore(
                client.config.get_session_root() if client is not None else None
            )
        return store

    def require_chat_service() -> ChatService:
        nonlocal chat_service
        active_client = require_client()
        if chat_service is None:
            chat_service = ChatService(active_client, require_store())
        return chat_service

    async def execute_chat(
        *,
        message: str,
        model: str | None = None,
        session_id: str | None = None,
        name: str | None = None,
        images: list[str] | None = None,
        image_output_dir: str | None = None,
        image_path: str | None = None,
        include_raw_response: bool = False,
    ) -> CallToolResult:
        if image_output_dir is not None and image_path is not None:
            _raise_public_error(
                ConferLLMError(
                    "invalid_request",
                    "image_output_dir and deprecated image_path "
                    "cannot be used together.",
                )
            )
        try:
            _validate_mcp_images(images)
        except Exception as error:
            _raise_public_error(error)
        output_dir = image_output_dir if image_output_dir is not None else image_path
        try:
            service = require_chat_service()
            result = await asyncio.to_thread(
                service.chat,
                message,
                model=model,
                session_id=session_id,
                name=name,
                images=images,
                image_output_dir=output_dir,
                include_raw_response=include_raw_response,
            )
            return await asyncio.to_thread(_chat_tool_result, result, require_store())
        except Exception as error:
            _raise_public_error(error)

    @server.tool()
    async def create_chat(
        message: str,
        model: str,
        name: str | None = None,
        images: list[str] | None = None,
        image_output_dir: str | None = None,
        include_raw_response: bool = False,
    ) -> CallToolResult:
        """Create a persistent chat using one configured model.

        ``images`` preserves input order and accepts server-local absolute paths
        or image data URLs. The result includes a reusable session ID.
        """
        return await execute_chat(
            message=message,
            model=model,
            name=name,
            images=images,
            image_output_dir=image_output_dir,
            include_raw_response=include_raw_response,
        )

    @server.tool()
    async def continue_chat(
        message: str,
        session_id: str,
        images: list[str] | None = None,
        image_output_dir: str | None = None,
        include_raw_response: bool = False,
    ) -> CallToolResult:
        """Continue a persistent chat; ConferLLM loads its stored history."""
        return await execute_chat(
            message=message,
            session_id=session_id,
            images=images,
            image_output_dir=image_output_dir,
            include_raw_response=include_raw_response,
        )

    @server.tool()
    async def chat(
        message: str,
        model: str | None = None,
        session_id: str | None = None,
        name: str | None = None,
        images: list[str] | None = None,
        image_output_dir: str | None = None,
        image_path: str | None = None,
        include_raw_response: bool = False,
    ) -> CallToolResult:
        """Create or continue a named chat with a configured model.

        Supply ``model`` for the first message or ``session_id`` to continue.
        New integrations should use ``create_chat`` or ``continue_chat``.
        ``image_path`` is a deprecated alias for ``image_output_dir``.
        """
        if (model is None) == (session_id is None):
            _raise_public_error(
                ConferLLMError(
                    "invalid_request",
                    "Exactly one of model or session_id is required.",
                )
            )
        return await execute_chat(
            message=message,
            model=model,
            session_id=session_id,
            name=name,
            images=images,
            image_output_dir=image_output_dir,
            image_path=image_path,
            include_raw_response=include_raw_response,
        )

    @server.tool()
    async def list_sessions(
        query: str | None = None,
        model: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 50,
    ) -> dict[str, object]:
        """List stored sessions, optionally filtered by name, model, or date."""
        try:
            result = await asyncio.to_thread(
                require_store().list_sessions_detailed,
                query=query,
                model=model,
                since=_optional_date(since, "since"),
                until=_optional_date(until, "until"),
                limit=limit,
            )
            return {
                "schema_version": "conferllm.sessions.response.v1",
                "sessions": [session.to_dict() for session in result.sessions],
                "warnings": [warning.to_dict() for warning in result.warnings],
            }
        except Exception as error:
            _raise_public_error(error)

    @server.tool()
    async def list_models() -> list[str]:
        """List the configured model names."""
        try:
            return require_client().list_models()
        except Exception as error:
            _raise_public_error(error)

    @server.tool()
    async def get_model_info(model: str) -> dict[str, object]:
        """Get non-secret configuration metadata for a model."""
        try:
            return require_client().get_model_info(model)
        except Exception as error:
            _raise_public_error(error)

    @server.resource(
        "conferllm://sessions/{session_id}/artifacts/{artifact_id}",
        name="ConferLLM session artifact",
        description="Read one image artifact referenced by a ConferLLM session.",
        mime_type="application/octet-stream",
    )
    def read_session_artifact(session_id: str, artifact_id: str) -> bytes:
        try:
            return require_store().read_artifact(session_id, artifact_id)
        except Exception as error:
            _raise_public_error(error, resource=True)

    return server


def run_server(
    *,
    config_path: Path | None = None,
    transport: Transport = "stdio",
    host: str = "localhost",
    port: int = 3001,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
) -> None:
    """Load configuration and run the MCP server."""
    if transport not in ("stdio", "sse", "http"):
        raise ValueError(f"Unsupported transport: {transport}")

    logging.getLogger().setLevel(getattr(logging, log_level))
    client = initialize_client(config_path)
    server = create_mcp_server(
        client,
        session_store=SessionStore(client.config.get_session_root()),
        log_level=log_level,
    )

    if transport == "stdio":
        logger.info("Starting ConferLLM MCP server with stdio transport")
        server.run("stdio")
    elif transport == "sse":
        logger.info("Starting ConferLLM MCP server with SSE on %s:%d", host, port)
        server.run("sse", host=host, port=port)
    elif transport == "http":
        logger.info("Starting ConferLLM MCP server with HTTP on %s:%d", host, port)
        server.run("streamable-http", host=host, port=port)


def main() -> None:
    """Compatibility entry point for ``python -m conferllm.server``."""
    from .cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
