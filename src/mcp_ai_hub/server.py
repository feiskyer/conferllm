"""MCP AI Hub Server - Unified AI provider access via LiteLM."""

import argparse
import asyncio
import base64
import logging
import re
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .ai_client import AIClient
from .config import AIHubConfig

# Configure logging to stderr (MCP requirement)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

logger = logging.getLogger(__name__)

# Global instances
ai_client: AIClient | None = None


def extract_and_save_base64_images(content: str, output_dir: Path | None = None) -> str:
    """Extract base64 images from content and save them to temporary files.

    Args:
        content: Content that may contain base64-encoded images

    Returns:
        Content with base64 images replaced by file paths
    """
    # Pattern to match base64 image data URLs
    base64_pattern = re.compile(
        r"data:image/([a-zA-Z]+);base64,([A-Za-z0-9+/=]+)", re.IGNORECASE
    )

    def replace_image(match: re.Match[str]) -> str:
        image_format = match.group(1).lower()
        base64_data = match.group(2)

        try:
            # Decode the base64 data
            image_data = base64.b64decode(base64_data)

            target_dir = output_dir or Path(tempfile.gettempdir())
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
            except Exception as mkdir_error:
                logger.error(
                    "Failed to create image output directory %s: %s",
                    target_dir,
                    mkdir_error,
                )
                return match.group(0)

            # Construct a unique filename inside the target directory
            temp_file_path = (
                target_dir / f"ai_hub_image_{uuid.uuid4().hex}.{image_format}"
            )

            # Write the image data to the file
            with temp_file_path.open("wb") as temp_file:
                temp_file.write(image_data)

            logger.info(f"Saved base64 image to: {temp_file_path}")

            # Return the file path instead of the base64 data
            return str(temp_file_path)

        except Exception as e:
            logger.error(f"Failed to process base64 image: {e}")
            # Return original content if processing fails
            return match.group(0)

    # Replace all base64 images with file paths
    return base64_pattern.sub(replace_image, content)


def process_response_for_images(
    response_dict: dict[str, Any], output_dir: Path | None = None
) -> dict[str, Any]:
    """Process response dictionary to extract base64 images and save them to files.

    Args:
        response_dict: Response dictionary from LiteLM

    Returns:
        Modified response dictionary with base64 images replaced by file paths
    """
    # Deep copy to avoid modifying the original
    import copy

    processed_response = copy.deepcopy(response_dict)

    # Process choices content for images
    if "choices" in processed_response:
        for choice in processed_response["choices"]:
            if "message" not in choice or "content" not in choice["message"]:
                continue

            content = choice["message"]["content"]
            if isinstance(content, str):
                # Extract and save any base64 images in the content
                choice["message"]["content"] = extract_and_save_base64_images(
                    content, output_dir
                )
            elif isinstance(content, list):
                for index, item in enumerate(content):
                    if isinstance(item, dict):
                        if item.get("type") == "image_url":
                            image_url = item.get("image_url")
                            if isinstance(image_url, dict):
                                url = image_url.get("url")
                                if isinstance(url, str):
                                    image_url["url"] = extract_and_save_base64_images(
                                        url, output_dir
                                    )
                        elif item.get("type") == "text":
                            text_value = item.get("text")
                            if isinstance(text_value, str):
                                item["text"] = extract_and_save_base64_images(
                                    text_value, output_dir
                                )
                    elif isinstance(item, str):
                        # Some providers may return bare strings inside the list
                        processed = extract_and_save_base64_images(item, output_dir)
                        if processed != item:
                            content[index] = processed

    return processed_response


async def initialize_client(config_path: Path | None = None) -> None:
    """Initialize the AI client with configuration."""
    global ai_client
    try:
        config = AIHubConfig.load_config(config_path)
        ai_client = AIClient(config)
        logger.info(f"Loaded configuration with {len(config.model_list)} models")
        for model in config.list_available_models():
            logger.info(f"Available model: {model}")
    except Exception as e:
        logger.error(f"Failed to initialize AI client: {e}")
        raise


def create_mcp_server(host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    """Create and configure the FastMCP server."""
    mcp = FastMCP("ai-hub", host=host, port=port)

    @mcp.tool()
    async def chat(
        model: str, messages: list[dict[str, Any]], image_path: str | None = None
    ) -> dict[str, Any]:
        """Chat with specified AI model using OpenAI-compatible messages format.

        Args:
            model: Model name from configuration. Call list_models() tool to see available models.
            messages: List of messages in OpenAI format. Each message should have 'role' and 'content' keys.
                     Content can be:
                     - String for text messages
                     - List of content objects for multimodal messages (text, image_url, etc.)

                     IMPORTANT: Image handling
                     -------------------------
                     For local images: Simply use the absolute file path as the URL.
                     The server will automatically detect local paths and convert them to base64.

                     Image formats supported:
                     - Remote URL: {"url": "https://example.com/image.jpg"}
                     - Local file path: {"url": "/path/to/local/image.jpg"} (auto-converted to base64)
                     - Base64: {"url": "data:image/jpeg;base64,<base64_string>"}

                     Example formats:
                     - Text only: [{"role": "user", "content": "Hello!"}]
                     - Multiple messages: [
                         {"role": "user", "content": "Hello!"},
                         {"role": "assistant", "content": "Hi there!"},
                         {"role": "user", "content": "How are you?"}
                       ]
                     - With remote image URL: [{"role": "user", "content": [
                         {"type": "text", "text": "What's in this image?"},
                         {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
                       ]}]
                     - With local image (absolute path): [{"role": "user", "content": [
                         {"type": "text", "text": "What's in this image?"},
                         {"type": "image_url", "image_url": {"url": "/Users/john/Desktop/screenshot.png"}}
                       ]}]
                     - With local image (Windows path): [{"role": "user", "content": [
                         {"type": "text", "text": "What's in this image?"},
                         {"type": "image_url", "image_url": {"url": "C:\\Users\\john\\Pictures\\photo.jpg"}}
                       ]}]
            image_path: Optional directory to save generated images. When omitted, images
                are stored in the system temporary directory. If the directory does not
                exist, it will be created. This is only used when the model returns
                images; purely text responses skip this option.

        Returns:
            Complete LiteLM ModelResponse as a dictionary containing:
            - id: Response ID
            - object: Response object type ('chat.completion')
            - created: Timestamp
            - model: Model used
            - choices: List of completion choices with message content
            - usage: Token usage statistics
        """
        global ai_client

        if ai_client is None:
            raise RuntimeError("AI client not initialized")

        image_output_dir: Path | None = None
        if image_path:
            image_output_dir = Path(image_path).expanduser()
            if image_output_dir.exists():
                if not image_output_dir.is_dir():
                    raise ValueError(
                        f"image_path must be a directory, got existing file: {image_output_dir}"
                    )
            else:
                try:
                    image_output_dir.mkdir(parents=True, exist_ok=True)
                except Exception as mkdir_error:
                    raise ValueError(
                        f"Failed to create image directory '{image_output_dir}': {mkdir_error}"
                    ) from mkdir_error

        try:
            response = ai_client.chat(model, messages)
            # Convert ModelResponse to dictionary for MCP serialization
            response_dict = response.model_dump()
            # Process response to extract and save any base64 images
            return process_response_for_images(response_dict, image_output_dir)
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise

    @mcp.tool()
    async def list_models() -> list[str]:
        """List all available AI models.

        Returns:
            List of available model names
        """
        global ai_client

        if ai_client is None:
            raise RuntimeError("AI client not initialized")

        return ai_client.list_models()

    @mcp.tool()
    async def get_model_info(model: str) -> dict[str, Any]:
        """Get information about a specific model.

        Args:
            model: Model name to get info for

        Returns:
            Dictionary with model information
        """
        global ai_client

        if ai_client is None:
            raise RuntimeError("AI client not initialized")

        return ai_client.get_model_info(model)

    return mcp


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI options."""
    parser = argparse.ArgumentParser(
        description="MCP AI Hub Server - Unified AI provider access via LiteLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Transport Types:
  stdio         Standard input/output (default for MCP clients)
  sse           Server-Sent Events (requires --host and --port)
  http          HTTP transport (requires --host and --port)

Examples:
  %(prog)s                           # Run with stdio transport
  %(prog)s --transport sse           # Run with SSE on default host/port
  %(prog)s --transport http --port 8080  # Run HTTP on port 8080
        """,
    )

    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="Transport type to use (default: stdio)",
    )

    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to for sse/http transports (default: localhost)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=3001,
        help="Port to bind to for sse/http transports (default: 3001)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file (default: ~/.ai_hub.yaml)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser


def main() -> None:
    """Main entry point for the MCP server."""
    parser = create_parser()
    args = parser.parse_args()

    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Initialize the AI client synchronously
    async def init_client() -> None:
        await initialize_client(args.config)

    try:
        # Initialize the AI client
        asyncio.run(init_client())

        # Create MCP server with host/port configuration
        mcp = create_mcp_server(host=args.host, port=args.port)

        # Run the MCP server with appropriate transport
        if args.transport == "stdio":
            logger.info("Starting MCP server with stdio transport")
            mcp.run("stdio")
        elif args.transport == "sse":
            logger.info(
                f"Starting MCP server with SSE transport on {args.host}:{args.port}"
            )
            mcp.run("sse")
        elif args.transport == "http":
            logger.info(
                f"Starting MCP server with streamable-http transport on {args.host}:{args.port}"
            )
            mcp.run("streamable-http")

    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
