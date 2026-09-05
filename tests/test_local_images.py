"""Tests for local image handling functionality."""

import base64
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conferllm.client import LLMClient
from conferllm.config import ConferLLMConfig, ModelConfig
from conferllm.images import (
    ImageProcessingError,
    ImageValidationError,
    detect_image_mime,
    load_image_inputs,
    process_response_for_images,
    replace_embedded_image_data,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\nminimal-png"
JPEG_BYTES = b"\xff\xd8\xffminimal-jpeg"
WEBP_BYTES = b"RIFF\x04\x00\x00\x00WEBPdata"
SVG_BYTES = b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = MagicMock(spec=ConferLLMConfig)
    config.model_list = [
        ModelConfig(
            model_name="test-model",
            litellm_params={"model": "openai/gpt-4-vision-preview"},
        )
    ]
    config.global_system_prompt = None
    config.list_available_models.return_value = ["test-model"]
    config.get_model_config.return_value = config.model_list[0]
    return config


@pytest.fixture
def client(mock_config):
    """Create an AI client with mock configuration."""
    return LLMClient(mock_config)


def test_is_local_path(client):
    """Test local path detection."""
    # Local paths - should return True
    assert client._is_local_path("/path/to/image.jpg") is True
    assert client._is_local_path("/Users/john/Desktop/photo.png") is True
    assert client._is_local_path("C:\\Users\\john\\Pictures\\image.jpg") is True
    assert client._is_local_path("D:\\photos\\vacation.png") is True

    # Non-local paths - should return False
    assert client._is_local_path("https://example.com/image.jpg") is False
    assert client._is_local_path("http://example.com/image.jpg") is False
    assert client._is_local_path("data:image/jpeg;base64,abc123") is False
    assert client._is_local_path("relative/path/image.jpg") is False


def test_read_and_encode_image(client):
    """Test reading and encoding a local image file."""
    # Create a temporary image file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        test_data = JPEG_BYTES
        tmp_file.write(test_data)
        tmp_file_path = tmp_file.name

    try:
        # Test successful encoding
        result = client._read_and_encode_image(tmp_file_path)
        assert result is not None
        assert result.startswith("data:image/jpeg;base64,")

        # Decode and verify the data
        base64_part = result.split(",")[1]
        decoded_data = base64.b64decode(base64_part)
        assert decoded_data == test_data

        # Test non-existent file
        with pytest.raises(ValueError, match="Failed to read local image"):
            client._read_and_encode_image("/non/existent/file.jpg")

    finally:
        # Clean up
        Path(tmp_file_path).unlink()


def test_process_content_item(client):
    """Test processing individual content items."""
    # Create a temporary image file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_file.write(PNG_BYTES)
        tmp_file_path = tmp_file.name

    try:
        # Test local image path conversion
        item = {"type": "image_url", "image_url": {"url": tmp_file_path}}
        processed = client._process_content_item(item)
        assert processed["image_url"]["url"].startswith("data:image/png;base64,")

        # Test remote URL (should not be modified)
        item = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.jpg"},
        }
        processed = client._process_content_item(item)
        assert processed["image_url"]["url"] == "https://example.com/image.jpg"

        # Test text content (should not be modified)
        item = {"type": "text", "text": "Hello"}
        processed = client._process_content_item(item)
        assert processed == item

        # Test non-dict item
        processed = client._process_content_item("plain string")
        assert processed == "plain string"

    finally:
        Path(tmp_file_path).unlink()


def test_process_messages_for_local_images(client):
    """Test processing messages with local images."""
    # Create temporary image files
    with (
        tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp1,
        tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp2,
    ):
        tmp1.write(JPEG_BYTES)
        tmp2.write(PNG_BYTES)
        path1, path2 = tmp1.name, tmp2.name

    try:
        # Test message with multiple local images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at these images"},
                    {"type": "image_url", "image_url": {"url": path1}},
                    {"type": "image_url", "image_url": {"url": path2}},
                ],
            },
            {"role": "assistant", "content": "I'll analyze them"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "And this one"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/remote.jpg"},
                    },
                ],
            },
        ]

        processed = client._process_messages_for_local_images(messages)

        # Check first message - local images should be converted
        assert processed[0]["content"][0]["type"] == "text"
        assert processed[0]["content"][1]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,"
        )
        assert processed[0]["content"][2]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )

        # Check second message - string content should be unchanged
        assert processed[1]["content"] == "I'll analyze them"

        # Check third message - remote URL should be unchanged
        assert processed[2]["content"][0]["type"] == "text"
        assert (
            processed[2]["content"][1]["image_url"]["url"]
            == "https://example.com/remote.jpg"
        )

    finally:
        Path(path1).unlink()
        Path(path2).unlink()


def test_chat_with_local_images(client, mock_config):
    """Test the chat method with local images."""
    # Create a temporary image file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file.write(JPEG_BYTES)
        tmp_file_path = tmp_file.name

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": tmp_file_path}},
                ],
            }
        ]

        # Mock the litellm.completion call
        with patch("conferllm.client.litellm.completion") as mock_completion:
            mock_response = MagicMock()
            mock_response.model_dump.return_value = {
                "choices": [{"message": {"content": "I see an image"}}]
            }
            mock_completion.return_value = mock_response

            # Call chat method
            client.chat("test-model", messages)

            # Verify that litellm was called with base64-encoded image
            called_messages = mock_completion.call_args[1]["messages"]
            assert len(called_messages) == 1
            assert called_messages[0]["content"][1]["image_url"]["url"].startswith(
                "data:image/jpeg;base64,"
            )

    finally:
        Path(tmp_file_path).unlink()


def test_mixed_image_types(client):
    """Test handling of mixed image types in a single message."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file.write(JPEG_BYTES)
        local_path = tmp_file.name

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these images"},
                    {"type": "image_url", "image_url": {"url": local_path}},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/remote.jpg"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                    },
                ],
            }
        ]

        processed = client._process_messages_for_local_images(messages)

        # Local path should be converted
        assert processed[0]["content"][1]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,"
        )
        # Remote URL should remain unchanged
        assert (
            processed[0]["content"][2]["image_url"]["url"]
            == "https://example.com/remote.jpg"
        )
        # Base64 should remain unchanged
        assert (
            processed[0]["content"][3]["image_url"]["url"]
            == "data:image/png;base64,iVBORw0KGgo="
        )

    finally:
        Path(local_path).unlink()


@pytest.mark.parametrize(
    ("data", "mime_type"),
    [
        (PNG_BYTES, "image/png"),
        (JPEG_BYTES, "image/jpeg"),
        (b"GIF89aimage", "image/gif"),
        (WEBP_BYTES, "image/webp"),
        (b"BMimage", "image/bmp"),
        (SVG_BYTES, "image/svg+xml"),
    ],
)
def test_detect_image_mime_from_content(data: bytes, mime_type: str) -> None:
    """Trust image bytes rather than a filename extension."""
    assert detect_image_mime(data) == mime_type


def test_load_multiple_images_preserves_order_and_detected_metadata(
    tmp_path: Path,
) -> None:
    """Load three ordered images with content-derived MIME and hashes."""
    first = tmp_path / "misleading.txt"
    second = tmp_path / "second.jpg"
    third = tmp_path / "third.svg"
    first.write_bytes(PNG_BYTES)
    second.write_bytes(JPEG_BYTES)
    third.write_bytes(SVG_BYTES)

    payloads = load_image_inputs(
        [first, second, third],
        max_count=3,
        max_bytes_per_image=1024,
        max_total_bytes=4096,
    )

    assert [payload.mime_type for payload in payloads] == [
        "image/png",
        "image/jpeg",
        "image/svg+xml",
    ]
    assert [payload.source_name for payload in payloads] == [
        "misleading.txt",
        "second.jpg",
        "third.svg",
    ]
    assert all(len(payload.sha256) == 64 for payload in payloads)


def test_load_data_url_rejects_mime_mismatch() -> None:
    """Reject a declared PNG whose bytes are actually JPEG."""
    value = "data:image/png;base64," + base64.b64encode(JPEG_BYTES).decode("ascii")

    with pytest.raises(ImageValidationError) as exc_info:
        load_image_inputs(
            [value],
            max_count=1,
            max_bytes_per_image=1024,
            max_total_bytes=1024,
        )

    assert exc_info.value.code == "invalid_image_data"
    assert exc_info.value.details["detected_mime_type"] == "image/jpeg"


def test_load_images_enforces_count_per_image_and_total_limits(
    tmp_path: Path,
) -> None:
    """Apply deterministic limits before a provider request."""
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(PNG_BYTES)
    second.write_bytes(PNG_BYTES)

    with pytest.raises(ImageValidationError) as count_error:
        load_image_inputs(
            [first, second],
            max_count=1,
            max_bytes_per_image=1024,
            max_total_bytes=4096,
        )
    assert count_error.value.code == "image_limit_exceeded"

    with pytest.raises(ImageValidationError) as size_error:
        load_image_inputs(
            [first],
            max_count=1,
            max_bytes_per_image=4,
            max_total_bytes=4096,
        )
    assert size_error.value.code == "image_too_large"

    with pytest.raises(ImageValidationError) as total_error:
        load_image_inputs(
            [first, second],
            max_count=2,
            max_bytes_per_image=1024,
            max_total_bytes=len(PNG_BYTES),
        )
    assert total_error.value.code == "image_limit_exceeded"


def test_recursive_output_extraction_preserves_multiple_image_order() -> None:
    """Find multiple generated images across nested response fields."""
    first = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")
    second = "data:image/svg+xml;base64," + base64.b64encode(SVG_BYTES).decode("ascii")
    saved: list[tuple[str, int]] = []

    processed, uris = replace_embedded_image_data(
        {
            "choices": [{"message": {"content": f"first {first}"}}],
            "provider_images": [{"url": second}],
        },
        lambda payload, index: (
            saved.append((payload.mime_type, index))
            or f"conferllm://sessions/20260904-{'a' * 32}/"
            f"artifacts/t0001-output-{index + 1:03d}"
        ),
    )

    assert saved == [("image/png", 0), ("image/svg+xml", 1)]
    assert uris[0].endswith("t0001-output-001")
    assert uris[1].endswith("t0001-output-002")
    assert processed["provider_images"][0]["url"] == uris[1]


def test_response_image_batch_failure_removes_previously_moved_files(
    tmp_path: Path,
) -> None:
    """Leave no partial final batch when a later atomic move fails."""
    first = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")
    second = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")
    real_replace = __import__("os").replace
    calls = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        real_replace(source, destination)

    with (
        patch("conferllm.images.os.replace", side_effect=fail_second_replace),
        pytest.raises(OSError, match="disk full"),
    ):
        process_response_for_images(
            {"choices": [{"message": {"content": first + second}}]},
            tmp_path,
        )

    assert list(tmp_path.glob("conferllm_image_*")) == []
    assert list(tmp_path.glob(".conferllm-images-*")) == []


def test_response_rejects_invalid_declared_image_content() -> None:
    """Reject generated data URLs whose declared MIME does not match bytes."""
    bad = "data:image/png;base64," + base64.b64encode(JPEG_BYTES).decode("ascii")

    with pytest.raises(ImageProcessingError, match="does not match"):
        replace_embedded_image_data({"image": bad}, lambda payload, index: "unused")
