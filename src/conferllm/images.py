"""Image validation, normalization, and provider-response helpers."""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import logging
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MIME_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/svg+xml": ".svg",
}
SUPPORTED_IMAGE_MIME_TYPES = frozenset(MIME_EXTENSIONS)
MIME_ALIASES = {
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
    "image/x-png": "image/png",
    "image/x-ms-bmp": "image/bmp",
}

DATA_URL_PATTERN = re.compile(
    r"data:(?P<mime>image/[A-Za-z0-9.+-]+);base64,"
    r"(?P<data>[A-Za-z0-9+/]+={0,2})",
    re.IGNORECASE,
)
DATA_URL_HEADER_PATTERN = re.compile(
    r"data:image/[A-Za-z0-9.+-]+;base64,", re.IGNORECASE
)
ARTIFACT_URI_PATTERN = re.compile(
    r"conferllm://sessions/(?P<session_id>\d{8}-[0-9a-f]{32})/"
    r"artifacts/(?P<artifact_id>t\d{4}-(?:input|output)-\d{3})"
)
_SVG_PREFIX = re.compile(
    r"""
    \A\s*
    (?:<\?xml[^>]*\?>\s*)?
    (?:(?:<!--.*?-->)\s*)*
    (?:<!DOCTYPE\s+svg[^>]*>\s*)?
    <svg(?:\s|>)
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)


class ImageProcessingError(RuntimeError):
    """Raised when generated image content cannot be processed safely."""

    def __init__(self, message: str, *, code: str = "provider_error") -> None:
        super().__init__(message)
        self.code = code


class ImageValidationError(ValueError):
    """Raised when an input is not a supported image."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_image_data",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True)
class ImagePayload:
    """One validated image held in memory exactly once."""

    data: bytes
    mime_type: str
    extension: str
    sha256: str
    source_name: str

    @property
    def size_bytes(self) -> int:
        """Return the byte length of the image."""
        return len(self.data)


def normalize_image_mime(mime_type: str) -> str:
    """Normalize supported MIME aliases."""
    normalized = mime_type.strip().lower()
    return MIME_ALIASES.get(normalized, normalized)


def detect_image_mime(data: bytes) -> str | None:
    """Detect a supported image MIME type from its bytes."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"

    try:
        prefix = data[:8192].decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    if _SVG_PREFIX.search(prefix):
        return "image/svg+xml"
    return None


def extension_for_mime(mime_type: str) -> str:
    """Return a stable file extension for a supported image MIME type."""
    normalized = normalize_image_mime(mime_type)
    try:
        return MIME_EXTENSIONS[normalized]
    except KeyError as error:
        raise ImageValidationError(
            f"Unsupported image MIME type: {mime_type}",
            code="unsupported_image_type",
            details={"mime_type": mime_type},
        ) from error


def decode_image_data_url(value: str, *, index: int | None = None) -> ImagePayload:
    """Decode and validate one complete image data URL."""
    match = DATA_URL_PATTERN.fullmatch(value.strip())
    details = {} if index is None else {"index": index}
    if match is None:
        raise ImageValidationError(
            "Image data URL must use data:image/...;base64,... format.",
            code="invalid_image_data",
            details=details,
        )

    declared_mime = normalize_image_mime(match.group("mime"))
    if declared_mime not in SUPPORTED_IMAGE_MIME_TYPES:
        raise ImageValidationError(
            f"Unsupported image MIME type: {declared_mime}",
            code="unsupported_image_type",
            details={**details, "mime_type": declared_mime},
        )
    try:
        data = base64.b64decode(match.group("data"), validate=True)
    except (binascii.Error, ValueError) as error:
        raise ImageValidationError(
            "Image data URL contains invalid base64.",
            code="invalid_image_data",
            details=details,
        ) from error

    return image_payload_from_bytes(
        data,
        declared_mime=declared_mime,
        source_name=f"data-url-{index + 1}" if index is not None else "data-url",
        index=index,
    )


def image_payload_from_bytes(
    data: bytes,
    *,
    declared_mime: str | None = None,
    source_name: str = "image",
    index: int | None = None,
) -> ImagePayload:
    """Validate bytes and return canonical image metadata."""
    details: dict[str, Any] = {} if index is None else {"index": index}
    detected_mime = detect_image_mime(data)
    if detected_mime is None:
        display_index = "" if index is None else f" {index + 1}"
        raise ImageValidationError(
            f"Input image{display_index} is not a supported image.",
            code="unsupported_image_type",
            details=details,
        )

    if declared_mime is not None:
        normalized_declared = normalize_image_mime(declared_mime)
        if normalized_declared != detected_mime:
            raise ImageValidationError(
                "Declared image MIME type does not match its content.",
                code="invalid_image_data",
                details={
                    **details,
                    "declared_mime_type": normalized_declared,
                    "detected_mime_type": detected_mime,
                },
            )

    return ImagePayload(
        data=data,
        mime_type=detected_mime,
        extension=extension_for_mime(detected_mime),
        sha256=hashlib.sha256(data).hexdigest(),
        source_name=source_name,
    )


def load_image_inputs(
    sources: Sequence[str | Path] | None,
    *,
    max_count: int,
    max_bytes_per_image: int,
    max_total_bytes: int,
) -> list[ImagePayload]:
    """Read and validate ordered image inputs under deterministic limits."""
    if not sources:
        return []
    if len(sources) > max_count:
        raise ImageValidationError(
            f"At most {max_count} input images are allowed per turn.",
            code="image_limit_exceeded",
            details={"count": len(sources), "max_count": max_count},
        )

    payloads: list[ImagePayload] = []
    total_bytes = 0

    def check_size(size: int, index: int) -> None:
        if size > max_bytes_per_image:
            raise ImageValidationError(
                f"Input image {index + 1} exceeds the per-image byte limit.",
                code="image_too_large",
                details={
                    "index": index,
                    "size_bytes": size,
                    "max_bytes": max_bytes_per_image,
                },
            )
        if total_bytes + size > max_total_bytes:
            raise ImageValidationError(
                "Input images exceed the aggregate byte limit.",
                code="image_limit_exceeded",
                details={
                    "index": index,
                    "total_bytes": total_bytes + size,
                    "max_total_bytes": max_total_bytes,
                },
            )

    for index, source in enumerate(sources):
        if isinstance(source, str) and source.lower().startswith("data:"):
            # Bound allocation before base64 decoding. A valid encoded length
            # determines its decoded size exactly (invalid syntax is checked next).
            encoded = source.strip().partition(",")[2]
            padding = len(encoded) - len(encoded.rstrip("="))
            check_size(len(encoded) // 4 * 3 - min(padding, 2), index)
            payload = decode_image_data_url(source, index=index)
        else:
            path = Path(source).expanduser()
            try:
                resolved = path.resolve(strict=True)
            except OSError as error:
                raise ImageValidationError(
                    f"Image file is not readable: {path}",
                    details={"index": index},
                ) from error
            if not resolved.is_file():
                raise ImageValidationError(
                    f"Image path is not a file: {resolved}",
                    details={"index": index},
                )
            try:
                stat_size = resolved.stat().st_size
            except OSError as error:
                raise ImageValidationError(
                    f"Image file is not readable: {resolved}",
                    details={"index": index},
                ) from error
            check_size(stat_size, index)
            try:
                with resolved.open("rb") as source_file:
                    # The file may grow after stat(). Never perform an unbounded
                    # read, even when the original size was below both limits.
                    data = source_file.read(
                        min(max_bytes_per_image, max_total_bytes - total_bytes) + 1
                    )
            except OSError as error:
                raise ImageValidationError(
                    f"Image file is not readable: {resolved}",
                    details={"index": index},
                ) from error
            check_size(len(data), index)
            payload = image_payload_from_bytes(
                data,
                source_name=resolved.name,
                index=index,
            )

        check_size(payload.size_bytes, index)
        total_bytes += payload.size_bytes
        payloads.append(payload)
    return payloads


def image_bytes_to_data_url(data: bytes, mime_type: str) -> str:
    """Encode trusted image bytes for a provider request."""
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{normalize_image_mime(mime_type)};base64,{encoded}"


def replace_image_refs(
    messages: list[dict[str, Any]],
    resolver: Callable[[str], str],
) -> list[dict[str, Any]]:
    """Replace logical image_ref items with provider image_url items."""
    resolved_messages = copy.deepcopy(messages)
    for message in resolved_messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        replacement: list[Any] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_ref":
                artifact_id = item.get("artifact_id")
                if not isinstance(artifact_id, str):
                    raise ImageValidationError(
                        "Stored image reference is missing an artifact ID.",
                        code="invalid_image_data",
                    )
                replacement.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": resolver(artifact_id)},
                    }
                )
            else:
                replacement.append(item)
        message["content"] = replacement
    return resolved_messages


def replace_embedded_image_data(
    value: Any,
    save_image: Callable[[ImagePayload, int], str],
) -> tuple[Any, list[str]]:
    """Replace every embedded image data URL recursively, in encounter order."""
    copied = copy.deepcopy(value)
    saved_uris: list[str] = []

    def replace_string(text: str) -> str:
        def replacement(match: re.Match[str]) -> str:
            index = len(saved_uris)
            try:
                data = base64.b64decode(match.group("data"), validate=True)
            except (binascii.Error, ValueError) as error:
                raise ImageProcessingError(
                    f"Generated image {index + 1} contains invalid base64."
                ) from error
            try:
                payload = image_payload_from_bytes(
                    data,
                    declared_mime=match.group("mime"),
                    source_name=f"generated-{index + 1}",
                    index=index,
                )
                uri = save_image(payload, index)
            except ImageValidationError as error:
                raise ImageProcessingError(str(error)) from error
            saved_uris.append(uri)
            return uri

        headers = list(DATA_URL_HEADER_PATTERN.finditer(text))
        pieces: list[str] = []
        position = 0
        for number, header in enumerate(headers):
            end = (
                headers[number + 1].start() if number + 1 < len(headers) else len(text)
            )
            match = DATA_URL_PATTERN.match(text, header.start(), end)
            if match is None or (
                match.end() < end
                and text[match.end()] not in " \t\r\n\"'<>()[].,;!?:{}"
            ):
                raise ImageProcessingError(
                    f"Generated image {len(saved_uris) + 1} contains invalid base64."
                )
            pieces.append(text[position : match.start()])
            pieces.append(replacement(match))
            position = match.end()
        pieces.append(text[position:])
        return "".join(pieces)

    def walk(item: Any) -> Any:
        if isinstance(item, str):
            return replace_string(item)
        if isinstance(item, list):
            return [walk(child) for child in item]
        if isinstance(item, dict):
            return {key: walk(child) for key, child in item.items()}
        return item

    return walk(copied), saved_uris


def normalize_assistant_content(
    content: Any,
    *,
    artifact_uris: set[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Convert assistant content containing artifact URIs to logical content."""
    normalized: list[dict[str, Any]] = []
    text_parts: list[str] = []

    def add_text(text: str) -> None:
        if not text:
            return
        text_parts.append(text)
        if normalized and normalized[-1].get("type") == "text":
            normalized[-1]["text"] += text
        else:
            normalized.append({"type": "text", "text": text})

    def add_string(text: str) -> None:
        position = 0
        for match in ARTIFACT_URI_PATTERN.finditer(text):
            if artifact_uris is not None and match.group(0) not in artifact_uris:
                continue
            add_text(text[position : match.start()])
            normalized.append(
                {
                    "type": "image_ref",
                    "artifact_id": match.group("artifact_id"),
                }
            )
            position = match.end()
        add_text(text[position:])

    if isinstance(content, str):
        add_string(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                add_string(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text" and isinstance(item.get("text"), str):
                add_string(item["text"])
                continue
            if item_type == "image_ref" and isinstance(item.get("artifact_id"), str):
                normalized.append(
                    {
                        "type": "image_ref",
                        "artifact_id": item["artifact_id"],
                    }
                )
                continue
            if item_type == "image_url":
                image_url = item.get("image_url")
                if isinstance(image_url, dict) and isinstance(
                    image_url.get("url"), str
                ):
                    match = ARTIFACT_URI_PATTERN.fullmatch(image_url["url"])
                    if match is not None and (
                        artifact_uris is None or image_url["url"] in artifact_uris
                    ):
                        normalized.append(
                            {
                                "type": "image_ref",
                                "artifact_id": match.group("artifact_id"),
                            }
                        )
    elif content is not None:
        add_text(str(content))

    return "".join(text_parts), normalized


def prepare_image_output_dir(image_path: str | Path | None) -> Path | None:
    """Validate and create an explicitly requested image export directory."""
    if image_path is None:
        return None

    output_dir = Path(image_path).expanduser()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(
            f"image output path must be a directory, got existing file: {output_dir}"
        )

    existed = output_dir.exists()
    try:
        output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not existed:
            os.chmod(output_dir, 0o700)
    except OSError as error:
        raise ValueError(
            f"Failed to create image directory '{output_dir}': {error}"
        ) from error
    return output_dir


def _persist_embedded_images(
    value: Any,
    output_dir: Path,
) -> Any:
    """Persist a complete generated-image batch or leave no final files."""
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".conferllm-images-", dir=output_dir))
    staged_to_final: list[tuple[Path, Path]] = []
    moved: list[Path] = []

    def save_image(payload: ImagePayload, index: int) -> str:
        filename = f"conferllm_image_{uuid.uuid4().hex}{payload.extension}"
        staged = staging / filename
        final = output_dir / filename
        try:
            staged.write_bytes(payload.data)
            staged.chmod(0o600)
        except OSError as error:
            raise ImageProcessingError(
                f"Failed to save generated image {index + 1}: {error}"
            ) from error
        staged_to_final.append((staged, final))
        return str(final)

    try:
        processed, _ = replace_embedded_image_data(value, save_image)
        for staged, final in staged_to_final:
            os.replace(staged, final)
            moved.append(final)
            final.chmod(0o600)
        return processed
    except Exception:
        for final in moved:
            try:
                final.unlink()
            except OSError:
                logger.warning("Unable to remove partial image output: %s", final)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def extract_and_save_base64_images(content: str, output_dir: Path | None = None) -> str:
    """Replace all embedded image data URLs with safely persisted file paths."""
    target_dir = output_dir or Path(tempfile.gettempdir())
    processed = _persist_embedded_images(content, target_dir)
    if not isinstance(processed, str):  # pragma: no cover - type invariant
        raise ImageProcessingError("Generated image processing returned invalid text.")
    return processed


def process_response_for_images(
    response_dict: dict[str, Any], output_dir: Path | None = None
) -> dict[str, Any]:
    """Return a copy with every embedded image persisted transactionally."""
    target_dir = output_dir or Path(tempfile.gettempdir())
    processed = _persist_embedded_images(response_dict, target_dir)
    if not isinstance(processed, dict):  # pragma: no cover - type invariant
        raise ImageProcessingError("Generated image processing returned invalid data.")
    return processed
