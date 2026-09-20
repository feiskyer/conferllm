"""Opt-in real image-generation and multi-image input acceptance through the CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
import time
from io import StringIO
from pathlib import Path
from typing import Any

import httpx
import litellm

from conferllm import cli
from conferllm.chat import ChatService
from conferllm.client import LLMClient
from conferllm.errors import ConferLLMError
from conferllm.images import image_payload_from_bytes
from conferllm.session import SessionStore
from conferllm.tools import ToolRuntime


def run(model: str, directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    os.chdir(directory)
    requests: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    tools_attempted: list[str] = []
    original_service, original_chat = cli.ChatService, LLMClient.chat
    original_execute, original_send = ToolRuntime.execute, httpx.Client.send
    phase = ""

    def save() -> None:
        (directory / "report.json").write_text(
            json.dumps(
                {
                    "model": model,
                    "phases": phases,
                    "requests": requests,
                    "tools_attempted": tools_attempted,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    class IsolatedChatService(ChatService):
        def __init__(self, client: LLMClient) -> None:
            super().__init__(client, SessionStore(directory / "sessions"))

    def observed_chat(
        client: LLMClient, alias: str, messages: list[dict[str, Any]]
    ) -> Any:
        print(
            json.dumps({"model": alias, "phase": phase, "calling_real_model": True}),
            flush=True,
        )
        return original_chat(client, alias, messages)

    def observed_send(
        client: httpx.Client, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        response = original_send(client, request, **kwargs)
        requests.append(
            {
                "phase": phase,
                "method": request.method,
                "status": response.status_code,
            }
        )
        save()
        return response

    def no_local_tools(_runtime: ToolRuntime, name: str, _arguments: str) -> str:
        tools_attempted.append(name)
        save()
        raise ConferLLMError(
            "image_smoke_scope",
            "Image acceptance requires native image generation, not local tool execution.",
        )

    def invoke(arguments: list[str]) -> dict[str, Any]:
        before = time.monotonic()
        output, error = StringIO(), StringIO()
        status = cli.run_cli(
            ["chat", *arguments, "--json"], output=output, error_output=error
        )
        try:
            result = json.loads(output.getvalue() if status == 0 else error.getvalue())
        except json.JSONDecodeError:
            result = {"ok": False, "error": {"code": "invalid_cli_output"}}
        record: dict[str, Any] = {
            "phase": phase,
            "exit_code": status,
            "duration_s": round(time.monotonic() - before, 3),
            "response": result,
            "images": [],
        }
        for artifact in result.get("artifacts", []):
            if artifact["direction"] != "output":
                continue
            path = Path(artifact["saved_path"])
            data = path.read_bytes()
            payload = image_payload_from_bytes(data)
            record["images"].append(
                {
                    "path": str(path),
                    "size": len(data),
                    "mime_type": payload.mime_type,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "path_in_message": str(path) in result["message"]["text"],
                }
            )
        record["no_embedded_images_in_result"] = "data:image/" not in json.dumps(result)
        phases.append(record)
        save()
        print(
            json.dumps(
                {
                    "phase": phase,
                    "exit_code": status,
                    "image_count": len(record["images"]),
                    "paths": [image["path"] for image in record["images"]],
                }
            ),
            flush=True,
        )
        return record

    def timeout(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    cli.ChatService = IsolatedChatService
    LLMClient.chat = observed_chat
    ToolRuntime.execute = no_local_tools
    httpx.Client.send = observed_send
    litellm.telemetry = False
    logging.disable(logging.CRITICAL)
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(600)
    try:
        phase = "default_tmp_output"
        first = invoke(
            [
                "--model",
                model,
                "--prompt",
                "Generate an actual image: a clean flat red circle centered on a white "
                "square background. Return the generated image, not code or instructions. "
                "Do not invoke any local shell or file tool. This is an image-output acceptance test.",
            ]
        )
        phase = "specified_output_directory"
        second = invoke(
            [
                "--model",
                model,
                "--prompt",
                "Generate an actual image: a clean flat blue triangle centered on a yellow "
                "square background. Return the generated image, not code or instructions. "
                "Do not invoke any local shell or file tool. This is an image-output acceptance test.",
                "--image-output-dir",
                str(directory / "chosen-output"),
            ]
        )
        if first["images"] and second["images"]:
            phase = "multiple_image_inputs"
            invoke(
                [
                    "--model",
                    model,
                    "--image",
                    first["images"][0]["path"],
                    "--image",
                    second["images"][0]["path"],
                    "--prompt",
                    "Inspect both attached images in order. Reply with text only: "
                    "name the foreground shape/color and background color of image 1 and image 2. "
                    "Do not generate images, write code, or call local tools.",
                ]
            )
            phase = "image_history_followup"
            invoke(
                [
                    "--session",
                    first["response"]["session"]["id"],
                    "--prompt",
                    "Using the image already in our conversation, describe its "
                    "foreground shape/color and background color. Text only. Do not generate "
                    "new images or call any local tool.",
                ]
            )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        cli.ChatService, LLMClient.chat = original_service, original_chat
        ToolRuntime.execute, httpx.Client.send = original_execute, original_send
        save()
    return {"phases": phases, "requests": requests}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    run(args.model, args.directory)


if __name__ == "__main__":
    main()
