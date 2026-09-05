"""Command-line interface for ConferLLM."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any, TextIO, cast

from . import __version__
from .chat import ChatResult, ChatService
from .client import LLMClient
from .config import ConferLLMConfig
from .errors import normalize_error
from .server import Transport, run_server
from .session import SessionListWarning, SessionMetadata, SessionStore

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _add_common_options(
    parser: argparse.ArgumentParser, *, suppress_defaults: bool = False
) -> None:
    default: Any = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument(
        "--config",
        type=Path,
        default=default,
        help="Configuration file (default: ~/.conferllm/config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default=argparse.SUPPRESS if suppress_defaults else "INFO",
        help="Logging level (default: INFO)",
    )


def _add_server_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "http"),
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for sse/http transports (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3001,
        help="Port for sse/http transports (default: 3001)",
    )


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid date '{value}'; expected YYYY-MM-DD"
        ) from error


def _parse_limit(value: str) -> int:
    try:
        limit = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("limit must be an integer") from error
    if limit < 0:
        raise argparse.ArgumentTypeError("limit must be non-negative")
    return limit


def create_parser() -> argparse.ArgumentParser:
    """Create the ConferLLM argument parser."""
    parser = argparse.ArgumentParser(
        prog="conferllm",
        description="Talk to locally configured AI models or run an MCP server.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    _add_common_options(parser)

    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Run the MCP server")
    _add_common_options(serve_parser, suppress_defaults=True)
    _add_server_options(serve_parser)

    chat_parser = subparsers.add_parser(
        "chat", help="Create or continue a model conversation"
    )
    _add_common_options(chat_parser, suppress_defaults=True)
    target_group = chat_parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--model", help="Configured model for a new session")
    target_group.add_argument("--session", help="Existing session ID to continue")
    chat_parser.add_argument(
        "--name",
        help="Optional name for a new session; valid only with --model",
    )
    prompt_group = chat_parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Prompt text")
    prompt_group.add_argument(
        "--prompt-file",
        type=Path,
        help="UTF-8 file containing the prompt",
    )
    chat_parser.add_argument(
        "--image",
        action="append",
        type=Path,
        default=[],
        help="Local image to include; may be repeated",
    )
    chat_parser.add_argument(
        "--image-output-dir",
        type=Path,
        help="Directory for images returned by the model",
    )
    chat_parser.add_argument(
        "--json",
        action="store_true",
        help="Print a structured result including the session ID",
    )
    chat_parser.add_argument(
        "--include-raw-response",
        action="store_true",
        help="Include the sanitized provider response in JSON output",
    )

    sessions_parser = subparsers.add_parser(
        "sessions", help="Inspect stored chat sessions"
    )
    sessions_subparsers = sessions_parser.add_subparsers(
        dest="sessions_command", required=True
    )
    list_parser = sessions_subparsers.add_parser("list", help="List chat sessions")
    _add_common_options(list_parser, suppress_defaults=True)
    list_parser.add_argument(
        "--query",
        help="Case-insensitive substring match against session ID or name",
    )
    list_parser.add_argument("--model", help="Exact configured model alias")
    list_parser.add_argument(
        "--since",
        type=_parse_date,
        help="Include sessions created on or after YYYY-MM-DD",
    )
    list_parser.add_argument(
        "--until",
        type=_parse_date,
        help="Include sessions created on or before YYYY-MM-DD",
    )
    list_parser.add_argument(
        "--limit",
        type=_parse_limit,
        default=50,
        help="Maximum results; 0 means unlimited (default: 50)",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Print complete session metadata as JSON",
    )

    models_parser = subparsers.add_parser("models", help="List configured model names")
    _add_common_options(models_parser, suppress_defaults=True)
    models_parser.add_argument("--json", action="store_true", help="Print a JSON array")

    info_parser = subparsers.add_parser(
        "model-info", help="Show non-secret model configuration metadata"
    )
    _add_common_options(info_parser, suppress_defaults=True)
    info_parser.add_argument("model", help="Configured model name")

    skill_parser = subparsers.add_parser("skill", help="Manage the bundled Agent Skill")
    skill_subparsers = skill_parser.add_subparsers(dest="skill_command", required=True)
    install_parser = skill_subparsers.add_parser(
        "install", help="Install the bundled ConferLLM Agent Skill"
    )
    install_parser.add_argument(
        "--target",
        choices=("agents", "codex"),
        default="agents",
        help="Standard skill root to use (default: agents)",
    )
    install_parser.add_argument(
        "--destination",
        type=Path,
        help="Install to this exact directory instead of the target default",
    )
    install_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing installation",
    )

    doctor_parser = subparsers.add_parser(
        "doctor", help="Check package, configuration, and Skill setup"
    )
    _add_common_options(doctor_parser, suppress_defaults=True)
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable diagnostic report",
    )

    return parser


def _load_client(config_path: Path | None) -> LLMClient:
    return LLMClient(ConferLLMConfig.load_config(config_path))


def _load_session_store(config_path: Path | None) -> SessionStore:
    config = ConferLLMConfig.load_config(config_path)
    return SessionStore(config.get_session_root())


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return cast(str, args.prompt)
    return cast(Path, args.prompt_file).read_text(encoding="utf-8")


def _print_json(value: Any, output: TextIO) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str), file=output)


def _print_answer(answer: Any, output: TextIO) -> None:
    if isinstance(answer, str):
        print(answer, file=output)
    else:
        _print_json(answer, output)


def _print_chat_result(
    result: ChatResult, output: TextIO, error_output: TextIO
) -> None:
    print(f"Session: {result.session_id}", file=output)
    print(f"Name: {result.name}", file=output)
    print(file=output)
    _print_answer(result.answer, output)
    for artifact in result.artifacts:
        if artifact["direction"] == "output":
            location = (
                artifact.get("exported_path")
                or artifact.get("local_path")
                or artifact.get("uri")
            )
            print(f"Image: {location}", file=output)
    for warning in result.warnings:
        print(f"conferllm: warning: {warning}", file=error_output)


def _serialize(value: Any) -> Any:
    """Convert diagnostic or installation results into JSON-compatible values."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _print_sessions(
    sessions: list[SessionMetadata],
    warnings: list[SessionListWarning],
    output: TextIO,
    error_output: TextIO,
    *,
    as_json: bool,
) -> None:
    if as_json:
        _print_json(
            {
                "schema_version": "conferllm.sessions.response.v1",
                "sessions": [session.to_dict() for session in sessions],
                "warnings": [warning.to_dict() for warning in warnings],
            },
            output,
        )
        return

    print(f"{'SESSION_ID':<41}  NAME", file=output)
    for session in sessions:
        print(f"{session.session_id:<41}  {session.name}", file=output)
    for warning in warnings:
        print(f"conferllm: warning: {warning.message}", file=error_output)


def _install_skill(*, target: str, destination: Path | None, force: bool) -> Any:
    from .skill import SkillTarget, install_skill

    typed_target = cast(SkillTarget, target)
    return install_skill(
        target=typed_target,
        destination=destination,
        force=force,
    )


def _run_doctor(config_path: Path | None) -> Any:
    from .doctor import run_doctor

    return run_doctor(config_path=config_path)


def _print_command_result(result: Any, output: TextIO, *, as_json: bool) -> None:
    serialized = _serialize(result)
    if as_json:
        _print_json(serialized, output)
    elif isinstance(serialized, str):
        print(serialized, file=output)
    elif isinstance(serialized, dict) and isinstance(serialized.get("message"), str):
        print(serialized["message"], file=output)
    else:
        _print_json(serialized, output)


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    output: TextIO | None = None,
    error_output: TextIO | None = None,
) -> int:
    """Run the CLI and return a process exit code."""
    parser = create_parser()
    args = parser.parse_args(argv)
    stdout = output or sys.stdout
    stderr = error_output or sys.stderr

    if args.command is None:
        parser.print_help(file=stdout)
        return 0

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=stderr,
    )
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    json_errors = bool(getattr(args, "json", False))
    previous_logging_disable = logging.root.manager.disable
    if json_errors:
        # Runtime error JSON owns stderr in machine mode. Library log records
        # must not precede the envelope and make it impossible to parse.
        logging.disable(logging.CRITICAL)

    try:
        if args.command == "serve":
            run_server(
                config_path=args.config,
                transport=cast(Transport, args.transport),
                host=args.host,
                port=args.port,
                log_level=args.log_level,
            )
            return 0

        if args.command == "sessions":
            listing = _load_session_store(args.config).list_sessions_detailed(
                query=args.query,
                model=args.model,
                since=args.since,
                until=args.until,
                limit=args.limit,
            )
            _print_sessions(
                listing.sessions,
                listing.warnings,
                stdout,
                stderr,
                as_json=args.json,
            )
            return 0

        if args.command == "skill":
            result = _install_skill(
                target=args.target,
                destination=args.destination,
                force=args.force,
            )
            _print_command_result(result, stdout, as_json=False)
            return 0

        if args.command == "doctor":
            result = _run_doctor(args.config)
            _print_command_result(result, stdout, as_json=args.json)
            return 1 if result.get("ok") is False else 0

        client = _load_client(args.config)

        if args.command == "models":
            models = client.list_models()
            if args.json:
                _print_json(models, stdout)
            else:
                for model in models:
                    print(model, file=stdout)
            return 0

        if args.command == "model-info":
            _print_json(client.get_model_info(args.model), stdout)
            return 0

        if args.command == "chat":
            result = ChatService(client).chat(
                _read_prompt(args),
                model=args.model,
                session_id=args.session,
                name=args.name,
                images=args.image,
                image_output_dir=args.image_output_dir,
                include_raw_response=args.include_raw_response,
            )
            if args.json:
                _print_json(
                    result.to_dict(include_local_paths=True),
                    stdout,
                )
            else:
                _print_chat_result(result, stdout, stderr)
            return 0

        parser.error(f"Unknown command: {args.command}")
    except KeyboardInterrupt:
        print("conferllm: interrupted", file=stderr)
        return 130
    except Exception as error:
        public_error = normalize_error(error)
        if json_errors:
            _print_json(public_error.to_dict(), stderr)
        else:
            print(f"conferllm: error: {public_error.message}", file=stderr)
        return 1
    finally:
        if json_errors:
            logging.disable(previous_logging_disable)


def main() -> None:
    """Console-script entry point."""
    raise SystemExit(run_cli())
