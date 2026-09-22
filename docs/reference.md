# ConferLLM reference

For a first conversation, start with the [README](../README.md#quick-start). This page covers configuration, integration contracts, and operational details.

## Installation

Install with [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended), which keeps the CLI in an isolated tool environment:

```bash
uv tool install conferllm
```

Alternatively, use pip in an activated Python virtual environment:

```bash
python -m pip install conferllm
```

Verify the installation:

```bash
conferllm --version
conferllm --help
```

For source checkout and editable installation instructions, see [Development](#development).

The Python package, import package, executable, and Agent Skill are all named `conferllm`. Python 3.10+ and POSIX file locking are required; native Windows is not supported.

## Configuration

The default path is `~/.conferllm/config.yaml`. Start with the README's single-model configuration, or copy the [multi-provider example](../config_example.yaml) from a source checkout:

```bash
mkdir -p ~/.conferllm
chmod 700 ~/.conferllm
if [ ! -e ~/.conferllm/config.yaml ]; then
  cp config_example.yaml ~/.conferllm/config.yaml
fi
chmod 600 ~/.conferllm/config.yaml
```

Replace placeholder credentials for the models you want to use and remove unused entries. The example includes the `gpt-6-astra` and `claude-opus-5` aliases used throughout the README. Model IDs are examples, not a guarantee of provider availability. The example file is part of the source checkout, not installed beside the CLI.

Each `model_name` must be non-empty and unique. It is the local alias passed to `--model`, not necessarily the provider's model ID. `litellm_params.model` is required and identifies the provider model.

Parameters in `litellm_params` are forwarded to LiteLLM, with format-specific mapping for Responses. ConferLLM owns `messages`/`input` and always sets `stream=false`, its built-in `tools`, and `tool_choice=auto`; configured message/input payloads are rejected. Legacy `functions`/`function_call` parameters are ignored, `drop_params` is forced off, and `additional_drop_params` cannot remove `tools` or `tool_choice`. Unknown application configuration fields are also rejected.

Each model has an `api_format` field accepting exactly `responses` (the default) or `chat_completion`. This setting selects the transport only for the `openai` provider, including custom endpoints configured as `openai/MODEL`; other providers retain their existing integration. Set it beside `model_name`, not inside `litellm_params`. `model-info` reports the configured value and `uses_responses_api`.

```yaml
model_list:
  - model_name: gpt-6-astra
    api_format: responses
    litellm_params:
      model: openai/gpt-6-astra
      api_key: "replace-with-your-key"
  - model_name: legacy-chat
    api_format: chat_completion
    litellm_params:
      model: openai/your-legacy-model
      api_base: "https://your-endpoint.example/v1"
      api_key: "replace-with-your-key"
```

Responses sends flat function definitions and typed `input` items. It preserves native function `call_id` values and all returned reasoning items when replaying tool results; it does not substitute the separate output-item `id`. Local conversation state is used with `store=false` by default and encrypted reasoning requested. Tool schemas explicitly use `strict=false` to preserve existing optional/default parameters, while local argument validation remains enabled. `max_tokens`/`max_completion_tokens` map to `max_output_tokens`, `reasoning_effort` maps to `reasoning`, and `response_format` maps to `text.format`. Chat-only parameters such as `stop`, `logprobs`, or `n>1` produce an explicit configuration error; select `chat_completion` when the endpoint requires that protocol. There is no automatic protocol downgrade.

Replayed assistant text uses `output_text`, with its original `phase` retained; user/system/developer text uses `input_text`. Optional message/function output-item IDs are not echoed, avoiding oversized IDs returned by some compatible endpoints. Required reasoning state remains intact. Generated images are supplied as separate image inputs explicitly attributed to the assistant, since assistant output messages do not accept `input_image`. Their order and the original internal history are preserved.

GPT-6 Astra requires Responses for tool calling. See the official [function-calling guide](https://developers.openai.com/api/docs/guides/function-calling) and [Responses migration guide](https://developers.openai.com/api/docs/guides/migrate-to-responses) for the protocol differences.

Optional settings:

- `global_system_prompt`: applied to every request.
- A model's `system_prompt`: overrides the global prompt; `""` disables it.
- `sessions_dir`: overrides the session root. Relative YAML paths resolve beside the configuration file, independently of the caller's working directory.
- `image_limits`: limits new images per turn. Defaults are 16 images, 20 MiB per image, and 50 MiB in total.
- A model's `capabilities`: declares input/output modalities and an optional `max_input_images`. These declarations guide validation, not provider discovery.

Use `--config PATH` to select another file. An explicitly selected missing file is an error. ConferLLM does not create, populate, or automatically migrate provider credentials.

## JSON output

Request `--json` for scripts and agents. The stable chat response envelope is `conferllm.chat.response.v1`. Its primary fields are illustrated below:

```json
{
  "schema_version": "conferllm.chat.response.v1",
  "ok": true,
  "session": {
    "id": "20260905-0123456789abcdef0123456789abcdef",
    "name": "Raft notes",
    "model": "gpt-6-astra",
    "turn": 1
  },
  "message": {
    "text": "The answer...",
    "content": [{"type": "text", "text": "The answer..."}]
  },
  "artifacts": [],
  "warnings": []
}
```

The session ID above is illustrative; continue with the ID returned by your own call. Compatibility fields `session_id`, `name`, `model`, and `answer` are also returned. New integrations should use the nested fields. Raw provider data is excluded unless `--include-raw-response` is supplied; when included, image payloads are replaced by saved paths. All Chat Completions choices and all Responses output items are processed, rather than only the first candidate. Text and images retain their order within a response, and every valid tool call is executed in returned order. Duplicate call IDs are rejected before executing the batch.

For completion integrations, an in-memory, per-request capture preserves native output fields before SDK convenience conversion. Gemini candidates are converted independently so a text-only candidate cannot inherit another candidate's tool calls, images, or reasoning. Interleaved Gemini text/image parts retain their order. A complete Chat Completions response with content arrays can be recovered if the SDK rejects its string-only content field; upstream errors and incomplete responses are not recovered this way. Request bodies and credentials are not retained by this capture.

Successful JSON results go to stdout. Runtime failures exit nonzero and write a `conferllm.error.v1` envelope to stderr. Library logging is suppressed in JSON mode so it does not contaminate the envelope. Argument-parser failures use the normal CLI usage error and exit code 2.

Without `--json`, output includes the session ID, name, answer, and output-image locations; warnings go to stderr. A name is derived locally from the first prompt. Use `--name` to choose one when creating a conversation.

## Session listing

```bash
conferllm sessions list
conferllm sessions list --query raft --json
conferllm sessions list --model gpt-6-astra --since 2026-09-01 --limit 20 --json
```

Filters combine with AND semantics. `--query` matches the ID or name without case sensitivity; model aliases match exactly. `--since` and `--until` are inclusive creation dates. The default limit is 50; `--limit 0` is unlimited. Listing searches metadata, not message bodies, and reports corrupt headers as warnings without hiding unrelated valid sessions.

`--json` returns a `conferllm.sessions.response.v1` object with `sessions` and `warnings` arrays.

For chat commands, exactly one of `--model` and `--session` is required. A session retains its model and name; `--name` is only accepted for a new chat. Prior successful turns, including tool calls and results, are restored automatically without executing old calls again. Compare models by creating independent sessions with the same prompt and attributing each answer.

## Built-in tools

All configured models receive the same native function definitions on every request. There is no opt-in flag, model allowlist, approval prompt, sandbox, command classifier, or read-before-write gate. Tools run on the ConferLLM host using its OS user's access, including for chats invoked through MCP. A read-only prompt is guidance to the model, not an enforced boundary.

| Tools | Operations |
| --- | --- |
| `run_shell`, `git_command` | Noninteractive bash/sh or Git commands, with an optional working directory and timeout |
| `run_powershell` | Noninteractive commands through installed `pwsh`, `powershell`, or `powershell.exe` |
| `shell_output`, `shell_kill` | Read new output/status or stop an owned background command |
| `read_file`, `list_directory` | Read line-numbered UTF-8 text or list entries with filename glob exclusions |
| `write_file`, `append_file`, `edit_file` | Create/overwrite, append, or perform exact string replacements |

The provider must support native function calling: Responses uses `function_call` / `function_call_output`; Chat Completions uses `tool_calls` / tool messages. Typed native content blocks such as `tool_use` are also normalized. ConferLLM does not interpret tool-shaped ordinary text or use a text-protocol fallback. LiteLLM's legacy `ollama/` route is mapped to `ollama_chat/` for the same model so Ollama uses native calling. An unsupported provider produces an explicit error rather than silently omitting tools.

Tool batches execute in order. Each result is returned with the original `tool_call_id`; invalid arguments, missing executables/files, and command failures become error results that the model can correct. Invalid protocol envelopes and duplicate call IDs fail the turn before executing that batch. A turn permits up to 1000 tool rounds, then one final model response; another requested tool round fails with `tool_call_limit_exceeded`.

Relative paths resolve against the invocation's working directory. Shell tools accept `cwd` without changing the server's process-wide directory. Commands have no interactive stdin; foreground timeouts default to 120 seconds (60 for Git) and accept 1–600 seconds. Shell output retains at most 30,000 unread bytes plus a truncation notice.

Background processes and their handles belong to the current chat invocation only. Use `shell_output`/`shell_kill` before the model finishes. Completion, failure, and interruption stop owned process groups and join output readers. A successful turn reports a warning if a background command was still running and had to be stopped. Handles cannot be reused in another session or a later CLI invocation. PowerShell is not installed automatically; the package's existing POSIX platform requirement still applies.

MCP cancellation signals the worker and its subprocesses, and prevents a late provider response from starting more tools. An in-flight blocking provider HTTP request may still complete; cancellation is not proof that it was never billed. A file operation already in progress may finish before cancellation is observed.

File tools handle UTF-8 files up to 10 MiB. Reads default to 2,000 lines and support a 1-based `offset` and `limit`; line-number prefixes are display-only. Listings default to 1,000 entries. Text output/diffs are capped at 30,000 characters. Writes create parent directories, follow symlink targets, preserve existing modes and dominant line endings, and replace files atomically; newly created files use a private mode. `edit_file` requires a unique exact match unless `replace_all=true`; an empty `old_string` creates only a new/empty file. It does not apply unified diffs.

These are reliability limits, not permission controls. Tool results can contain file contents, command output, or host paths and are sent to the model. Commands and file changes are not part of the session-storage transaction and cannot be rolled back by ConferLLM.

## Images

Repeat `--image` to preserve input order. Accepted images are copied into private, session-owned storage, so follow-ups use those copies even if the originals change or disappear. PNG, JPEG, GIF, WebP, BMP, and SVG MIME types are detected from content rather than file extensions. Provider format support may be narrower.

Declared capabilities are checked before reading images or calling the provider. Application limits bound new attachments; a model's `max_input_images` also counts images replayed from history. Unknown image capability produces a warning, not a claim of provider support.

Models can return multiple images, text/image mixtures, and tool calls in the same response. Every image is saved. Public `message.content` uses `image_url` blocks whose `image_url.url` is the saved absolute filesystem path, retaining `artifact_id`; `message.text` substitutes the same paths for images and separates otherwise adjacent paths/text with newlines. Output artifacts expose `saved_path`, MIME type, size, SHA-256, and a stable `conferllm://sessions/.../artifacts/...` URI. These are paths on the CLI/server host, not on a remote MCP client's machine.

With no `image_output_dir` / `--image-output-dir`, images are exported to `/tmp`. Specify a directory for a durable or project-specific location. Immutable canonical copies remain in private session storage and internal history uses artifact references, so follow-ups survive deletion of temporary exports. If an export fails, the returned path falls back to the canonical copy and a warning explains the failure; do not repeat a successful model call to repair an export.

Supported image outputs include embedded data URLs, common base64 image blocks, Responses image-generation items, and HTTP(S) image URLs. Remote image downloads use a fresh client without provider credentials and are limited to 20 MiB per image. Arbitrary provider-supplied local paths are not read as image outputs. Images from every candidate and tool round get distinct artifact IDs. Tool argument strings and encrypted reasoning are not scanned or rewritten as image output; unsupported typed blocks fail explicitly instead of disappearing.

## Agent Skill

Install the Skill bundled with the executable:

```bash
conferllm skill install
conferllm skill install --target codex
conferllm skill install --destination /custom/skills/conferllm
```

These are alternative destinations. The default is `~/.agents/skills/conferllm`; `--target codex` selects `~/.codex/skills/conferllm`. Installation is idempotent. A different existing installation is preserved unless `--force` is explicitly requested. Home, workspace, package directories, and their ancestors are never valid replacement targets.

The [source bundle](../skills/conferllm/) is the single source of truth:

```text
skills/conferllm/
├── SKILL.md
├── agents/openai.yaml
└── references/
    ├── installation.md
    ├── multimodal.md
    └── errors.md
```

The wheel embeds the same files; the source distribution preserves this layout. Source and installed Skills use identical relative reference paths. Installation copies the files: after editing the bundle, rerun `conferllm skill install --force` for the destination you use. Editable Python installation alone does not update an already-installed Skill.

The Skill discovers configured aliases, retains session IDs, preserves image order, and attributes answers. It uses the CLI rather than opening credential or session files.

## MCP server

The default transport is stdio:

```bash
conferllm serve
```

Configure your MCP client to launch that command as shown in the [README](../README.md#use-as-an-mcp-server). To select a non-default configuration, append `--config` and an absolute file path to the client's command arguments.

HTTP and SSE are also supported:

```bash
conferllm serve --transport http --host localhost --port 3001
conferllm serve --transport sse --host localhost --port 3001
```

Run one transport at a time. `http` selects streamable HTTP. Keep network transports bound to localhost unless you have separately arranged appropriate access controls.

Available tools:

- `list_models()` and `get_model_info(model)` discover configured aliases and non-secret metadata.
- `create_chat(message, model, ...)` starts a conversation.
- `continue_chat(message, session_id, ...)` restores and continues a conversation.
- `list_sessions(query, model, since, until, limit)` finds session metadata; all filters are optional.
- `chat(...)` is a compatibility wrapper. Prefer the separate create/continue tools in new integrations.

Chat results include the JSON envelope as both structured content and the first text content block. MCP image inputs must be data URLs or absolute paths on the server's filesystem, not relative paths on the client's machine. Generated images are returned as saved host paths in the envelope, plus resource links for remote access. MCP no longer inlines generated image base64. The `conferllm://sessions/{session_id}/artifacts/{artifact_id}` resource still provides canonical image bytes on an explicit read.

## Storage and reliability

Sessions are UTF-8 JSONL files below `~/.conferllm/sessions/YYYY/MM/DD/`. Every committed turn contains a user message, final assistant answer, optional ordered `tool_messages`, and its artifact metadata. Existing v1/v2 sessions remain readable; tool transcripts are an additive field. Replay validates call/result pairing. Turn usage totals include all model calls, while optional raw response data represents the last provider response. Directories use `0700`; session and image files use `0600`. The configuration file remains user-managed.

Continuation holds a thread/process lock across loading, model/tool execution, and atomic JSONL replacement. Stored image size/hash checks precede replay. Failed provider calls do not append a partial turn, but commands and file edits that already ran remain in effect. Errors after tool attempts include `tool_calls_attempted`, `side_effects_may_remain`, and the attempted session/turn. A new session ID in a failure is not proof that its JSONL was committed.

JSONL is the commit record. A locked continuation can recover the next uncommitted image directory left by a crash; it never removes committed turns. Early staging leftovers are not automatically swept. This is not an exactly-once guarantee for provider or tool execution. A crash can leave external side effects without a committed transcript; do not blindly repeat a failed request. Long histories are not silently summarized or truncated.

Session content can be sensitive. Keep credentials and conversations out of version control. Configuration/provider errors do not echo raw input or provider exception payloads.

## Troubleshooting

Start with diagnostics that do not print configuration values:

```bash
conferllm doctor --json
conferllm models --json
conferllm model-info gpt-6-astra
```

Doctor reports safe metadata and suggested next steps. It exits nonzero when configuration is missing, invalid, or has no model aliases, while still returning its diagnostic report.

- **Command not found:** for uv tool installs, run `uv tool update-shell` and reopen the terminal. MCP clients may need an absolute executable path.
- **Model not found:** use an alias from `conferllm models`, or add the alias to your configuration. A provider model ID is not automatically a local alias.
- **Configuration error:** check the path, YAML syntax, and field names locally. Keep aliases unique and supply `litellm_params.model` for each.
- **Provider error:** check your account's model access, endpoint, credentials, and native tool-calling support. A successful configuration check does not test provider connectivity.
- **Tool-round limit:** inspect the task's scope and any reported side effects before submitting another request. Repeating it can repeat file changes or commands.
- **PowerShell missing:** install PowerShell separately if needed; ConferLLM returns the missing-executable error to the model and does not change the host configuration.
- **Image export warning after success:** the conversation is already saved; retain its session ID instead of making the same provider call again.

## Development

For development, use `uv>=0.12.9,<0.13`. Clone the repository, or skip the first two commands if you already have a checkout:

```bash
git clone https://github.com/feiskyer/mcp-ai-hub.git conferllm
cd conferllm
uv sync --extra dev
uv run ruff format --check .
uv run ruff check .
uv run mypy src/
uv run pytest
uv lock --check
uv build
git diff --check
```

Tests isolate the user home and block unmocked provider calls. They cover persistence, process/thread locking, fault recovery, protocol integration, Skill installation, documentation examples, and package contents.

To link the globally available CLI to this checkout:

```bash
uv tool install --editable . --force
```

The tool environment is separate from the development environment. Python source changes take effect on the next invocation; running servers need a restart. Reinstall after changing dependencies or command entry points.

Dependency major versions are bounded, and `uv.lock` records the development resolution. Python 3.10 uses LiteLLM 1.97.x with a guarded response-type rebuild for its nested forward-reference regression; Python 3.11+ permits LiteLLM 1.x from 1.99.0. The project enables uv's centralized environments to avoid hidden `.pth` problems in synced macOS folders. Session-only imports do not load the provider SDK, keeping spawned lock workers independent of provider initialization.
