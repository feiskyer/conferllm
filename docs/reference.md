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

Replace placeholder credentials for the models you want to use and remove unused entries. The example includes the `gpt-4o` alias used throughout the README. Model IDs are examples, not a guarantee of provider availability. The example file is part of the source checkout, not installed beside the CLI.

Each `model_name` must be non-empty and unique. It is the local alias passed to `--model`, not necessarily the provider's model ID. `litellm_params.model` is required and identifies the provider model.

Parameters in `litellm_params` are forwarded to LiteLLM. ConferLLM owns `messages` and always sets `stream` to `false`; a configured `messages` field is rejected. Unknown application configuration fields are also rejected.

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
    "model": "gpt-4o",
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

The session ID above is illustrative; continue with the ID returned by your own call. Compatibility fields `session_id`, `name`, `model`, and `answer` are also returned. New integrations should use the nested fields. Raw provider data is excluded unless `--include-raw-response` is supplied.

Successful JSON results go to stdout. Runtime failures exit nonzero and write a `conferllm.error.v1` envelope to stderr. Library logging is suppressed in JSON mode so it does not contaminate the envelope. Argument-parser failures use the normal CLI usage error and exit code 2.

Without `--json`, output includes the session ID, name, answer, and output-image locations; warnings go to stderr. A name is derived locally from the first prompt. Use `--name` to choose one when creating a conversation.

## Session listing

```bash
conferllm sessions list
conferllm sessions list --query raft --json
conferllm sessions list --model gpt-4o --since 2026-09-01 --limit 20 --json
```

Filters combine with AND semantics. `--query` matches the ID or name without case sensitivity; model aliases match exactly. `--since` and `--until` are inclusive creation dates. The default limit is 50; `--limit 0` is unlimited. Listing searches metadata, not message bodies, and reports corrupt headers as warnings without hiding unrelated valid sessions.

`--json` returns a `conferllm.sessions.response.v1` object with `sessions` and `warnings` arrays.

For chat commands, exactly one of `--model` and `--session` is required. A session retains its model and name; `--name` is only accepted for a new chat. Prior successful turns are restored automatically. Compare models by creating independent sessions with the same prompt and attributing each answer.

## Images

Repeat `--image` to preserve input order. Accepted images are copied into private, session-owned storage, so follow-ups use those copies even if the originals change or disappear. PNG, JPEG, GIF, WebP, BMP, and SVG MIME types are detected from content rather than file extensions. Provider format support may be narrower.

Declared capabilities are checked before reading images or calling the provider. Application limits bound new attachments; a model's `max_input_images` also counts images replayed from history. Unknown image capability produces a warning, not a claim of provider support.

Models that return images can produce multiple ordered output artifacts, including image-only replies. `message.content` contains their references and `artifacts` contains their MIME type, size, SHA-256, local path, and stable `conferllm://sessions/.../artifacts/...` URI. Local paths are included in CLI responses, not exposed through MCP results.

Use `--image-output-dir PATH` for additional exported copies. If export fails, the canonical artifacts and successful conversation remain available. Keep the session ID and handle the warning rather than repeating the model call.

Generated images must be embedded data URLs. ConferLLM does not download remote-only image outputs, execute provider tool calls, or silently discard unsupported content.

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

Chat results include the JSON envelope as both structured content and the first text content block. MCP image inputs must be data URLs or absolute paths on the server's filesystem, not relative paths on the client's machine. Generated images may be returned inline or through `conferllm://sessions/{session_id}/artifacts/{artifact_id}` resources.

## Storage and reliability

Sessions are UTF-8 JSONL files below `~/.conferllm/sessions/YYYY/MM/DD/`. Every committed turn contains a complete user/assistant pair and its artifact metadata. Directories use `0700`; session and image files use `0600`. The configuration file remains user-managed.

Continuation holds a thread/process lock across loading, model execution, and atomic JSONL replacement. Stored image size/hash checks precede replay. Failed provider calls do not append a partial turn.

JSONL is the commit record. A locked continuation can recover the next uncommitted image directory left by a crash; it never removes committed turns. Early staging leftovers are not automatically swept. This is not an exactly-once guarantee for provider calls. Long histories are not silently summarized or truncated.

Session content can be sensitive. Keep credentials and conversations out of version control. Configuration/provider errors do not echo raw input or provider exception payloads.

## Troubleshooting

Start with diagnostics that do not print configuration values:

```bash
conferllm doctor --json
conferllm models --json
conferllm model-info gpt-4o
```

Doctor reports safe metadata and suggested next steps. It exits nonzero when configuration is missing, invalid, or has no model aliases, while still returning its diagnostic report.

- **Command not found:** for uv tool installs, run `uv tool update-shell` and reopen the terminal. MCP clients may need an absolute executable path.
- **Model not found:** use an alias from `conferllm models`, or add the alias to your configuration. A provider model ID is not automatically a local alias.
- **Configuration error:** check the path, YAML syntax, and field names locally. Keep aliases unique and supply `litellm_params.model` for each.
- **Provider error:** check your account's model access, endpoint, and credentials. A successful configuration check does not test provider connectivity.
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

Dependency major versions are bounded, and `uv.lock` records the development resolution. Python 3.10 uses LiteLLM 1.97.x; Python 3.11+ permits LiteLLM 1.x from 1.99.0. The project enables uv's centralized environments to avoid hidden `.pth` problems in synced macOS folders.
