# ConferLLM

ConferLLM is a local command-line tool for asking configured AI models questions,
continuing conversations, and working with images. Its bundled Agent Skill lets
another agent use the same commands for a second opinion or a model comparison.

The Python package, import package, executable, and Skill are all named
`conferllm`. LiteLLM handles provider integration; callers select a local model
alias and keep the returned session ID.

## Install

From an authorized ConferLLM source checkout:

```bash
uv tool install .
```

If you already use a Python virtual environment:

```bash
python -m pip install .
```

A supplied ConferLLM wheel can be used instead of `.`. Once a release of this
project has been published and its identity verified, it can be installed with
`uv tool install conferllm`. Do not assume that a same-named package is this project.

Verify the executable:

```bash
conferllm --version
conferllm --help
```

ConferLLM requires Python 3.10+ and POSIX file locking (macOS/Linux). Native Windows
is not supported.

## Configure models

Configuration belongs in `~/.conferllm/config.yaml`, outside the repository.
From a source checkout, create a private directory and copy the example only
when no configuration exists:

```bash
mkdir -p ~/.conferllm
chmod 700 ~/.conferllm
test ! -e ~/.conferllm/config.yaml && cp config_example.yaml ~/.conferllm/config.yaml
chmod 600 ~/.conferllm/config.yaml
```

Alternatively, create the file using this minimal configuration and supply
your own credentials:

```yaml
model_list:
  - model_name: reasoning
    capabilities:
      input_modalities: [text]
      output_modalities: [text]
    litellm_params:
      model: openai/gpt-5
      api_key: "replace-with-your-key"

  - model_name: vision
    capabilities:
      input_modalities: [text, image]
      output_modalities: [text]
      max_input_images: 8
    litellm_params:
      model: openai/gpt-4o
      api_key: "replace-with-your-key"
```

Provider model IDs and parameters are examples; availability depends on your
provider account. Each local alias must be non-empty and unique, and
`litellm_params.model` must identify a provider model. See
[config_example.yaml](config_example.yaml) for additional configuration shapes.

Values in `litellm_params` are forwarded to LiteLLM, except that ConferLLM owns
`messages` and always sets `stream` to `false`. Unknown application configuration
fields are rejected so typos do not silently change behavior.

Optional settings:

- `global_system_prompt`: applied to every request.
- A model's `system_prompt`: overrides the global prompt; `""` disables it.
- `sessions_dir`: overrides the session root. Relative YAML paths resolve
  beside the configuration file, independently of the caller's working directory.
- `image_limits`: limits newly attached images per turn. Defaults are 16 images,
  20 MiB per image, and 50 MiB in total.

Use `--config PATH` to select another file. An explicitly selected missing file
is an error. ConferLLM does not create, populate, or automatically migrate provider
credentials.

Check configuration without printing its values:

```bash
conferllm doctor --json
conferllm models --json
conferllm model-info reasoning
```

Doctor reports safe metadata and suggested next steps. It exits nonzero when
configuration is missing, invalid, or has no model aliases, while still
returning its diagnostic report.

## Ask a model

Create a conversation:

```bash
conferllm chat --model reasoning --prompt "Explain Raft leader election."
```

Human-readable output includes the session ID, name, answer, output-image
locations, and any warnings. A name is derived locally from the first prompt;
use `--name` to choose one explicitly.

For scripts and agents, request structured output:

```bash
conferllm chat \
  --model reasoning \
  --name "Raft notes" \
  --prompt "Explain Raft leader election." \
  --json
```

The stable response envelope is `conferllm.chat.response.v1`. Its authoritative
fields are:

```json
{
  "schema_version": "conferllm.chat.response.v1",
  "ok": true,
  "session": {
    "id": "20260905-0123456789abcdef0123456789abcdef",
    "name": "Raft notes",
    "model": "reasoning",
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

Compatibility fields `session_id`, `name`, `model`, and `answer` are also
returned. New integrations should use the nested fields. Raw provider data
is excluded unless `--include-raw-response` is supplied.

For long or multiline prompts:

```bash
conferllm chat --model reasoning --prompt-file ./prompt.md --json
```

Successful JSON results are written to stdout. Runtime failures exit nonzero
and write a `conferllm.error.v1` envelope to stderr; library logging is suppressed
in JSON mode so the envelope remains parseable. Argument-parser failures use
the normal CLI usage error and exit code 2.

## Continue and find conversations

Use the actual session ID returned by the preceding call:

```bash
conferllm chat \
  --session 20260905-0123456789abcdef0123456789abcdef \
  --prompt "Now compare it with Paxos." \
  --json
```

Exactly one of `--model` and `--session` is required. A session retains its
model and name; `--name` is only accepted when creating a conversation.
ConferLLM restores prior successful turns automatically.

Find conversations using metadata:

```bash
conferllm sessions list
conferllm sessions list --query raft --json
conferllm sessions list --model reasoning --since 2026-09-01 --limit 20 --json
```

Filters combine with AND semantics. `--query` matches the ID or name without
case sensitivity; model aliases match exactly. `--since` and `--until` are
inclusive creation dates. The default limit is 50; `--limit 0` is unlimited.
Listing never searches message bodies and reports corrupt headers as warnings
without hiding unrelated valid sessions.

For a model comparison, create one independent session per model using the
same prompt. Attribute each answer and retain each session ID; a conversation
does not switch models.

## Work with images

Repeat `--image` to preserve input order:

```bash
conferllm chat \
  --model vision \
  --prompt "Compare these screenshots." \
  --image ./before.png \
  --image ./after.png \
  --json
```

Accepted images are copied into private, session-owned storage. Follow-ups
use those copies even if the original files change or disappear. PNG, JPEG,
GIF, WebP, BMP, and SVG MIME types are detected from content rather than file
extensions.

Declared capabilities are checked before reading images or calling the
provider. Application limits bound new attachments; a model's
`max_input_images` also counts images replayed from history. Unknown image
capability produces a warning, not a claim of provider support.

Models that return images can produce multiple ordered output artifacts,
including image-only replies. `message.content` contains their references and
`artifacts` contains their MIME type, size, SHA-256, local path, and stable
`conferllm://sessions/.../artifacts/...` URI.

Use `--image-output-dir PATH` for additional exported copies. If export fails,
the canonical artifacts and successful conversation remain available. Preserve
the session ID and handle the warning instead of repeating the model call.

Generated images must be embedded data URLs. ConferLLM does not download
remote-only image outputs, execute provider tool calls, or silently discard
unsupported content.

## Install the Agent Skill

The source layout is:

```text
skills/conferllm/
├── SKILL.md
├── agents/openai.yaml
└── references/
    ├── installation.md
    ├── multimodal.md
    └── errors.md
```

[skills/conferllm/](skills/conferllm/) is the single source of truth for the Skill.
The wheel embeds the same files; the source distribution preserves this layout.
Source and installed Skills use identical relative reference paths.

Install the version bundled with the executable:

```bash
conferllm skill install
conferllm skill install --target codex
conferllm skill install --destination /custom/skills/conferllm
```

Defaults are `~/.agents/skills/conferllm` and `~/.codex/skills/conferllm`. Installation
is idempotent. A different existing installation is preserved unless `--force`
is explicitly requested. Home, workspace, package directories, and their
ancestors are never valid replacement targets.

An agent invoking `$conferllm` discovers configured aliases, retains session IDs,
preserves image order, and attributes answers. It uses the CLI rather than
opening credential or session files.

## Storage and reliability

Sessions are UTF-8 JSONL files below `~/.conferllm/sessions/YYYY/MM/DD/`. Every
committed turn contains a complete user/assistant pair and its artifact
metadata. Directories use mode `0700`; session and image files use `0600`.
The configuration file remains user-managed.

Continuation holds a thread/process lock across loading, model execution, and
atomic JSONL replacement. Stored image size/hash checks precede replay.
Failed provider calls do not append a partial turn.

JSONL is the commit record. A locked continuation can recover the next
uncommitted image directory left by a crash; it never removes committed turns.
Early staging leftovers are not automatically swept. This is not an
exactly-once guarantee for provider calls, and long histories are not silently
summarized or truncated.

Session content can be sensitive. Keep credentials and conversations out of
version control. Configuration/provider errors do not echo raw input or
provider exception payloads.

## Development

Use `uv>=0.12.9,<0.13`:

```bash
uv sync --extra dev
uv run ruff format --check .
uv run ruff check .
uv run mypy src/
uv run pytest
uv lock --check
uv build
git diff --check
```

Tests isolate the user home and block unmocked provider calls. They cover
persistence, process/thread locking, fault recovery, protocol integration,
Skill installation, and wheel/source-distribution content.

Key dependency major versions are bounded, and `uv.lock` records the development
resolution. Python 3.10 uses the verified LiteLLM 1.97.x line; Python 3.11+
permits LiteLLM 1.x from 1.99.0. The project uses uv's centralized environment
support so editable installs do not depend on hidden `.pth` files in synced
macOS folders.

## License

ConferLLM is released under the [MIT License](LICENSE).
