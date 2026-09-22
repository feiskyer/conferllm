# ConferLLM

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MCP Ready](https://img.shields.io/badge/MCP-2.x-green.svg)](https://modelcontextprotocol.io/)
[![uv](https://img.shields.io/badge/installed%20with-uv-purple.svg)](https://docs.astral.sh/uv/)

ConferLLM is a minimalist agent harness with multi-model support and essential host tools. It enables AI agents (such as Claude Code, Codex, Cursor, and Cline) and human developers to consult, delegate, and collaborate across models on complex tasks with full session state and local execution capabilities.

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│  AI Agent / Developer (Claude Code, Codex, Cursor, CLI, Script)             │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │ CLI (--json) or MCP (stdio/sse/http)
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ConferLLM Engine                                                            │
│  • Stateful Session Store (atomic JSONL + fcntl locking)                     │
│  • Self-Correcting Tool Runtime (up to 1,000 tool rounds per turn)           │
│  • Multimodal Pipeline (artifact tracking + auto-saving)                     │
└──────────────┬───────────────────────────────────────────────┬───────────────┘
               │ Native Tool Calls                             │ LiteLLM /
               ▼                                               │ Responses API
┌──────────────────────────────┐                ┌──────────────▼───────────────┐
│ Host Tools (Unrestricted)    │                │ Configured AI Models         │
│ • Shell: bash, pwsh, git     │                │ • OpenAI GPT-6 Astra         │
│ • Background: output, kill   │                │ • Claude Opus/Sonnet         │
│ • Files: read, write, edit,  │                │ • DeepSeek Flash / Pro       │
│   append, list_directory     │                │ • Google Gemini              │
│                              │                │ • Local Ollama               │
└──────────────────────────────┘                └──────────────────────────────┘
```

---

## Highlights

- **Multi-Model Delegation:** Connect to 100+ providers via LiteLLM (OpenAI, Anthropic, Gemini, DeepSeek, local Ollama, Azure, Bedrock, etc.) using clean local aliases.
- **10 Native Built-in Host Tools:** Delegated models get access to shell (`run_shell`, `run_powershell`, `shell_output`, `shell_kill`, `git_command`) and filesystem tools (`read_file`, `write_file`, `append_file`, `edit_file`, `list_directory`) without extra flags.
- **Self-Correcting Tool Loop:** Tool execution errors are returned as native results so the model can inspect errors and self-correct across up to 1,000 rounds in a single turn.
- **Persistent Multi-Turn Sessions:** Conversations are stored as atomic JSONL files with process-safe locking. Resume any session at any time with `--session <id>`.
- **First-Class Multimodal Support:** Send input images (`--image`); model-generated images are automatically saved to disk, linked via URIs, and safely replayed in history.
- **Agent-Ready Surfaces:** Dual interface out of the box—bundled **Agent Skill** (`conferllm skill install`) for Claude Code/Codex and standard **MCP Server** (`conferllm serve`) for Claude Desktop/Cursor/Cline.

---

## Quick Start

Requires Python 3.10+ on macOS or Linux (POSIX file locking required; native Windows is not supported).

### 1. Install

Install with [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended):

```bash
uv tool install conferllm
```

Or install with pip in an activated virtual environment:

```bash
pip install conferllm
```

Verify the installation:

```bash
conferllm --version
conferllm doctor --json
```

*Note:* If the command is not found after `uv tool install`, run `uv tool update-shell` and restart your terminal.

### 2. Configure Models

Create the configuration directory and file:

```bash
mkdir -p ~/.conferllm
chmod 700 ~/.conferllm
touch ~/.conferllm/config.yaml
chmod 600 ~/.conferllm/config.yaml
```

Populate `~/.conferllm/config.yaml` with your preferred providers. For example:

```yaml
model_list:
  - model_name: gpt-6-astra
    api_format: responses
    capabilities:
      input_modalities: [text, image]
      output_modalities: [text]
    litellm_params:
      model: openai/gpt-6-astra
      api_key: "your-openai-key"

  - model_name: claude-opus-5
    capabilities:
      input_modalities: [text, image]
      output_modalities: [text]
    litellm_params:
      model: anthropic/claude-opus-5
      api_key: "your-anthropic-key"

  - model_name: DeepSeek-V4.1-Flash
    capabilities:
      input_modalities: [text, image]
      output_modalities: [text]
    litellm_params:
      model: deepseek/deepseek-flash
      api_key: "your-deepseek-key"

  - model_name: gemini-3.8-flash
    capabilities:
      input_modalities: [text, image]
      output_modalities: [text]
    litellm_params:
      model: gemini/gemini-3.8-flash
      api_key: "your-gemini-key"

  - model_name: local-qwen3
    litellm_params:
      model: ollama_chat/qwen3-coder:30b
      api_base: "http://localhost:11434"
```

*Tip:* See [config_example.yaml](config_example.yaml) for more provider templates including Gemini 3.8 Flash, Qwen 3 Coder, Mistral Large, Together AI, Hugging Face, Azure, and AWS Bedrock.

### 3. Run Your First Chat

```bash
conferllm chat --model gpt-6-astra --prompt "Explain Raft leader election in two sentences."
```

Output includes the answer and a reusable `Session` ID:

```text
Session: 20260905-0123456789abcdef0123456789abcdef
Name: Explain Raft leader election in two sentences.

Raft leader election ensures a cluster chooses a single leader through randomized election timeouts and majority voting. Nodes transition to candidates, request votes, and become leaders upon securing a majority.
```

---

## AI Agent Integration

ConferLLM is designed from the ground up for agent consumption. AI agents use ConferLLM to obtain attributed second opinions, delegate specialized tasks (e.g., deep math proofs to DeepSeek V4.1 Flash, visual analysis to GPT-6 Astra, or private offline tasks to local Llama 4), and let external models interact with the host repository.

### Option A: Use as an Agent Skill (Claude Code & Codex)

Install the bundled skill into your agent's skill directory:

```bash
# For Claude Code and general agents (~/.agents/skills/conferllm):
conferllm skill install

# For Codex (~/.codex/skills/conferllm):
conferllm skill install --target codex
```

Once installed, your agent automatically knows how to discover configured models, run queries with `--json`, parse outputs, and continue conversations.

**Example agent instruction:**
> *"Use $conferllm to ask DeepSeek-V4.1-Flash to review our Raft consensus implementation in src/consensus.py and check for election split-vote edge cases."*

### Option B: Use as an MCP Server (Claude Desktop, Cursor, Cline, Windsurf)

ConferLLM includes a full-featured MCP (Model Context Protocol) server over stdio, SSE, or streamable HTTP.

Add to your MCP configuration (e.g. `claude_desktop_config.json` or Cursor Settings):

```json
{
  "mcpServers": {
    "conferllm": {
      "command": "conferllm",
      "args": ["serve"]
    }
  }
}
```

#### MCP Tools Provided

| MCP Tool | Description |
| --- | --- |
| `create_chat(message, model, name=None, images=None, ...)` | Start a new conversation with host shell and file tools enabled. |
| `continue_chat(message, session_id, images=None, ...)` | Continue an existing session with conversation history restored. |
| `list_sessions(query=None, model=None, since=None, until=None, limit=50)` | Search past sessions by ID, name, model, or creation date. |
| `list_models()` | Return all configured model aliases. |
| `get_model_info(model)` | Return declared capabilities, modalities, and format details. |

Generated images are accessible via the MCP resource `conferllm://sessions/{session_id}/artifacts/{artifact_id}`.

---

## Common Workflows

### 1. Multi-Turn Conversation (Follow-Up)

Pass `--session` with the returned session ID to continue with full context:

```bash
conferllm chat --session 20260905-0123456789abcdef0123456789abcdef --prompt "Now compare it with Paxos."
```

*Note:* A session retains its original model alias. History and past tool calls are restored automatically without re-executing old commands.

### 2. Machine-Readable JSON Mode

AI agents and scripts should always pass `--json` for predictable parsing:

```bash
conferllm chat --model gpt-6-astra --prompt "Explain Raft." --json
```

**JSON Response Contract (`conferllm.chat.response.v1`):**

```json
{
  "schema_version": "conferllm.chat.response.v1",
  "ok": true,
  "session": {
    "id": "20260905-0123456789abcdef0123456789abcdef",
    "name": "Explain Raft.",
    "model": "gpt-6-astra",
    "turn": 1
  },
  "message": {
    "text": "Raft is a consensus algorithm...",
    "content": [
      {
        "type": "text",
        "text": "Raft is a consensus algorithm..."
      }
    ]
  },
  "artifacts": [],
  "warnings": []
}
```

Key fields:
- `session.id`: Unique session identifier to pass to `--session` on follow-up.
- `session.turn`: Turn counter (starts at 1).
- `message.text`: The complete assistant answer (with image payloads replaced by saved paths).
- `artifacts`: List of turn artifacts (images input/output, paths, mime types, hashes).
- `warnings`: Non-fatal notices (e.g. background processes stopped at turn end).

### 3. Long Prompts from Files

For complex multi-line prompts, markdown instructions, or code snippets:

```bash
conferllm chat --model DeepSeek-V4.1-Flash --prompt-file ./review_prompt.md --json
```

### 4. Multimodal Analysis (Images)

Attach one or more images using repeated `--image` flags:

```bash
conferllm chat \
  --model gpt-6-astra \
  --prompt "Analyze the architectural bottleneck shown in this diagram." \
  --image ./architecture.png \
  --json
```

- Images are validated against model capabilities and copied into session-owned storage for safe replay across turns.
- Model-generated images are automatically written to `--image-output-dir` (defaults to `/tmp`) and listed in `artifacts` with `direction: "output"`.

### 5. Local File and Shell Operations

Delegated models have native access to host files and commands without extra flags:

```bash
conferllm chat \
  --model gpt-6-astra \
  --prompt "Read pyproject.toml and tell me what dependencies need attention." \
  --json
```

*Safety note:* If you want an opinion or review only without risking file modifications, include in your prompt: *"Provide an analysis only; do not edit files or execute modifying commands."*

### 6. Search and Inspect Past Sessions

```bash
# List recent sessions
conferllm sessions list

# Search by keyword in name or ID
conferllm sessions list --query raft --json

# Filter by model alias and date range
conferllm sessions list --model gpt-6-astra --since 2026-09-01 --limit 10 --json
```

### 7. Compare Models Side-by-Side

To compare how different models handle the same challenge, start independent chats:

```bash
conferllm chat --model DeepSeek-V4.1-Flash --prompt-file ./challenge.md --json
conferllm chat --model claude-opus-5 --prompt-file ./challenge.md --json
```

---

## Built-in Host Tools

ConferLLM provides every configured model with 10 native tools. Native tool calls execute on the host machine with the user's OS permissions:

| Tool | Purpose | Key Parameters |
| --- | --- | --- |
| `run_shell` | Run bash/sh commands | `command`, `timeout` (1–600s, def 120s), `cwd`, `run_in_background` |
| `run_powershell` | Run PowerShell commands | `command`, `timeout`, `cwd`, `run_in_background` |
| `shell_output` | Read unread output from a background command | `shell_id`, `filter_str` (optional regex) |
| `shell_kill` | Terminate an owned background process group | `shell_id` |
| `git_command` | Execute git commands | `command`, `timeout` (1–600s, def 60s), `cwd` |
| `read_file` | Read UTF-8 file with 1-indexed line numbers | `path`, `offset` (def 1), `limit` (def 2000 lines) |
| `write_file` | Create or atomically overwrite a file | `path`, `content` (up to 10 MiB) |
| `append_file` | Append content to a file | `path`, `content` |
| `edit_file` | Exact unique string replacement | `path`, `old_string`, `new_string`, `replace_all` (def false) |
| `list_directory` | Directory listing with glob exclusions | `path` (def `.`), `ignore` (patterns), `limit` (def 1000) |

### Tool Execution Rules
- **No confirmation gates:** Operations execute immediately to enable seamless multi-step autonomous problem solving.
- **Self-correction:** Missing files, invalid arguments, and command errors return as structured tool results so the model can rectify errors itself.
- **Turn boundaries:** Background processes belong to the turn and are cleanly stopped at turn completion. Up to 1,000 tool rounds are permitted per turn.

---

## Diagnostics & Troubleshooting

Run doctor to inspect system status, configuration, models, and permissions without exposing credentials:

```bash
conferllm doctor --json
```

List available model aliases:

```bash
conferllm models --json
```

Inspect non-secret details of a specific model alias:

```bash
conferllm model-info gpt-6-astra
```

### Common Issues

- **`command not found: conferllm`:** Run `uv tool update-shell` and restart terminal, or check your virtualenv `PATH`.
- **`model_not_found`:** Run `conferllm models` to see available aliases. Model aliases in `config.yaml` are the identifiers used on the CLI, not raw provider names.
- **`configuration_error`:** Verify YAML syntax in `~/.conferllm/config.yaml`. Permissions should be `0700` for directory and `0600` for the configuration file.
- **Provider Responses API vs Chat Completions:** OpenAI models use Responses by default. For legacy OpenAI-compatible endpoints that only support `/chat/completions`, add `api_format: chat_completion` to the model config. GPT-6 Astra tool calling requires Responses.

---

## Development

```bash
# Clone repository
git clone https://github.com/feiskyer/mcp-ai-hub.git conferllm
cd conferllm

# Setup development environment
uv sync --extra dev

# Run quality checks & test suite
uv run ruff format --check .
uv run ruff check .
uv run mypy src/
uv run pytest

# Build packages
uv build
```

To link an editable checkout globally to your tool environment:

```bash
uv tool install --editable . --force
```

---

## License

ConferLLM is open source software released under the [MIT License](LICENSE).
