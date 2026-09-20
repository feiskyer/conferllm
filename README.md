# ConferLLM

ConferLLM is a minimalist agent harness with multi-model support and only essential file and shell tools.

Run models with minimal prompts and tools to get the most out of their capabilities.

Use it as an Agent Skill or MCP server to help Claude Code, Codex, and other agents collaborate across models on complex tasks.

## Quick start

Requires Python 3.10+ and API access to a model.

### 1. Install

Install with [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended):

```bash
uv tool install conferllm
```

Alternatively, use pip in an activated Python virtual environment:

```bash
pip install conferllm
```

If your shell cannot find the uv-installed command, run `uv tool update-shell` and reopen the terminal. See the [installation reference](docs/reference.md#installation) for details.

### 2. Configure one model

Create the configuration directory:

```bash
mkdir -p ~/.conferllm
```

Create `~/.conferllm/config.yaml` with the following content and replace the example API key:

```yaml
model_list:
  - model_name: gpt-4o
    capabilities:
      input_modalities: [text, image]
      output_modalities: [text]
    litellm_params:
      model: openai/gpt-4o
      api_key: "replace-with-your-key"
```

`model_name` is the alias used in commands. For more models and endpoints, see [config_example.yaml](config_example.yaml).

OpenAI models use Responses by default. For older compatible endpoints, add `api_format: chat_completion` beside `model_name`.

### 3. Ask a question

```bash
conferllm chat --model gpt-4o --prompt "Explain Raft leader election."
```

The output includes an answer and a session ID. Keep the ID to ask follow-up questions with the conversation history restored.

## Common tasks

### Continue or find a conversation

Replace `SESSION_ID` with the ID returned by your chat:

```bash
conferllm chat --session SESSION_ID --prompt "Now compare it with Paxos."
conferllm sessions list
conferllm sessions list --query raft --json
```

A session keeps its original model. To compare models, start a separate chat for each configured alias using the same prompt. Use `--name "Raft notes"` when creating a chat to give it a memorable name.

### Use JSON or a prompt file

```bash
conferllm chat --model gpt-4o --prompt "Explain Raft leader election." --json
conferllm chat --model gpt-4o --prompt-file ./prompt.md --json
```

Write a long or multiline prompt into `prompt.md` before using `--prompt-file`. JSON output includes `session.id`, `message.text`, `artifacts`, and `warnings`. See the [response and error reference](docs/reference.md#json-output) for the full contract.

### Include images

With a model that supports images, repeat `--image` to attach your files in order:

```bash
conferllm chat \
  --model gpt-4o \
  --prompt "Compare these screenshots." \
  --image ./before.png \
  --image ./after.png \
  --json
```

Images are copied into the session so follow-ups can reuse them. Generated images are saved to `/tmp` by default; use `--image-output-dir ./output` to choose another directory. Responses replace image data with saved file paths in `message.content`, `message.text`, and output artifacts' `saved_path`. See [image handling](docs/reference.md#images).

### Work with local files and commands

No extra flag or per-model setting is needed:

```bash
conferllm chat \
  --model gpt-4o \
  --prompt "Read README.md and list the files in this directory." \
  --json
```

Relative paths use the command's working directory. PowerShell requires an installed `pwsh` or `powershell` executable. See [built-in tools](docs/reference.md#built-in-tools).

## Use from an agent

Install the bundled Skill into `~/.agents/skills/conferllm`:

```bash
conferllm skill install
```

For `~/.codex/skills/conferllm`, use `conferllm skill install --target codex`. Then ask your agent to use ConferLLM to delegate tasks to other models.

See [Skill installation details](docs/reference.md#agent-skill) for custom destinations and updates.

## Use as an MCP server

The stdio server starts with `conferllm serve`. For clients that use an `mcpServers` configuration, add:

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

The client launches the server; you do not need to start it separately. If the client cannot find the command, use the absolute path from `command -v conferllm`. See the [MCP reference](docs/reference.md#mcp-server) for tools and transports.

## Configuration and troubleshooting

Check configuration and available models:

```bash
conferllm doctor --json
conferllm models --json
conferllm model-info gpt-4o
```

Use `--config PATH` with a command to select another configuration file. If a model is not found, use an alias listed by `conferllm models`. See [configuration options](docs/reference.md#configuration) and [troubleshooting](docs/reference.md#troubleshooting).

## Development and contributing

Report bugs in [GitHub Issues](https://github.com/feiskyer/mcp-ai-hub/issues); pull requests are welcome too. Include reproduction steps and relevant output. See the [source setup and development checks](docs/reference.md#development).

To keep an installed CLI linked to this checkout while editing:

```bash
uv tool install --editable . --force
```

This replaces an existing ConferLLM tool install. Python source edits take effect on the next invocation; restart any running server after edits. Reinstall when dependencies or command entry points change.

## License

ConferLLM is released under the [MIT License](LICENSE).
