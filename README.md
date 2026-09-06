# ConferLLM

ConferLLM is a CLI for people and agents to consult AI models, continue conversations, and work with images. Use it to get a second opinion, compare answers across models, or call a model from a script with structured JSON output.

It connects to providers through LiteLLM and stores conversations locally. A bundled Agent Skill and an MCP server expose the same conversation features.

## Quick start

Requires Python 3.10+. Bring credentials for the provider you want to use.

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

Create a private configuration directory:

```bash
mkdir -p ~/.conferllm
chmod 700 ~/.conferllm
```

Create `~/.conferllm/config.yaml` with the following content and replace the example API key. If you already have a configuration, add the model without duplicating an existing alias or overwriting your settings.

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

```bash
chmod 600 ~/.conferllm/config.yaml
```

`model_name` is the alias used in commands; substitute your own if it differs. This example uses OpenAI, and model availability depends on your account. For other providers and local endpoints, see [config_example.yaml](config_example.yaml).

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

Images are copied into the session so follow-ups can reuse them. For models that return images, `--image-output-dir ./output` exports additional copies. See [image support and limits](docs/reference.md#images).

## Use from an agent

Install the bundled Skill into `~/.agents/skills/conferllm`:

```bash
conferllm skill install
```

For `~/.codex/skills/conferllm`, use `conferllm skill install --target codex`. Then ask your agent to use the ConferLLM Skill for a second opinion or model comparison. The Skill discovers configured aliases and uses the CLI without opening credential or session files.

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

Inspect configuration without printing credentials:

```bash
conferllm doctor --json
conferllm models --json
conferllm model-info gpt-4o
```

Use `--config PATH` with a command to select another configuration file. If a model is not found, use an alias listed by `conferllm models`. See [configuration options](docs/reference.md#configuration) and [troubleshooting](docs/reference.md#troubleshooting).

## Privacy and limits

Prompts, history, and images go to the provider endpoint you configure; a local CLI does not imply offline inference. Sessions and image copies remain under `~/.conferllm/sessions/` by default. Keep them and your credentials out of version control. Stored directories use `0700`; session and image files use `0600`.

Responses are non-streaming. ConferLLM does not execute provider tool calls, download remote-only image outputs, or automatically shorten long histories. See [storage and reliability](docs/reference.md#storage-and-reliability).

## Development and contributing

Report bugs in [GitHub Issues](https://github.com/feiskyer/mcp-ai-hub/issues); pull requests are welcome too. Include reproduction steps and sanitized output, never credentials or private conversations. See the [source setup and development checks](docs/reference.md#development).

To keep an installed CLI linked to this checkout while editing:

```bash
uv tool install --editable . --force
```

This replaces an existing ConferLLM tool install. Python source edits take effect on the next invocation; restart any running server after edits. Reinstall when dependencies or command entry points change.

## License

ConferLLM is released under the [MIT License](LICENSE).
