---
name: conferllm
description: Query locally configured AI models through ConferLLM. Use for another model's answer, specialized reasoning, multimodal visual analysis, persistent follow-up, or attributed cross-model comparisons.
---

# ConferLLM Agent Skill

Use the `conferllm` CLI to consult the user's configured models (e.g., DeepSeek R1, GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, local Ollama models). The CLI executes locally, dispatching requests to configured provider endpoints with persistent session state and native host tools.

## Critical Safety & Operational Rules

1. **Native host tools are active on every chat:** Every delegated model receives 10 native tools (`run_shell`, `run_powershell`, `shell_output`, `shell_kill`, `git_command`, `read_file`, `write_file`, `append_file`, `edit_file`, `list_directory`). Tools run on the host with the OS user's permissions without an approval prompt or sandbox.
2. **Read-only vs. modifying intent:** If the user wants a review, second opinion, or analysis only, explicitly instruct the model in your prompt: *"Provide an analysis/opinion only; do not edit files or run modifying commands."*
3. **Never inspect or leak secrets:** Never read, echo, migrate, or dump `~/.conferllm/config.yaml` or credentials. Use `conferllm doctor --json` and `conferllm models --json` for safe inspection. Never inspect raw JSONL session files.
4. **Attribute responses:** Always attribute answers to the specific model alias queried when reporting findings back to the user.

---

## 1. Verify Availability (Preflight)

Check whether ConferLLM is installed:

```bash
command -v conferllm
```

If missing, consult [installation reference](references/installation.md). If installed, verify readiness safely:

```bash
conferllm doctor --json
```

Doctor returns non-secret diagnostics (`ok`, `model_aliases`, `permissions`, `next_steps`). If `ok` is false, review suggested next steps.

---

## 2. Model Discovery

List configured model aliases:

```bash
conferllm models --json
```

Inspect specific capabilities (modalities, limits, provider format):

```bash
conferllm model-info MODEL
```

*Rule:* Use only configured aliases returned by `conferllm models`. Never invent or silently substitute an alias.

---

## 3. Starting a New Conversation

Run a new chat and parse machine-readable JSON:

```bash
conferllm chat --model MODEL --prompt "YOUR PROMPT HERE" --json
```

For long or multiline prompts, write to a UTF-8 file first and use `--prompt-file`:

```bash
conferllm chat --model MODEL --prompt-file ./prompt.md --json
```

Attach local images by repeating `--image`:

```bash
conferllm chat --model MODEL --prompt "Explain this diagram" --image ./diag.png --json
```

Save generated images to a specific directory using `--image-output-dir ./output`.

### Parsing the JSON Response (`conferllm.chat.response.v1`)

```json
{
  "schema_version": "conferllm.chat.response.v1",
  "ok": true,
  "session": {
    "id": "20260905-0123456789abcdef0123456789abcdef",
    "name": "Prompt title",
    "model": "gpt-4o",
    "turn": 1
  },
  "message": {
    "text": "Answer text here...",
    "content": [{"type": "text", "text": "Answer text here..."}]
  },
  "artifacts": [
    {
      "id": "artifact-id",
      "direction": "output",
      "mime_type": "image/png",
      "saved_path": "/tmp/output.png",
      "uri": "conferllm://sessions/.../artifacts/..."
    }
  ],
  "warnings": []
}
```

- **Answer text:** `response["message"]["text"]`
- **Session ID:** `response["session"]["id"]` (save this for follow-up turns!)
- **Output images:** `[a["saved_path"] for a in response["artifacts"] if a.get("direction") == "output"]`

---

## 4. Continuing a Conversation

Continue a stored session by passing `--session` with the saved session ID:

```bash
conferllm chat --session SESSION_ID --prompt "FOLLOW-UP PROMPT" --json
```

*Rules for continuation:*
- Exactly one of `--model` and `--session` is required.
- Do NOT provide `--model` when continuing; the session permanently retains its original model alias.
- Prior conversation history and tool executions are automatically restored without re-running past tools.
- Never attempt to manually concatenate prior history into the prompt.

---

## 5. Finding Past Sessions

Find prior conversations through metadata queries:

```bash
conferllm sessions list --query QUERY --json
conferllm sessions list --model MODEL --limit 20 --json
```

`--query` matches session IDs and names (case-insensitive). Filter optionally with `--since YYYY-MM-DD` and `--until YYYY-MM-DD`.

---

## 6. Comparing Models

To compare models on the same task, run independent sessions with each model alias and compare the attributed results:

```bash
conferllm chat --model MODEL_A --prompt-file ./task.md --json
conferllm chat --model MODEL_B --prompt-file ./task.md --json
```

Report both answers clearly attributed to their respective models. Read [multimodal reference](references/multimodal.md) for image comparisons.

---

## 7. Error Handling & Diagnostics

- **Exit code 0:** Success. Check `warnings` array for non-fatal issues (e.g. background process stopped).
- **Exit code 1:** Handled failure. Stderr contains a `conferllm.error.v1` envelope with `error.code` and `error.message`.
- **Exit code 2:** CLI usage or argument error.

Consult [errors reference](references/errors.md) for full error codes and recovery boundaries.

## 8. MCP Alternative

When using ConferLLM via MCP (`conferllm serve`), call:
- `create_chat(message, model, ...)` to start a chat.
- `continue_chat(message, session_id, ...)` to continue.
- `list_sessions(...)` to search sessions.
- `list_models()` and `get_model_info(model)` for discovery.
