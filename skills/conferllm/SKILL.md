---
name: conferllm
description: Consult the user's locally configured AI models through the ConferLLM CLI. Use when the user asks for another model's answer or review, a cross-model comparison, delegation to a specific model, or a follow-up in an existing ConferLLM session.
---

# ConferLLM Agent Skill

The `conferllm` CLI sends a prompt to one of the user's configured model aliases, stores the conversation as a session, and returns JSON. Every delegated model can call 10 native host tools (`run_shell`, `run_powershell`, `shell_output`, `shell_kill`, `git_command`, `read_file`, `write_file`, `append_file`, `edit_file`, `list_directory`). They run with the OS user's permissions, without approval prompts or a sandbox, so the delegated prompt is the only thing that sets the model's scope.

Use `conferllm doctor --json` and `conferllm models --json` to learn about the setup. Credentials live in `~/.conferllm/config.yaml` and session contents live in `~/.conferllm/sessions/`; do not read, print, or copy either. Attribute each answer to the model alias that produced it.

## Commands

```bash
conferllm models --json                     # configured aliases; use only these
conferllm model-info MODEL                  # modalities, image limits, API format
conferllm chat --model MODEL --prompt-file ./task.md --json
conferllm chat --session SESSION_ID --prompt "Follow-up" --json
conferllm sessions list --query TEXT --json # also --model, --since/--until YYYY-MM-DD, --limit
```

- Use `--prompt TEXT` for short prompts and `--prompt-file` for long or multiline ones.
- Attach images with repeated `--image PATH`; generated images go to `--image-output-dir` (default `/tmp`).
- A continued session keeps its original model, so pass `--session` without `--model`. Stored history, including past tool calls, is replayed automatically; do not paste earlier turns into the prompt.
- To compare models, run the same prompt file in one new session per alias.

In the JSON result (`conferllm.chat.response.v1`), read the answer from `message.text`, keep `session.id` for follow-ups, and read generated image paths from `artifacts[].saved_path` where `direction` is `"output"`. Check `warnings` even when the call succeeds.

## Writing the delegated prompt

The delegated model sees only your prompt and its session history, and you see only its final message. It cannot ask you questions mid-turn. Write the prompt the way you would brief a capable colleague who has no other context:

- **Goal and context:** what you need and why, plus the paths, constraints, and findings it would otherwise have to rediscover.
- **Scope and permissions:** say what it may change. For a review or opinion, write something like "Read whatever you need, but do not edit files or run commands that change state." For an implementation, say what it may modify and what it may run without asking, for example "The tests use temporary fixtures; run them, fix failures caused by this change, and rerun them."
- **Done criteria:** what finished looks like, such as tests passing or a specific question answered, and where exploration should stop.
- **Response shape:** what the final message should contain, for example findings with file:line references, a diff summary, or a verdict with reasons.

Treat the model's final message as a claim to verify, especially when it reports file changes or passing tests.

## When something goes wrong

- Exit code 0 means success; 1 means a handled failure, with a `conferllm.error.v1` envelope on stderr (`error.code`, `error.message`); 2 means a CLI usage error. See [errors reference](references/errors.md) for codes, provider/API-format failures, and when retrying is unsafe. A failure after tool calls may have left side effects, so do not blindly retry it.
- If `conferllm` is missing or doctor reports `ok: false`, follow [installation reference](references/installation.md).
- For image inputs, limits, and generated images, see [multimodal reference](references/multimodal.md).

## MCP

With `conferllm serve`, the same operations are available as MCP tools: `create_chat(message, model, ...)`, `continue_chat(message, session_id, ...)`, `list_sessions(...)`, `list_models()`, and `get_model_info(model)`.
