---
name: conferllm
description: Consult configured models via ConferLLM for a requested second opinion, model comparison, named-model task, or session follow-up. Not for ordinary coding/review or Codex setup.
---

# ConferLLM Agent Skill

The `conferllm` CLI sends a prompt to a configured model alias and stores the conversation. Delegated models can read/write files and run shell, Git, and PowerShell commands on the host with the OS user's permissions. There is no approval gate or sandbox; prompt restrictions are guidance, not enforced access control.

Preserve the user's chosen model, session, task, and authorized scope. Do not read, print, or copy real configuration or private session files, including custom paths. Default locations are `~/.conferllm/config.yaml` and `~/.conferllm/sessions/`.

## Choose the call

```bash
conferllm models --json
conferllm model-info MODEL
conferllm chat --model MODEL --prompt-file ./task.md --json
conferllm chat --session SESSION_ID --prompt "Follow-up" --json
conferllm sessions list --query TEXT --json
```

- Use a known alias or returned session ID directly. Discover aliases with `models --json` when needed; inspect `model-info` for relevant capabilities or limits. If the requested model is unavailable, report alternatives without silently switching.
- Continue with `--session` and no `--model`; stored history, including tool results, is replayed automatically. Use filtered `sessions list` only when the requested session ID needs locating.
- Use `--prompt TEXT` for short prompts, `--prompt-file` for longer ones, and `--json` for machine-readable results. Pass any selected `--config PATH` consistently.
- For comparisons, use the same task in independent sessions and attribute each actual answer to its alias. Keep shared inputs read-only or isolate authorized edits so one model does not change another's inputs.

## Brief the delegated model

The parent agent's conversation and instructions are not passed automatically, and the caller cannot answer questions mid-turn. Supply the outcome, necessary context and paths, permitted actions, and observable completion criteria. Specify a response format only when the task needs one.

- For a review, name the files or questions to inspect and keep the work read-only; analysis does not authorize implementation.
- For implementation, state the allowed write scope, constraints to preserve, and relevant checks. Allow in-scope fixes and checks to finish; if missing authority or a material decision blocks progress, require a concrete handoff instead of expanded scope.
- Include information the model cannot infer; avoid generic role descriptions or a fixed sequence of tools when the result does not depend on that sequence.

## Read the result

In `conferllm.chat.response.v1`, read `message.text`, retain `session.id` for follow-ups, and check `warnings` even on success. Attribute the answer to its model alias and distinguish the model's claims from independently verified changes or checks.

A successful chat is committed even with export warnings. Do not repeat it to recover an artifact. After a timeout or failed call, check the known outcome and possible side effects before retrying; file changes and commands are not rolled back.

Read only the reference needed for the current task:

- [Errors and safe diagnostics](references/errors.md): failed commands, warnings, exit/JSON contracts, or configuration diagnosis with `doctor --json`.
- [Installation](references/installation.md): a missing executable, incomplete setup, or an authorized install/update.
- [Images](references/multimodal.md): image inputs, limits, generated files, or MCP image resources.

## MCP

If ConferLLM is already exposed through MCP, use the corresponding `create_chat`, `continue_chat`, `list_sessions`, `list_models`, and `get_model_info` tools. Do not start another server just to make the same call.
