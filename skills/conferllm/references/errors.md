# Errors and safe diagnostics

Use this reference when a command fails, a result contains warnings, or an agent needs machine-readable diagnostics.

Run:

```bash
conferllm doctor --json
```

Doctor reports safe package/runtime metadata, configuration parse state, aliases, declared capabilities, paths, permission checks, detected Skill installations, and suggested next steps. It does not expose credentials, prompts, or session content, and it does not test provider connectivity.

## Output and exit status

- **Successful `chat --json` calls:** exit 0 and write a `conferllm.chat.response.v1` result to stdout. Warnings can accompany a committed success.
- **Handled runtime failures in JSON mode:** exit 1 and write a `conferllm.error.v1` envelope to stderr. Read `error.code`, `error.message`, and optional `error.details`.
- **`doctor --json` diagnostic reports:** normally write the `conferllm.doctor.v1` diagnostic report to stdout, even when exiting 1 for missing, invalid, legacy, or empty configuration. It can exit 0 with permission warnings; inspect the report rather than treating exit 0 as proof of provider access or secure permissions.
- **Argument-parser failures:** use ordinary CLI usage text on stderr and exit 2, even with `--json`. An interrupted command prints an interruption notice and exits 130. Do not assume every nonzero exit contains JSON.
- **MCP failures:** return protocol tool/resource errors containing the public code and message in `[code] message` format; do not parse them as CLI stderr envelopes.

## Error envelope structure

```json
{
  "schema_version": "conferllm.error.v1",
  "ok": false,
  "error": {
    "code": "model_not_found",
    "message": "Model 'vision' not found. Available models: gpt-6-astra, claude-opus-5, DeepSeek-V4.1-Flash",
    "details": {
      "model": "vision",
      "available_models": ["gpt-6-astra", "claude-opus-5", "DeepSeek-V4.1-Flash"]
    }
  }
}
```

## Error codes

Prefer the stable code over parsing prose. Current codes include:

- `invalid_request`: Malformed parameters, incompatible flags, or mutually exclusive arguments.
- `model_not_found`: Specified model alias does not exist in configuration.
- `model_capability_mismatch`: Request exceeds declared capabilities (e.g. image sent to text-only model).
- `image_limit_exceeded`: Count of images exceeds configured limit (default: 16).
- `image_too_large`: Single image exceeds byte limit (default: 20 MiB) or turn total exceeds limit (50 MiB).
- `unsupported_image_type`: File is not a recognized image type (PNG, JPEG, GIF, WebP, BMP, SVG).
- `invalid_image_data`: Image file is corrupt or unreadable.
- `session_not_found`: Specified session ID was not found in storage.
- `session_corrupt`: Session JSONL is malformed or unparseable.
- `artifact_not_found`: Requested session artifact does not exist or has been deleted.
- `provider_error`: Upstream LLM provider returned an error, bad payload, or connectivity failed.
- `tool_call_limit_exceeded`: Turn exceeded maximum allowed tool rounds (1000 rounds).
- `chat_cancelled`: Operation cancelled by user or client.
- `storage_error`: Host filesystem read, write, or locking error.
- `configuration_error`: Configuration file is missing, invalid YAML, or fails schema validation.

Report the code, concise message, selected alias or session ID when known, and safe details. Do not open configuration contents or private session storage, including user-selected paths. Do not dump environment variables or provider parameters to diagnose authentication failures.

## Recovery boundaries

- **Missing executable:** follow [installation and configuration](installation.md), then verify with `command -v conferllm`.
- **Missing configuration (`configuration_error`):** ask the user to create `~/.conferllm/config.yaml` and supply provider credentials themselves.
- **Invalid configuration:** ask the user to correct its syntax/schema locally without sharing secrets.
- **Missing alias (`model_not_found`):** inspect `conferllm models --json`; choose an available alias or ask the user. Do not invent or silently substitute a model.
- **Capability or image-limit mismatch:** explain the rejected input or replayed-history constraint. Do not silently omit images, drop history, change limits, or switch models.
- **Session/artifact corruption:** stop the affected continuation or read, preserve the files, and report the failure. Do not rewrite private storage.
- **Provider error:** it can represent an upstream failure, a wrong API format for the endpoint/model, malformed tool calls, incomplete Responses output, or an image download failure. Do not assume it is an authentication problem. OpenAI endpoints default to Responses; older endpoints can explicitly select model-level `api_format: chat_completion`. GPT-6 Astra tool calling requires Responses.
- **Tool error results:** missing files/executables, invalid arguments, and command failures are returned to the model so it can correct them within the same turn. PowerShell is not installed automatically.
- **Tool-round limit:** the turn exceeded 1000 rounds. Check scope and side effects before asking the model to continue the task.

A timeout or interruption does not prove that a provider or tool never ran. File changes and commands are not rolled back when the final response or session commit fails. Errors after tool attempts include `tool_calls_attempted` and `side_effects_may_remain`; a failed new session's returned ID does not prove it was committed. Do not blindly repeat ambiguous or completed calls; preserve any returned IDs and explain the uncertainty.

For image-export warnings after success, retain `session.id`, `session.turn`, and the artifact URIs. MCP does not inline images; a client can read an artifact resource without another model call. The CLI has no standalone artifact-read or re-export command; use an already-exported file in an authorized output directory, or report that export recovery is unavailable through the CLI. See [image handling](multimodal.md).
