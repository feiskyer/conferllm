# Errors and safe diagnostics

Use this reference when a command fails, a result contains warnings, or an agent needs machine-readable diagnostics.

Run:

```bash
conferllm doctor --json
```

Doctor reports safe package/runtime metadata, configuration parse state, aliases, declared capabilities, paths, permission checks, detected Skill installations, and suggested next steps. It does not expose credentials, prompts, or session content, and it does not test provider connectivity.

## Output and exit status

- Successful `chat --json` calls exit 0 and write a `conferllm.chat.response.v1` result to stdout. Warnings can accompany a committed success.
- Runtime failures handled by the CLI in JSON mode exit 1 and write a `conferllm.error.v1` envelope to stderr. Read `error.code`, `error.message`, and optional `error.details`.
- `doctor --json` normally writes its `conferllm.doctor.v1` diagnostic report to stdout, even when it exits 1 for missing, invalid, legacy, or empty configuration. It can exit 0 with permission warnings; inspect the report rather than treating exit 0 as proof of provider access or secure permissions.
- Argument-parser failures use ordinary usage text on stderr and exit 2, even with `--json`. An interrupted command prints an interruption message and exits 130. Do not assume every nonzero exit contains JSON.
- MCP failures use protocol tool/resource errors containing the public code and message; do not parse them as CLI stderr envelopes.

## Error codes

Prefer the stable code over parsing prose. Current codes include:

- `invalid_request`
- `model_not_found`
- `model_capability_mismatch`
- `image_limit_exceeded`
- `image_too_large`
- `unsupported_image_type`
- `invalid_image_data`
- `session_not_found`
- `session_corrupt`
- `artifact_not_found`
- `provider_error`
- `storage_error`
- `configuration_error`

Report the code, concise message, selected alias or session ID when known, and safe details. Do not open configuration contents or private session storage, including user-selected paths. Do not dump environment variables or provider parameters to diagnose authentication failures.

## Recovery boundaries

- Missing executable: follow [installation and configuration](installation.md), then verify it.
- Missing configuration: ask the user to create and populate it.
- Invalid configuration: ask the user to correct its syntax/schema locally.
- Missing alias: inspect `conferllm models --json`; do not silently replace the requested model.
- Capability or image-limit mismatch: explain the rejected input or replayed-history constraint. Do not silently omit images, drop history, change limits, or switch models.
- Session/artifact corruption: stop the affected continuation or read, preserve the files, and report the failure. Do not rewrite private storage.
- Provider error: it can represent an upstream failure or an unsupported response, including remote-only output images or tool calls. Do not assume it is an authentication problem.

A timeout or interruption does not prove the provider never ran. Do not blindly repeat ambiguous or completed calls; preserve any returned IDs and explain the uncertainty.

For export or MCP inline-display warnings after success, retain `session.id`, `session.turn`, and the artifact URIs. An MCP client can retry reading an artifact resource without another model call. The CLI has no standalone artifact-read or re-export command; use an already-exported file in an authorized output directory, or report that export recovery is unavailable through the CLI. See [image handling](multimodal.md).
