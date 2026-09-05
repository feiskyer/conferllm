# Errors and safe diagnostics

Use this reference when a ConferLLM command fails or an agent needs
machine-readable diagnostics.

Run:

```bash
conferllm doctor --json
```

Doctor reports only package/runtime metadata, configuration parse state, model
aliases, declared capability summaries, paths, permission checks, detected
Skill installations, and suggested next steps. It does not expose provider
parameter values, prompts, or session content.
An invalid or missing configuration makes `doctor` exit nonzero; its JSON
report is still on stdout. Runtime `chat --json` errors are on stderr and
also exit nonzero.

JSON command failures use the `conferllm.error.v1` envelope. Prefer its stable
error code over parsing prose. Initial codes include:

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

Report the code, concise message, model alias or session ID, and safe details.
Do not open `~/.conferllm/config.yaml` or files under `~/.conferllm/sessions/`.
Do not print environment variables or provider parameters to diagnose
authentication failures.

Useful recovery choices:

- Missing executable: follow the installation reference, then verify it.
- Missing configuration: ask the user to create and populate it.
- Invalid configuration: ask the user to correct its syntax/schema locally.
- Capability mismatch: choose a declared compatible model or omit images.
- Session/artifact corruption: fail closed and preserve the files for manual
  recovery; do not silently rewrite them.
- Provider error: report it as provider-facing after local validation has
  passed; do not expose credentials while troubleshooting.

Model image limits also account for replayed history, even if the current
turn attaches no new images. Do not silently drop history to bypass a limit.

A successful response can contain warnings about optional exports or MCP
inline display. Preserve the committed session ID and turn; retry the artifact
operation instead of repeating the model call. Unsupported provider tool calls
or remote-only image outputs fail explicitly instead of becoming empty answers.
