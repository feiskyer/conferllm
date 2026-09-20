# Multimodal conversations

Use this reference for requests with input images, generated images, or comparisons across configured models. Replace `MODEL` with the selected alias from `conferllm models --json`, not a presumed alias such as `vision`.

## Multiple input and output images

Preserve the user's image order by repeating `--image` in that order:

```bash
conferllm chat \
  --model MODEL \
  --prompt "Compare these screenshots." \
  --image ./first.png \
  --image ./second.jpg \
  --image ./third.webp \
  --json
```

Use only files supplied or authorized for the task, and preserve their order. ConferLLM copies accepted inputs into session-owned storage so later turns reuse those copies rather than the original files.

The response contains ordered `message.content` and `artifacts` arrays. `artifacts` describes the current turn and may include both input and output images; select `direction == "output"` when identifying generated images. Image blocks have `type: image_url`, an `artifact_id`, and an absolute saved path in `image_url.url`. `message.text` also contains saved paths in place of image data, including image-only replies. Each artifact retains a `conferllm://sessions/.../artifacts/...` resource URI.

Generated images are automatically saved to `/tmp` when no output directory is provided. Include `--image-output-dir PATH` for a task-authorized, durable location outside private session storage. Use output artifacts' `saved_path`; CLI JSON also retains `exported_path` and canonical `local_path` for compatibility. Do not inspect private session storage directly. If export failed and `saved_path` points into private storage, report the warning and use an MCP resource read rather than opening that private file.

There may be zero, one, or many generated images, depending on the model and response. All candidates and supported content blocks are processed. Embedded data URLs, base64 blocks, native image-generation items, and HTTP(S) image outputs are saved; remote downloads are bounded and do not receive provider credentials. Native tool calls execute automatically, and images produced across tool rounds receive distinct artifact IDs. Tool argument strings and encrypted reasoning are not treated as image output.

Image generation success does not imply reliable instruction following or tool use. An image model may return another image even when a follow-up requests text; inspect the actual content types and report what it returned.

An export warning does not undo the committed chat or remove its canonical artifacts. Do not repeat the model call to repair an export. The CLI has no standalone artifact-read or re-export command; use an available MCP artifact resource, or report the warning and preserved IDs. See [recovery boundaries](errors.md).

## Continue a multimodal session

Continue with the returned session ID:

```bash
conferllm chat \
  --session SESSION_ID \
  --prompt "Focus on the second image." \
  --image ./new-context.png \
  --json
```

Responses history conversion preserves assistant text phases and supplies generated images as image inputs explicitly attributed to the assistant. This conversion is automatic; do not rebuild the history or reattach old images just to change the API format.

Repeated `--image` values on a continuation are new images for that turn. Never reconstruct history or read JSONL session files. A model's declared image limit also counts input and output images replayed from earlier turns, even when the new turn attaches no images. If a limit is reached, explain it; start a new session only when a fresh context fits the user's intent. Do not silently discard history or switch model aliases.

Application image limits apply to newly attached images. Use `conferllm model-info MODEL` to inspect declared capabilities and effective limits. Do not send images when the declared `input_modalities` excludes `image`; missing capability metadata is not proof of image support.

## MCP image handling

The CLI's `--image` accepts local file paths. MCP image arguments instead accept data URLs or absolute paths on the server's filesystem; a relative path on the client is not a valid MCP image argument.

MCP returns the chat envelope in structured content and its first text block, followed by resource links. Images are not inlined as base64. `saved_path` and image content paths refer to the server host; a remote client should read the artifact URI through MCP when it cannot access that filesystem. Canonical `local_path` and compatibility `exported_path` fields remain omitted from MCP artifact metadata.

## Compare models

A session retains its original model alias, not a frozen provider configuration. Do not reconfigure aliases as part of a comparison. Check `provider_model` from `conferllm model-info MODEL`; distinct aliases can point to the same underlying model.

For each requested model, make an independent call with the same prompt and authorized images:

```bash
conferllm chat --model MODEL_A --prompt-file PROMPT --image IMAGE --json
conferllm chat --model MODEL_B --prompt-file PROMPT --image IMAGE --json
```

`PROMPT` in these examples is a UTF-8 file path. Omit `--image IMAGE` for text-only comparisons. Retain and attribute each returned alias, session ID, answer, and output artifact.

Compare actual returned answers after the requested calls complete or fail. Report failed calls as gaps rather than inventing another model's response. Follow-ups must use the session ID belonging to that model.
