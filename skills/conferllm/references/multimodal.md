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

The response contains ordered `message.content` and `artifacts` arrays. `artifacts` describes the current turn and may include both input and output images; select `direction == "output"` when identifying generated images. `message.content` refers to images by `artifact_id`, and each artifact has a `conferllm://sessions/.../artifacts/...` URI. An empty `message.text` is valid for an image-only reply.

If the user needs viewable local output files, include `--image-output-dir PATH` with the original chat request, choosing a task-authorized output directory outside private session storage. Use the returned `exported_path` when available. CLI JSON may also contain a canonical `local_path`; do not open private session files directly.

There may be zero, one, or many generated images, depending on the model and response. Output images must be embedded data URLs; remote-only image outputs and unsupported provider tool calls fail explicitly.

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

Repeated `--image` values on a continuation are new images for that turn. Never reconstruct history or read JSONL session files. A model's declared image limit also counts input and output images replayed from earlier turns, even when the new turn attaches no images. If a limit is reached, explain it; start a new session only when a fresh context fits the user's intent. Do not silently discard history or switch model aliases.

Application image limits apply to newly attached images. Use `conferllm model-info MODEL` to inspect declared capabilities and effective limits. Do not send images when the declared `input_modalities` excludes `image`; missing capability metadata is not proof of image support.

## MCP image handling

The CLI's `--image` accepts local file paths. MCP image arguments instead accept data URLs or absolute paths on the server's filesystem; a relative path on the client is not a valid MCP image argument.

MCP returns the chat envelope in structured content and its first text block, followed by inline output images or resource links. It omits CLI filesystem fields such as `local_path` and `exported_path`. Read returned artifact URIs through the MCP client's resource interface, not as filesystem paths or HTTP download URLs.

## Compare models

A session retains its original model alias, not a frozen provider configuration. Do not reconfigure aliases as part of a comparison. Check `provider_model` from `conferllm model-info MODEL`; distinct aliases can point to the same underlying model.

For each requested model, make an independent call with the same prompt and authorized images:

```bash
conferllm chat --model MODEL_A --prompt-file PROMPT --image IMAGE --json
conferllm chat --model MODEL_B --prompt-file PROMPT --image IMAGE --json
```

`PROMPT` in these examples is a UTF-8 file path. Omit `--image IMAGE` for text-only comparisons. Retain and attribute each returned alias, session ID, answer, and output artifact.

Compare actual returned answers after the requested calls complete or fail. Report failed calls as gaps rather than inventing another model's response. Follow-ups must use the session ID belonging to that model.
