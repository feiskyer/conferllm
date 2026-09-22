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

The response contains ordered `message.content` and `artifacts` arrays:

```json
{
  "schema_version": "conferllm.chat.response.v1",
  "ok": true,
  "session": {
    "id": "20260905-0123456789abcdef0123456789abcdef",
    "name": "Compare screenshots",
    "model": "gpt-6-astra",
    "turn": 1
  },
  "message": {
    "text": "The differences are...\n/tmp/output-diagram.png",
    "content": [
      {"type": "text", "text": "The differences are..."},
      {"type": "image_url", "image_url": {"url": "/tmp/output-diagram.png"}, "artifact_id": "0123456789abcdef"}
    ]
  },
  "artifacts": [
    {
      "id": "0123456789abcdef",
      "direction": "output",
      "mime_type": "image/png",
      "size_bytes": 1048576,
      "saved_path": "/tmp/output-diagram.png",
      "uri": "conferllm://sessions/20260905-0123456789abcdef0123456789abcdef/artifacts/0123456789abcdef"
    }
  ],
  "warnings": []
}
```

- **Output images:** filter `artifacts` where `direction == "output"`. Read `saved_path` for the local absolute file path.
- **`message.text`:** automatically replaces raw image payloads with saved file paths, separating text and paths with newlines.
- **`--image-output-dir PATH`:** specify a dedicated directory for generated images. When omitted, images default to `/tmp`.

Embedded data URLs, base64 blocks, native Responses image-generation items, and HTTP(S) image outputs are saved. Remote downloads are bounded (20 MiB max) and do not receive provider credentials.

An export warning does not undo the committed chat or remove its canonical artifacts. Do not repeat the model call to repair an export.

## Continue a multimodal session

Continue with the returned session ID:

```bash
conferllm chat \
  --session SESSION_ID \
  --prompt "Focus on the second image." \
  --image ./new-context.png \
  --json
```

- Replayed history automatically attributes previous generated images as assistant inputs.
- Repeated `--image` values on a continuation are new images for that turn.
- A model's declared `max_input_images` counts both replayed history images and newly attached images. If a limit is reached, explain it; do not silently drop history or switch models.
- Inspect effective limits with `conferllm model-info MODEL`. Do not send images to a model whose declared `input_modalities` excludes `image`.

## MCP image handling

- The CLI's `--image` accepts local filesystem paths.
- MCP image arguments (`create_chat`, `continue_chat`, `chat`) accept **data URLs** or **absolute paths on the server's filesystem**. Relative client paths are rejected with `invalid_request`.
- MCP returns the chat envelope in structured content and in its first text block, accompanied by resource links (`conferllm://sessions/{session_id}/artifacts/{artifact_id}`). Generated images are not inlined as base64 over the protocol.

## Compare models

A session retains its original model alias. To compare models, make independent calls with the same prompt and authorized images:

```bash
conferllm chat --model MODEL_A --prompt-file ./prompt.md --image ./input.png --json
conferllm chat --model MODEL_B --prompt-file ./prompt.md --image ./input.png --json
```

1. Run each model independently and capture JSON output.
2. Read `message.text` and `session.id` from each response.
3. Compare actual returned answers and clearly attribute each answer to its respective model alias.
4. If one model fails, report the error honestly as a gap rather than inventing an answer.
5. Follow-ups must use the respective session ID belonging to that model.
