# Multimodal conversations

Use this reference for requests that include input images, expect generated
images, or compare several configured models.

## Multiple input and output images

Preserve the user's image order by repeating `--image` in that order:

```bash
conferllm chat \
  --model vision \
  --prompt "Compare these screenshots." \
  --image ./first.png \
  --image ./second.jpg \
  --image ./third.webp \
  --json
```

Use only files the user supplied or explicitly authorized. Do not search the
filesystem for additional images. ConferLLM copies accepted inputs into immutable
session-owned storage, so later turns use stable artifact references rather
than the original files.

The structured response contains ordered `message.content` and `artifacts`
arrays. Each generated image has a `conferllm://sessions/.../artifacts/...` URI.
Treat `artifacts` as authoritative. There may be zero, one, or many output
images.

Use `--image-output-dir PATH` only when the user needs exported local copies.
Canonical session artifacts remain available even if an export fails.
Check `warnings` without treating a committed chat as a failed model call.
Generated images must be embedded data URLs; remote output URLs are not
downloaded automatically.

## Continue a multimodal session

Continue with the returned session ID:

```bash
conferllm chat \
  --session SESSION_ID \
  --prompt "Focus on the second image." \
  --image ./new-context.png \
  --json
```

Repeated `--image` values on a continuation are new images for that turn.
Never reconstruct history or read JSONL session files.
Declared model image limits include the images replayed from prior turns.
If a limit is reached, explain it and start a new session when appropriate;
do not silently remove context or switch models.

MCP returns the same envelope in structured content and in its first text
block, followed by inline images or resource links. Even clients that only
read text can retain the session ID and continue.

## Compare models

A session has one immutable model. For a real comparison, create one
independent session per model using the same prompt and images:

```bash
conferllm chat --model MODEL_A --prompt-file PROMPT --image IMAGE --json
conferllm chat --model MODEL_B --prompt-file PROMPT --image IMAGE --json
```

Retain and attribute each returned session ID. Compare the model answers and
artifacts only after every independent call completes. Follow-ups must use the
session ID belonging to that model; never attempt to switch a session's model.

Before sending images, use `conferllm model-info MODEL` when capability metadata
would help. A declared text-only model must not receive images. Unknown image
capability is not proof of support.
