# Installation and configuration

Use this reference when the executable is missing, diagnostics show incomplete configuration, or the user requests an installation update. ConferLLM requires Python 3.10+ and POSIX file locking on macOS/Linux; native Windows is not supported.

## Install the binary

Check for the executable first:

```bash
command -v conferllm
```

If missing, install by package name with uv (recommended):

```bash
uv tool install conferllm
```

Alternatively, use pip in an activated Python virtual environment:

```bash
python -m pip install conferllm
```

Do not use both installers. Verify the selected installation:

```bash
conferllm --version
conferllm doctor --json
```

If a uv-installed command is not on `PATH`, suggest `uv tool update-shell` and reopening the terminal. For pip, check that the intended virtual environment is active. Do not change shell startup files without authorization. If installation is blocked or the package cannot be resolved, report the actual error instead of silently choosing a different package or source.

## Configure models

When doctor reports a missing configuration, tell the user to create:

```text
~/.conferllm/config.yaml
```

The README and repository `config_example.yaml` document the format; the example file is not installed beside the executable. This self-contained template uses the same `gpt-4o` alias as the README:

```yaml
model_list:
  - model_name: gpt-4o
    capabilities:
      input_modalities: [text, image]
      output_modalities: [text]
    litellm_params:
      model: openai/gpt-4o
      api_key: "replace-with-your-key"
```

This is an OpenAI example, not a guarantee of account access. The user must select an available provider model and add credentials themselves. Keep aliases unique. Capability declarations describe the intended configuration; they do not probe provider support.

Never read, infer, migrate, populate, copy, or print credential values. Do not inspect the configuration file to troubleshoot it; use:

```bash
conferllm doctor --json
conferllm models --json
```

Expected permissions are `0700` for the application directory and `0600` for the configuration file:

```bash
mkdir -p ~/.conferllm
chmod 700 ~/.conferllm
chmod 600 ~/.conferllm/config.yaml
```

The user should create the configuration file before applying its file permissions. To use a different configuration, pass `--config PATH` consistently to diagnostics, model discovery, chat, and session commands. See [safe diagnostics](errors.md) for report fields and failure handling.

## Development installs

Only when the user requests development from a local checkout, run from that checkout's root:

```bash
uv tool install --editable .
```

Python source changes take effect on the next invocation; restart a running server after edits. Reinstall after changing dependencies or command entry points. Add `--force` only when replacing an existing tool installation is intended.

## Install or update the Skill

Install the Skill bundled with the executable into the selected target:

```bash
conferllm skill install
conferllm skill install --target codex
```

These are alternatives: the default is `~/.agents/skills/conferllm`; `--target codex` selects `~/.codex/skills/conferllm`. For an exact custom directory, use `--destination PATH`. A custom destination must be a dedicated Skill directory, not the home directory, workspace, package directory, or an ancestor.

The installer copies the bundle. Updating the Python package does not refresh an already-copied Skill; replacing a different installed copy requires the user's authorization and `--force`.

Check whether the installed directory is a user-managed symlink before replacing it. A symlink to the source bundle already reflects source-file edits. Preserve that link: a forced installation would replace it with a copied directory. An agent that already loaded the previous instructions may need to reload the Skill.
