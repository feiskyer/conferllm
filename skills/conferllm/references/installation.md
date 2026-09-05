# Installation and configuration

Use this reference only when the `conferllm` executable is missing or diagnostics
show that configuration is incomplete.

## Install the binary

Check for the executable first:

```bash
command -v conferllm
```

For a user-authorized source checkout, run from its repository root:

```bash
uv tool install .
```

A supplied ConferLLM wheel can be used as the installation source instead.
For a verified published release of **this project**, if `uv` is available:

```bash
uv tool install conferllm
```

Otherwise install the same selected source into the current user's Python
environment (`.` for a checkout; `conferllm` only for the verified release):

```bash
python -m pip install --user conferllm
```

Do not use both installers. Verify the selected installation:

```bash
conferllm --version
conferllm doctor --json
```

If the command is still unavailable after a pip user installation, report that
the Python user-script directory must be added to `PATH`; do not guess or edit
shell startup files without the user's request.

## Configure models

When doctor reports a missing configuration, tell the user to create:

```text
~/.conferllm/config.yaml
```

The project README and repository `config_example.yaml` document the format.
The example file is not installed beside the executable. A minimal template is:

```yaml
model_list:
  - model_name: reasoning
    litellm_params:
      model: provider/model-id
      api_key: "user-supplied-key"
```

The user must select their provider model and add credentials themselves.
Never read, infer, migrate, populate, copy, or print credential values. Never
inspect the configuration file to troubleshoot it; use:

```bash
conferllm doctor --json
conferllm models --json
```

Expected permissions are:

```bash
chmod 700 ~/.conferllm
chmod 600 ~/.conferllm/config.yaml
```

Install the version-matched Skill when useful:

```bash
conferllm skill install
conferllm skill install --target codex
```

Do not overwrite an existing different Skill unless the user explicitly asks
to use the force option. A custom destination must be a dedicated Skill
directory, never the home directory, workspace, or an ancestor.
