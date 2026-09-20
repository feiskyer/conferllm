"""Resolve an installed PowerShell without changing host configuration."""

import shutil


def powershell_argv(command: str) -> list[str]:
    """Build a noninteractive command line for PowerShell Core or Windows PS."""
    executable = (
        shutil.which("pwsh")
        or shutil.which("powershell")
        or shutil.which("powershell.exe")
    )
    if executable is None:
        raise FileNotFoundError(
            "PowerShell executable not found. Install PowerShell (pwsh) to use "
            "run_powershell."
        )
    return [executable, "-NoProfile", "-NonInteractive", "-Command", command]
