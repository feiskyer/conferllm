#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "packaging",
#   "requests",
#   "tomli; python_version < '3.11'",
# ]
# ///
"""Raise dependency floors using declared constraints and PyPI metadata."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import requests
from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

try:
    import tomllib
except ImportError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

PYPROJECT = Path("pyproject.toml")
PYPI_PACKAGE_URL = "https://pypi.org/pypi/{package}/json"
REQUIREMENT_PARTS = re.compile(
    r"^(?P<prefix>[A-Za-z0-9_.-]+(?:\[[^\]]+\])?)"
    r"(?P<specifier>[^;]*)"
    r"(?P<marker>\s*;.*)?$"
)
LOWER_BOUND = re.compile(r"^(===|==|~=|>=|>)(.+)$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Update runtime, optional, and build dependency floors to the "
            "newest stable releases allowed by their constraints and PyPI "
            "Python metadata."
        )
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=PYPROJECT,
        help="pyproject.toml to update (default: ./pyproject.toml)",
    )
    parser.add_argument(
        "--python-version",
        default="3.10",
        help="Minimum Python version that releases must support (default: 3.10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show available updates without writing the file",
    )
    return parser.parse_args(argv)


def dependency_entries(data: dict[str, Any]) -> list[str]:
    """Collect runtime, optional, and build-system requirement strings."""
    entries: list[str] = []
    project = data.get("project", {})
    if isinstance(project, dict):
        dependencies = project.get("dependencies", [])
        if isinstance(dependencies, list):
            entries.extend(item for item in dependencies if isinstance(item, str))

        optional = project.get("optional-dependencies", {})
        if isinstance(optional, dict):
            for group in optional.values():
                if isinstance(group, list):
                    entries.extend(item for item in group if isinstance(item, str))

    build_system = data.get("build-system", {})
    if isinstance(build_system, dict):
        requirements = build_system.get("requires", [])
        if isinstance(requirements, list):
            entries.extend(item for item in requirements if isinstance(item, str))

    return list(dict.fromkeys(entries))


def release_supports_python(
    files: list[dict[str, Any]], target_python: Version
) -> bool:
    """Check artifact metadata for support of the target Python."""
    for file_info in files:
        if file_info.get("yanked"):
            continue

        requires_python = file_info.get("requires_python")
        if not requires_python:
            return True

        try:
            if SpecifierSet(requires_python).contains(target_python, prereleases=True):
                return True
        except InvalidSpecifier:
            continue

    return False


def fetch_releases(package: str) -> dict[str, Any] | None:
    """Fetch release metadata for one package."""
    try:
        response = requests.get(
            PYPI_PACKAGE_URL.format(package=package),
            timeout=15,
        )
        response.raise_for_status()
        releases = response.json().get("releases", {})
    except (requests.RequestException, ValueError) as error:
        print(f"  Warning: could not query {package}: {error}")
        return None

    return releases if isinstance(releases, dict) else None


def latest_compatible_version(
    target_python: Version,
    allowed_versions: SpecifierSet,
    releases: dict[str, Any],
) -> str | None:
    """Select the newest allowed stable release supported by PyPI metadata."""
    candidates: list[Version] = []
    for version_text, files in releases.items():
        try:
            version = Version(version_text)
        except InvalidVersion:
            continue

        if version.is_prerelease or version.is_devrelease:
            continue
        if not allowed_versions.contains(version, prereleases=False):
            continue
        if not isinstance(files, list):
            continue
        if release_supports_python(files, target_python):
            candidates.append(version)

    return str(max(candidates)) if candidates else None


def applies_to_python(requirement: Requirement, target_python: Version) -> bool:
    """Evaluate a dependency marker for the requested Python version."""
    if requirement.marker is None:
        return True

    release = (*target_python.release, 0, 0, 0)
    environment = default_environment()
    environment["python_version"] = f"{release[0]}.{release[1]}"
    environment["python_full_version"] = f"{release[0]}.{release[1]}.{release[2]}"
    return requirement.marker.evaluate(environment)


def upgraded_requirement(requirement_text: str, version: str) -> str | None:
    """Return a requirement with its first version floor updated."""
    try:
        requirement = Requirement(requirement_text)
    except InvalidRequirement as error:
        print(f"  Warning: skipping invalid requirement {requirement_text!r}: {error}")
        return None

    if requirement.url is not None:
        print(f"  Warning: skipping direct URL requirement {requirement_text!r}")
        return None

    match = REQUIREMENT_PARTS.fullmatch(requirement_text)
    if match is None:
        print(f"  Warning: could not rewrite requirement {requirement_text!r}")
        return None

    specifier = (match.group("specifier") or "").strip()
    parts = [part.strip() for part in specifier.split(",") if part.strip()]
    for index, part in enumerate(parts):
        lower_bound = LOWER_BOUND.fullmatch(part)
        if lower_bound is not None:
            parts[index] = f"{lower_bound.group(1)}{version}"
            break
    else:
        parts.insert(0, f">={version}")

    return f"{match.group('prefix')}{','.join(parts)}{match.group('marker') or ''}"


def replace_quoted_requirement(text: str, old: str, new: str) -> tuple[str, int]:
    """Replace an exact single- or double-quoted TOML requirement."""
    pattern = re.compile(rf"(?P<quote>[\"']){re.escape(old)}(?P=quote)")

    def replacement(match: re.Match[str]) -> str:
        quote = match.group("quote")
        return f"{quote}{new}{quote}"

    return pattern.subn(replacement, text)


def update_pyproject(
    pyproject_path: Path,
    *,
    target_python: Version,
    dry_run: bool,
) -> int:
    """Update dependency floors in a pyproject file."""
    if not pyproject_path.is_file():
        print(f"Error: {pyproject_path} not found", file=sys.stderr)
        return 1

    with pyproject_path.open("rb") as pyproject_file:
        data = tomllib.load(pyproject_file)
    text = pyproject_path.read_text(encoding="utf-8")
    requirements = dependency_entries(data)
    if not requirements:
        print("No dependencies found.")
        return 0

    print(
        f"Checking {len(requirements)} declared dependencies for Python "
        f"{target_python} metadata compatibility..."
    )
    release_cache: dict[str, dict[str, Any] | None] = {}
    replacements: list[tuple[str, str]] = []

    for requirement_text in requirements:
        try:
            requirement = Requirement(requirement_text)
        except InvalidRequirement as error:
            print(
                f"  Warning: skipping invalid requirement {requirement_text!r}: {error}"
            )
            continue

        if not applies_to_python(requirement, target_python):
            continue

        package = requirement.name
        if package not in release_cache:
            release_cache[package] = fetch_releases(package)
        releases = release_cache[package]
        if releases is None:
            continue

        version = latest_compatible_version(
            target_python,
            requirement.specifier,
            releases,
        )
        if version is None:
            print(
                f"  Warning: no release of {package} satisfies "
                f"{requirement.specifier or '(any version)'}"
            )
            continue

        updated = upgraded_requirement(requirement_text, version)
        if updated is None or updated == requirement_text:
            continue
        replacements.append((requirement_text, updated))
        print(f"  {requirement_text} -> {updated}")

    if not replacements:
        print("All applicable dependency floors are already current.")
        return 0

    for old, new in replacements:
        text, count = replace_quoted_requirement(text, old, new)
        if count == 0:
            print(
                f"Error: could not find quoted requirement {old!r}",
                file=sys.stderr,
            )
            return 1

    if dry_run:
        print(f"\nDry run: {len(replacements)} updates available.")
        return 0

    pyproject_path.write_text(text, encoding="utf-8")
    print(f"\nUpdated {len(replacements)} dependencies in {pyproject_path}.")
    print("Run 'uv lock --upgrade' to refresh the lockfile.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the dependency updater."""
    args = parse_args(argv)
    try:
        target_python = Version(args.python_version)
    except InvalidVersion as error:
        print(
            f"Error: invalid Python version {args.python_version!r}: {error}",
            file=sys.stderr,
        )
        return 2

    return update_pyproject(
        args.pyproject,
        target_python=target_python,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
