#!/usr/bin/env python3
"""Compute the next semver from existing v* git tags."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

SEMVER_RE = re.compile(r"^v?(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")


def latest_tag() -> tuple[int, int, int] | None:
    result = subprocess.run(
        ["git", "tag", "-l", "v*", "--sort=-v:refname"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        tag = line.strip()
        match = SEMVER_RE.match(tag)
        if match:
            return (
                int(match.group("major")),
                int(match.group("minor")),
                int(match.group("patch")),
            )
    return None


def bump(version: tuple[int, int, int], release_type: str) -> tuple[int, int, int]:
    major, minor, patch = version
    if release_type == "major":
        return major + 1, 0, 0
    if release_type == "minor":
        return major, minor + 1, 0
    if release_type == "patch":
        return major, minor, patch + 1
    raise ValueError(f"unknown release_type: {release_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "release_type",
        choices=["patch", "minor", "major"],
        help="Semver bump to apply to the latest v* tag",
    )
    args = parser.parse_args()

    current = latest_tag()
    if current is None:
        new_version = (0, 1, 0)
    else:
        new_version = bump(current, args.release_type)

    version_str = f"{new_version[0]}.{new_version[1]}.{new_version[2]}"
    tag = f"v{version_str}"
    print(f"version={version_str}")
    print(f"tag={tag}")
    if current is None:
        print("previous=none")
    else:
        print(f"previous=v{current[0]}.{current[1]}.{current[2]}")


if __name__ == "__main__":
    main()
