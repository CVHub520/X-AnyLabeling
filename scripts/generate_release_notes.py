#!/usr/bin/env python3

import argparse
import re
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
VERSION_PATTERN = re.compile(
    r'^__version__\s*=\s*["\']([^"\']+)["\']', re.MULTILINE
)
CHANGELOG_PATTERN = re.compile(r"^## `([^`]+)`[^\n]*$", re.MULTILINE)


def read_version() -> str:
    app_info = ROOT_DIR / "anylabeling" / "app_info.py"
    match = VERSION_PATTERN.search(app_info.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"Failed to read version from {app_info}")
    return match.group(1)


def read_changelog_section(tag: str) -> str:
    changelog_path = ROOT_DIR / "CHANGELOG.md"
    changelog = changelog_path.read_text(encoding="utf-8")
    matches = list(CHANGELOG_PATTERN.finditer(changelog))
    if not matches:
        raise ValueError(f"No release entries found in {changelog_path}")
    if matches[0].group(1) != tag:
        raise ValueError(
            f"Latest changelog version is {matches[0].group(1)}, expected {tag}"
        )
    end = matches[1].start() if len(matches) > 1 else len(changelog)
    section = changelog[matches[0].end() : end].strip()
    if not section:
        raise ValueError(f"Changelog entry for {tag} is empty")
    return section


def find_previous_tag(tag: str) -> str:
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0", f"{tag}^"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def generate_notes(tag: str, repository: str) -> str:
    version = read_version()
    expected_tag = f"v{version}"
    if tag != expected_tag:
        raise ValueError(f"Tag is {tag}, expected {expected_tag}")

    changelog = read_changelog_section(tag)
    previous_tag = find_previous_tag(tag)
    compare_url = (
        f"https://github.com/{repository}/compare/{previous_tag}...{tag}"
    )
    return (
        "> PyPI: https://pypi.org/project/x-anylabeling-cvhub/\n"
        "> Baidu Cloud: "
        "https://pan.baidu.com/s/1pgaw02inCvbEgOme9ajDJA?pwd=e528\n\n"
        "> [!NOTE]\n"
        "> Due to compatibility issues across different systems, if the "
        "precompiled version doesn’t work properly on your machine, you can "
        "try building and running it from source instead. For details, check "
        "out the official installation guide and user documentation.\n\n"
        f"{changelog}\n\n"
        f"**Full Changelog**: {compare_url}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    notes = generate_notes(args.tag, args.repository)
    args.output.write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
