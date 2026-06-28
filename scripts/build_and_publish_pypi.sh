#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Clean previous build artifacts.
rm -rf dist/ build/ ./*.egg-info wheels_dist/

# Build distributions.
python -m build --no-isolation

# Validate distributions before publishing.
twine check dist/*

# Publish to PyPI.
twine upload dist/*
