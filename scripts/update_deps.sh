#!/usr/bin/env bash

# Turn on bash strict mode.
set -euo pipefail

# Delete the old lock file.
rm -f uv.lock

# Remove the current virtual environment.
rm -rf .venv || true

# Create a new virtual environment.
uv venv -p 3.11

# Install the latest versions of the dependencies.
uv lock && uv sync
