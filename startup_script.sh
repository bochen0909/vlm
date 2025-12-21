#!/bin/bash
# Startup script for remote instance

set -e
export TERM=xterm-256color
echo "Updating package lists..."
apt-get update -y

echo "Installing vim, tmux, and essentials..."
apt-get install -y vim tmux htop git curl wget build-essential

echo "Installing uv (Fast Python package manager)..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure uv is in the path for the rest of the script
export PATH="$HOME/.cargo/bin:$PATH"
source $HOME/.local/bin/env
echo "Found pyproject.toml, running uv sync..."
uv sync
