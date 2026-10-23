#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
uv run --locked python -u scripts/smoke_nf4.py "$@"
uv run --locked python -u scripts/smoke_nf4.py --reload "$@"
