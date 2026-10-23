#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV
mkdir -p outputs logs
trap 'code=$?; printf "%s\n" "$code" > logs/evals.exit_code' EXIT
exec 9>outputs/sft.lock
flock -n 9 || { echo 'Another training or evaluation job is running.' >&2; exit 1; }
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NLTK_DATA="$PWD/data/eval-sources/nltk"
export PYTORCH_ALLOC_CONF=expandable_segments:True
uv run --locked --group eval python -u scripts/evaluate_sft.py "$@"
