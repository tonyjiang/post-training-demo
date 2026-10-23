#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV
run_dir=$1
shift
trap 'code=$?; printf "%s\n" "$code" > "$run_dir/exit_code"' EXIT
exec 9>outputs/sft.lock
flock -n 9 || { echo 'Another SFT job is already running.' >&2; exit 1; }
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
uv run --locked python -u scripts/train_sft.py --output-dir "$run_dir" "$@"
