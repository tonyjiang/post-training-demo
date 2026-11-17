#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV
run_dir=$1
shift
trap 'code=$?; printf "%s\n" "$code" > "$run_dir/exit_code"' EXIT
exec 9>outputs/sft.lock
flock -n 9 || { echo 'Another training or evaluation job is running.' >&2; exit 1; }
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
uv run --locked --group rlvr python -u -m rlvr.rollout_check --output-dir "$run_dir" "$@"
