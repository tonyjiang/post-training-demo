#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV
run_name=${1:-rlvr-pilot}
if [[ $# -gt 0 ]]; then shift; fi
[[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'Invalid run name' >&2; exit 1; }
run_dir="$PWD/outputs/$run_name"
[[ ! -e "$run_dir" ]] || { echo "Refusing to overwrite $run_dir" >&2; exit 1; }
uv run --locked --group rlvr python -m rlvr.train --output-dir "$run_dir" --check-only "$@"
mkdir -p outputs
mkdir "$run_dir"
nohup bash scripts/run_rlvr.sh "$run_dir" "$@" > "$run_dir/train.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$run_dir/pid"
echo "Started $pid: $run_dir"
