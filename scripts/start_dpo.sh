#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
run_name=dpo
train_args=()
while (($#)); do
    case "$1" in
        --run-name)
            if (($# < 2)); then echo '--run-name requires a name.' >&2; exit 1; fi
            run_name=$2; shift 2 ;;
        --run-name=*) run_name=${1#*=}; shift ;;
        --output-dir|--output-dir=*|--check-only)
            echo 'The launcher manages output directories and preflight; use --run-name NAME.' >&2
            exit 1 ;;
        *) train_args+=("$1"); shift ;;
    esac
done
set -- ${train_args[@]+"${train_args[@]}"}
if [[ ! "$run_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
    echo 'Run names must start with a letter or digit and contain only letters, digits, - or _.' >&2
    exit 1
fi
run_dir="$PWD/outputs/$run_name"
if [[ -e "$run_dir" || -L "$run_dir" ]]; then
    printf 'Refusing to overwrite %s; choose another --run-name.\n' "$run_dir" >&2
    exit 1
fi
uv run --locked --group dpo python -m dpo.train --check-only "$@"
mkdir -p outputs
mkdir "$run_dir"
nohup bash scripts/run_dpo.sh "$run_dir" "$@" > "$run_dir/train.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$run_dir/pid"
printf 'Submitted PID: %s\nRun directory: %s\nLog: %s/train.log\n' "$pid" "$run_dir" "$run_dir"
