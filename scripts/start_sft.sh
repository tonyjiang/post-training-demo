#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
run_name=sft
train_args=()
while (($#)); do
    case "$1" in
        --run-name)
            if (($# < 2)); then
                echo '--run-name requires a name.' >&2
                exit 1
            fi
            run_name=$2
            shift 2
            ;;
        --run-name=*) run_name=${1#*=}; shift ;;
        --output-dir|--output-dir=*)
            echo 'Use --run-name NAME; the launcher manages the output directory.' >&2
            exit 1
            ;;
        *) train_args+=("$1"); shift ;;
    esac
done
# Preserve an empty argument list on Bash 3.2 with nounset enabled, too.
set -- ${train_args[@]+"${train_args[@]}"}
if [[ ! "$run_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
    echo 'Run names must start with a letter or digit and contain only letters, digits, - or _.' >&2
    exit 1
fi
run_dir="$PWD/outputs/$run_name"
if [[ -e "$run_dir" || -L "$run_dir" ]]; then
    printf 'Output already exists: %s\nChoose a new name with --run-name NAME. Existing runs are never overwritten.\n' "$run_dir" >&2
    exit 1
fi
# Check the actual --data and options synchronously, before reporting a submitted job.
uv run --locked python -m sft.train --check-only "$@"
mkdir -p outputs
# Atomic creation also protects against two simultaneous launches with the same name.
mkdir "$run_dir"
nohup bash scripts/run_sft.sh "$run_dir" "$@" > "$run_dir/train.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$run_dir/pid"
printf 'Submitted PID: %s\nRun directory: %s\nLog: %s/train.log\n' "$pid" "$run_dir" "$run_dir"
printf 'You can close this terminal. exit_code appears when the run finishes (0 = success).\n'
