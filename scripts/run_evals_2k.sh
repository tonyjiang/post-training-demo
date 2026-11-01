#!/usr/bin/env bash
# Evaluate both saved adapters with 2,048 output tokens, without rerunning base.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs logs
for path in outputs/evals-sft-2k-b1 outputs/evals-dpo-2k-b1 \
  logs/evals-sft-2k-b1.log logs/evals-dpo-2k-b1.log \
  logs/evals-sft-2k-b1.exit_code logs/evals-dpo-2k-b1.exit_code logs/evals-2k-b1.exit_code; do
  if [[ -e "$path" ]]; then
    echo "Refusing to overwrite $path" >&2
    exit 1
  fi
done
trap 'code=$?; printf "%s\n" "$code" > logs/evals-2k-b1.exit_code' EXIT

run_stage() {
  local stage=$1
  shift
  local code=0
  echo "Starting $stage evaluation with 2048 output tokens"
  bash scripts/run_evals.sh "$@" > "logs/evals-$stage-2k-b1.log" 2>&1 || code=$?
  printf '%s\n' "$code" > "logs/evals-$stage-2k-b1.exit_code"
  return "$code"
}

run_stage sft --adapter-only \
  --adapter outputs/sft/final-adapter --adapter-name sft \
  --max-new-tokens 2048 --output outputs/evals-sft-2k-b1
run_stage dpo --baseline-results outputs/evals-sft-2k-b1 --baseline-name sft \
  --adapter outputs/dpo/final-adapter --adapter-name dpo \
  --max-new-tokens 2048 --output outputs/evals-dpo-2k-b1
echo 'SFT and DPO evaluation complete: outputs/evals-dpo-2k-b1/results.json'
