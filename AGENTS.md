# AGENTS.md

Guidance for AI agents (and humans) working in this repo.

## Project goal

A learning exercise: reproduce the **Olmo 3 Think 7B** post-training pipeline as closely as possible,
using only Ai2's released artifacts.

    Olmo-3-1025-7B (base)  ->  SFT  ->  DPO  ->  RLVR  ->  our "Olmo-3-7B-Think" reproduction

Out of scope: RL-Zero runs, the Instruct track, pretraining / midtraining.

## Ground rules

- Before starting any task, read `README.md` for the artifact list, decisions, and open questions.
- Prefer Ai2's code and configs over writing our own. Deviate only when forced (e.g. no Beaker access,
  smaller GPU budget), and record every deviation in `README.md` under "Deviations".
- Use the exact Hugging Face IDs listed in `README.md`. Pin dataset and model revisions (commit SHAs)
  once we start training.
- Every stage has an official Ai2 checkpoint. Use it as the reference to compare against, and as a
  fallback starting point (e.g. start DPO from `Olmo-3-7B-Think-SFT` if our SFT is not ready).
- Upstream repos are cloned into `third_party/` and not edited in place. Our reusable stage code lives in `sft/`, `dpo/`, and `rlvr/`,
  launchers and standalone utilities in `scripts/`, and configs in `configs/`.
- Keep `README.md` current: decisions, open questions, run log.

## Hardware

**Fixed constraint: this is our only GPU.** Plan and implement all experiments for
the RTX 5070 below. Do not propose buying, renting, or switching to another GPU.
Fit its memory budget using quantization, shorter sequences/truncation, microbatch
size, gradient accumulation/checkpointing, and verified memory-efficient kernels.
The user permits truncation and restarting SFT from the base on combined shuffled data.

Verified over SSH on 2026-10-21:
- Training host: `gaming-server` (SSH alias), Linux x86_64.
- GPU: 1 x NVIDIA GeForce RTX 5070, 12 GB VRAM (`nvidia-smi`: 12,227 MiB).
- CPU: Intel Core i7-14790F, 24 logical CPUs.
- System RAM: 31 GiB usable (32 GB class); 7 GiB swap.
- NVIDIA driver at verification: 595.84. Recheck runtime/CUDA compatibility before training.
- This GPU cannot hold ordinary full-parameter 7B SFT training in VRAM. A smaller dataset
  reduces runtime, not model/optimizer memory. The user chose NF4 QLoRA with Transformers,
  bitsandbytes, and PEFT as a documented deviation; do not assume OLMo-core supports QLoRA.
- Remote project: `/root/post-training-demo`; official base weights:
  `/root/models/Olmo-3-1025-7B`. Recheck available disk and GPU memory before runs.

## Environment

Environment setup on gaming-server (Linux x86_64, Python 3.12):
`cd /root/post-training-demo && uv sync --locked`.
Use `uv run --locked python ...` for scripts. If uv is not on PATH, use
`/root/.local/bin/uv` or add `/root/.local/bin` to PATH. This CUDA environment targets the server,
not the local Mac. Dependencies live in `pyproject.toml`; resolved versions live in `uv.lock`.

## Layout

    AGENTS.md        this file
    README.md         setup, usage, artifact inventory, decisions, run log
    docs/workflow.html  pipeline diagram (open in a browser); update when the plan changes
    third_party/     upstream clones (open-instruct, OLMo-core, olmes)
    evaluate_and_compare.py  shared evaluation across training stages
    sft/             reusable SFT preparation and training
    dpo/             reusable preference preparation and DPO training
    rlvr/            math data preparation, verification, and GRPO training
    scripts/         shell launchers, hardware diagnostics, and standalone utilities
    configs/         our config overrides
    tests/           automated tests grouped under sft/, dpo/, and rlvr/
