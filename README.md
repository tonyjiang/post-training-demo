# Olmo 3 Think 7B post-training demo

A learning exercise following Ai2's pipeline: **base model → SFT → DPO → RLVR**.
SFT is implemented with QLoRA; DPO and RLVR are planned.

## Hardware and setup

All training must fit our **single RTX 5070 (12 GB VRAM)** on `gaming-server`.
The Linux workspace is `/root/post-training-demo`; model weights are in
`/root/models/Olmo-3-1025-7B` (13.60 GiB). Hardware details are in [AGENTS.md](AGENTS.md).

On the server:

```bash
cd /root/post-training-demo
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked
nvidia-smi
df -h /root
```

Dependencies are pinned in `pyproject.toml` and `uv.lock`. Create the Linux `.venv`
on the server; don't copy the Mac environment.

## Dataset

We use **4,241 training conversations and 64 held-out conversations** sampled from
[Dolci Think SFT 7B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B).
The prepared dataset is `data/dolci-combined-4k/` on the server.

| Source | Training | Held-out |
|---|---:|---:|
| Aya | 494 | 20 |
| CoCoNot | 250 | 0 |
| Dolci Precise IF | 501 | 15 |
| Dolci Think Persona IF | 250 | 0 |
| Dolci Think Python | 503 | 13 |
| Nemotron code | 250 | 0 |
| OpenAssistant Guanaco | 250 | 0 |
| OpenThoughts 3 | 250 | 0 |
| SYNTHETIC-2 | 250 | 0 |
| TableGPT | 250 | 0 |
| WildChat | 493 | 16 |
| WildGuardMix | 250 | 0 |
| WildJailbreak | 250 | 0 |
| **Total** | **4,241** | **64** |

Whole conversations are shuffled across sources each epoch; message order stays intact.
Sequences are capped at **4,096 tokens**. Truncation can cut off reasoning or final answers;
examples with no remaining assistant targets are excluded. Only assistant tokens contribute
to the loss. Train and evaluation IDs are disjoint; evaluation covers four source families.
Exact labels, counts and hashes are in the [data manifest](reports/sft-combined-4k/data-manifest.json).

<details>
<summary>Rebuild the dataset and download weights</summary>

Run once on the server when preparing a new workspace:

```bash
uv run --locked python scripts/download_base.py
uv run --locked python scripts/download_base.py \
  --manifest configs/sft-data.json --output data/dolci-shards
uv run --locked python scripts/download_base.py \
  --manifest configs/sft-source-shards.json --output data/dolci-source-shards
uv run --locked python scripts/prepare_sft.py
uv run --locked python scripts/sample_sft_sources.py --exclude-data data/dolci-small
uv run --locked python scripts/prepare_combined_sft.py
```

If Hugging Face downloads time out, add `--endpoint https://hf-mirror.com` to the
download commands. Files are checked against pinned hashes. Sampling and final preparation
refuse to overwrite existing output directories.

</details>

## Training

| Setting | Value |
|---|---|
| Base model | Frozen NF4 weights, quantized during loading; BF16 compute |
| Trainable weights | Rank-8 LoRA adapters on all linear layers, about 20M parameters |
| Batch | One conversation at a time; accumulate gradients over 8 conversations per update |
| Duration | One epoch, 531 optimizer updates |
| Optimizer | AdamW; peak learning rate 0.0002; 27 warmup updates, then cosine decay; no weight decay |
| Memory controls | Gradient checkpointing, SDPA attention, Liger fused loss |
| Checkpoints and evaluation | Every 25 updates; keep the latest two checkpoints |

The completed SFT is in `/root/post-training-demo/outputs/sft/` on the server:

```text
outputs/
└── sft/
    ├── final-adapter/            # Trained adapter for inference, with the base model
    ├── checkpoint-525/           # Full training state for resuming
    ├── checkpoint-531/           # Full training state at the final update
    ├── snapshots/checkpoint-100/ # Preserved adapter for intermediate evaluation
    └── ...                      # Logs, metrics, configs and exit status
```

Only the latest two resumable checkpoints are retained; older ones are automatically
removed. The step-100 snapshot was saved separately and has no optimizer state.

SFT completed successfully on October 22, 2026: **531 updates in 3h 13m**, final
held-out loss **0.9288**, peak allocated VRAM **10.13 GiB** ([metrics](reports/sft/all_results.json)).

To launch on a new workspace **from the Mac**:

```bash
ssh gaming-server 'cd /root/post-training-demo && bash scripts/start_sft.sh --data data/dolci-combined-4k --liger-fused-loss'
```

The launcher defaults to `outputs/sft/` and refuses to overwrite existing results.
For another experiment, explicitly add `--run-name sft-v2`.
It validates inputs and prints a PID and run directory. You can then close
SSH, your terminal and your laptop. The server must stay on; jobs do not restart after
a reboot. `--liger-fused-loss` reduces temporary memory used to calculate training loss.

On the server, use the printed run directory:

```bash
RUN_DIR=/root/post-training-demo/outputs/sft
tail -f "$RUN_DIR/train.log"       # Ctrl-C only stops viewing the log.
cat "$RUN_DIR/exit_code"          # 0 means success; written when the job exits.
cat "$RUN_DIR/all_results.json"   # Loss, runtime, throughput and peak VRAM.
```

`process_runtime_seconds` includes loading, training, saving and final evaluation;
`train_runtime` covers the training call. The trained adapter is saved in
`final-adapter/` and requires the base model for inference.

To resume an interrupted run on the same dataset, run on the server:

```bash
bash scripts/start_sft.sh --run-name sft-resumed --data data/dolci-combined-4k --liger-fused-loss \
  --resume-from-checkpoint /absolute/path/to/checkpoint-N
```

## Small evaluations

Compare the frozen base with the SFT adapter on the same **16 [GSM8K](https://huggingface.co/datasets/openai/gsm8k) math problems
and 16 [IFEval](https://huggingface.co/datasets/HuggingFaceH4/ifeval) instruction-following prompts**. Samples are seeded and checked for
exact prompt overlap with retained SFT text. Both use the same chat template,
NF4 loading and greedy decoding, capped at 1,024 generated tokens.

These are learning-exercise scores: GSM8K checks the final boxed number; IFEval uses
Ai2's strict instruction checkers. Small samples, short generations and chat prompting
limit conclusions about general capability or the base model's best performance.

| Evaluation (16 prompts each) | Base | SFT adapter |
|---|---:|---:|
| GSM8K: correct boxed answer | 13/16 | 13/16 |
| IFEval: all instructions followed strictly | 5/16 | 8/16 |

The comparison finished in **12m 21s**, peaking at **8.75 GiB allocated VRAM**.
Token limits affected 2/16 math responses for each model and 9/16 versus 8/16
instruction responses, so these scores need that qualification.
[Results](reports/evals/results.json) · [Every generated answer](reports/evals/predictions.jsonl)

<details>
<summary>Set up and run evaluations on the server</summary>

```bash
uv sync --locked --group eval
git clone https://github.com/allenai/olmes.git third_party/olmes
git -C third_party/olmes checkout 5a51f502d463b8cdc4a2dcad7d7096c41ff1197e
uv run --locked --group eval python -m nltk.downloader -d data/eval-sources/nltk punkt punkt_tab
uv run --locked --group eval python scripts/prepare_evals.py
mkdir -p logs
nohup bash scripts/run_evals.sh > logs/evals.log 2>&1 < /dev/null &
```

Preparation refuses to overwrite the sampled data. Evaluation refuses to overwrite
`outputs/evals/`; use `--output outputs/evals-v2` for a new run.

</details>

Scores and every generated answer go to `outputs/evals/results.json` and
`outputs/evals/predictions.jsonl`. Progress is in `logs/evals.log`; completion status
is in `logs/evals.exit_code` (`0` means success). Token-limit hits are reported.
Revisions, source hashes and settings are pinned in [configs/eval-small.json](configs/eval-small.json).

## Script reference

| Script | Purpose |
|---|---|
| [download_base.py](scripts/download_base.py) | Download pinned model files or dataset shards; resume downloads and verify hashes. |
| [prepare_sft.py](scripts/prepare_sft.py) | Prepare short conversation splits and assistant loss masks; provide shared tokenizer helpers. |
| [sample_sft_sources.py](scripts/sample_sft_sources.py) | Sample conversations across source families, exclude specified IDs and remove duplicates. |
| [profile_sft_sample.py](scripts/profile_sft_sample.py) | Report token lengths and context-limit coverage by source family. |
| [prepare_combined_sft.py](scripts/prepare_combined_sft.py) | Assemble the training dataset, apply truncation, shuffle records and retain the held-out split. |
| [start_sft.sh](scripts/start_sft.sh) | Validate inputs and launch detached training in a named directory; refuse to overwrite results. |
| [run_sft.sh](scripts/run_sft.sh) | Set the runtime environment, prevent concurrent SFT jobs and record the exit status. |
| [train_sft.py](scripts/train_sft.py) | Train and evaluate QLoRA adapters; save checkpoints and metrics; support continuation and resume. |
| [smoke_nf4.py](scripts/smoke_nf4.py) | Test quantized loading, synthetic training, adapter saving/reloading and generation. |
| [run_smoke.sh](scripts/run_smoke.sh) | Run the synthetic training and reload checks in separate processes. |
| [probe_sft_context.py](scripts/probe_sft_context.py) | Test GPU memory fit at different sequence lengths with disposable adapter updates. |
| [prepare_evals.py](scripts/prepare_evals.py) | Download pinned benchmarks, sample evaluation prompts and exclude exact SFT matches. |
| [evaluate_sft.py](scripts/evaluate_sft.py) | Compare base and adapter; save answers, scores, token-limit hits and runtime. |
| [run_evals.sh](scripts/run_evals.sh) | Run evaluation with the GPU lock and record its exit status. |

Training call chain: `start_sft.sh` → `run_sft.sh` → `train_sft.py`.

## Ai2 artifacts

All artifacts below are from `allenai` on Hugging Face.

| Stage | Dataset | Official checkpoint |
|---|---|---|
| Base | — | [Olmo-3-1025-7B](https://huggingface.co/allenai/Olmo-3-1025-7B) |
| SFT | [Dolci-Think-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B) | [Olmo-3-7B-Think-SFT](https://huggingface.co/allenai/Olmo-3-7B-Think-SFT) |
| DPO | [Dolci-Think-DPO-7B](https://huggingface.co/datasets/allenai/Dolci-Think-DPO-7B) | [Olmo-3-7B-Think-DPO](https://huggingface.co/allenai/Olmo-3-7B-Think-DPO) |
| RLVR | [Dolci-Think-RL-7B](https://huggingface.co/datasets/allenai/Dolci-Think-RL-7B) | [Olmo-3-7B-Think](https://huggingface.co/allenai/Olmo-3-7B-Think) |

Model, SFT data and chat-template revisions are pinned in [configs/](configs/).
Pin DPO and RLVR revisions before using them. Upstream code:
[OLMo-core](https://github.com/allenai/OLMo-core),
[open-instruct](https://github.com/allenai/open-instruct),
[olmes](https://github.com/allenai/olmes).

## Deviations and validation

Ai2 uses full-parameter OLMo-core SFT with packing and up to 32K context. We use
Transformers + bitsandbytes + PEFT, QLoRA, a sampled source mixture, and 4K truncation
to fit our GPU. The official SFT chat template is retained.

The small evaluation runner reuses pinned OLMES instruction checkers and reasoning-output
processing. OLMES's full dependency set requires older Torch/Transformers than our runtime;
we use our existing NF4 loader, small subsets and stricter boxed-number math scoring.
These settings differ from Ai2's full benchmark runs.

A two-update 4K smoke test passed training, evaluation and checkpoint saves at
**10.10 GiB peak allocated VRAM**. [Validation reports](reports/sft-combined-4k/)
record the results; they establish execution compatibility, not model quality.

Next: compare with the official SFT checkpoint, then implement DPO and RLVR.
