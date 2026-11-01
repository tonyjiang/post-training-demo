# Olmo 3 Think 7B post-training demo

A learning exercise: we retrace how Ai2 turned its Olmo 3 7B base model into
**Olmo 3 7B Think**, one stage at a time: **base model → SFT → DPO → RLVR**.
SFT, DPO, and the 1,024- and 2,048-token evaluation runs are complete; RLVR comes next.

## What we build on

Everything we start from is released by Ai2 (`allenai` on Hugging Face). For each
stage we take a small sample of Ai2's dataset, because our single 12 GB GPU can't
train on the full sets, and compare our result with Ai2's own checkpoint for that stage.

| Stage | Ai2 dataset | Our sample (train / held-out) | Ai2 checkpoint to compare against |
|---|---|---|---|
| Base | — | — | [Olmo-3-1025-7B](https://huggingface.co/allenai/Olmo-3-1025-7B) (our starting point) |
| SFT | [Dolci-Think-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B) | 4,241 / 64 conversations | [Olmo-3-7B-Think-SFT](https://huggingface.co/allenai/Olmo-3-7B-Think-SFT) |
| DPO | [Dolci-Think-DPO-7B](https://huggingface.co/datasets/allenai/Dolci-Think-DPO-7B) | 1,024 / 64 preference pairs | [Olmo-3-7B-Think-DPO](https://huggingface.co/allenai/Olmo-3-7B-Think-DPO) |
| RLVR | [Dolci-Think-RL-7B](https://huggingface.co/datasets/allenai/Dolci-Think-RL-7B) | 1,024 / 128 math prompts | [Olmo-3-7B-Think](https://huggingface.co/allenai/Olmo-3-7B-Think) |

We reuse Ai2's code and recipes from [OLMo-core](https://github.com/allenai/OLMo-core),
[open-instruct](https://github.com/allenai/open-instruct) and
[olmes](https://github.com/allenai/olmes) wherever our hardware allows; every place we
had to differ is listed under [Deviations and validation](#deviations-and-validation).
Model, dataset, chat-template and recipe revisions are pinned in [configs/](configs/);
RLVR dataset, upstream recipe, verifier sources, and official reference checkpoint
are pinned in `configs/rlvr-data.json` and `configs/rlvr.json`.

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
uv run --locked python -m sft.prepare
uv run --locked python -m sft.sample_sources --exclude-data data/dolci-small
uv run --locked python -m sft.prepare_combined
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

## DPO

DPO continues `outputs/sft/final-adapter` using chosen/rejected conversations from
[Dolci-Think-DPO-7B](https://huggingface.co/datasets/allenai/Dolci-Think-DPO-7B).
The policy and a frozen copy of the **SFT adapter** share one NF4 base. Reference
scores are cached before any updates; disabling the SFT adapter would incorrectly
use the base model as the reference. The final DPO adapter replaces the SFT adapter
at inference and uses the same original base weights.

The default experiment samples **1,024 training pairs and 64 held-out pairs** from
one pinned shard. IDs and exact conversation prompts are unique across splits.
Both branches use the official SFT chat template and are right-truncated to
**2,048 tokens including the prompt**. Empty or identical retained responses are
dropped. Truncation can remove final answers and weaken the preference signal;
the data manifest records exclusions and truncated-branch counts.

A [random 10-pair length audit](reports/dpo-data/random-10-pairs.json) (seed
20261031) found chosen responses longer in 7/10 original pairs, but 5/10 after
training truncation (two ties). Across all 1,024 retained training pairs, chosen
responses average **1,252 tokens versus 962 rejected** (30% longer): 645 chosen
longer, 139 rejected longer, 240 ties. Counts exclude the shared prompt and include
reasoning and response end tokens. This supports a response-length imbalance in
the training data, but does not establish why the trained model became more verbose.


| Setting | Value |
|---|---|
| Objective | Length-normalized DPO (`sigmoid_norm` in TRL, `dpo_norm` in Ai2), beta 5 |
| Trainable weights | Continue our rank-8 SFT adapter; FP32 adapter/optimizer weights, BF16 compute |
| Batch | One preference pair (two sequences); accumulate over 8 pairs |
| Schedule | One epoch; peak LR `1e-5`, 10% warmup, linear decay, no weight decay |
| Memory | NF4 base, SDPA, gradient checkpointing, TRL's chunked fused LM head |
| Evaluation/checkpoints | Every 25 updates plus final evaluation; keep two checkpoints |

The LR is an untuned adapter-training starting point. Override it with
`--learning-rate`, `--warmup-ratio`, `--lr-scheduler`, or `--beta`.
[configs/dpo.json](configs/dpo.json) records the pinned Ai2 recipe, official
reference checkpoints, experiment defaults, and deviations.

Run from the repository root on the server:

```bash
uv sync --locked --group dpo
uv run --locked python scripts/download_base.py \
  --manifest configs/dpo-data.json --output data/dolci-dpo-shards
uv run --locked --group dpo python -m dpo.prepare
bash scripts/start_dpo.sh
```

Use `--endpoint https://hf-mirror.com` for the downloader if needed. Preparation
and the launcher refuse to overwrite existing outputs. The detached launcher
writes to `outputs/dpo/`; inspect `train.log`, `exit_code`, and `all_results.json`.
The SFT, DPO, and evaluation launchers share the same GPU lock. Successful runs
save `final-adapter/`, cached SFT reference scores, and resumable checkpoints.

For a short **real-data compatibility test**, prepare a separate dataset and run:

```bash
uv run --locked --group dpo python -m dpo.prepare \
  --train-size 4 --eval-size 2 --output data/dolci-dpo-smoke
bash scripts/start_dpo.sh --run-name dpo-smoke --data data/dolci-dpo-smoke \
  --max-steps 2 --gradient-accumulation 1 --save-steps 1
```

Resume into a new run directory with the same data and training arguments, adding
`--resume-from-checkpoint /absolute/path/to/checkpoint-N`. The original SFT adapter
must remain available: it defines the fixed reference even when resuming.

Run the DPO checks with `uv run --locked --group dpo python -m unittest discover -s tests/dpo -t . -v`.
See [Tests](#tests) for the complete suite.

The first subset run completed successfully on **2026-10-31** on `gaming-server`:
**1,024 pairs, one epoch, 128 optimizer updates in 1h 37m** including reference
caching and final evaluation. Exit status was `0`, final held-out DPO loss was
**0.4118**, and peak allocated VRAM was **10.19 GiB**. The frozen SFT reference
remained unchanged. [Full run metrics](reports/dpo/all_results.json).

The saved adapter is `/root/post-training-demo/outputs/dpo/final-adapter/`;
resumable checkpoints are `checkpoint-125/` and `checkpoint-128/` in the same run
directory. Its SHA-256 matches the recorded final adapter hash. The GPU is now idle.
Small SFT-versus-DPO task evaluations are complete (below). Comparison to Ai2's
official `Olmo-3-7B-Think-DPO` checkpoint remains pending.

Validation on `gaming-server`: a real-data **two-update 2K smoke test passed**,
including evaluation and checkpoint saves, at **9.93 GiB peak allocated VRAM**.
The SFT reference stayed unchanged, and the final adapter reloaded and generated
successfully. Seven CPU tests cover data integrity, the normalized objective,
frozen reference/base weights, adapter reload, and checkpoint continuation.
[Smoke metrics](reports/dpo-smoke/all_results.json) ·
[Prepared 1,024/64-pair dataset](reports/dpo-data/data-manifest.json).
GPU resume from update 2 completed update 3 with the same cached reference scores
and an unchanged SFT reference. Replay was not bit-identical: parameter RMS
difference `1.31e-8`, maximum `3.58e-6` versus the uninterrupted run
([comparison](reports/dpo-resume/comparison.json)).
This validates execution, not a quality improvement.

## Next experiments: RLVR and human-feedback RLHF

RLVR is implemented in `rlvr/`, continuing `outputs/dpo/final-adapter` with
NF4 QLoRA and TRL 1.17.1 GRPO. The prepared math-only dataset has **1,024 training
and 128 held-out prompts** from the first pinned Dolci shard. Prompts are sampled
with seed 42, deduplicated, checked against the fixed benchmark prompts, and
dropped if over 512 tokens; they are never truncated. The manifest records the
three source families and all exclusions. Ground-truth answers are used only by
the reward function, never included in model inputs.

Each prompt receives **four sequential sampled rollouts**, temperature 1.0, up to
**2,048 completion tokens**. Updates use microbatch 1, accumulation 4, constant
LR `1e-6`, no weight decay, and beta 0 (no reference/KL term), following those parts
of the pinned Ai2 recipe. The DPO adapter remains preserved as the parent.
An answer earns 1 only when thinking closes, a boxed final answer matches using
Ai2's pinned math equivalence helpers, and the completion is not token-truncated.
Truncated completions are also masked from the GRPO loss. Reward and rollout
records make zero-variance groups and absence of learning visible. A pilot stops
early after eight updates without any mixed-reward group rather than spending
the remaining budget on zero-advantage updates.

The default run is a bounded **16-update pilot**, not a full pass over all 1,024
training prompts. It uses four fixed held-out prompts for greedy before/after
checks; the complete 128-prompt split remains separate. This is a compatibility
and learning-signal experiment, not a reliable quality benchmark. `--max-steps -1`
selects a full epoch (1,024 optimizer updates / 4,096 sampled completions), which
must be budgeted using measured pilot runtime. New run names preserve old results.

```bash
uv sync --locked --group rlvr --group eval
uv run --locked python scripts/download_base.py \
  --manifest configs/rlvr-data.json --output data/dolci-rlvr-shards
uv run --locked --group rlvr python -m rlvr.prepare
bash scripts/start_rlvr.sh rlvr-smoke-2k --max-steps 1 --eval-size 1
# After smoke validation:
bash scripts/start_rlvr.sh rlvr-pilot
```

Preparation and launch refuse to overwrite data or outputs. Runs share the SFT/DPO
GPU lock and save `exit_code`, `all_results.json`, `final-adapter/`, rollout logs,
before/after predictions, configs, and checkpoints every eight updates. Inspect
`mixed_reward_groups` and `policy_changed` before claiming that learning occurred.
A pinned upstream snapshot was exported unchanged from the local clone for the
server's math helpers and recipe, avoiding a slow full-network Git clone.
[Data manifest](reports/rlvr-data/data-manifest.json).

Real-model validation: the one-update 2K smoke test completed with exit status 0
in **6m 10s**, at **8.63 GiB peak allocated VRAM**. All four sampled answers were
truncated and received zero reward; the policy did not change. This validates
execution and memory fit, not learning. The tiny CPU integration test separately
verified a nonzero adapter update from mixed rewards with frozen base weights.
The implementation recognizes both answer-ending tokens for generation and
truncation masking. All 22 tests passed, with the five RLVR checks rechecked
before pilot launch. [Smoke metrics](reports/rlvr-smoke-2k/all_results.json).

The detached pilot is launched as `outputs/rlvr-pilot/`, capped at 16 updates
(four rollouts per prompt), stopping after eight updates if no mixed-reward group
appears. Monitor `train.log`; completion requires `exit_code` equal to 0, final
metrics, and `final-adapter/`. Review `mixed_reward_groups`, `policy_changed`, and
`stopped_for_no_learning_signal` before expanding the run. A full epoch is **not**
currently scheduled. The smoke timing suggests roughly an hour or more for the
pilot, and multiple days for a full 1,024-prompt epoch at these settings.


For the subsequent RLHF experiment, use the human `preference` subset of
`nvidia/HelpSteer3`: a proposed **1,024 training pairs plus 128 validation pairs**,
covering General, STEM, Code, and Multilingual domains. Use the official splits,
check for shared prompts, preserve preference strength, and audit response lengths
and truncation. Validate the reward model on held-out human preferences before
using it to train the policy. This is a pilot size, not evidence of sufficient data
for general quality improvements; no HelpSteer3 data has been sampled yet.

Compare independent RLVR and RLHF branches from the same DPO adapter, then run
RLHF on the RLVR adapter as a combined experiment. This additional human-feedback
experiment extends the original Olmo reproduction scope at the user's request.

## Small evaluations

Compare the frozen base with the SFT adapter on the same **16 [GSM8K](https://huggingface.co/datasets/openai/gsm8k) math problems
and 16 [IFEval](https://huggingface.co/datasets/HuggingFaceH4/ifeval) instruction-following prompts**. Samples are seeded and checked for
exact prompt overlap with retained SFT text. The initial runs used the same chat template,
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

DPO evaluation completed on **2026-10-31** in **13m 41s**, with exit status `0`.
The data hash matches the SFT run, and all 32 regenerated base predictions are
identical to the earlier run.

| Evaluation | SFT | DPO |
|---|---:|---:|
| GSM8K correct | 13/16 | 10/16 |
| IFEval strict prompt passes | 8/16 | 9/16 |
| IFEval individual instruction passes | 13/22 | 13/22 |
| GSM8K token-limit hits | 2/16 | 6/16 |
| IFEval token-limit hits | 8/16 | 13/16 |

All three newly failed math prompts hit the 1,024-token generation limit under
DPO. One computes the correct answer (83 boxes), but keeps rechecking until cut
off before a final boxed answer. IFEval has two prompt gains and one regression;
its aggregate individual-instruction score is unchanged. These results show a
math regression under the current budget and more truncation, not proof of lost
math ability or a broad instruction-following gain. A longer-budget paired run
is needed to distinguish unfinished answers from reasoning failures.
[DPO results](reports/evals-dpo/results.json) ·
[DPO generated answers](reports/evals-dpo/predictions.jsonl)

**2,048-token evaluation rerun completed successfully** on `gaming-server` on
2026-11-01 at approximately 01:18 UTC (October 31 Pacific). Both stages and the
pipeline exited `0`. The initial batch-2 attempt exhausted GPU memory after
12 SFT prompts; restarting at batch size 1 preserved the 2,048-token output budget.
SFT took 14m 24s and DPO 19m 42s, peaking at **8.55 GiB allocated VRAM**.
Five-minute monitoring checks found no further errors. The GPU is now idle.

| Metric | SFT, 1,024 output tokens | DPO, 1,024 | SFT, 2,048 | DPO, 2,048 |
|---|---:|---:|---:|---:|
| GSM8K correct | 13/16 | 10/16 | 13/16 | 12/16 |
| IFEval prompts passed | 8/16 | 9/16 | 7/16 | 7/16 |
| Individual instructions passed | 13/22 | 13/22 | 12/22 | 13/22 |
| Math responses hitting output cap | 2/16 | 6/16 | 2/16 | 4/16 |
| IFEval responses hitting output cap | 8/16 | 13/16 | 6/16 | 6/16 |

Both new runs use the same 32 prompts, batch size 1, greedy decoding, and
4,096-token total context. Historical runs used batch size 2: differences from
those runs include batching as well as output budget. Some previously completed
answers also changed. The new comparison shows DPO one math answer behind SFT,
with equal strict IFEval prompt scores; it does not establish a broad DPO gain.
DPO still generates more tokens overall. Samples remain too small for strong
claims, and some responses still reach the output cap.

The evaluator runs SFT alone, then reuses its validated predictions for DPO.
No base generation or training was performed. The default generation budget is
now 2,048; `--max-new-tokens` overrides it. Effective settings are recorded
separately from the unchanged prepared-data manifest.

- SFT: `outputs/evals-sft-2k-b1/`, log `logs/evals-sft-2k-b1.log`.
- DPO plus cached SFT: `outputs/evals-dpo-2k-b1/`, log `logs/evals-dpo-2k-b1.log`.
- Pipeline status: `logs/evals-2k-b1.exit_code`; per-stage status:
  `logs/evals-{sft,dpo}-2k-b1.exit_code`.

[2K comparison results](reports/evals-dpo-2k-b1/results.json) ·
[2K generated answers](reports/evals-dpo-2k-b1/predictions.jsonl) ·
[Validation](reports/evals-dpo-2k-b1/validation.json).
All previous results are preserved. Ten scorer/cache tests passed before launch;
final validation checked all 64 predictions, matching prompts, cached SFT answers,
score aggregates, effective configs, adapter hashes, and exit statuses.

<details>
<summary>Set up and run evaluations on the server</summary>

```bash
uv sync --locked --group eval
git clone https://github.com/allenai/olmes.git third_party/olmes
git -C third_party/olmes checkout 5a51f502d463b8cdc4a2dcad7d7096c41ff1197e
uv run --locked --group eval python -m nltk.downloader -d data/eval-sources/nltk punkt punkt_tab
uv run --locked --group eval python -m sft.prepare_evals
mkdir -p logs
nohup bash scripts/run_evals.sh --compare-base --adapter-name sft > logs/evals.log 2>&1 < /dev/null &
```

Preparation refuses to overwrite the sampled data. Evaluation refuses to overwrite
`outputs/evals/`; use `--output outputs/evals-v2` for a new run.

</details>

Scores and every generated answer go to `outputs/evals/results.json` and
`outputs/evals/predictions.jsonl`. Progress is in `logs/evals.log`; completion status
is in `logs/evals.exit_code` (`0` means success). Token-limit hits are reported.
Revisions, source hashes and settings are pinned in [configs/eval-small.json](configs/eval-small.json).

To evaluate a new DPO adapter against saved SFT answers, run on the server:

```bash
mkdir -p logs
nohup bash scripts/run_evals.sh \
  --adapter outputs/dpo/final-adapter \
  --adapter-name dpo \
  --baseline-results outputs/evals-sft-2k-b1 --baseline-name sft \
  --max-new-tokens 2048 \
  --output outputs/evals-dpo-new \
  > logs/evals-dpo-new.log 2>&1 < /dev/null &
```

The launcher invokes `evaluate_and_compare`. Choose `--baseline-results` to
reuse saved answers, or explicitly choose `--compare-base` to generate base answers.
DPO comparisons should reuse SFT results with the same output budget; only the
new adapter generates answers. Use `--adapter-only` to establish a new SFT baseline
without generating base answers. Cached results must match the prompts, generation and
scoring config, base revision, chat template, and recorded package versions.
Changing the token budget requires new evaluations of both adapters.

The original 1,024-token runs are complete; compare them without loading a model:

```bash
uv run --locked --group eval python -m evaluate_and_compare \
  --baseline-results outputs/evals --baseline-name sft \
  --candidate-results outputs/evals-dpo --adapter-name dpo \
  --output outputs/sft-vs-dpo
```

[Saved SFT-versus-DPO comparison](reports/sft-vs-dpo/results.json) reuses both
completed runs. Comparison reports record source file hashes and never overwrite
existing outputs. Keep each path on one line; use a trailing backslash to continue
a shell command.


## Code layout

Reusable stage code lives in the `sft/`, `dpo/`, and `rlvr/` Python packages. Shared
evaluation lives in the root-level `evaluate_and_compare.py`. `scripts/` holds shell
launchers, hardware diagnostics, and the standalone download utility; `configs/` holds pinned inputs.
Automated tests are grouped by stage under `tests/sft/`, `tests/dpo/`, and `tests/rlvr/`.
Run SFT modules from the repository root with `uv run --locked python -m sft.<module>`
(add `--group eval` to `uv run` for evaluation modules).

| File | Purpose |
|---|---|
| [download_base.py](scripts/download_base.py) | Download pinned model files or dataset shards; resume downloads and verify hashes. |
| [sft/prepare.py](sft/prepare.py) | Prepare short conversation splits and assistant loss masks; provide shared tokenizer helpers. |
| [sft/sample_sources.py](sft/sample_sources.py) | Sample conversations across source families, exclude specified IDs and remove duplicates. |
| [sft/profile_sample.py](sft/profile_sample.py) | Report token lengths and context-limit coverage by source family. |
| [sft/prepare_combined.py](sft/prepare_combined.py) | Assemble the training dataset, apply truncation, shuffle records and retain the held-out split. |
| [start_sft.sh](scripts/start_sft.sh) | Validate inputs and launch detached training in a named directory; refuse to overwrite results. |
| [run_sft.sh](scripts/run_sft.sh) | Set the runtime environment, prevent concurrent SFT jobs and record the exit status. |
| [sft/train.py](sft/train.py) | Train and evaluate QLoRA adapters; save checkpoints and metrics; support continuation and resume. |
| [scripts/smoke_nf4.py](scripts/smoke_nf4.py) | Test quantized loading, synthetic training, adapter saving/reloading and generation. |
| [run_smoke.sh](scripts/run_smoke.sh) | Run the synthetic training and reload checks in separate processes. |
| [scripts/probe_sft_context.py](scripts/probe_sft_context.py) | Test GPU memory fit at different sequence lengths with disposable adapter updates. |
| [sft/prepare_evals.py](sft/prepare_evals.py) | Download pinned benchmarks, sample evaluation prompts and exclude exact SFT matches. |
| [evaluate_and_compare.py](evaluate_and_compare.py) | Compare base and adapter; save answers, scores, token-limit hits and runtime. |
| [run_evals.sh](scripts/run_evals.sh) | Run evaluation with the GPU lock and record its exit status. |
| [run_evals_2k.sh](scripts/run_evals_2k.sh) | Evaluate SFT then DPO with 2,048 output tokens, reusing SFT results and preserving old runs. |
| [rlvr/prepare.py](rlvr/prepare.py) | Prepare the pinned, disjoint math prompt splits. |
| [rlvr/rewards.py](rlvr/rewards.py) | Verify boxed final answers with pinned Ai2 math helpers. |
| [rlvr/train.py](rlvr/train.py) | Run sequential-rollout QLoRA GRPO with before/after checks and auditable rewards. |
| [start_rlvr.sh](scripts/start_rlvr.sh) | Validate and launch a detached named RLVR pilot. |
| [run_rlvr.sh](scripts/run_rlvr.sh) | Run RLVR under the shared GPU lock and record exit status. |
| [dpo/data.py](dpo/data.py) | Validate preference pairs, tokenize/truncate both branches, and verify prepared splits. |
| [dpo/prepare.py](dpo/prepare.py) | Sample pinned Dolci preference data and write disjoint tokenized splits. |
| [dpo/train.py](dpo/train.py) | Continue the SFT adapter with TRL DPO, a frozen SFT reference, checkpoints, and metrics. |
| [start_dpo.sh](scripts/start_dpo.sh) | Validate DPO inputs and launch a detached named run. |
| [run_dpo.sh](scripts/run_dpo.sh) | Run DPO under the shared GPU lock and record exit status. |

Training call chain: `scripts/start_sft.sh` → `scripts/run_sft.sh` → `python -m sft.train`.

## Tests

Run from the repository root. The suite uses CPU checks and a tiny model; it does
not run the 7B training jobs. `-t .` keeps the test packages separate from the
production `sft` and `dpo` packages during discovery.

```bash
# All automated tests (requires the OLMES checkout and NLTK data prepared above).
uv run --locked --group dpo --group rlvr --group eval python -m unittest discover -s tests -t . -v

# One stage at a time.
uv run --locked --group eval python -m unittest discover -s tests/sft -t . -v
uv run --locked --group dpo python -m unittest discover -s tests/dpo -t . -v
```

`tests/sft/test_evaluate.py` holds the math and instruction-scoring checks previously
embedded in the evaluator. `tests/dpo/` holds preference-data and
trainer integration checks. Tests requiring unavailable optional dependencies,
the OLMES checkout, or cached NLTK data are reported as skipped; discovery does
not download resources.

Hardware diagnostics remain explicit server commands:

```bash
bash scripts/run_smoke.sh
uv run --locked python -m scripts.probe_sft_context \
  --adapter outputs/sft/final-adapter --liger --lengths 2048 4096 \
  --output outputs/sft-context-probe-new
```

## Deviations and validation

Ai2 uses full-parameter OLMo-core SFT with packing and up to 32K context. We use
Transformers + bitsandbytes + PEFT, QLoRA, a sampled source mixture, and 4K truncation
to fit our GPU. The official SFT chat template is retained.

The small evaluation runner reuses pinned OLMES instruction checkers and reasoning-output
processing. OLMES's full dependency set requires older Torch/Transformers than our runtime;
we use our existing NF4 loader, small subsets and stricter boxed-number math scoring.
These settings differ from Ai2's full benchmark runs.

DPO follows the length-normalized objective, beta, warmup fraction, and linear
schedule in the [pinned Ai2 recipe](https://github.com/allenai/open-instruct/blob/5db7f11bf7f34596ba9af4cfe5ade1e2dbff8cce/scripts/train/olmo3/7b_think_dpo.sh)
and its configuration defaults. We use TRL 1.17.1 instead of open-instruct's full
DeepSpeed/Beaker runtime: the upstream QLoRA path initializes a fresh adapter and
does not directly continue our saved SFT adapter. TRL supplies the loss, frozen
reference adapter, and memory-efficient LM head; our thin wrapper consumes the
already validated tokenized pairs. The 12 GB experiment uses QLoRA, a one-shard
subset, 2K truncation, an effective batch of 8 pairs, and peak LR `1e-5` instead
of Ai2's full-parameter `8e-8`, 16K context, and 150K-pair run. These are recorded
deviations, not a claim of equivalent model quality.
The wrapper also restores the policy adapter explicitly on resume: Transformers
5.20.2 otherwise sees the saved `ref/` directory and skips the policy at the
checkpoint root. The original SFT reference is retained.

A two-update 4K smoke test passed training, evaluation and checkpoint saves at
**10.10 GiB peak allocated VRAM**. [Validation reports](reports/sft-combined-4k/)
record the results; they establish execution compatibility, not model quality.

Next: inspect the bounded RLVR pilot for reward variation and policy updates, then decide whether a full epoch is justified.

RLVR deviations: TRL replaces the distributed open-instruct/vLLM/Beaker runtime;
we use QLoRA, the original GRPO loss, a first-shard math-only subset, four serial
rollouts, a 512-token prompt ceiling and 2,048-token completion cap instead of
Ai2's full mixture, eight rollouts and 32K completions. We add a boxed-answer
instruction and require closed thinking, with zero reward and loss masking for
truncation. These changes fit the hardware and define a narrower experiment;
they do not reproduce the full official RL recipe.
