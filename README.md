# Olmo 3 Think 7B post-training demo

A learning exercise: we retrace how Ai2 turned its Olmo 3 7B base model into
**Olmo 3 7B Think**, one stage at a time: **base model → SFT → DPO → RLVR**.
SFT, DPO, and the 1,024- and 2,048-token evaluation runs are complete; the ten-update three-answer WildChat trial is complete; RLVR is stopped at checkpoint-32 after the interactive judge timed out in `outputs/rlvr-3gen-full-discard-cutoffs`.

## What we build on

Everything we start from is released by Ai2 (`allenai` on Hugging Face). For each
stage we take a small sample of Ai2's dataset, because our single 12 GB GPU can't
train on the full sets, and compare our result with Ai2's own checkpoint for that stage.

| Stage | Ai2 dataset | Our sample (train / held-out) | Ai2 checkpoint to compare against |
|---|---|---|---|
| Base | — | — | [Olmo-3-1025-7B](https://huggingface.co/allenai/Olmo-3-1025-7B) (our starting point) |
| SFT | [Dolci-Think-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B) | 4,241 / 64 conversations | [Olmo-3-7B-Think-SFT](https://huggingface.co/allenai/Olmo-3-7B-Think-SFT) |
| DPO | [Dolci-Think-DPO-7B](https://huggingface.co/datasets/allenai/Dolci-Think-DPO-7B) | 1,024 / 64 preference pairs | [Olmo-3-7B-Think-DPO](https://huggingface.co/allenai/Olmo-3-7B-Think-DPO) |
| RLVR | [Dolci-Think-RL-7B](https://huggingface.co/datasets/allenai/Dolci-Think-RL-7B) | 1,024 / 128 WildChat English conversations | [Olmo-3-7B-Think](https://huggingface.co/allenai/Olmo-3-7B-Think) |

We reuse Ai2's code and recipes from [OLMo-core](https://github.com/allenai/OLMo-core),
[open-instruct](https://github.com/allenai/open-instruct) and
[olmes](https://github.com/allenai/olmes) wherever our hardware allows; every place we
had to differ is listed under [Deviations and validation](#deviations-and-validation).
Model, dataset, chat-template and recipe revisions are pinned in [configs/](configs/);
RLVR dataset, upstream recipe, reward sources, and official reference checkpoint
are pinned in `configs/rlvr*.json`; the WildChat experiment uses `configs/rlvr-wildchat.json`.

`reports/` keeps curated metrics, run configurations, manifests, audits, and selected
examples in Git. Raw predictions, rollouts, judge queues, logs, process markers,
Trainer state, source backups, and placeholder model cards remain local and ignored.

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

## RLVR and human-feedback RLHF

**Current status: stopped at checkpoint-32 (32/1,024 updates; 992 remaining).**
The full continuation `outputs/rlvr-3gen-full-discard-cutoffs` exited with code 1
while awaiting interactive quality judgments for the next prompt. It completed
only one new update, from checkpoint-31 to checkpoint-32. The next prompt has
three pending judge requests and was not committed as an optimizer update.
The failure was `TimeoutError: Interactive judge timed out; refusing to substitute
zero reward`. The GPU is idle. Checkpoint-32 retains the complete adapter, optimizer,
scheduler, RNG and Trainer state. All 11 checkpoint files have a hash-verified
local backup under `outputs/rlvr-3gen-full-discard-cutoffs/checkpoint-32/`.
A continuously served judge is required before
an unattended restart; the interactive chat judge does not keep grading between turns.

The user approved continuing the existing WildChat sample and excluding token-cutoff
responses. The `mask_truncated_completions=True` setting masks their tokens out of the loss
and skips external judging. Cutoff training rewards are now `None`, so they also
do not affect the GRPO group mean or standard deviation. Groups with fewer than
two finished rollouts get equal training rewards and zero advantages. Actual
judgments and cutoff diagnostics remain in logs, with a separate `training_reward`
field. The initial `outputs/rlvr-3gen-full` launch was stopped during baseline
evaluation to make this exclusion complete; it performed no optimizer updates. Checkpoints are
saved after every update. General-quality scores still require the interactive judge;
the server process alone does not provide unattended judging. The full-cutoff
exclusion change passed all 41 tests, including a CPU GRPO check that equal
finished answers plus one discarded answer produce zero gradient and no policy
change. Identical baseline-answer judgments from checkpoint-31 can be reused by
exact prompt-and-answer hash; new answers still require new judgments.

The first new update saved complete **checkpoint-32** with finite, nonzero
gradient norm `0.1148`. Its three rollouts used 1,935 / 3,061 / 3,072 tokens:
the two finished answers received `0.5` / `0.3`, and the cutoff had training
reward `None`. TRL reported mean reward `0.4` and standard deviation `0.14142`,
confirming that the cutoff was excluded from group normalization. The step took
291.7 seconds; this single long group does not establish a new average speed.
The run `outputs/rlvr-3gen-10` continued the saved policy and optimizer from
checkpoint-21 to **checkpoint-31**. The full epoch is **31/1,024 updates complete**,
with **993 remaining at this restart**; the full epoch has not finished.

| Measurement | Result |
|---|---:|
| New updates / sampled training answers | 10 / 30 |
| Mean complete update time, including judging | 3m 21s |
| Recent four-answer baseline | 3m 54s |
| Observed reduction in update time | 14.5% |
| Training token cutoffs / missing final answers | 3 / 1 |
| Finite, nonzero new gradients | 10/10 |
| Total runtime, including before/after evaluation | 46m 35s |
| Peak allocated VRAM, including evaluation | 10.88 GiB |

The timing comparison uses different prompts/completion lengths and includes
interactive judging; it is not a controlled same-prompt four-versus-three test.
The measured mean projects about **55 more active hours** for 993 updates;
allow roughly 55–65 hours because prompt length and judge availability vary.

| Four fixed held-out cases | Checkpoint-21 | Checkpoint-31 |
|---|---:|---:|
| Completed answers | 1/4 | 2/4 |
| Mean judge reward, with cutoffs scored zero | 0.175 | 0.225 |
| Score on the case completed by both models | 7/10 | 5/10 |

Completion coverage improved in this tiny check, but the shared completed answer
became repetitive and worse. The newly completed answer had unreliable citations.
**General quality improvement is unproven.** [Review and audit](reports/rlvr-3gen-10/review.json).
[Metrics](reports/rlvr-3gen-10/all_results.json).

All **39 tests passed**, including four-to-three CPU resume, restored adapter and
optimizer state, and next-unused-prompt checks. The actual optimizer reached step
31; new prompt IDs exactly matched seeded epoch positions 22–31. All 29 used
judgment bindings were verified. The final adapter matches checkpoint-31; both
checkpoint-21 and the DPO parent are unchanged. The allocator reported recoverable
memory pressure during evaluation; training/evaluation completed successfully.

Checkpoint-31 (adapter/config, optimizer, scheduler, RNG and Trainer state), the
final adapter, logs and metadata have hash-verified local backups under `outputs/`
as well as their server copies. The previous checkpoint-21 backup remains intact.
The config is `configs/rlvr-wildchat-short-3gen.json`: three sequential rollouts,
accumulation 3, LR `1e-6`, 3,072 output tokens and unchanged prepared data/rubric.
The baseline evaluation now restores the resume checkpoint before generation and
uses BF16 autocast on both passes. Earlier resumed-run before evaluations used
the DPO parent rather than their immediate preceding checkpoint.

Resume instructions, **only with a continuously active judge**. This command is
not launched. Use a fresh output name and resume the latest complete checkpoint:

```bash
cd /root/post-training-demo
bash scripts/start_rlvr.sh rlvr-3gen-resumed-32 \
  --config configs/rlvr-wildchat-short-3gen.json --judge-backend file \
  --max-steps -1 --eval-size 4 --save-steps 1 --judge-timeout-seconds 3600 \
  --resume-from-checkpoint outputs/rlvr-3gen-full-discard-cutoffs/checkpoint-32
```

The initial four-to-three continuation used explicit `--allow-generation-count-change`
and verified the 21 previously consumed prompts before proceeding. Future resumes
with the same three-answer config need no generation-count override.
[Change validation](reports/rlvr-3gen-10/validation.json).

The following math setup and run log record earlier experiments; the current
training dataset/config is the short-output WildChat sample described below.

RLVR is implemented in `rlvr/`, continuing `outputs/dpo/final-adapter` with
NF4 QLoRA and TRL 1.17.1 GRPO. The prepared math-only dataset has **1,024 training
and 128 held-out prompts** from the first pinned Dolci shard. Prompts are sampled
with seed 42, deduplicated, checked against the fixed benchmark prompts, and
dropped if over 512 tokens; they are never truncated. The manifest records the
three source families and all exclusions. Ground-truth answers are used only by
the reward function, never included in model inputs.

Each prompt receives **four sequential sampled rollouts**, temperature 1.0, up to
**4,096 completion tokens** in the current math configuration. Updates use microbatch 1, accumulation 4, constant
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

The initial detached 2K pilot was launched as `outputs/rlvr-pilot/`, capped at 16 updates
(four rollouts per prompt), stopping after eight updates if no mixed-reward group
appears. Monitor `train.log`; completion requires `exit_code` equal to 0, final
metrics, and `final-adapter/`. Review `mixed_reward_groups`, `policy_changed`, and
`stopped_for_no_learning_signal` before expanding the run. A full epoch is **not**
currently scheduled. The smoke timing suggests roughly an hour or more for such a
pilot, and multiple days for a full 1,024-prompt epoch at those settings.


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

The 2K RLVR pilot was stopped after 16 completed rollouts: all were truncated,
so our reward function forced every reward to zero and the logged gradients were
zero. This is a budget/termination failure, not evidence that the answers were
mathematically wrong. [Stop report](reports/rlvr-pilot-2k/stop-report.json).
The **4,096-token math probe** (`outputs/rlvr-probe-4k/`) also finished without
learning: all four sampled answers were truncated, every reward was zero, and the
adapter stayed unchanged. Peak allocated memory was **10.61 GiB**, runtime **14m 43s**.
Increasing the math completion budget alone did not resolve termination.

### WildChat English experiment

We now select **1,024 training conversations and 128 held-out conversations** from
`hamishivi/new-wildchat-english-general_filtered` in the same pinned Dolci release.
Seed 42; retain only `general-quality` records, deduplicate IDs and conversations,
exclude exact benchmark prompts, preserve valid conversation history, and drop
malformed turns or prompts exceeding 1,024 tokens. References are retained for audit,
never fed to the policy or used as exact-match rewards. No reference-length filter.
Data on the server: `data/dolci-rlvr-wildchat/`. Source and split hashes are recorded
in [the manifest](reports/rlvr-wildchat-data/data-manifest.json).

These records use Ai2's **LLM-judged general-quality reward**. The pinned
`judge_utils.py` supplies its rubric and score normalization (1–10 divided by 10).
API mode requires an explicitly configured OpenAI-compatible judge endpoint/model;
the interactive file-queue alternative is described below. API failures, malformed
scores, and truncated judge replies stop the run instead of
silently becoming zero rewards. The judge sees the conversation and completed final
answer, excluding the policy's thinking. Unfinished policy answers receive zero and
token-truncated completions remain masked from the loss. The 8-update pilot keeps
LR `1e-6`, four sequential rollouts, and a 2,048-token completion budget (the first 1K probe showed frequent truncation).
The [1K generation-only probe](reports/rlvr-wildchat-probe-1k/all_results.json)
finished with exit 0: **3/4 cutoffs**, lengths 1,024 / 1,024 / 1,024 / 817,
**5.67 GiB** peak allocated memory, **115 seconds**. The
[2K follow-up](reports/rlvr-wildchat-probe-2k/all_results.json) also finished with exit 0:
**2/4 cutoffs**, lengths 2,048 / 1,080 / 2,048 / 367, **6.19 GiB** peak, **162 seconds**.
These are tiny stochastic checks on the same four prompts, not quality comparisons.
Half the 2K answers still hit the cap, so a useful quality-reward signal must be
verified in the bounded pilot before attempting a full epoch. These probes have
no quality scores and make no adapter updates. All **28 tests passed** on the server,
including strict judge-error handling, conversation parsing, data-integrity checks,
and a tiny GRPO adapter-update test.

No API judge endpoint was configured at preparation time. The interactive judge
option below is now available; API mode still requires configuration. Set these on the server (keep credentials out of Git and logs):
`RLVR_JUDGE_BASE_URL` (including `/v1`), `RLVR_JUDGE_MODEL`, and optionally
`RLVR_JUDGE_API_KEY`. The API endpoint/model choice remains open.

```bash
uv run --locked python scripts/download_base.py \
  --manifest configs/rlvr-wildchat-data.json --output data/dolci-wildchat-inspection
uv run --locked --group rlvr python -m rlvr.wildchat_data
# After checking rollout termination and configuring the judge:
bash scripts/start_rlvr.sh rlvr-wildchat-pilot --config configs/rlvr-wildchat.json
# Only after a useful pilot: --max-steps -1 selects the full 1,024-prompt epoch.
```

### Interactive Codex judge pilot

The user selected this Codex session as the quality judge. `--judge-backend file`
uses an auditable request/response queue in the run directory instead of an API.
I read the original conversation and completed final answer, apply Ai2's pinned
1–10 quality rubric, and submit a score with a short rationale. Responses are
bound to the exact rubric/query/answer hash, cached per run, and written atomically.
Missing judgments time out after one hour by default (configurable) and fail the run; they never become
zero rewards. The policy's private thinking and reference answer are not shown
to the judge. Score and request files remain with the run for inspection.

This is a documented deviation from Ai2's hosted judge: **Codex interactive session**,
with no pinned exact judge-model version. It requires an active agent session and
is suited to a bounded pilot. The full epoch would need up to 4,096 rollout
judgments plus evaluation; server-side generation does not itself depend on the Mac,
but this interactive judge workflow depends on the session servicing its queue.

```bash
bash scripts/start_rlvr.sh rlvr-wildchat-agent-smoke \
  --config configs/rlvr-wildchat.json --judge-backend file --max-steps 1 --eval-size 2
```

The [one-update pilot](reports/rlvr-wildchat-agent-smoke/all_results.json) completed
with exit 0 in **11m 35s** (including interactive judging), at **8.62 GiB peak VRAM**.
All four training rollouts finished: 1,561 / 1,356 / 2,016 / 1,008 tokens. I scored
them **6 / 4 / 7 / 2 out of 10**, giving rewards 0.6 / 0.4 / 0.7 / 0.2, one mixed
reward group, and gradient norm **0.237**. The adapter changed; the DPO parent hash
remained unchanged. Request/response hashes and delivered rewards were audited.

The two held-out cases averaged 0.7 before the update, but both after-update
answers hit the 2,048-token cap and received zero. This validates the judge-to-update
integration, not a general quality gain. Review completion budgets and evaluation
conditions before interpreting score differences or launching a longer run.
The expanded 2K pilot stopped after **seven updates** when I missed the file
judge's 15-minute deadline. Five updates had nonzero gradients; two had all four
answers truncated. No checkpoint had been saved, so these weights were lost;
the scores and [interruption report](reports/rlvr-wildchat-agent-pilot-interrupted/interruption.json)
are preserved. This was an orchestration failure, not a successful completed run.

Recovery now saves **every update**, keeps the latest two checkpoints, and allows
**one hour** for file judgments by default. `--resume-from-checkpoint` validates
data, base model, DPO parent, optimizer/RNG state, and core training settings;
resume uses a fresh output directory and reports both global and newly performed
steps. Changing the completion budget is allowed. A tiny CPU integration test
restores deliberately perturbed adapter weights, optimizer state, and continues
training. All **33 tests passed**.

The new configuration `configs/rlvr-wildchat-3k.json` allows **3,072 completion
plus 1,024 prompt tokens**. The [one-update 3K checkpoint/recovery smoke](reports/rlvr-wildchat-agent-3k-smoke/all_results.json)
completed with exit 0 in **5m 46s**, at **8.96 GiB peak**.
The after-update held-out answer finished at **2,385 tokens** (the 2K run
cut this case off). Checkpoint-1 contains adapter, optimizer, scheduler, and RNG state.
The resumed 3K pilot reached **checkpoint-5** and was deliberately stopped there
when inspection found an all-cutoff prompt explicitly demanding a **2,000-word
blog**. Another all-cutoff case had 779 prompt tokens of formatting instructions.
The [saved checkpoint and stop record](reports/rlvr-wildchat-agent-3k-pilot-stopped/stop-report.json)
preserve its adapter/optimizer/RNG state. This was not a completed eight-update run.
Previous judge scores are reused only for exact rubric/query/answer hash matches.
A paired generation probe can compare 2K and 3K on actual failed cases using the
same saved policy and per-case seeds, checking that the token prefixes match.
Generation-only memory measurements are distinct from backpropagation memory.

The original 354 seconds per training update (including interactive waits) would
extrapolate to about **four days** for 1,024 updates; this is a rough estimate.
That broader-data run did not start a full epoch; the short-output run below supersedes it.

### Short-output WildChat sample

A separate, pinned-source resample keeps **1,024 training + 128 held-out records**.
Drop prompts over **512 tokens**, numeric word-count mentions over **600** in user
turns, and references over **768 tokens**. The numeric filter is conservative;
reference length is a budget proxy, not a guarantee of short model reasoning.
Original datasets and checkpoints remain intact. This changes the sampled task
mix and is an explicit hardware-driven deviation. [Manifest](reports/rlvr-wildchat-short-data/data-manifest.json).

`configs/rlvr-wildchat-short.json` uses **3,072 completion tokens** (at most 3,584
prompt-plus-completion tokens). The [four-update pilot](reports/rlvr-wildchat-short-pilot/operational-summary.json)
from the unchanged DPO parent completed with exit 0 in **26m 49s** (including
judging and before/after checks), at **10.06 GiB peak allocated VRAM**. It had
**2/16 token-limit hits**, 13 judged final answers, and one separate final-marker
failure. All four updates had finite, nonzero gradients; the final adapter hash
and every used judge-response binding were verified. The longest observed total
sequence was **3,579 tokens**, close to the configured 3,584 ceiling. Checkpoints
are saved each update, and checkpoint-4 is available for resume. It does not resume
the broader-data checkpoint; the resume guard rejects dataset changes.
All **37 tests passed**, including length screening, checkpoint restoration, and batched file-queue judging.
The single held-out case hit the 3K cap both before and after, so these results
validate training/memory/recovery, not a quality gain. The full **1,024-prompt epoch** continued at `outputs/rlvr-full-resumed`, resumed from
`outputs/rlvr-full/checkpoint-8`; 1,016 additional updates remain at launch. The first
full-run update completed with zero cutoffs but four equally poor judgments,
so its gradient was zero. It was intentionally stopped at its verified checkpoint
to publish all four file-judge requests together before awaiting scores. This
changes queue handling only; the rubric, reward mapping, data and recipe stay the same.
The preserved [prefix](reports/rlvr-prefix-unbatched/stop-report.json) has exit 143.
An active session judges the queue when training is running. All updates are checkpointed; a disconnected
judge still times out after one hour, so the training is not independent of judge
availability. The pilot's 354 seconds per update suggests about four days overall.

The continuation reached checkpoint-8: across the pilot and continuation there
were **5/32 token cutoffs**. One simple Linux-command adventure prompt caused
three cutoffs, all before closing thinking. The run was intentionally stopped
at its verified checkpoint for `outputs/rlvr-brief-reasoning-check`, a generation-only
comparison of original versus a brief-reasoning system instruction on two selected
training cases, holding checkpoint-8, per-case seed, and 3,072-token budget fixed.
The diagnostic completed with exit 0 in **5m 14s** and 6.46 GiB inference peak.
The game prompt hit 3,072 tokens in both conditions; the other prompt finished at
2,380 / 2,365 tokens. **Both conditions had 1/2 cutoffs**, so this small check
provides no reason to adopt the instruction. The checkpoint hash was unchanged.
Training resumed unchanged from checkpoint-8; no quality gain is claimed.
[Diagnostic results](reports/rlvr-brief-reasoning-check/all_results.json).
[Preserved prefix](reports/rlvr-full-prefix/stop-report.json).

At **2026-11-01 21:05 UTC**, [checkpoint-16 was audited](reports/rlvr-full-resumed-progress/checkpoint-16-audit.json):
**16/1,024 updates**, 64 training completions, **10 token cutoffs**, 52 judged final
answers and two separate missing-final-answer cases. All 16 gradient measurements
were finite; 14 were nonzero. Every used judgment binding, checkpoint file hash,
and unchanged DPO parent hash was verified. This is an execution snapshot, not
an established quality gain. The run subsequently reached checkpoint-21 and was paused as described above.

The [fixed-policy, matched-seed check](reports/rlvr-wildchat-short-budget-check/all_results.json)
completed with exit 0. One selected case ended at 2,019 tokens under both budgets;
the other hit 2K and stopped at 2,534 tokens under 3K. Token prefixes matched in
both pairs. The latter still lacked a final-answer marker, demonstrating a separate
format problem that more tokens alone do not fix. These are two selected training
cases and a generation-only diagnostic, not a quality benchmark.

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
[Results](reports/evals/results.json). Raw generated answers are local run artifacts
at `outputs/evals/predictions.jsonl`.

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
[DPO results](reports/evals-dpo/results.json). Raw generated answers are local run
artifacts at `outputs/evals-dpo/predictions.jsonl`.

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
[Validation](reports/evals-dpo-2k-b1/validation.json).
Raw generated answers are local run artifacts at
`outputs/evals-dpo-2k-b1/predictions.jsonl`.
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
| [rlvr/wildchat_data.py](rlvr/wildchat_data.py) | Prepare disjoint WildChat English conversations for general-quality rewards. |
| [rlvr/quality.py](rlvr/quality.py) | Call an explicitly configured judge with Ai2’s rubric; reject failed or malformed judgments. |
| [rlvr/file_judge.py](rlvr/file_judge.py) | Wait for auditable interactive judgments bound to exact request hashes. |
| [rlvr/judge_queue.py](rlvr/judge_queue.py) | Inspect the judge queue or atomically submit explicit scores. |
| [rlvr/rollout_check.py](rlvr/rollout_check.py) | Check completion lengths without scoring or updating the policy. |
| [run_rlvr_probe.sh](scripts/run_rlvr_probe.sh) | Run a bounded generation probe under the GPU lock and record exit status. |
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

Next: verify WildChat completion lengths, configure its general-quality judge, then inspect pilot reward variation and policy updates before a full epoch.

RLVR deviations: TRL replaces the distributed open-instruct/vLLM/Beaker runtime;
we use QLoRA, the original GRPO loss, a first-shard math-only subset, four serial
rollouts, a 512-token prompt ceiling and 2,048-token completion cap instead of
Ai2's full mixture, eight rollouts and 32K completions. We add a boxed-answer
instruction and require closed thinking, with zero reward and loss masking for
truncation. These changes fit the hardware and define a narrower experiment;
they do not reproduce the full official RL recipe.
