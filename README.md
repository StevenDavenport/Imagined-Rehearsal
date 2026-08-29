# Continual Imagined Rehearsal

Research branch for studying **Imagined Rehearsal (IR)** as an online policy-learning mechanism in continual reinforcement learning (CRL).

The current working direction is documented in
[CIR_RESEARCH_PLAN.md](CIR_RESEARCH_PLAN.md). It supersedes the earlier
assumption below that CIR would begin as an additional actor update inside an
otherwise jointly trained Continual-Dreamer agent.

The next executable stage is the
[controlled MiniGrid difficulty audition](experiments/minigrid/README.md): a
restart-safe six-task, three-seed screen used to choose a fast, non-trivial
continual sequence before CIR integration.

The executed RLScape evidence chain, canonical stage names, maintained
protocols, and future experiment reservations are indexed separately in the
[RLScape experiment registry](experiments/rlscape/README.md). That registry is
authoritative when older `M1`--`M4` engineering milestones or roadmap labels
conflict with the scientific stage names.

The original repository is a complete implementation of IR for test-time policy repair. This `crl` branch starts a new line of work: the world-model agent will train continually across a sequence of tasks, while IR supplies immediate, current-posterior actor updates between real environment interactions.

> [!IMPORTANT]
> The CRL method described below is a research plan, not a claim about the current implementation or experimental results. The current code still implements the original evaluation-time IR mechanism.

## Research Question

> Does just-in-time, posterior-conditioned Imagined Rehearsal accelerate transfer in a continually trained Dreamer agent without increasing forgetting, and can rehearsal be gated to retain that benefit under a fixed compute budget?

The primary hypothesis is that IR will improve the **speed of transfer** after a task change. At each real state, IR can generate many short imagined trajectories and improve the actor before the next real action. Standard Dreamer actor learning also uses imagined trajectories, but normally performs background updates from posterior states sampled from replay. Planned continual IR differs in the timing and selection of those states: it concentrates additional policy optimisation on the agent's current posterior belief, at the moment adaptation is needed.

## Existing Foundation

![Imagined Rehearsal Monte Carlo Posterior Batching diagram](assets/irmcpb_diagram.png)

The existing implementation provides:

- DreamerV3 world-model training;
- evaluation-time IR from the current recurrent posterior;
- actor-only imagined policy updates;
- Monte Carlo Posterior Batching (MCPB);
- actor corruption and checkpoint-transfer tools;
- multi-seed evaluation, trace extraction, and plotting scripts;
- a real-trajectory online actor-repair baseline.

In the current evaluation path, IR:

1. infers the current latent posterior from a real observation;
2. clones the checkpoint actor;
3. imagines short trajectories in the frozen world model;
4. updates only the cloned actor from predicted returns;
5. acts with that adapted actor;
6. accumulates adaptation within the episode; and
7. discards the adapted actor at the episode boundary.

MCPB repeats the current deterministic RSSM state, draws multiple stochastic states from the same posterior logits, and averages the resulting imagined actor loss in one batched update. It captures stochastic posterior and rollout uncertainty, not world-model parameter uncertainty.

## Planned Continual-Learning Setting

The first CRL experiments will use a single agent trained on a sequence of tasks or environment contexts. Unlike the existing evaluation-only procedure:

- the world model, reward model, critic, and actor continue learning from the real experience stream;
- a continual replay mechanism is used to reduce world-model forgetting;
- IR is integrated into the interaction/training loop rather than being confined to evaluation;
- the first implementation will apply IR updates directly to the persistent actor;
- evaluation will measure both rapid online behaviour and durable learning.

The continual world-model mechanism is deliberately not the intended novelty. It will be selected from established work and held constant across all IR ablations, so that differences can be attributed to rehearsal rather than to different replay systems.

### Initial actor semantics

The simplest first version treats IR as an additional persistent actor update:

```text
real observation
      |
      v
current posterior -----> short imagined rollouts
      |                           |
      |                           v
      |                  persistent actor update
      |                           |
      +---------------------------+
                  |
                  v
            next real action
```

This tests IR as a full training mechanism without immediately introducing a second actor. A fast/slow actor design remains a follow-up if persistent IR improves transfer but causes interference or drift:

- a fast actor would adapt immediately through IR;
- a slow actor would retain consolidated lifetime knowledge;
- consolidation would control which fast changes become persistent.

## Hypotheses

1. **Transfer speed:** Always-on IR improves return AUC and reduces the number of real interactions needed after a task transition.
2. **Retention:** Current-posterior actor updates can improve transfer without materially increasing forgetting when the world model is protected by continual replay.
3. **MCPB under change:** Posterior batching is most useful near task transitions or other periods of high uncertainty.
4. **Efficient rehearsal:** A gated method retains most of the transfer benefit of always-on IR while using fewer imagined transitions and less wall-clock time.
5. **Failure boundary:** IR becomes harmful when the world model is too stale or uncertain for its imagined returns to be trustworthy.

## Rehearsal Gating

Gating will be treated as three separate decisions rather than a single threshold.

### 1. Is actor adaptation needed?

Candidate signals:

- disagreement among an ensemble of actors;
- a recent return or value drop;
- large imagined advantages;
- large expected actor update or policy KL.

### 2. Can the current imagination be trusted?

Candidate signals:

- world-model prediction surprise;
- reconstruction or dynamics loss;
- disagreement among world-model predictors;
- poor agreement between predicted and recently observed rewards.

A large distribution shift may create an urgent need for adaptation while simultaneously making imagined updates unreliable. The gate should be able to defer or reduce IR in that case.

### 3. How much rehearsal compute is justified?

Candidate controls:

- posterior entropy determines MCPB batch size;
- gradient variance across posterior samples determines whether to draw more samples;
- actor disagreement determines the number of IR optimiser steps;
- adaptation stops when disagreement or update magnitude falls below a threshold.

The first experiments will log these candidate signals during always-on IR. Gating rules will be chosen only after testing which signals predict a positive marginal benefit from rehearsal.

## Historical CIR roadmap

The phase labels below predate the executed RLScape diagnostic programme. They
are retained as conceptual background, but they are not the canonical RLScape
experiment stages.

### Roadmap phase A: continual baseline

Build a task-sequence runner and reproduce a continual Dreamer baseline using one fixed world-model replay mechanism. Verify:

- task transitions and evaluation do not reset the learned agent;
- previous tasks can be evaluated without contaminating training state;
- replay composition can be inspected over the complete lifetime;
- per-task learning curves and a performance matrix are recorded.

### Roadmap phase B: always-on rehearsal

Integrate persistent IR into the training/interaction loop and compare:

| Method | Current-posterior IR | MCPB | Trigger |
|---|---:|---:|---|
| Continual Dreamer | No | No | Never |
| Always-on IR | Yes | No | Every eligible interaction |
| Always-on IR + MCPB | Yes | Yes | Every eligible interaction |
| Periodic IR | Yes | Optional | Fixed cadence |
| Compute-matched random IR | Yes | Optional | Random |
| Oracle boundary IR | Yes | Optional | Known task transitions |

The oracle-boundary method is an upper-bound diagnostic, not the final task-agnostic method.

### Roadmap phase C: gated rehearsal

Compare candidate gates under matched budgets:

- actor-ensemble disagreement;
- posterior entropy;
- world-model surprise;
- combined need/trust/compute gating;
- periodic and random controls with the same number of imagined transitions.

### Roadmap phase D: persistence and consolidation, if needed

If persistent IR learns quickly but forgets or drifts, evaluate:

- temporary per-episode IR;
- persistent IR;
- fast/slow actors;
- guarded consolidation;
- posterior-anchor rehearsal for retention.

These mechanisms are intentionally deferred until the simpler persistent-actor experiment establishes the failure mode they would address.

## Evaluation

Final return alone cannot distinguish rapid behavioural adaptation from durable continual learning. The planned measurements are:

- return throughout the complete task sequence;
- return AUC immediately after every task transition;
- real steps to a fixed performance threshold;
- forward transfer relative to isolated single-task learning;
- average and maximum forgetting;
- performance on every previous task after each training phase;
- recurrence gain when a task is encountered again;
- worst-task performance;
- imagined transitions and optimiser updates;
- wall-clock time and policy throughput;
- replay memory and total memory use;
- performance with IR disabled during evaluation.

The final measurement separates knowledge stored in persistent agent parameters from performance that only exists while online rehearsal is active.

### Experimental controls

- Match compute-heavy methods by imagined transitions and wall-clock time, not only by gradient steps.
- Use multiple task orders or at least a default and reversed order.
- Report absolute performance alongside forgetting; an agent that never learns cannot meaningfully forget.
- Keep the continual world-model mechanism identical across IR variants.
- Evaluate old tasks from cloned evaluation state so evaluation cannot train the agent.
- Separate task-aware oracle gating from task-agnostic gating.
- Log world-model error around task transitions to identify when IR is optimising stale imagination.

## Continual World-Model Learning

The initial implementation will reuse an established replay principle rather than propose a new world-model continual-learning algorithm.

Two leading candidates are:

### Continual-Dreamer

[The Effectiveness of World Models for Continual Reinforcement Learning](https://proceedings.mlr.press/v232/kessler23a.html) finds reservoir replay to be a strong retention mechanism for DreamerV2 and calls the resulting system Continual-Dreamer. The [official implementation](https://github.com/skezle/continual-dreamer) includes MiniGrid and MiniHack task-sequence runners, reservoir sampling, recent/past minibatch mixing, and evaluation tooling.

The repository is useful as a research reference, but it is based on TensorFlow 2.6, DreamerV2, Python 3.8, and older Gym/MiniHack dependencies. It is not a drop-in dependency for this JAX DreamerV3 codebase.

### ARROW

[ARROW: Augmented Replay for RObust World Models](https://openreview.net/forum?id=3FK2tFwNwK) combines a short-term FIFO buffer with a long-term reservoir/distribution-matching buffer under a fixed total memory budget. Its [official implementation](https://github.com/Cerenaut/ARROW) is MIT-licensed and includes DreamerV3 experiments for Atari and Procgen CoinRun.

ARROW is implemented as a separate PyTorch DreamerV3 rather than as a component compatible with this repository. Its dual-buffer design and experimental configurations are nevertheless directly relevant implementation references.

### Implemented integration

The original sliding-sequence FIFO replay remains unchanged. The optional
`embodied/core/episode_replay.py:EpisodeReplay` instead stages complete
episodes and applies Algorithm R independently within each goal. Training
samples a represented goal uniformly, then a retained episode and sequence
start. Short episodes are concatenated with explicit `is_first` boundaries so
successful short trajectories are not discarded.

The episode reservoir persists its admission and sampling RNGs, per-goal
episode counters, retained slots, partial episodes, and sampling counters. Its
manifest and retained files are included in the same immutable 250k-step
milestone archives as FIFO replay. Select it through
`--replay-kind episode_reservoir`; FIFO remains the default. An ARROW-style
dual buffer remains a possible later comparison rather than part of this first
reservoir experiment.

## Reading Before Development

1. [The Effectiveness of World Models for Continual Reinforcement Learning](https://proceedings.mlr.press/v232/kessler23a.html) — the closest conceptual baseline and the source of Continual-Dreamer.
2. [ARROW: Augmented Replay for RObust World Models](https://openreview.net/forum?id=3FK2tFwNwK) — the most directly relevant modern DreamerV3 continual replay implementation.
3. [Continual World: A Robotic Benchmark for Continual Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2021/hash/ef8446f35513a8d6aa2308357a268a7e-Abstract.html) — forward-transfer, forgetting, and task-sequence evaluation design.
4. [Towards Evaluating Adaptivity of Model-Based Reinforcement Learning Methods](https://proceedings.mlr.press/v162/wan22d.html) — why replay-based model learning can adapt poorly after local changes.
5. [Same State, Different Task: Continual Reinforcement Learning without Interference](https://arxiv.org/abs/2106.02940) — distinguishes catastrophic forgetting from incompatible objectives and replay interference.

## Current Installation

Python 3.11+ is recommended for the current IR code:

```bash
pip install -U -r requirements.txt
```

Depending on the available accelerator and CUDA installation, the JAX line in `requirements.txt` may need adjustment.

## Current IR Usage

The original actor-only IR path remains available. To enable critic-first IR,
add `--eval_adapt.train_critic True` and set its independent learning rate with
`--eval_adapt.critic_lr`. At the first observation of each episode, this runs
one critic-only MCPB update followed by the configured critic-to-actor update
before the first environment action. Later triggers run the critic-to-actor
update, and all adapted parameters are discarded at the episode boundary.

```bash
python dreamerv3/main.py \
  --script eval_only \
  --logdir logs/dmc_walker_walk/eval_ir \
  --configs dmc_vision \
  --task dmc_walker_walk \
  --run.from_checkpoint /path/to/checkpoint \
  --run.steps 50000 \
  --run.envs 1 \
  --jax.platform cuda \
  --eval_adapt.enabled True \
  --eval_adapt.train_critic True \
  --eval_adapt.steps 1 \
  --eval_adapt.imag_length 6 \
  --eval_adapt.lr 1e-4 \
  --eval_adapt.critic_lr 1e-4 \
  --eval_adapt.every_k 1 \
  --eval_adapt.actent 3e-4 \
  --eval_adapt.start_batch 512
```

### Cheetah critic-only intermediary experiment

The critic-only diagnostic described in the research plan has a staged,
resumable CPU runner. The command below collects a fixed healthy-policy
dataset, checks frozen-world-model fidelity, screens imagination horizons, and
confirms the most promising critic-repair conditions. The optional proposal
checkpoint measures the loss of imagined reward coverage caused by the damaged
actor, but never uses that actor for critic repair.

```bash
python scripts/critic_intermediary_experiment.py all \
  --outdir logs/cheetah_critic_intermediary \
  --clean_checkpoint checkpoints/cheetah_run_vision_seed0 \
  --proposal_checkpoint \
    checkpoints/cheetah_run_vision_seed0_corruptions/both_zero_mask_moderate_all_s0p2_seed0 \
  --jax_platform cpu \
  --skip_existing
```

Run `prepare`, `collect`, `probe`, `screen`, `confirm`, and `summarize`
separately to inspect each stage before paying for the next one. Results are
stored as CSV and JSON under the output directory; critic snapshots allow
completed horizon/seed conditions to be reused.

### Early Cheetah continual-IR pilot

The first end-to-end CIR runner implements the fixed A-to-B action-gain pilot
from the research plan. It starts from the healthy Cheetah checkpoint, gathers
A and B diagnostics, performs world-model-only replay repair, and then
alternates persistent in-episode actor-only IR with episode-boundary world-model
updates. The critic and slow critic are frozen and supply a fixed value
scaffold to the actor; a namespace hash verifies that actor rehearsal cannot
change either of them. The dedicated actor optimizer is identical to the one
used by the critic-first pilot, so the critic update is the only behavioural
ablation. The run is CPU compatible and resumes automatically when the same log
directory already contains a checkpoint.

```bash
python dreamerv3/main.py \
  --configs dmc_vision cir_cheetah \
  --logdir logs/cheetah_cir_actor_only_seed0 \
  --run.from_checkpoint \
    /home/staff/steven/crl_ir/checkpoints/cheetah_run_vision_seed0 \
  --jax.platform cpu \
  --jax.prealloc False
```

To reproduce the earlier critic-first control, append
`cir_cheetah_critic_first` to `--configs`. Always use a separate log directory
because the saved phase state and optimizer counters are resumable.

The matched world-model-only control freezes both actor and critic parameters,
performs no imagined rehearsal, and otherwise retains the complete schedule:

```bash
python dreamerv3/main.py \
  --configs dmc_vision cir_cheetah cir_cheetah_wm_only \
  --logdir logs/cheetah_cir_wm_only_seed0 \
  --run.from_checkpoint \
    /home/staff/steven/crl_ir/checkpoints/cheetah_run_vision_seed0 \
  --jax.platform cpu \
  --jax.prealloc False
```

Actor and critic namespace hashes are checked after every B collection episode.
Consequently, the only learned parameters in this control belong to the
encoder, RSSM, decoder, reward head, and continuation head.

Structured results are written under `cir_artifacts/`: `events.jsonl` records
phase transitions, optimizer work, resource use, namespace hashes, and stop
guards; `episodes.csv` contains return and aggregate diagnostics;
`probes.jsonl` and `probe_anchors.jsonl` contain critic and multi-horizon model
diagnostics; compressed trajectory files retain the latent anchors. The normal
`ckpt/` directory stores the live agent, replay, counters, and phase state.

### Stationary Walker critic-repair experiment

Train one fresh Walker baseline and retain its final 1.1M-step checkpoint:

```bash
python dreamerv3/main.py \
  --logdir logs/critic_ir_walker_seed0 \
  --configs dmc_vision \
  --task dmc_walker_walk \
  --seed 0 \
  --run.steps 1.1e6 \
  --run.ckpt_keep 1
```

Evaluate 20 clean episodes in total (four per evaluation seed) and require a
mean return of at least 800 before continuing:

```bash
python scripts/checkpoint_eval_sweep.py \
  --checkpoints logs/critic_ir_walker_seed0/ckpt \
  --configs dmc_vision \
  --task dmc_walker_walk \
  --seeds 0,1,2,3,4 \
  --run_steps 4000 \
  --jax_platform cuda
```

Create matched actor-and-critic corruptions. The damaged online critic is
copied into the slow target critic to prevent intact-target leakage:

```bash
python scripts/corrupt_actor_checkpoint.py \
  --source logs/critic_ir_walker_seed0/ckpt \
  --out_root checkpoints/critic_ir_walker \
  --target both \
  --preset zero_mask_severity_triplet \
  --seed 0
```

Run the clean safety check and the corrupted recovery ladder:

```bash
python scripts/eval_suite.py \
  --checkpoint logs/critic_ir_walker_seed0/ckpt \
  --configs dmc_vision \
  --task dmc_walker_walk \
  --seeds 0,1,2,3,4 \
  --run_steps 10000 \
  --jax_platform cuda \
  --suite_json scripts/critic_ir_clean_suite.json

python scripts/walker_recovery_suite.py \
  --checkpoints checkpoints/critic_ir_walker \
  --configs dmc_vision \
  --task dmc_walker_walk \
  --seeds 0,1,2,3,4 \
  --run_steps 10000 \
  --jax_platform cuda \
  --suite_json scripts/critic_ir_corrupt_suite.json
```

Core existing scripts include:

- `scripts/corrupt_actor_checkpoint.py` — corrupt the actor, critic, or both;
- `scripts/transfer_actor_checkpoint.py` — transplant actors between compatible checkpoints;
- `scripts/eval_suite.py` — multi-seed IR sweeps;
- `scripts/rehearsal_benchmark_suite.py` — original paper benchmark workflow;
- `scripts/online_actor_repair_eval.py` — real-trajectory actor-repair baseline;
- `scripts/walker_intra_episode_trace_suite.py` — per-step adaptation traces.

## Repository Layout

- `dreamerv3/` — Dreamer agent, RSSM, objectives, and IR implementation;
- `embodied/` — runtime, environments, replay, and training/evaluation loops;
- `scripts/` — checkpoint manipulation, experiment suites, analysis, and plots;
- `assets/` — project figures.

Key implementation files:

- `dreamerv3/agent.py`;
- `dreamerv3/configs.yaml`;
- `embodied/jax/agent.py`;
- `embodied/core/replay.py`;
- `embodied/run/train.py`;
- `embodied/run/eval_only.py`.

## Scope and Attribution

This repository uses DreamerV3 as its world-model agent foundation. The original contribution is Imagined Rehearsal and its evaluation workflow. The planned CRL contribution is the use and efficient gating of current-posterior rehearsal during continual training.

The repository does not ship trained checkpoints or large benchmark outputs.

If you use this work, cite both the Imagined Rehearsal paper and the upstream DreamerV3 work. Continual replay or benchmark components adopted during this project will retain their original attribution and licensing.
