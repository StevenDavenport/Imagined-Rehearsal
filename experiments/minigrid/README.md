# Controlled MiniGrid difficulty audition

This is the fast environment-selection stage before continual imagined
rehearsal (CIR). It does **not** test continual learning yet. It trains isolated
DreamerV3 agents to identify tasks that are learnable, non-trivial, and similar
enough in difficulty to form a defensible continual sequence.

## Candidate tasks

All candidates use the same `Discrete(7)` action interface, 64x64 partial RGB
observation, 100-step episode limit, and six-way goal-conditioned Dreamer
heads. The RSSM remains goal-agnostic, matching the RLScape architecture.

| Goal ID | Task | Family | Controlled objective |
|---:|---|---|---|
| 0 | `go_to_door` | navigation | Go to the red door |
| 1 | `fetch` | pickup | Fetch the blue ball among distractors |
| 2 | `go_to_object` | navigation | Go to the green key |
| 3 | `put_near` | pickup/place | Put the yellow ball near the blue box |
| 4 | `unlock_pickup` | hierarchical | Unlock a door and pick up the only box |
| 5 | `pickup_dist` | pickup | Pick up the grey box among four distractors |

The mission is functionally fixed within each task, so this audition does not
accidentally become a language-grounding experiment. Object positions, agent
positions/orientations, colors, and geometry where applicable still vary by
reset seed.
The controlled environments are defined in
[`embodied/envs/minigrid_tasks.py`](../../embodied/envs/minigrid_tasks.py).

The two candidate sequences retained for selection are:

- alternating: `go_to_door -> fetch -> go_to_object -> put_near`;
- progressive: `fetch -> put_near -> unlock_pickup`.

No sequence is selected until the isolated audition results are available.

## Prespecified full run

The default supervisor runs 6 tasks x 3 training seeds: 18 isolated training
runs with the 12M Dreamer preset, 100,000 environment steps per run, and exact
checkpoint/replay archives every 25,000 steps. Each of the four checkpoints is
then evaluated for 50 sampled-policy and 50 deterministic-policy episodes.
That is 144 evaluation conditions and 7,200 evaluation episodes.

Use a fresh absolute run root on the 4090 machine. First inspect the complete
matrix without launching a child process:

```bash
python scripts/minigrid_audition.py \
  --experiment-root /absolute/path/to/logs/minigrid/audition_12m_v1 \
  --dry-run
```

Then launch the same specification without `--dry-run`:

```bash
python scripts/minigrid_audition.py \
  --experiment-root /absolute/path/to/logs/minigrid/audition_12m_v1 \
  --stream-output
```

The supervisor is serial and restart-safe. Re-run the exact command after an
interruption; complete training milestones and evaluation conditions are
validated and skipped. Incomplete conditions get a new attempt directory, so
raw evidence is never overwritten. The experiment spec is immutable within a
run root, and the runner refuses to mix changed settings into it.

For a short end-to-end installation and GPU smoke test, use a separate root:

```bash
python scripts/minigrid_audition.py \
  --experiment-root /absolute/path/to/logs/minigrid/smoke_1m \
  --tasks fetch \
  --seeds 0 \
  --model-size 1m \
  --steps 1000 \
  --checkpoint-every 500 \
  --eval-episodes-per-mode 2 \
  --stream-output
```

Do not extend the smoke root into the full experiment. If the 12M audition
saturates too early, repeat the whole matrix with `--model-size 1m` and a new
root rather than mixing capacities.

## Extending selected 100k runs to 200k

The initial audition selected four runs for a longer learnability and stability
check: `go_to_door`, `fetch`, `go_to_object`, and `put_near`. Continue them in a
new experiment root so the completed 100k baseline remains immutable:

```bash
export MINIGRID_SOURCE_ROOT=/home/localadmin/experiment_logs/minigrid/audition_12m_v1
export MINIGRID_EXTENSION_ROOT=/home/localadmin/experiment_logs/minigrid/audition_12m_200k_extension_v1

python scripts/minigrid_audition.py \
  --source-experiment-root "$MINIGRID_SOURCE_ROOT" \
  --experiment-root "$MINIGRID_EXTENSION_ROOT" \
  --tasks go_to_door fetch go_to_object put_near \
  --seeds 0 1 2 \
  --steps 200000 \
  --checkpoint-every 25000 \
  --dry-run
```

Inspect the dry-run commands, then launch the identical specification without
`--dry-run`:

```bash
python scripts/minigrid_audition.py \
  --source-experiment-root "$MINIGRID_SOURCE_ROOT" \
  --experiment-root "$MINIGRID_EXTENSION_ROOT" \
  --tasks go_to_door fetch go_to_object put_near \
  --seeds 0 1 2 \
  --steps 200000 \
  --checkpoint-every 25000 \
  --stream-output
```

The extension validates each source 100k milestone, hard-links or copies its
replay chunks into the new run, and restores the complete agent, optimizer,
step counter, and replay state. It creates and evaluates only the 125k, 150k,
175k, and 200k milestones. A deterministic 100k environment-seed offset avoids
repeating the source run's initial MiniGrid layout stream. Restarting the
command uses the extension's own latest checkpoint; it does not reseed from
100k or duplicate experience.

The extension analysis combines the inherited 25k--100k source evidence with
the new 125k--200k evidence, while the raw artifacts remain in their respective
immutable roots. Never point `--experiment-root` at the source baseline.

## Monitoring and outputs

While the supervisor is running:

```bash
python scripts/minigrid_audition_status.py \
  --experiment-root /absolute/path/to/logs/minigrid/audition_12m_v1
```

The run root contains:

```text
experiment_spec.json
supervisor_status.json
active_child.json                    only while a child is active
runs/<task>/seed_<seed>/
  training/{episodes.jsonl,metrics.jsonl,ckpt/,console.log}
  milestones/step_<step>/{checkpoint/,replay/,manifest.json}
  evaluations/step_<step>/<policy_mode>/
    attempt_<n>/{episodes.jsonl,audit_env0.jsonl,eval_integrity.json}
    condition_status.json
  evaluations/step_<step>/pairing.json
analysis/{summary.json,*.csv}
```

The sampled and deterministic evaluations use identical environment reset
seeds and are never pooled. The sampled policy also receives an explicit
episode/step random-key schedule. Completion requires the exact episode and
reset counts plus unchanged checkpoint parameters. Each `pairing.json`
independently verifies the reset schedules across policy modes.

The final report quantifies:

- normalized AUC of rolling binary success during training;
- steps to 70% rolling success, after a full 20-episode window;
- final rolling success and time-discounted return;
- the first 90% saturation point and early-saturation warnings;
- across-seed variability;
- final sampled and deterministic success/return separately; and
- pairwise difficulty agreement within the prespecified 25% tolerance.

Generate the report again at any time after copying or recovering artifacts:

```bash
python scripts/minigrid_audition_report.py \
  --experiment-root /absolute/path/to/logs/minigrid/audition_12m_v1
```

## Dependencies

MiniGrid is pinned to 3.1.0 and uses the Gymnasium API. Install the updated
[`requirements.txt`](../../requirements.txt) in the existing Dreamer CUDA
environment before the smoke test (the repository's pinned JAX/NumPy stack is
best installed under Python 3.11 or 3.12). The supervisor checks the MiniGrid
and JAX versions, required runtime packages, available disk space, checkpoint
integrity, and spec consistency before or during execution.

## Visibility-first sequential baseline

The selected sequence is `go_to_door -> fetch -> go_to_object -> put_near`,
with 200,000 new environment steps per phase and no return to task A. This is
the pre-CIR baseline: it deliberately makes no counterfactual head update, no
critic regularization, and no imagined-rehearsal update.

Three replay arms answer different questions without conflating them:

- `fifo_100k` is the short-memory forgetting control;
- `uniform_reservoir` is the primary future-CIR baseline, retaining about
  1,000 complete episodes per model goal and sampling uniformly over goals
  represented so far (current-task shares 100%, 50%, 33%, then 25%);
- `current_old_50_50` has the same reservoir and update budget but always
  assigns 50% of samples to the current task, dividing the other half
  uniformly across old tasks. It tests whether simple current-biased replay
  already provides the benefit that CIR will later need to beat.

The default is 3 arms x 3 seeds x 800k steps = 7.2 million environment steps.
Only one environment is active at a time. Agent/optimizer snapshots are saved
every 25k; exact replay is added only at the four 200k phase boundaries.
The active task is evaluated every 25k and all four tasks every 50k, always as
50 sampled-policy plus 50 deterministic-policy episodes with paired reset
seeds. This produces 1,512 evaluation conditions (75,600 episodes).

After training, the final `uniform_reservoir` replay fixes one success/failure
state bank per seed. Every arm's 0/200k/400k/600k/800k snapshots are then
audited on those identical states. The head audit measures factual and
counterfactual reward/continuation selectivity and online/slow critic
calibration. The critic-provenance audit uses horizons 1/3/6/15 and separates
reward-only from bootstrap return under both actor and recorded-action
proposals. All audits are read-only and verify parameter integrity.

On the 4090, first inspect the exact matrix using a fresh absolute root:

```bash
export MINIGRID_SEQUENTIAL_ROOT=/home/localadmin/experiment_logs/minigrid/sequential_baseline_12m_v1

python scripts/minigrid_sequential_baseline.py \
  --experiment-root "$MINIGRID_SEQUENTIAL_ROOT" \
  --dry-run
```

Then launch the exact same command without `--dry-run`. It is designed to run
inside tmux and to be safely detached; child output is written under each run
root. Re-run the same command after interruption to adopt the exact live child
or validate and skip completed immutable work.

```bash
python scripts/minigrid_sequential_baseline.py \
  --experiment-root "$MINIGRID_SEQUENTIAL_ROOT"
```

Monitor it from any shell:

```bash
python scripts/minigrid_sequential_status.py \
  --experiment-root "$MINIGRID_SEQUENTIAL_ROOT"
```

The final machine-readable analysis can also be regenerated independently:

```bash
python scripts/minigrid_sequential_report.py \
  --experiment-root "$MINIGRID_SEQUENTIAL_ROOT"
```

The sequential run root adds this structure:

```text
runs/<arm>/seed_<seed>/
  training/{episodes.jsonl,metrics.jsonl,ckpt/,replay/,console.log}
  snapshots/step_<step>/{checkpoint/,manifest.json}
  milestones/step_<phase-end>/{checkpoint/,replay/,manifest.json}
  evaluations/step_<step>/<task>/<mode>/attempt_<n>/
diagnostics/
  banks/seed_<seed>/selection.json
  head/<arm>/seed_<seed>/step_<step>/
  critic_provenance/<arm>/seed_<seed>/step_<step>/
analysis/{summary.json,evaluation_curves.csv,transfer_and_forgetting.csv,
          arm_summary.csv,replay_allocation.csv,
          component_losses_by_goal.csv,diagnostic_inventory.csv,
          actor_same_state_drift.csv}
```

Training metrics retain per-goal sample totals and per-goal dynamics,
reconstruction, reward, continuation, policy, and value losses. This is
intentional: behavioural retention must not conceal a component that is
quietly losing the semantics or calibration CIR depends on.

## Counterfactual-head Phase-A pilot

Before a full sequential re-baseline, the pinned pilot reruns only seed 0 of
the primary uniform-reservoir arm on `go_to_door` for 200,000 steps. It adds an
equal-weight loss averaged over the five nonmatching goal queries: their
reward target is zero, and completing the factual goal is labelled as
continuing under those alternative queries. Physical terminals still stop
every goal. The added path receives frozen posterior features, so it updates
only the reward and continuation heads; the factual losses and all
actor/critic, encoder, decoder, and RSSM objectives are unchanged.

The pilot remains serial (`run.envs: 1`), checkpoints every 25,000 steps, runs
50 sampled and 50 deterministic evaluations per checkpoint, and performs the
same read-only head and critic-provenance audits at 0 and 200,000 steps. Use a
fresh run root:

```bash
export MINIGRID_CF_PILOT_ROOT=/home/localadmin/experiment_logs/minigrid/counterfactual_heads_phase_a_pilot_v1

python scripts/minigrid_counterfactual_pilot.py \
  --experiment-root "$MINIGRID_CF_PILOT_ROOT" \
  --dry-run

python scripts/minigrid_counterfactual_pilot.py \
  --experiment-root "$MINIGRID_CF_PILOT_ROOT"
```

The supervisor is restart-safe. Its immutable spec explicitly records the
counterfactual config, single task, seed, replay arm, snapshots, evaluation
schedule, and diagnostic plan. The go/no-go comparison is against the existing
standard-training `uniform_reservoir`, seed-0 Phase-A evidence at 200,000
steps; the pilot is not itself evidence about catastrophic forgetting.
