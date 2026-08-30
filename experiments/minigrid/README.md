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
