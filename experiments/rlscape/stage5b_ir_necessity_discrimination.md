# Stage 5B: IR Necessity Discrimination

## Status and question

Status: **complete; hypothesis rejected**.

Stage 5A starts every episode from a policy selected because it needs repair.
That can show whether a gate schedules useful updates, but not whether it knows
when to leave a competent policy alone. Stage 5B therefore asks a stricter
question: on one immutable checkpoint, does the frozen gate activate for an
IR-responsive weak task while abstaining on a distinct strong task?

“Needs IR” is defined behaviorally and only for the actor-recovery objective.
It is not inferred from task identity, training phase, privileged success
history, or a manually assigned strong/weak label.

## Frozen selection and conditions

A small calibration phase evaluates all three established RLScape goals with
no IR, always-on standard IR, and always-on reward-only IR. It selects:

- a weak task whose best always-on IR condition improves mean sampled and
  deterministic success by at least 0.05; and
- a distinct strong task whose no-IR mean competence is at least 0.60.

The resulting `task_selection.json` is hashed and frozen before confirmation.
If no task satisfies either criterion, the launcher stops rather than silently
changing the scientific definition.

Confirmation compares exactly seven conditions:

1. no IR;
2. always-on standard IR;
3. always-on reward-only IR;
4. JS-gated standard IR;
5. JS-gated reward-only IR;
6. two-stage-gated standard IR; and
7. two-stage-gated reward-only IR.

The JS and two-stage thresholds are read from the completed, digest-verified
Stage 5A artifact. Entropy-only gating is intentionally absent: Stage 5A is the
qualification experiment, while Stage 5B tests the two mechanisms with the
clearest posterior-disagreement and consequence interpretations.

## Evaluation and endpoints

Confirmation uses three paired replicates. Each strong/weak, condition, and
policy-mode cell receives 50 sampled and 25 deterministic episodes. This is 84
units and 3,150 confirmation episodes, following 270 calibration episodes.
Every comparison shares explicit episode-step policy/imagination random keys
and an audited RLScape reset sequence.

Primary endpoints are success on the weak and strong tasks and paired change
against the corresponding always-on objective. Secondary endpoints are gate
activation, actor updates, imagined transitions, wall time, and throughput.
The central result is a separation, not merely high activation: useful gating
should preserve the weak-task benefit while reducing needless strong-task
updates and damage.

Actor changes remain episode-local. The world model, reward/continuation heads,
critic, normalizers, and source checkpoint are read-only.

## Result

Calibration selected bury bones as weak and chop logs as competent. All 84
confirmation units and 3,150 episodes completed after 270 calibration episodes.
Always-on reward-only achieved `.653/.653` sampled/deterministic success on
bury. JS-gated reward-only fell to `.320/.480`, while two-stage reward-only fell
to `.280/.240`. JS did activate more often on weak bury than strong chop, but
withheld too much useful repair; the two-stage gate activated roughly ten times
more often on strong chop than weak bury. The necessity-discrimination
hypothesis is rejected. Compact results and figures are in
`results/rlscape/m4_reservoir_3goal_500k_seed0/stage5b_need_discrimination` and
Study IX of the research diary.

## Execution

After Stage 5A has written `stage5a_complete`:

```bash
xvfb-run -a -s "-screen 0 1280x1024x24" \
  python scripts/rlscape_stage5b_need_discrimination.py \
  --checkpoint "$CHECKPOINT" \
  --stage5a-root "$STAGE5A_ROOT" \
  --output-root "$STAGE5B_ROOT"
```

The same command safely resumes interrupted units. Use `--phase calibrate` or
`--phase confirm` only when deliberately separating the phases. Inspect with:

```bash
python scripts/rlscape_stage5_status.py "$STAGE5B_ROOT"
```

## Claim boundary

Stage 5B tests per-state necessity discrimination under episode-local repair.
It does not establish persistent recovery, trust-region safety, or continual
learning. Those belong to Stages 5C, 5D, and 5E respectively.
