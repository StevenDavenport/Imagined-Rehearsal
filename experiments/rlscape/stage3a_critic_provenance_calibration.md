# RLScape Stage 3A: Critic Provenance and Calibration Audit

**Status:** Complete (August 2026). Hypothesis supported and refined: critic
error is present on real posterior states and amplified by actor-selected
imagination, but does not progressively inflate with horizon.

## Outcome

Both immutable checkpoints completed the full 384-episode audit with unchanged
parameter digests. On the same 48,299 factual states, the weighted online-value
bias grows from +1.184 at 1.25M to +1.595 at 1.50M, while mean rank correlation
falls sharply. The slow critic reproduces the same bias. Full imagined targets
plateau by `H=3`--`H=6`, ruling out long-horizon accumulation as the primary
origin. Actor-selected trajectories nevertheless exceed recorded-action
targets by approximately 0.13--0.21, almost entirely through larger critic
bootstrap rather than predicted reward.

The complete analysis and portable aggregate artifacts are in
[`STAGE3A_FINDINGS.md`](../../results/rlscape/m4_reservoir_3goal_500k_seed0/stage3a_critic_provenance_calibration/STAGE3A_FINDINGS.md).

## Hypothesis

Standard imagined rehearsal fails because erroneous critic bootstrap targets
dominate otherwise useful imagined rewards. The audit must distinguish whether
the error is already present on real posterior states, accumulates through
bootstrapping, or is amplified when the actor enters exploitable imagined
states.

## Fixed sources

- Pinned 1.25M strong and 1.5M collapsed checkpoints from canonical Stage 0.
- The immutable 384-episode Stage 1A state bank.
- The paired posterior anchors and random-key scheme from Stage 1B.
- 128 MCPB posterior samples per anchor--goal proposal.
- All agent parameters remain read-only.

## Required measurements

1. Sweep the active goal and all counterfactual goals at identical real
   posterior states for online value, slow value, reward, and continuation.
2. Compare online and slow value against observed discounted return and
   episode success, reporting calibration and ranking separately.
3. Decompose every imagined target into predicted reward, online/slow value,
   reward-only lambda return, and critic-bootstrap contribution.
4. Evaluate prefixes at horizons 1, 3, 6, 15, and 20 to locate the onset of inflation.
5. Compare policy-selected imagined actions with recorded replay-action controls from
   the same anchors. This separates general drift from actor-selected model
   exploitation.

The implementation generates one 20-step actor trajectory per anchor--goal
pair and analytically recomputes every shorter prefix target using Dreamer's
exact lambda-return recurrence. It similarly generates one longest available
recorded-action control and reuses its valid prefixes. This avoids five
separate world-model rollouts per proposal. Actor-versus-recorded contrasts use
only anchors where both proposals exist; actor-only horizon summaries retain
all anchors. The raw tables expose this scope explicitly.

Realized return for the factual recorded trajectory is taken directly from the
replay episode. Counterfactual goals have no oracle realized-return target.
Actor-selected trajectories also have no realized-return target because their
actions differ from the recorded behavior; they are compared with the paired
recorded-action proposal instead of being mislabeled as calibrated outcomes.
Executing actor-selected imagined actions in live RLScape clones is deliberately
left as a later targeted validation after this audit identifies suspicious
anchor/horizon cells.

## Decision outcomes

- **Real-state miscalibration:** prioritize bounded success-probability value
  targets and counterfactual goal supervision.
- **Horizon-dependent bootstrap inflation:** test clipped, mixed, or
  reward-only tail targets.
- **Policy-only inflation:** prioritize conservative model-use constraints and
  action-distribution trust regions.
- **Slow critic repeats online error:** treat it as a stabilizer rather than an
  independent source of truth.

This stage diagnoses the teacher. It does not update the critic, actor, or
world model and does not establish a continual-learning improvement.

## Run

Stage 1A's selection and summaries must remain at the historical
`<experiment-root>/head_audit/` path:

```bash
python scripts/rlscape_stage3a_critic_provenance.py \
  --experiment-root logs/rlscape/m4_reservoir_3goal_500k_seed0 \
  --dry-run

python scripts/rlscape_stage3a_critic_provenance.py \
  --experiment-root logs/rlscape/m4_reservoir_3goal_500k_seed0
```

No X server or RLScape process is required because this is a checkpoint/replay
audit. The restart-safe output root is
`<experiment-root>/stage3a_critic_provenance_calibration/`, containing atomic
per-episode shards, raw and aggregate rollout tables, real-state calibration,
paired actor-versus-recorded contrasts, figures, integrity hashes, and the
`stage3a_complete` marker. The immutable specification records the source Git
commit, Python/RLScape versions, checkpoint seed, posterior seed schedule, and
its own companion SHA-256 digest; `commands.json` records both exact child
commands.
