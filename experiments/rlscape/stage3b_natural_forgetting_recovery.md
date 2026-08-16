# RLScape Stage 3B: Natural Forgetting Recovery

**Status:** Complete (14 August 2026). Hypothesis supported with important
task- and horizon-specific exceptions.

**Execution priority:** Run before Stage 3A.

## Outcome

All 60 evaluation units and 4,500 paired real-environment episodes completed
without a retry or integrity mismatch. Reward-only IR improved mean success at
both natural checkpoints. The clearest recovery was 1.50M bury bones at
`H=15`: sampled success rose from 14% to 91% and deterministic success from 4%
to 86%. Standard `H=6` IR remained unsafe, reducing the 1.25M sampled mean from
70.3% to 40.3% even though it helped 1.50M kill goblin.

The complete result matrix, paired intervals, interpretation, compact CSV/JSON
artifacts, and figures are in
[`results/rlscape/m4_reservoir_3goal_500k_seed0/stage3b_natural_forgetting_recovery`](../../results/rlscape/m4_reservoir_3goal_500k_seed0/stage3b_natural_forgetting_recovery/STAGE3B_FINDINGS.md).

## Hypothesis

Critic-free reward-only imagined rehearsal can recover naturally forgotten
goal-conditioned behaviour from continual-training checkpoints, rather than
only repairing the synthetic actor corruption used in Stage 2.

## Fixed sources

- Existing immutable Stage 0 checkpoints; no continual retraining.
- Initial focus on the 1.25M and 1.5M checkpoints.
- Kill-goblin, bury-bones, and chop-logs evaluation clones.
- Episode-local actor adaptation discarded after every episode.

## Primary comparison

For each checkpoint, goal, policy mode, and paired reset seed:

1. frozen checkpoint actor;
2. standard critic-bootstrapped IR at the historical six-step setting;
3. reward-only IR at horizons 6, 15, and 20.

Use 100 sampled and 50 deterministic episodes per cell. The fixed matrix has
60 paired units and 4,500 real-environment episodes. Keep the encoder, RSSM, decoder, reward,
continuation, online critic, slow critic, normalizers, replay, and persistent
checkpoint state frozen. Use one actor update per real action, 128 MCPB
posterior samples, and learning rate `4e-5`, matching Stage 2.

## Primary outcomes

- Absolute success after adaptation.
- Paired change from the frozen actor.
- Recovery on goals classified as naturally weak at that checkpoint.
- Damage on goals classified as retained/competent at that checkpoint.
- Actor KL, entropy, update norm, imagined reward incidence, and computation.

Weak versus competent status is fixed before the new evaluation from Stage 0's
historical sampled-policy success. The prespecified threshold is 50%:

| Checkpoint | Kill | Bury | Chop |
|---|---:|---:|---:|
| 1.25M | 56% (competent) | 49% (weak) | 88% (competent) |
| 1.50M | 16% (weak) | 8% (weak) | 5% (weak) |

This classification is an oracle mechanism-analysis label, not a deployable
gate. The exact values and rule are persisted in `experiment_spec.json` before
any child evaluation starts.

## Interpretation boundary

A positive result establishes checkpoint-local behavioural recovery without
new training experience. It does not establish persistent retention. A gate
or trust region belongs in Stage 5 only after this experiment identifies where
unconditional reward-only IR helps and harms.

## Run

The command reuses the existing immutable milestones and does not retrain the
1.5M-step continual learner:

```bash
xvfb-run -a -s "-screen 0 1280x1024x24" \
  python scripts/rlscape_stage3b_natural_forgetting.py \
  --experiment-root logs/rlscape/m4_reservoir_3goal_500k_seed0
```

Inspect the exact plan without launching RLScape:

```bash
python scripts/rlscape_stage3b_natural_forgetting.py \
  --experiment-root logs/rlscape/m4_reservoir_3goal_500k_seed0 \
  --dry-run
```

The restart-safe workflow writes under
`<experiment-root>/stage3b_natural_forgetting_recovery/`: immutable
specification, queue state, exact commands, per-unit logs, paired reset audit,
raw and baseline-relative result tables, aggregate weak/competent tables,
figures, and `stage3b_complete` only after every integrity check passes.
The immutable specification also records the source Git commit, Python and
RLScape versions, complete reset-seed schedule, and a companion SHA-256 digest.
