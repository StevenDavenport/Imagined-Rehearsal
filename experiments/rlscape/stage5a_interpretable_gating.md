# Stage 5A: Interpretable Gating for Imagined Rehearsal

## Status and question

Status: **complete; hypothesis rejected**.

Stage 5A asks whether a state-dependent gate can improve episode-local IR by
avoiding unnecessary or harmful actor updates. Compute saving is an important
secondary endpoint, but gate selection is based on behavior and safety rather
than a required compute reduction.

The fixed source is the Stage 4A `counterfactual_bounded` checkpoint. The
world model, repaired reward/continuation heads, bounded value head, historical
critic, normalizers, and persistent actor remain unchanged. Temporary actor
adaptation is discarded at every episode boundary.

## Signals

At an eligible factual posterior, the audit records:

1. normalized entropy of the factual actor distribution;
2. mean normalized actor entropy over 128 Monte Carlo posterior samples;
3. normalized Jensen--Shannon disagreement among those policies; and
4. consequence evidence from H=15 imagination: predicted goal-success rate,
   concentration of successful first actions, and divergence between the
   actor mixture and the success-conditioned action distribution.

Entropy and JS are deliberately separate. High entropy with low JS denotes
shared action ambiguity, whereas high JS denotes disagreement induced by
posterior uncertainty. Consequence evidence tests whether uncertainty contains
a consistent model-predicted corrective direction.

## Gate families

- **Entropy gate:** adapt when factual normalized actor entropy crosses its
  frozen threshold.
- **JS gate:** adapt when normalized MCPB disagreement crosses its threshold.
- **Two-stage gate:** run the cheap entropy-or-JS screen first. Only passing
  states pay for consequence imagination; IR runs only when at least 4/128
  rollouts predict success and the success-conditioned action divergence also
  crosses its threshold.

The consequence rollout is passed directly into the actor update. It is never
recomputed after the gate accepts it. Gating is disabled by default outside
explicit Stage 5A configurations.

## Workflow

The restart-safe launcher has three ordered phases:

1. `audit`: no IR plus always-on standard, reward-only, and beta=.05 IR, with
   all features recorded. Candidate thresholds are the 50th, 75th, and 90th
   empirical quantiles.
2. `qualify`: entropy and JS candidates are assessed first on reward-only and
   beta=.05. Their selected thresholds define the cheap two-stage screen, after
   which three consequence thresholds are assessed. Thresholds maximize
   sampled macro success subject to a deterministic worst-goal floor of -0.05
   versus the corresponding always-on objective. Standard IR is excluded from
   selection and remains a negative-control generalization test.
3. `confirm`: freeze the selected threshold artifact and run no IR; always-on
   standard, reward-only, and beta=.05; and entropy-, JS-, and two-stage-gated
   versions of all three objectives.

Confirmation uses three held-out replicates, three goals, 50 sampled and 25
deterministic episodes per replicate/cell: 234 units and 8,775 episodes. It
retains Stage 4B's explicit episode-step RNG pairing and reset audit.

## Compute endpoints

Each episode records cheap-gate and consequence-gate calls, posterior samples,
imagined transitions, actor updates, gate/imagination/update time, total
evaluation time, throughput, and host/device usage where available. JAX work is
synchronized by fetching gate/update metrics before timers stop. Runtime
comparisons must be made within one physical machine; the planned execution
host is the available 4090.

There is no minimum compute-saving requirement. A gate advances when it
improves sampled macro success over its always-on counterpart, stays within the
-0.05 deterministic/worst-goal safety floor, and improves in at least two of
three confirmation replicates.

## Result

All 234 confirmation units and 8,775 episodes completed. Always-on reward-only
IR achieved `.822` sampled and `.653` deterministic macro success. Entropy-,
JS-, and two-stage-gated reward-only achieved `.611/.516`, `.611/.596`, and
`.549/.338`, respectively. No threshold family passed the prespecified
advancement rule. The gates reduced direct updates and imagined work, but the
two-stage consequence screen remained costly despite applying only about
4--5 updates per 100 real steps. The fixed gate hypothesis is rejected; compact
results and figures are in
`results/rlscape/m4_reservoir_3goal_500k_seed0/stage5a_gating` and Study VIII of
the research diary.

## Execution

Inspect the immutable plan without launching RLScape:

```bash
python scripts/rlscape_stage5a_gating.py \
  --stage4a-root logs/rlscape/m4_reservoir_3goal_500k_seed0/stage4a_counterfactual_head_repair \
  --dry-run
```

Run all phases inside `tmux` under Xvfb:

```bash
xvfb-run -a -s "-screen 0 1280x1024x24" \
  python scripts/rlscape_stage5a_gating.py \
  --stage4a-root logs/rlscape/m4_reservoir_3goal_500k_seed0/stage4a_counterfactual_head_repair
```

The launcher can be safely restarted with the identical command. Phase-only
runs use `--phase audit`, `--phase qualify`, or `--phase confirm`. Status is:

```bash
python scripts/rlscape_stage5a_status.py \
  logs/rlscape/m4_reservoir_3goal_500k_seed0/stage5a_gating
```

## Claim boundary

Stage 5A evaluates an interpretable gate, not a learned gate, a KL trust
region, or persistent actor adaptation. Stage 5B tests whether the frozen gate
can distinguish states where IR is behaviorally useful from states belonging
to an already-competent policy. Stage 5C then tests persistent recovery. Hard-KL
update safety is reserved for Stage 5D.
