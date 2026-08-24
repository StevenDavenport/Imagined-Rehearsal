# Stage 5B.2: Paired Virtual-Update Gate

## Status and question

Status: **implemented; awaiting execution**.

Stage 5B.1 found that absolute JS disagreement did not transfer as a direct
performance threshold. It also separated two questions that a useful IR gate
must answer:

1. does the current actor appear likely to underperform; and
2. would the proposed IR intervention improve that actor?

Stage 5B.2 tests the second question causally inside imagination while retaining
the first as a measured baseline-performance signal. It asks:

> Can a paired, held-out imagined-performance comparison safely decide whether
> to commit a tentative reward-only IR update?

This remains an investigation. Every accepted and rejected proposal is logged,
and retrospective labels expose both successes and failure modes.

## Gate algorithm

At a probe state, the evaluator:

1. clones evaluation-local actor and optimizer state if necessary;
2. evaluates the current actor on held-out imagination batch `B`, recording
   reward-only return, imagined success rate, JS, and entropy;
3. generates independent imagination batch `A` and performs one tentative
   reward-only actor update on `A`;
4. evaluates the tentative actor on batch `B` using exactly the same posterior
   and action-sampling random numbers as step 2;
5. commits the actor and optimizer state only when the mean held-out
   reward-return difference is strictly positive; otherwise it restores the
   exact pre-update actor and optimizer state.

The critic, reward/continuation heads, representation, dynamics, normalizers,
and the source checkpoint remain frozen. Batch `A` prevents approval through
training-batch overfitting; common random numbers on `B` reduce comparison
variance. The source parameters remain immutable and adaptation is local to an
episode.

The baseline score is logged on every probe. Its hard screen is deliberately
disabled in this first causal test (`max_baseline_score=1e30`) because Stage
5B.1 rejected a portable absolute threshold. This lets the experiment measure
whether baseline score later adds useful discrimination without contaminating
the primary virtual-update result.

## Lifecycle and cadence

Each episode starts dormant and probes immediately.

- A rejected dormant probe schedules the next probe after either 25 or 100
  real steps, depending on condition.
- An accepted probe activates IR. While active, every real step proposes and
  checks one update.
- Three consecutive rejected active proposals return the gate to dormancy.
- Terminal observations never trigger unused adaptation.

The 25- and 100-step dormant variants test whether occasional checks retain the
benefit while reducing rejected-probe compute. Active cadence and hysteresis
are identical, isolating dormant cadence.

## Prespecified conditions

All three established goals are evaluated in sampled and deterministic modes:

1. `no_ir`;
2. `always_reward_only`;
3. `virtual_reward_dormant_25`; and
4. `virtual_reward_dormant_100`.

Defaults are three paired replicates, 50 sampled episodes and 25 deterministic
episodes per goal, condition, and replicate. This is 2,700 recorded episodes,
plus one unrecorded compilation warm-up per unit. Environment reset seeds and
agent random streams are paired within every replicate/goal/mode cell.

IR uses `H=15`, 128 Monte Carlo posterior samples, one actor update, learning
rate `4e-5`, zero actor-entropy bonus, and the reward-only objective.

## Measurements

Every probe records:

- held-out reward return before and after the tentative update;
- predicted improvement and accept/reject decision;
- success fraction, entropy, and JS before and after;
- active/dormant state and rejection streak;
- tentative, committed, and rejected actor updates;
- rollout, update, wall-clock, posterior-sample, and imagined-transition cost.

Real outcomes provide retrospective episode classes only after execution:
`ir_needed`, `ir_unnecessary`, `ir_harmful`, and `not_repaired`. These labels
are never visible to the gate. Analysis reports false rejection of known repair
opportunities, updates applied to already competent episodes, actual gated
success, successful completion length, and compute relative to no IR and
always-on IR.

## Execution

```bash
CHECKPOINT="logs/rlscape/m4_reservoir_3goal_500k_seed0/stage4a_counterfactual_head_repair/checkpoints/counterfactual_bounded"
STAGE5B1_ROOT="logs/rlscape/m4_reservoir_3goal_500k_seed0/stage5b1_performance_gate_audit"
STAGE5B2_ROOT="logs/rlscape/m4_reservoir_3goal_500k_seed0/stage5b2_virtual_update_gate"

xvfb-run -a -s "-screen 0 1280x1024x24" \
  python scripts/rlscape_stage5b2_virtual_update_gate.py \
  --checkpoint "$CHECKPOINT" \
  --stage5b1-root "$STAGE5B1_ROOT" \
  --output-root "$STAGE5B2_ROOT" \
  --max-attempts 4
```

The runner is immutable-spec, restartable, and retry-aware. Re-running the same
command resumes completed units. A changed specification requires a new output
root.

## Claim boundary

A positive result qualifies the virtual comparison as a prospective episodic
gate under these three tasks and this checkpoint. It does not yet establish
multi-episode persistent recovery, full continual-learning integration, or
generalisation to unseen goals. A negative result remains informative because
the paired trace distinguishes inaccurate imagined scoring, harmless rejected
updates, false accepts, false rejects, and excessive compute.
