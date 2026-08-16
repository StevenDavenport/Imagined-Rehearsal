# Stage 3A findings: critic provenance and calibration

## Result in one sentence

The critic failure begins on real posterior states, worsens between the 1.25M
and 1.50M checkpoints, and is then amplified by actor-selected imagination;
it is not primarily created by increasing imagination horizon, and the slow
critic reproduces rather than corrects the error.

## Question and fixed design

Stage 3A diagnoses why standard critic-bootstrapped imagined rehearsal is an
unsafe actor teacher. It reuses the immutable Stage 0 checkpoints and the
immutable 384-episode Stage 1A state bank:

- **strong checkpoint:** 1.25M environment steps;
- **weak checkpoint:** 1.50M environment steps;
- 1,433 valid posterior anchors from 384 replay episodes;
- all five one-hot goal queries at every anchor;
- 128 Monte Carlo posterior samples per anchor--goal proposal;
- actor-selected and recorded-action imagined proposals;
- analytically reused prefixes at `H = 1, 3, 6, 15, 20` from one maximum
  20-step rollout;
- online value, slow value, reward-only return, bootstrap contribution,
  continuation, entropy, and imagined-state diagnostics;
- no actor, critic, world-model, optimizer, replay, or checkpoint update.

Real-state calibration uses the factual goal and the realized discounted
return remaining in the recorded episode. Counterfactual goals and
actor-selected trajectories are not assigned false oracle targets. Instead,
actor-selected trajectories are compared with recorded-action proposals from
the same posterior anchors wherever sufficient future recorded actions exist.

## Execution and integrity

- Git source: `f41f48747d225dc2957c2d675aaef9f4a962bc04`.
- RLScape 0.1.6; Python 3.11.15; checkpoint seed 0.
- Both checkpoint audits completed and wrote `audit_complete`.
- Each checkpoint processed 384 episodes, 35,825 actor prefix summaries and
  26,365 paired recorded-action prefix summaries: 62,190 rows per checkpoint,
  124,380 total.
- The actor proposal alone represents 7,165 anchor--goal maximum-horizon
  rollouts and 18,342,400 imagined latent transitions per checkpoint.
- Parameter digests before and after each audit are identical.
- The source selection hash and canonical experiment-specification digest
  match their persisted companions.
- `stage3a_complete` was written only after the combined tables and figures
  were produced.

The raw 58 MB rollout tables and per-episode shards remain under the matching
`logs/rlscape/` root on office-gpu. This portable directory contains the
complete aggregate evidence and integrity summaries.

## 1. The critic is already wrong on real posterior states

The same 48,299 factual replay states and realized-return targets are evaluated
under both checkpoints. The targets average 0.176 when weighted by state
count. The online critic instead predicts 1.359 at 1.25M and 1.771 at 1.50M.

| Checkpoint | Target mean | Online prediction | Bias | MAE | Mean Pearson | Mean Spearman |
|---|---:|---:|---:|---:|---:|---:|
| 1.25M strong | 0.176 | 1.359 | +1.184 | 1.192 | 0.198 | 0.213 |
| 1.50M weak | 0.176 | 1.771 | +1.595 | 1.595 | 0.058 | 0.081 |

The bias is not a small calibration offset. In this environment the factual
episode can produce at most one sparse completion reward, yet mean predicted
remaining values exceed one for the aggregated strong checkpoint and all
three weak-checkpoint goals. Ranking is also poor, and it nearly disappears
at 1.50M. The critic therefore cannot be repaired for IR merely by subtracting
a global offset.

### Goal-level calibration

| Checkpoint | Goal | Target | Prediction | Bias | Pearson | Spearman |
|---|---|---:|---:|---:|---:|---:|
| 1.25M | Kill goblin | 0.198 | 1.618 | +1.421 | 0.182 | 0.121 |
| 1.25M | Bury bones | 0.104 | 1.725 | +1.621 | 0.107 | 0.121 |
| 1.25M | Chop logs | 0.215 | 0.794 | +0.579 | 0.307 | 0.396 |
| 1.50M | Kill goblin | 0.198 | 1.991 | +1.793 | 0.048 | 0.057 |
| 1.50M | Bury bones | 0.104 | 2.005 | +1.900 | 0.064 | 0.103 |
| 1.50M | Chop logs | 0.215 | 1.357 | +1.142 | 0.061 | 0.082 |

The deterioration is synchronized with behavioral collapse. On the identical
state bank, weighted critic bias grows by 0.412 (34.8%) from the strong to the
weak checkpoint, while mean rank correlation falls sharply.

## 2. The slow critic is not an independent source of truth

The slow critic has weighted bias +1.190 at 1.25M and +1.603 at 1.50M, almost
the same as the online critic. Their mean absolute prediction gap is only
0.0098 and 0.0136 respectively. Goal-level online--slow gaps remain tiny beside
biases of 0.58--1.90.

This rules out using the existing slow critic as a trustworthy replacement
teacher. It is a temporal stabilizer for the same learned target, not an
independently calibrated evaluator.

## 3. Longer imagination does not create runaway target inflation

The actor-selected factual IR target is already dominated by bootstrap at
`H=1` and changes little after `H=3`.

| Checkpoint | H | Full target | Reward-only | Bootstrap | Mean bootstrap fraction |
|---|---:|---:|---:|---:|---:|
| 1.25M | 1 | 1.443 | 0.052 | 1.391 | 95.9% |
| 1.25M | 6 | 1.480 | 0.072 | 1.407 | 94.4% |
| 1.25M | 20 | 1.482 | 0.106 | 1.376 | 91.7% |
| 1.50M | 1 | 1.848 | 0.036 | 1.812 | 97.5% |
| 1.50M | 6 | 1.891 | 0.039 | 1.852 | 97.4% |
| 1.50M | 20 | 1.891 | 0.050 | 1.841 | 96.9% |

There is no progressive explosion from `H=6` to `H=20`. At the strong
checkpoint, added horizon increases predicted reward while the bootstrap
component slightly falls. At the weak checkpoint, full return and bootstrap
are essentially flat after `H=3`. The weak model imagines *less* reward than
the strong model while assigning much larger total return. This directly
localizes the checkpoint difference to value bootstrap rather than reward
accumulation.

Recorded-action factual proposals are also severely optimistic relative to
their realized returns. Their mean return bias ranges from +1.17 to +1.37 at
1.25M and +1.50 to +1.73 at 1.50M across the five horizons. Actor exploitation
is therefore not required for critic failure: the error exists along replay
actions and on real posterior starts.

## 4. The actor selects critic-optimistic trajectories

Relative to paired recorded actions, the actor selects imagined trajectories
with higher full return at every checkpoint and horizon.

| Checkpoint | H | Actor minus recorded full target | Bootstrap difference | Reward-only difference |
|---|---:|---:|---:|---:|
| 1.25M | 1 | +0.139 | +0.211 | -0.072 |
| 1.25M | 6 | +0.126 | +0.138 | -0.012 |
| 1.25M | 20 | +0.142 | +0.129 | +0.013 |
| 1.50M | 1 | +0.212 | +0.298 | -0.086 |
| 1.50M | 6 | +0.175 | +0.194 | -0.019 |
| 1.50M | 20 | +0.182 | +0.196 | -0.015 |

At the collapsed checkpoint, the actor's apparent advantage is entirely a
larger bootstrap value while its trajectories receive *lower* predicted
reward. This is the expected signature of critic/model exploitation rather
than genuine imagined task progress. The effect is strongest for bury bones:
at `H=6`, actor minus recorded bootstrap is +0.266 at 1.25M and +0.289 at
1.50M, while the reward-only differences are -0.036 and -0.043.

The actor amplification is secondary rather than causal origin. It adds about
0.13--0.14 full-return units at the strong checkpoint and 0.17--0.21 at the
weak checkpoint, on top of real-state biases already exceeding one.

## 5. Goal semantics remain weak in the imagined target

Factual and counterfactual actor-selected aggregates are nearly identical. At
`H=6`, their mean full targets differ by only 0.0008 at both checkpoints; at
`H=20`, the difference remains below 0.008. Mean reward-only differences are
below 0.005 at the headline horizons. This agrees with the direct Stage 1A
same-state goal sweep: the requested one-hot goal has little influence on the
teaching signal compared with checkpoint identity and critic scale.

This aggregate comparison is supportive rather than a replacement for the
Stage 1A counterfactual confusion matrix, because Stage 3A pools the four
alternative goal queries into the counterfactual relationship.

## Mechanistic conclusion

The audit distinguishes the candidate causes:

1. **Real-state miscalibration is primary.** The critic is structurally
   overoptimistic and poorly ranked before imagination begins.
2. **Continual checkpoint drift worsens it.** The identical replay state bank
   receives substantially higher and less informative values at 1.50M.
3. **Actor-selected imagination amplifies it.** The actor preferentially enters
   higher-bootstrap trajectories, especially on bury, without increasing
   predicted reward.
4. **Longer horizons are not the main source.** The full target plateaus early;
   there is no `H=6` to `H=20` runaway.
5. **The slow critic offers no protection.** It closely reproduces online
   critic predictions and errors.

This refines the earlier model-exploitation hypothesis. Exploitation is real,
but it is an amplifier of an already corrupted value landscape rather than the
sole origin of the critic's optimism.

## Consequences for the experimental ladder

- Do not use the current online or slow critic as an unconditional IR teacher.
- Reward-only IR remains the justified default for the next intervention.
- A future conservative critic mixture must first bound values to the task's
  success-return semantics and demonstrate calibration on real states.
- Counterfactual goal supervision is still required: neither factual replay
  nor the present critic distinguishes requested goals strongly enough.
- A Stage 5 gate/trust region should monitor adaptation need and policy
  movement, not trust absolute critic magnitude or online--slow disagreement.
- The next constructive experiment should repair counterfactual goal
  conditioning and bounded value targets, then repeat Stages 1A and 3A before
  another costly continual-training run.

## Portable artifacts

- `real_state_calibration.csv`: real-posterior online/slow critic calibration.
- `combined_rollout_summary.csv`: checkpoint, proposal, relationship, goal,
  and horizon target decompositions.
- `actor_vs_recorded.csv`: paired actor-versus-recorded proposal contrasts.
- `comparison_summary.json`: complete audit summaries and integrity metadata.
- `strong/` and `weak/`: checkpoint-local aggregate tables, specifications,
  summaries, and terminal markers.
- `figures/`: calibration, horizon provenance, and proposal-comparison plots.
