# Stage 4B: Conservative Bounded-Value Dose and Stability

## Status and question

Status: **ready to run**.

Stage 4A repaired counterfactual reward semantics and learned a bounded
success-value head, but full bounded-value IR remained task-dependent and was
especially unsafe on `chop_logs`. Stage 4B asks a narrower causal question:

> Can a small amount of bounded value advice improve the already successful
> reward-only IR teacher without reducing worst-goal performance?

This is an episode-local mechanism experiment. It does not yet claim continual
learning, gating, or persistent actor updates.

## Fixed source

All conditions load the same Stage 4A `counterfactual_bounded` checkpoint:

```text
<stage4a-root>/checkpoints/counterfactual_bounded
```

Its actor, encoder, RSSM, decoder, repaired reward/continuation heads, and
bounded-value head are held fixed at load. Only a temporary, episode-local actor
clone is changed by IR. The live checkpoint is hashed before the immutable
Stage 4B specification is written.

## Teaching objective

Let the critic-free reward-only advantage be

\[
  A_R = R_R^\lambda / \sigma_R,
\]

and let the bounded-value advantage be

\[
  A_B = (R_B^\lambda - V_B) / \sigma_R.
\]

Stage 4B uses the interpolation

\[
  A_\beta = A_R + \beta\,
  \operatorname{clip}(A_B-A_R,-c,c).
\]

Thus `beta=0` is exactly reward-only IR, while `beta=1` with clipping disabled
is exactly the Stage 4A bounded-value objective. The added term is only the
incremental value correction; imagined reward is not counted twice. Entropy is
zero in every adaptation arm. The world model, reward, continuation, bounded
value, and historical critic are frozen. One actor update is made at every
nonterminal environment step using 128 Monte Carlo posterior samples.

## Prespecified conditions

| Condition | H | beta | Correction clip | Update RMS cap | Purpose |
|---|---:|---:|---:|---:|---|
| `frozen` | -- | -- | -- | -- | No-adaptation reference |
| `reward_only_h15` | 15 | 0 | off | off | Primary IR reference |
| `mixed_b005_h15` | 15 | 0.05 | off | off | Very small value dose |
| `mixed_b010_h15` | 15 | 0.10 | off | off | Small value dose |
| `mixed_b025_h15` | 15 | 0.25 | off | off | Moderate value dose |
| `mixed_b025_clip050_h15` | 15 | 0.25 | 0.50 | off | Conservative correction cap |
| `mixed_b025_ucap15e6_h15` | 15 | 0.25 | off | 1.5e-5 | Conservative actor-step cap |
| `mixed_b025_h6` | 6 | 0.25 | off | off | Horizon control |
| `mixed_b100_h15` | 15 | 1.00 | off | off | Full bounded-objective anchor |

The update cap was fixed from the scale of Stage 4A actor updates before any
Stage 4B outcomes were observed. It is applied after Adam scaling and preserves
the update direction. No condition is selected or altered after looking at
interim outcomes.

## Pairing and evaluation budget

The assay uses three prespecified replicates. Each replicate has a distinct
environment seed and a distinct internal-agent seed. Within a replicate, all
conditions share:

- the exact RLScape goal/reset-seed sequence;
- posterior-sampling keys indexed by episode and environment step; and
- sampled-policy action keys indexed by the same episode and step.

The explicit episode-step keying prevents different earlier episode lengths
from shifting later random streams. Existing experiments retain their legacy
process-counter randomness; this behavior is opt-in for Stage 4B.

For each of three goals and each condition, every replicate runs 50 sampled and
25 deterministic episodes. This is 675 episodes per condition and 6,075 total
episodes. The launcher is restart-safe, retries failed child processes up to
three times, verifies checkpoint integrity, and refuses to combine mismatched
reset schedules or RNG contracts.

## Endpoints and decision rule

Primary endpoints are:

1. macro mean success across the three goals;
2. worst-goal success;
3. paired macro change from `reward_only_h15`; and
4. worst-goal paired change from `reward_only_h15`.

Per-goal paired bootstrap intervals, improved/harmed discordant episode counts,
and between-replicate ranges are supporting diagnostics. A small-dose condition
is considered safe enough to advance only if it has positive sampled macro
change, sampled and deterministic worst-goal changes no lower than -0.05, and
positive sampled macro change in at least two of three replicates. Otherwise
reward-only IR remains the teaching signal for later gating experiments.

## Artifacts

The canonical run root contains the immutable specification, restart queue,
exact child commands, raw evaluation logs, pairing audit, raw and aggregated
CSV/JSON tables, four figure families, and `stage4b_complete`. The status helper
is `scripts/rlscape_stage4b_status.py`.

## Claim boundary

A positive result would establish a conservative episode-local actor teaching
objective suitable for a later gate. It would not establish that the same
objective remains safe under persistent updates, changing world models, longer
task sequences, or multiple task orders. Those claims belong to Stages 5 and 6.
