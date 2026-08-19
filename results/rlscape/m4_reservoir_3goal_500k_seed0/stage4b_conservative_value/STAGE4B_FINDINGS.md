# Stage 4B findings: conservative bounded-value dose and stability

## Completion and integrity

Stage 4B completed all 162 prespecified evaluation units and all 6,075
requested RLScape episodes. The matrix contains nine conditions, three goals,
three independent evaluation/adaptation replicates, 50 sampled episodes, and
25 deterministic episodes per condition--goal--replicate cell. No child
attempt failed. The pairing audit passed: within each replicate, every
condition used the same environment reset identities and explicit
episode-step-indexed posterior and policy random keys.

All conditions loaded the same Stage 4A `counterfactual_bounded` checkpoint.
The world model and all prediction heads were frozen. Adaptation was
episode-local, used one actor update at each nonterminal real step, 128 Monte
Carlo posterior samples, learning rate `4e-5`, and zero entropy bonus. The
mixed teaching advantage was

```text
A_beta = A_reward + beta * clip(A_bounded - A_reward, -c, c).
```

Consequently `beta=0` is exactly reward-only IR and unclipped `beta=1` is
exactly the Stage 4A bounded-value objective. Intermediate doses add only the
incremental bounded-value correction and do not count imagined reward twice.

## Primary result

Small bounded-value doses substantially improved sampled-policy performance:

| Condition | Sampled mean | Sampled worst | Change vs reward-only | Worst-goal change |
|---|---:|---:|---:|---:|
| Frozen | 0.620 | 0.500 | -0.118 | -0.300 |
| Reward-only, H=15 | 0.738 | 0.513 | 0.000 | 0.000 |
| Mixed beta=.05, H=15 | **0.876** | **0.787** | **+0.138** | **+0.040** |
| Mixed beta=.10, H=15 | 0.824 | 0.600 | +0.087 | +0.020 |
| Mixed beta=.25, H=15 | 0.840 | 0.760 | +0.102 | -0.020 |
| Mixed beta=.25, clip=.50 | 0.862 | 0.747 | +0.124 | 0.000 |
| Mixed beta=.25, update cap | 0.836 | 0.767 | +0.098 | -0.080 |
| Mixed beta=.25, H=6 | 0.811 | 0.773 | +0.073 | -0.027 |
| Pure bounded value, beta=1 | 0.722 | 0.613 | -0.016 | -0.187 |

The beta=.05 sampled macro improvement was positive in all three replicates
(+0.067, +0.133, and +0.213). Its paired goal changes relative to reward-only
were +0.273 kill goblin, +0.040 bury bones, and +0.100 chop logs. Thus the
headline gain is neither one-task compensation nor one lucky adaptation seed.

## Sampled--deterministic divergence

Reward-only remained the strongest deterministic treatment:

| Condition | Deterministic mean | Deterministic worst | Change vs reward-only | Worst-goal change |
|---|---:|---:|---:|---:|
| Frozen | 0.587 | 0.427 | -0.227 | -0.333 |
| Reward-only, H=15 | **0.813** | **0.760** | 0.000 | 0.000 |
| Mixed beta=.05, H=15 | 0.796 | 0.627 | -0.018 | -0.133 |
| Mixed beta=.10, H=15 | 0.796 | 0.667 | -0.018 | -0.093 |
| Mixed beta=.25, H=15 | 0.720 | 0.653 | -0.093 | -0.120 |
| Mixed beta=.25, clip=.50 | 0.716 | 0.587 | -0.098 | -0.173 |
| Mixed beta=.25, update cap | 0.751 | 0.707 | -0.062 | -0.213 |
| Mixed beta=.25, H=6 | 0.649 | 0.613 | -0.164 | -0.280 |
| Pure bounded value, beta=1 | 0.724 | 0.573 | -0.089 | -0.187 |

For beta=.05 the deterministic loss is concentrated in kill goblin
(-0.133), while chop logs improves by +0.093 and bury bones changes by -0.013.
The mixed update therefore places substantially more useful probability mass
in the sampled policy without reliably moving the modal action sequence in the
same direction. This distinction matters for later persistent adaptation: a
sampled gain is real behavior, but it is not yet evidence of a uniformly safer
policy update.

## Controls and prespecified decision

The controls sharpen the interpretation.

- Reducing the moderate-dose horizon from 15 to 6 lowered sampled mean success
  from 0.840 to 0.811 and deterministic mean from 0.720 to 0.649. The useful
  correction is not reproduced by simply shortening imagination.
- Pure bounded-value teaching (`beta=1`) was slightly worse than reward-only
  sampled and clearly worse deterministic. The bounded head contains useful
  information, but is not trustworthy as a replacement teacher.
- Correction clipping and the actor-update RMS cap did not remove the
  sampled--deterministic trade-off. The update cap improved neither the
  deterministic mean nor the worst-goal safety threshold.

The advancement rule required positive sampled macro change, positive sampled
change in at least two of three replicates, and sampled and deterministic
worst-goal changes no lower than -0.05. Beta=.05 and beta=.10 satisfy the
sampled clauses, but their deterministic worst-goal changes are -0.133 and
-0.093. Every other mixed condition also violates at least one safety clause.
Therefore **no condition passes the complete prespecified advancement rule**,
and reward-only H=15 remains the approved teaching signal for Stage 5.

## Scientific interpretation and boundary

Stage 4B establishes that the repaired bounded head contains behaviorally
useful information: a five-percent correction produces a large, replicated,
all-goal sampled improvement over reward-only IR. It simultaneously shows
that the information is not calibrated strongly enough to use without a gate
or additional policy constraint. Increasing its dose is non-monotonic, and
pure bounded-value teaching loses the benefit.

This is an episode-local, single-source-checkpoint mechanism result. It does
not establish persistent continual updates, robustness across independently
trained world models, or safety under changing replay distributions. The next
algorithmic experiment should retain reward-only IR as its prespecified base,
use the Stage 4B result to motivate—but not automatically activate—a small
value correction, and test a gate/trust region against both sampled gain and
deterministic worst-task harm. Training-time critic regularisation is now a
separate well-motivated intervention rather than something to fold into this
completed assay post hoc.
