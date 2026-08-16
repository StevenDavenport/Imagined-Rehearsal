# Stage 3B findings: natural forgetting recovery

## Result in one sentence

Critic-free reward-only imagined rehearsal transferred from the synthetic
actor-corruption assay to naturally forgotten checkpoint policies: it improved
mean real-environment success at both audited checkpoints and produced a large
recovery of bury-bones behavior at 1.50M, whereas standard critic-bootstrapped
IR remained strongly task-dependent and sometimes destructive.

## Question and precommitted design

Stage 3B asks whether the successful Stage 2 mechanism repairs natural
continual-learning damage without collecting new training experience. It
reuses the immutable 1.25M and 1.50M Stage 0 reservoir checkpoints and compares,
for every checkpoint, goal, policy mode, and paired reset seed:

1. the frozen checkpoint actor;
2. standard critic-bootstrapped IR at `H=6`;
3. reward-only IR at `H=6`, `H=15`, and `H=20`.

Each condition uses 100 sampled and 50 deterministic episodes per goal. IR is
episode-local, starts at the first action, takes one actor update per real
action, uses 128 Monte Carlo posterior samples and learning rate `4e-5`, and is
discarded on reset. Encoder, RSSM, decoder, reward, continuation, online critic,
slow critic, normalizers, replay, and persistent checkpoint parameters remain
frozen. Reward-only IR removes critic bootstrap and actor entropy but still
uses learned transition, reward, and continuation predictions.

The weak/competent label was fixed before this evaluation from Stage 0 sampled
success using the threshold `< 0.50`. At 1.25M, bury was labelled weak and kill
and chop competent. All three 1.50M goals were labelled weak. This label is an
oracle analysis stratum, not a deployable gate. In particular, the new paired
frozen evaluation measured 71% sampled success for 1.25M bury, so the
precommitted historical label and the current baseline are deliberately kept
distinct.

## Execution and integrity

- Git source: `f41f48747d225dc2957c2d675aaef9f4a962bc04`
- RLScape 0.1.6; Python 3.11.15; 28 x 18 click grid; 200-action episode limit.
- 60/60 evaluation units completed on their first attempt.
- 4,500/4,500 requested real-environment episodes completed: 3,000 sampled and
  1,500 deterministic.
- 4,500 reset identities matched across conditions; `pairing.json` reports no
  mismatch.
- Every row reports exact episode count, successful return code, and unchanged
  persistent parameter digest.
- Total wall-clock workflow time was 23.60 hours (13 August 17:40 to 14 August
  17:16, local execution-host time).
- The evaluations consumed approximately 623,247 real actions. Adapted arms
  performed approximately 489,099 actor-update triggers in total and about 689M
  imagined latent transitions; these are compute diagnostics rather than
  additional environment experience.

The exact immutable specification and its SHA-256 companion are
`experiment_spec.json` and `experiment_spec_digest.json`. The authoritative raw
episodes and checkpoints remain under the matching `logs/rlscape/` root on the
execution host.

## Complete real-environment success matrix

### Sampled policy (100 episodes per cell)

| Checkpoint | Condition | Kill | Bury | Chop | Mean |
|---|---|---:|---:|---:|---:|
| 1.25M | Frozen | 0.56 | 0.71 | 0.84 | 0.703 |
| 1.25M | Standard IR, H=6 | 0.61 | 0.07 | 0.53 | 0.403 |
| 1.25M | Reward-only, H=6 | 0.74 | 0.95 | 0.75 | 0.813 |
| 1.25M | Reward-only, H=15 | 0.76 | 0.99 | 0.76 | 0.837 |
| 1.25M | Reward-only, H=20 | 0.73 | 1.00 | 0.88 | **0.870** |
| 1.50M | Frozen | 0.23 | 0.14 | 0.03 | 0.133 |
| 1.50M | Standard IR, H=6 | 0.46 | 0.02 | 0.09 | 0.190 |
| 1.50M | Reward-only, H=6 | 0.37 | 0.17 | 0.11 | 0.217 |
| 1.50M | Reward-only, H=15 | 0.20 | **0.91** | 0.08 | 0.397 |
| 1.50M | Reward-only, H=20 | 0.44 | 0.78 | 0.08 | **0.433** |

### Deterministic policy (50 episodes per cell)

| Checkpoint | Condition | Kill | Bury | Chop | Mean |
|---|---|---:|---:|---:|---:|
| 1.25M | Frozen | 0.64 | 0.36 | 0.78 | 0.593 |
| 1.25M | Standard IR, H=6 | 0.60 | 0.12 | 0.34 | 0.353 |
| 1.25M | Reward-only, H=6 | 0.80 | 0.90 | 0.70 | 0.800 |
| 1.25M | Reward-only, H=15 | 0.60 | 0.96 | 0.78 | 0.780 |
| 1.25M | Reward-only, H=20 | 0.76 | 0.98 | 0.80 | **0.847** |
| 1.50M | Frozen | 0.26 | 0.04 | 0.04 | 0.113 |
| 1.50M | Standard IR, H=6 | 0.36 | 0.00 | 0.12 | 0.160 |
| 1.50M | Reward-only, H=6 | 0.24 | 0.12 | 0.14 | 0.167 |
| 1.50M | Reward-only, H=15 | 0.38 | **0.86** | 0.04 | **0.427** |
| 1.50M | Reward-only, H=20 | 0.18 | 0.78 | 0.10 | 0.353 |

## Paired effects

The primary estimates compare episode outcomes at the same checkpoint, goal,
policy mode, and reset-seed position. The reported 95% intervals are paired
episode bootstrap intervals; they quantify evaluation-episode uncertainty, not
variation across independently trained agents.

At 1.25M, reward-only IR improves sampled aggregate success from 70.3% to
81.3%, 83.7%, and 87.0% as the horizon increases from 6 to 20. H=20 therefore
adds 16.7 percentage points in aggregate. Goal-level effects include kill
`+0.20 [0.08, 0.32]` at H=15, bury `+0.29 [0.20, 0.38]` at H=20, and chop
`+0.04 [-0.06, 0.14]` at H=20. Shorter reward-only horizons slightly damage
the already-strong chop policy: `-0.09 [-0.19, 0.01]` at H=6 and
`-0.08 [-0.19, 0.03]` at H=15.

At 1.50M, frozen sampled mean is only 13.3%. Reward-only IR raises it to 21.7%,
39.7%, and 43.3%. The decisive cell is bury bones: H=15 changes sampled success
from 0.14 to 0.91, a paired `+0.77 [0.68, 0.85]`, and deterministic success
from 0.04 to 0.86, `+0.82 [0.70, 0.92]`. H=20 remains strong but is lower:
sampled `+0.64 [0.53, 0.74]` and deterministic `+0.74 [0.62, 0.86]`. H=20 also
improves sampled kill from 0.23 to 0.44, `+0.21 [0.09, 0.33]`, but changes its
deterministic result from 0.26 to 0.18, `-0.08 [-0.26, 0.10]`.

Across the 12 preclassified weak sampled condition-horizon cells,
reward-only IR improves 11 and harms one. Across the six competent sampled
cells it improves four and harms two. This is promising evidence for a future
gate, but the current label uses privileged historical knowledge and cannot be
treated as a deployable detection rule.

Standard H=6 IR is not simply ineffective. At 1.25M it reduces sampled mean by
30.0 points and deterministic mean by 24.0 points. Bury falls by
`-0.64 [-0.74, -0.54]` sampled, and chop by `-0.31 [-0.42, -0.20]`. At 1.50M,
however, standard IR raises sampled kill by `+0.23 [0.10, 0.36]`. The critic
teacher is therefore conditionally useful but not safe enough for unconditional
test-time adaptation.

## Interpretation

Stage 3B supports its main hypothesis. Stage 2 did not merely exploit an
artificial zero-mask corruption: critic-free imagined reward can repair at
least one severe natural forgetting failure and improve aggregate behavior at
both checkpoints using no new reward-labelled training transitions.

The result also sharpens the mechanism claim. Standard IR and reward-only IR
use the same checkpoint, posterior starts, actor optimizer, and learned world
model. Their principal difference is the critic-bootstrap teaching signal
(with the historical entropy term also absent in reward-only IR). The reversal
from large standard-IR damage to reward-only recovery is consistent with the
Stage 1B and Stage 2 conclusion that value bootstrap, not the adaptation
machinery itself, is the main unsafe component.

This is not yet a complete continual-learning algorithm. Adaptation resets at
every episode, the best horizon depends on checkpoint, goal, and action mode,
and chop remains difficult after the final collapse. A single training seed
cannot establish population-level reliability. The next immediate experiment
remains Stage 3A, which is already implemented and audits where critic error
enters the target. Stage 3B also supplies a concrete design target for the
later gate/trust-region stage: preserve the large weak-task gains while
preventing unnecessary changes to competent policies.

## Portable artifacts

- `main_results.csv`: all 60 absolute result rows and adaptation diagnostics.
- `comparison_vs_frozen.csv`: paired deltas, intervals, discordance counts, and
  prespecified status labels.
- `aggregate_results.csv`: checkpoint/status/family summaries.
- `pairing.json`: reset-pairing integrity record.
- `figures/`: sampled/deterministic success matrices and paired-effect plots.
