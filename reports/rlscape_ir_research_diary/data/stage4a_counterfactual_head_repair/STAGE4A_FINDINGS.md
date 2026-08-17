# Stage 4A findings: counterfactual goal-head repair

## Status and provenance

All 56 scientific units completed: five held-out head audits, three repair
arms, and 48 real-environment evaluation cells. The evaluation contains all
3,600 requested episodes (100 sampled and 50 deterministic episodes for each
condition--goal pair), identical reset identities across conditions, and no
failed child attempt. The immutable specification digest is
`5a8d39b2bf3ce957a77c935178ff0bd4ceb5a75f609028ded0a919e3dc69ee28`;
the source code revision recorded by the run is
`9aa45a70d975cc79c430353480e451dfe504597d`.

The original supervisor wrote the authoritative result tables but stopped
during final plotting because an integer goal index was passed to
`str.replace`. This post-processing-only defect left `queue.json` marked
`running` and prevented `stage4a_complete` from being written; it did not
affect any audit, repair, checkpoint, environment seed, or evaluation result.
The plotting typo is fixed in the maintained launcher, and the research diary
regenerates all report figures directly from the compact CSV/JSON artifacts.

## Design

The experiment uses the reservoir 1.25M checkpoint and a disjoint,
episode-balanced state bank. Repair uses 96 successful and 96 failed episodes
per trained goal; the held-out audit uses 64 successful and 64 failed episodes
per goal. Reward and continuation receive 2,000 updates with batch size 256,
learning rate `4e-5`, 24 sampled states per episode, and an event fraction of
0.5. The factual control repeats the factual query five times. The
counterfactual arm queries all five goal IDs and labels a completion reward and
stop only for the completed goal. The bounded arm additionally fits a separate
Bernoulli success-value head for 2,000 updates to factual discounted completion
returns. Encoder, RSSM, decoder, actor, original online/slow critics, and
normalizers remain frozen.

The eight paired actor-assay conditions are frozen original; original standard
IR at H=6; original, factual-repaired, counterfactual-repaired, and
counterfactual-plus-bounded reward-only IR at H=15; and bounded-bootstrap IR at
H=6 and H=15. Adaptation uses one temporary actor update per real action, 128
Monte Carlo posterior samples, and actor learning rate `4e-5`. Entropy is zero
for repaired/reward-only objectives; the historical standard control retains
`3e-4`. Adaptation resets at every episode.

## Held-out head repair

Counterfactual supervision repairs goal semantics decisively. On 192 held-out
completion events, reservoir-original top-1 goal identification is 33.3% and
its matching-minus-nonmatching reward margin is 0.004. Factual-only matched
training reaches 34.4% and 0.020. Counterfactual training reaches 97.4% and
0.567. At the same events, predicted continuation is 0.108 for the completed
goal and 0.698 for other goals, compared with 0.043 and 0.057 in the reservoir
original. Thus the original head tends to stop every queried goal, whereas the
counterfactual head learns goal-specific stopping. Factual event detection
remains excellent (reward AUROC 0.9996), although the positive-event reward
mean falls from 0.959 to 0.785.

The separate bounded head repairs value scale. For kill, bury, and chop, the
unchanged original critic predicts means 1.623, 1.694, and 0.819 against
realized targets 0.194, 0.078, and 0.224. The bounded head predicts 0.180,
0.069, and 0.226. Absolute bias therefore falls from 1.429/1.616/0.595 to
0.014/0.009/0.002. Initial-state success AUROC improves from
0.400/0.109/0.570 to 0.568/0.848/0.632. Calibration is not perfect--bounded
value MAE is 0.179/0.084/0.173--but the scale and bury ranking failure are
substantially repaired.

## Real-environment actor assay

Mean success across kill, bury, and chop is:

| Condition | Sampled | Deterministic |
|---|---:|---:|
| Frozen original | 69.7% | 59.3% |
| Original standard IR, H=6 | 34.0% | 38.0% |
| Original reward-only, H=15 | 80.3% | **78.0%** |
| Factual reward-only, H=15 | **88.0%** | 76.0% |
| Counterfactual reward-only, H=15 | 81.7% | 77.3% |
| Counterfactual + bounded reward-only, H=15 | 78.7% | 74.0% |
| Counterfactual + bounded bootstrap, H=6 | 73.3% | 68.7% |
| Counterfactual + bounded bootstrap, H=15 | 78.3% | 56.0% |

The standard critic-backed objective reproduces its known failure: sampled
bury falls from 69% to 7% and chop from 85% to 49%. Original reward-only IR
raises sampled kill by 14 percentage points and bury by 28, but lowers chop by
10. Factual-repaired reward-only IR gives the best sampled profile
(86%/96%/82%), including a worst-goal success of 82%. Counterfactual
reward-only IR remains strong (75%/96%/74%) but does not outperform its matched
factual control. This rejects the simple hypothesis that better completion
goal selectivity must immediately produce better actor adaptation in this
arena.

The repaired bounded bootstrap is much safer than the original critic in
aggregate, especially at H=6, but it is not uniformly safe. H=6 improves
sampled kill by 32 points and deterministic kill by 22, while reducing sampled
chop by 24. H=15 improves sampled kill/bury by 27/19 points but reduces sampled
chop by 20 and deterministic chop by 32. The bounded head therefore restores a
usable value scale without yet supplying a universally reliable actor teacher.

The counterfactual and counterfactual-plus-bounded checkpoints have identical
actors and bit-identical reward/continuation heads. Their reward-only outcomes
nevertheless differ because the separate evaluation processes pair
environment reset identities, not the Monte Carlo posterior/random-action
streams used inside adaptation. The difference between those two reward-only
rows must therefore be treated as a stochastic reproducibility diagnostic,
not a causal bounded-head effect. A future matched assay should explicitly
pair all agent RNG streams or evaluate repeated adaptation seeds.

## Conclusion and claim boundary

Stage 4A establishes three results. First, explicit counterfactual event labels
can repair reward and continuation goal semantics on held-out real states.
Second, a separately bounded success head can remove the original critic's
gross optimism and materially improve success ranking. Third, neither repair
automatically dominates the simpler reward-only actor objective. Factual-only
reward repair is the strongest sampled teacher, while bounded-bootstrap IR
still trades kill/bury gains against chop damage.

The results justify conservative value reintroduction as an ablation target,
not deployment of the current bounded objective. The next stage should isolate
value dose and stochastic adaptation variance before designing a persistent
gate and trust region. These are single-checkpoint, single-training-seed,
episode-local results and do not yet establish continual actor updates.
