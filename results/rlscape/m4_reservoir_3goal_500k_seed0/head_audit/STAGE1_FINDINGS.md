# Stage 1 head-audit findings

The audit compares the 1.25M strong checkpoint against the 1.5M weak
checkpoint on the identical immutable 1.25M replay bank: 64 successful and 64
failed episodes for each trained goal (384 episodes, 48,299 states). Parameter
digests are identical before and after both audits.

## Main findings

1. **Factual reward prediction is excellent and survives the collapse.**
   Reward AUROC is 0.99993 at 1.25M and 0.99991 at 1.5M. Mean reward on genuine
   completion states is 0.961 and 0.957, while ordinary-state false positives
   remain very small.

2. **The reward is not functionally goal-selective.** On completion states,
   changing only the one-hot goal leaves the prediction near one. The matching
   versus nonmatching reward margin is only 0.0033 at 1.25M and 0.0018 at
   1.5M. Correct-goal top-1 identification is 39.6% and 38.0% despite five goal
   slots (three trained and two unused controls).

3. **Continuation has the same semantic failure.** It accurately stops at
   factual episode completion and continues on ordinary states, but it also
   predicts a 93.9%/94.3% stop probability for the wrong goals. The matching
   stop probabilities are only slightly higher: 95.6%/95.9%.

4. **The critic is severely overoptimistic.** Recorded discounted returns lie
   in [0, 1]. At 1.25M, mean factual critic predictions are 1.62, 1.73, and
   0.79 for kill, bury, and chop versus empirical means 0.20, 0.10, and 0.21.
   At 1.5M they rise to 1.99, 2.00, and 1.36 even though behavioral success
   collapses.

5. **State ranking also deteriorates.** Factual value--return Pearson
   correlations change from 0.182/0.107/0.307 to 0.048/0.064/0.061. At episode
   starts, value scarcely separates successful from failed kill trajectories,
   reverses the ordering for bury, and weakens for chop.

6. **The slow critic is not an independent safeguard.** Mean absolute online
   versus slow-value disagreement is only 0.006--0.017. Both critics share the
   same optimism and drift.

7. **There is no obvious scalar posterior collapse.** Posterior categorical
   entropy changes modestly from 0.564 to 0.543 and deterministic-state RMS
   from 0.256 to 0.262. These summaries cannot rule out task-relevant
   representation drift, but they do rule out a gross entropy or magnitude
   collapse.

## Interpretation

Reservoir replay preserved factual examples but did not identify the desired
counterfactual semantics. Every observed completion was paired only with its
active goal. The model could therefore minimise factual training loss by using
the visual completion event and largely ignoring the one-hot goal. It was
never taught that the same state should have zero reward and continue under a
different goal.

This is directly hostile to imagined rehearsal: an imagined completion-like
state can produce reward and termination for almost any requested goal. The
critic then bootstraps through those model predictions and becomes confidently
optimistic. The weak checkpoint's higher values during worse real behavior are
consistent with this failure mechanism.

The Monte Carlo calibration comparison is diagnostic rather than an exact
critic loss reconstruction: replay actions are off-policy relative to each
checkpoint and Dreamer trains lambda returns in imagination. However, the
directional strong-to-weak comparison uses identical states, and values above
the environment's observed maximum return plus the inverse relationship with
actual evaluation success remain serious evidence of critic miscalibration.

## Recommended next audit

Audit six-step policy-imagined rollouts from these same posterior anchors.
Record reward events, continuation, lambda returns, bootstrap contribution,
and critic values under all goal IDs. This will locate whether optimism enters
through imagined reward hallucination, continuation timing, value bootstrap,
or a combination. Actor corruption/recovery should follow only after this
objective is shown to be trustworthy or deliberately replaced by an oracle
diagnostic objective.
