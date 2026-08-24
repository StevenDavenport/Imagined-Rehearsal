# Stage 5B.1: Offline Performance-Gate Audit

## Status and question

Status: **implemented; awaiting execution**.

Stage 5B showed that posterior JS disagreement was the most promising cheap
signal, but a global binary threshold did not preserve reward-only IR on the
weak task. Stage 5B.1 returns to the underlying question:

> Can a signal available before an outcome predict that the current policy
> will underperform, and does that relationship transfer across tasks?

This is a strictly offline audit. It performs no environment interaction, no
imagination updates, and no policy changes.

## Data and labels

Stage 5A's no-IR audit prospectively recorded at every state:

- factual actor entropy;
- mean posterior-conditioned actor entropy;
- Monte Carlo Posterior Batching JS disagreement;
- H=15 imagined success frequency;
- successful-action concentration; and
- success-conditioned action divergence.

Only prefixes of 1, 5, 10, and 25 factual states are used as predictors. Each
episode is labelled later using its real outcome and its reset-matched
always-on reward-only episode:

- `ir_needed`: no IR fails and reward-only IR succeeds;
- `ir_unnecessary`: both succeed;
- `ir_harmful`: no IR succeeds and reward-only IR fails; or
- `not_repaired`: both fail.

The labels are retrospective, but every feature is prospective. They are used
to audit information content, never as deployable gate inputs.

## Analyses

For each task and policy mode, the audit produces:

1. JS distributions separated by eventual real success/failure;
2. empirical calibration of JS against underperformance and completion time;
3. JS versus imagined-success planes coloured by paired necessity class;
4. ROC AUC for JS, entropy, predicted failure, normalized JS, and simple
   JS--predicted-failure combinations;
5. task-specific optimal JS thresholds and a full cross-task threshold-transfer
   matrix; and
6. the performance of the frozen Stage 5A global threshold on each task.

The principal diagnostic is cross-task transfer. A high within-task score with
poor off-diagonal transfer means the threshold is task-specific rather than a
general estimate of performance. Strong overlap between all labelled groups
means JS cannot be the complete gate regardless of threshold tuning.

## Candidate next mechanism

The historical traces do not contain a held-out estimate of whether a proposed
IR update improves imagined performance. If JS and predicted performance remain
insufficient, the next small intervention is cross-validated imagined
improvement:

1. propose an IR actor update on imagination batch A;
2. evaluate the current and proposed actors on independent batch B; and
3. apply the update only when a conservative lower bound on the predicted
   performance difference is positive.

This relative comparison is intended to avoid task-specific absolute
thresholds. It will be tested separately; Stage 5B.1 does not manufacture this
missing signal from retrospective outcomes.

## Execution

```bash
python scripts/rlscape_stage5b1_performance_gate_audit.py \
  --stage5a-root "$STAGE5A_ROOT" \
  --stage5b-root "$STAGE5B_ROOT" \
  --output-root "$STAGE5B1_ROOT"
```

The output contains `episode_features.csv`, `feature_separability.csv`,
`threshold_transfer.csv`, `js_calibration.csv`, figures in PNG and PDF,
`STAGE5B1_FINDINGS.md`, an immutable source manifest, and a terminal completion
marker.

## Claim boundary

This audit can reject a threshold or identify candidate feature geometry. It
cannot prove causal usefulness, qualify an online gate, or justify persistent
Stage 5C by itself. Any nominated rule must be frozen and tested on held-out
episodes before persistent adaptation.
