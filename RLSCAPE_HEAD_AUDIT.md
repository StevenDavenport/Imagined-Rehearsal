# RLScape Stage 1: goal-conditioned head audit

This audit compares the strong 1.25M-step checkpoint with the weak 1.5M-step
checkpoint without training or interacting with RLScape. It answers whether
the reward, continuation, online value, and slow-value heads still provide a
usable goal-conditioned learning signal for imagined rehearsal.

## Controlled comparison

The 1.25M milestone replay is the immutable data source for both checkpoints.
A persisted selection contains, when available, 64 successful and 64 failed
complete episodes for each of the three trained goals. The selection records
the replay-manifest hash and exact episode IDs. Both checkpoints therefore see
the same observations, actions, targets, and episode ordering.

Each checkpoint re-encodes the old images through its own encoder and RSSM.
At every posterior state, the latent is held fixed while the one-hot goal is
swept through all five model goal slots. The first three slots are trained
goals; light-fire and catch-fish are deliberately retained as unused negative
controls.

The audit reports:

- reward discrimination, calibration, false positives, and completion-event
  goal-confusion;
- continuation calibration, with goal completion separated from physical
  termination and ordinary states;
- online and slow critic calibration against both inclusive and exclusive
  discounted Monte Carlo returns;
- critic success ranking at episode starts under every counterfactual goal;
- online-versus-slow critic disagreement and strong-to-weak per-state drift;
- head uncertainty and posterior entropy/RMS diagnostics;
- SHA-256 parameter digests before and after inference, proving that the
  checkpoint weights were not mutated.

The critic targets are diagnostic Monte Carlo targets, not claims about the
exact Dreamer lambda-return training target. Both common timestep alignments
are retained so an apparent off-by-one convention cannot determine the result.

## Run

Run this inside the `crl-rlscape` environment on the GPU machine. Replace the
experiment root with the directory containing `milestones/` from the completed
three-goal reservoir experiment.

```bash
python scripts/rlscape_head_audit.py \
  --experiment-root logs/rlscape/<reservoir-experiment-root> \
  --strong-step 1250000 \
  --weak-step 1500000 \
  --data-step 1250000 \
  --stream-output
```

The workflow is restart-safe. It reuses a valid completed checkpoint audit and
refuses to combine results produced from a different replay selection.

Outputs are written under `<experiment-root>/head_audit/`:

- `selection.json` and `audit_spec.json`: exact comparison specification;
- `strong/` and `weak/`: raw prediction arrays, JSON summary, and CSV tables;
- `figures/`: completion confusion, head separation, critic calibration, and
  per-state checkpoint-drift plots;
- `comparison_summary.json` and `comparison_complete`: combined result and
  terminal marker.

Before launching the full audit, the command can be checked without loading a
model:

```bash
python scripts/rlscape_head_audit.py \
  --experiment-root logs/rlscape/<reservoir-experiment-root> \
  --dry-run
```
