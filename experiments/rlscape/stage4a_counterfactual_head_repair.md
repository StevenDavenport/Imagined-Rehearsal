# RLScape Stage 4A: Counterfactual Head Repair

**Status:** Implemented; GPU execution pending. No result claimed.

## Purpose

Stage 4A asks whether explicit goal semantics can repair the learned teaching
signal before imagined rehearsal is made persistent across episodes. It has
five linked phases:

1. audit FIFO and reservoir checkpoints on one identical held-out state bank;
2. clone the reservoir checkpoint into matched repair conditions;
3. train reward and continuation with factual-only or counterfactual labels;
4. optionally train a separate bounded goal-success value head from factual
   Monte Carlo returns;
5. repeat the held-out audit and compare paired real-environment IR outcomes.

The experiment does not update either historical checkpoint and does not yet
perform persistent multi-episode IR.

## Source checkpoints and shared state bank

The default comparison uses the 1.25M FIFO and reservoir milestones. The
reservoir milestone supplies a deterministic success/failure-balanced episode
bank because the FIFO replay need not retain complete balanced episodes for
old goals. The bank is split by episode identity before any repair:

- training: 96 successful and 96 failed episodes per trained goal;
- held out: 64 successful and 64 failed episodes per trained goal;
- goals: kill goblin, bury bones, and chop logs;
- no episode may occur in both splits.

Both original agents and every repaired clone are audited on the exact same
held-out observations and recorded actions. This controls state distribution
across checkpoint conditions. Because the shared bank originates from the
reservoir archive, FIFO results measure cross-bank generalization and must not
be interpreted as a native-replay estimate.

## Repair arms

| Arm | Reward/continuation training | Value training | IR use |
|---|---|---|---|
| `original` | None | Existing critic unchanged | Standard and reward-only controls |
| `factual_only` | Factual goal repeated across all five query slots | None | Reward-only |
| `counterfactual` | All five goals with event-valid targets | None | Reward-only |
| `counterfactual_bounded` | Same counterfactual targets | New bounded success head, factual targets only | Reward-only and bounded-bootstrap |

The factual-only arm repeats each factual example across five slots. It
therefore matches the counterfactual arm's optimizer steps, tensor shape, and
example count while withholding alternative-goal information.

## Counterfactual targets

For latent state (s_t), observed task goal (g_t), probed goal (g), sparse
reward (r_t\in\{0,1\}), factual completion indicator (e_t), and physical
terminal indicator (d_t):

\[
  r_t^{(g)} = r_t\,\mathbf{1}[g=g_t],
\]

\[
  c_t^{(g)} = 1 - \mathbf{1}[d_t \lor (e_t \land g=g_t)].
\]

Thus completion of goal A is a positive reward and stopping event for A, but
not for B--E. Physical terminal events stop every goal. These labels concern
the event represented by the current transition and are semantically valid.

The experiment deliberately does **not** label alternative-goal values as
zero. An episode ending after goal A censors what might have happened under a
goal-B policy. Counterfactual values are therefore policy-dependent and not
identified by the recorded trajectory.

## Bounded success-value head

The bounded arm adds a separate Bernoulli head

\[
  b_\psi(s_t,g_t) \in [0,1]
\]

and fits it to the factual discounted Monte Carlo completion return remaining
in the recorded episode. Training is goal- and outcome-balanced. The original
Dreamer online and slow critics remain unchanged. Bounded-bootstrap IR uses
this head directly and never silently clips the historical critic.

## Frozen components and integrity

During repair, the encoder, RSSM, decoder, actor, original online critic,
original slow critic, and all normalizers are frozen. Reward and continuation
are the only parameters updated in the first two repair arms. The bounded arm
also updates only the new bounded-value head. Dedicated repair optimizer state
is removed from exported evaluation checkpoints.

Every repair artifact records component digests before and after training and
must reject any change outside its permitted namespaces.
The supervisor also requires the counterfactual and
counterfactual-plus-bounded reward/continuation digests to be bit-identical;
their only permitted difference is the subsequent bounded-value update.

## Held-out audits

The existing same-state audit is run on:

- FIFO original;
- reservoir original;
- reservoir factual-only clone;
- reservoir counterfactual clone;
- reservoir counterfactual-plus-bounded clone.

Primary head endpoints are completion-event top-1 goal accuracy, matching
minus nonmatching reward margin, continuation stopping selectivity, ordinary
state false-positive rate, and factual calibration. The bounded head is
additionally evaluated for factual Brier/MAE, calibration, range, and initial
success ranking.

## IR assay

The default paired evaluation uses 100 sampled and 50 deterministic episodes
per goal and condition. The prespecified conditions are:

- frozen original;
- original standard IR at H=6;
- original reward-only IR at H=15;
- factual-only reward-only IR at H=15;
- counterfactual reward-only IR at H=15;
- counterfactual-plus-bounded reward-only IR at H=15;
- counterfactual-plus-bounded bounded-bootstrap IR at H=6 and H=15.

All conditions share checkpoint step, task, policy mode, reset seed schedule,
one update per real action, 128 Monte Carlo posterior samples, actor learning
rate (4\times10^{-5}), and zero entropy for the repaired objectives. Repair
checkpoints change heads only; the actor begins identically in every arm.

## Primary comparisons

1. FIFO versus reservoir original held-out goal selectivity.
2. Counterfactual versus factual-only head selectivity after matched updates.
3. Counterfactual-plus-bounded versus original critic calibration.
4. Counterfactual versus factual-only reward-only IR in real RLScape.
5. Bounded-bootstrap versus reward-only IR from the same repaired checkpoint.

## Claim boundary

Stage 4A can establish whether explicit goal supervision repairs head semantics
and improves episode-local IR. It cannot establish safe persistent adaptation,
learned gating, or population-level robustness from one training seed. Those
belong to Stages 4B and 5 after this mechanism test succeeds.

## Canonical execution

Run the entire workflow under one virtual display from the repository root:

```bash
xvfb-run -a -s "-screen 0 1280x1024x24" \
  python scripts/rlscape_stage4a_counterfactual_repair.py \
  --fifo-root logs/rlscape/e1_fifo_seq_actor_ir_seed0 \
  --reservoir-root logs/rlscape/m4_reservoir_3goal_500k_seed0
```

The launcher writes an immutable specification and resumable `queue.json`,
adopts its exact live child after a supervisor restart, retries failed units up
to three times, and writes `stage4a_complete` only after every audit,
evaluation, pairing check, table, and figure succeeds. Progress can be read
without attaching to the training terminal:

```bash
python scripts/rlscape_stage4a_status.py \
  logs/rlscape/m4_reservoir_3goal_500k_seed0/stage4a_counterfactual_head_repair
```
