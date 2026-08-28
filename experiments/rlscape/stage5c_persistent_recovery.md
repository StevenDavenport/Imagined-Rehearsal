# Stage 5C: Persistent Self-Terminating Recovery

## Status and question

Status: **implemented; execution deferred after Stage 5B.2**.

Stage 5B.2 showed that the current prospective gate can approve severe
real-environment damage despite paired held-out imagination. Persisting that
decision across episodes would amplify a known error, so this workflow remains
available but must not be launched until an episode-local gate qualifies.

Stage 5C asks whether IR can repair an actor over multiple episodes and whether
a frozen gate naturally stops updating once repair is no longer required. It
also measures the complementary failure mode: whether always-on or gated
recovery damages a retained competent task.

Two source assays are run independently:

- **natural:** the naturally weak checkpoint used by Stage 5B; and
- **corrupted:** a controlled actor-only corruption derived from that exact
  checkpoint, verified through its immutable corruption manifest.

Separate roots allow these assays to run concurrently on different machines
without sharing mutable queue state.

## Persistence contract

Persistence is a new opt-in evaluation setting. The repository default remains
`eval_adapt.persistence=episode`, which discards all adapted parameters at each
episode boundary exactly as before. Stage 5C explicitly requests `run` mode.

In `run` mode, the actor clone and the dedicated IR optimizer state survive
episode boundaries. Recovery is divided into ten blocks of ten episodes. Each
block atomically exports an ordinary Dreamer checkpoint plus a manifest; the
next block reloads the actor and optimizer state. World-model, reward,
continuation, critic, representation, and normalizer digests are checked before
publication. A block with any frozen-component change is invalid.

Each output checkpoint has its own `done` marker and attempt directory. The
supervisor can recover a completed child after disconnection and never resumes
from a partial checkpoint.

## Conditions and fixed budget

The same seven Stage 5B conditions are used: no IR; always-on standard and
reward-only IR; JS-gated standard and reward-only IR; and two-stage-gated
standard and reward-only IR. This exposes recovery speed for both IR objectives
rather than presuming reward-only is always superior.

Every trajectory receives exactly 100 recovery episodes. There is no online
early stopping: stopping at the first apparent recovery would hide later damage
or renewed gate activation. “Self-termination” is measured post hoc as two
consecutive blocks below a 0.05 activation rate.

Before recovery and after every ten episodes, an independent clean evaluation
loads the current exported actor and performs no adaptation. It measures:

- target-task success in 20 sampled and 10 deterministic episodes;
- retained-task success under the same budgets;
- first stable recovery above 0.70 for two consecutive blocks;
- first stable gate shutoff for two consecutive blocks;
- actor updates after recovery, total imagined transitions, and wall time; and
- retained-task change from the initial checkpoint.

Clean evaluation never writes back into the recovery chain. Reset and random
keys are paired by block, task role, and policy mode across all conditions.

## Execution

Run the natural source:

```bash
xvfb-run -a -s "-screen 0 1280x1024x24" \
  python scripts/rlscape_stage5c_persistent_recovery.py \
  --source-kind natural \
  --source-checkpoint "$NATURAL_CHECKPOINT" \
  --stage5a-root "$STAGE5A_ROOT" \
  --stage5b-root "$STAGE5B_ROOT" \
  --output-root "$STAGE5C_NATURAL_ROOT"
```

Run the controlled-corruption assay in a different root (and, optionally, on a
different host):

```bash
xvfb-run -a -s "-screen 0 1280x1024x24" \
  python scripts/rlscape_stage5c_persistent_recovery.py \
  --source-kind corrupted \
  --source-checkpoint "$CORRUPTED_CHECKPOINT" \
  --stage5a-root "$STAGE5A_ROOT" \
  --stage5b-root "$STAGE5B_ROOT" \
  --output-root "$STAGE5C_CORRUPTED_ROOT"
```

Both commands are restart-safe. Status is:

```bash
python scripts/rlscape_stage5_status.py "$STAGE5C_NATURAL_ROOT"
```

## Claim boundary

Stage 5C is a persistent actor-recovery assay, not full continual training. It
does not mix current-task gradient learning with reservoir preservation, and it
does not yet apply hard-KL rollback. Stage 5D studies update safety; Stage 5E
then integrates the supported mechanisms into the continual learner.
