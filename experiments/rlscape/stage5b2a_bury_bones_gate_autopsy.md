# Stage 5B.2a: Bury-Bones Virtual-Gate Autopsy

## Status and question

Status: **ready to run; hypothesis unresolved**.

Stage 5B.2 produced a qualitatively exceptional failure. On sampled
`bury_bones`, no IR achieved `.373`, always-on reward-only IR achieved `.660`,
and the 25-step virtual gate achieved exactly `.000`. The gated actor received
IR in every episode and its frozen learned model predicted improvement, yet no
episode recorded any task progress. This study asks:

> Does virtual gating collapse bury bones because it applies an insufficient
> dose, selectively accepts model-exploiting updates, or follows a hallucinated
> reward attractor in the learned world model?

This is a targeted failure autopsy, not a new general gate qualification.

## Prespecified design

Only `bury_bones` is run, with sampled policies as the primary and sole default
endpoint. Three replicates use identical RLScape reset schedules and explicit
episode-step policy random keys. The `121000/131000` environment/actor seed
bases exactly reproduce the shared Stage 5B.2 Table 15 cells before the new
controls branch away. Each main condition records 50 episodes; a
non-intervening world-model audit records 20 episodes with probes every 25
steps.

The reward-only matrix is:

1. no IR;
2. no IR with frozen imagined-reward auditing;
3. always-on IR;
4. first-update caps of 16, 46, 80, and 127 updates per episode;
5. 46 randomly scheduled updates in a full 200-step episode;
6. the original positive virtual selector with 25-step dormant cadence; and
7. its negative complement, which commits candidates predicted not to improve.

A secondary exact `beta=0.005` mixed-bounded signal includes always-on,
46-update cap, matched random schedule, and positive virtual selection. This is
not the historical `mixed_b005` label from Stage 4B, which denoted
`beta=0.05`; the distinction is stored explicitly in the experiment spec.

All IR conditions use `H=15`, 128 Monte Carlo posterior samples, one actor
update at learning rate `4e-5`, zero entropy bonus, an episode-local actor, and
frozen critic/world model/representation/reward/continuation components.

## Hallucinated-reward audit

Every consequence imagination now reports raw frozen-model diagnostics:

- predicted reward mean, sum, maximum, and 95th-percentile peak;
- predicted reward-event rate and peak/first-event timestep;
- the modal imagined action;
- the modal action immediately preceding peak predicted reward; and
- concentration of both action distributions.

These measurements do not affect optimization or gate decisions. They are
paired with real click-grid heatmaps, click entropy, top-cell concentration,
task progress, factual `item_buried` events, update count, and episode length.
The runner also verifies that merely probing the world model does not change
the no-IR sampled action sequence.

## Interpretation contract

- Collapse under both the 46-update cap and random schedule supports a
  partial-dose recovery valley.
- Success under random scheduling but failure under virtual-positive selection
  identifies harmful selection by the imagined score.
- Improvement under the negative selector shows that the model ranks candidate
  updates in the wrong real-world direction.
- A repeated imagined reward/action attractor without real progress supports
  reward-model or dynamics exploitation.
- Protection from exact `beta=0.005` mixed IR indicates that a small independent
  value signal regularizes that exploitation.

No single comparison is permitted to establish world-model hallucination by
itself; the claim requires predicted-reward, action-attractor, and real-outcome
evidence to agree.

## Outputs

The resumable runner writes immutable specification and queue files, per-unit
console and integrity logs, `episode_diagnostics.csv`,
`imagined_reward_diagnostics.csv`, `real_action_histograms.csv`, an explicit
action-pairing audit, aggregate tables, generated findings, and PDF/PNG figures.

## Execution

```bash
CHECKPOINT="logs/rlscape/m4_reservoir_3goal_500k_seed0/stage4a_counterfactual_head_repair/checkpoints/counterfactual_bounded"
STAGE5B2_ROOT="logs/rlscape/m4_reservoir_3goal_500k_seed0/stage5b2_virtual_update_gate"
STAGE5B2A_ROOT="logs/rlscape/m4_reservoir_3goal_500k_seed0/stage5b2a_bury_autopsy"

xvfb-run -a -s "-screen 0 1280x1024x24" \
  python scripts/rlscape_stage5b2a_bury_autopsy.py \
  --checkpoint "$CHECKPOINT" \
  --stage5b2-root "$STAGE5B2_ROOT" \
  --output-root "$STAGE5B2A_ROOT" \
  --max-attempts 4
```

Re-running the identical command resumes completed units. Any changed
scientific setting requires a new output root.
