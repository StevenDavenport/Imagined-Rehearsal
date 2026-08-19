# Imported RLScape result artifacts

This directory contains compact audited results copied from the execution
host. It is not a substitute for the authoritative raw `logs/rlscape/` run
roots and immutable milestone archives.

The directory `m4_reservoir_3goal_500k_seed0` retains its historical execution
name. Canonical experiment identities are defined in
[`experiments/rlscape/registry.json`](../../experiments/rlscape/registry.json):

- the parent reservoir/FIFO comparison is **Stage 0**;
- `head_audit/` is **Stage 1A**;
- `imagination_audit/` is **Stage 1B**;
- the separately imported controlled-recovery tables are **Stage 2** and live
  in the self-contained research-diary package.
- `stage3b_natural_forgetting_recovery/` is the complete **Stage 3B** portable
  evidence package: immutable specification, all 60 compact result rows,
  paired effects, integrity record, figures, and the findings narrative.
- `stage3a_critic_provenance_calibration/` is the complete **Stage 3A**
  portable evidence package: real-state critic calibration, exact horizon
  target decomposition, actor-versus-recorded proposal contrasts, integrity
  summaries, figures, and the findings narrative.
- `stage4a_counterfactual_head_repair/` is the compact **Stage 4A** evidence
  package: immutable specification, held-out FIFO/reservoir/repaired head
  audits, all 48 paired evaluation cells, repair-integrity manifests, and the
  findings narrative.
- `stage4b_conservative_value/` is the compact **Stage 4B** evidence package:
  immutable nine-condition specification, complete 162-unit restart queue,
  all 6,075 episode outcomes, paired goal/condition summaries, pairing audit,
  five figure families, and the findings narrative.

Stage 3B completed all 4,500 episodes without retry or pairing failure. Its
principal result is that reward-only IR transfers from synthetic actor
corruption to natural forgetting, including a 14% to 91% sampled recovery for
1.50M bury bones at horizon 15. See
[`STAGE3B_FINDINGS.md`](m4_reservoir_3goal_500k_seed0/stage3b_natural_forgetting_recovery/STAGE3B_FINDINGS.md).

Stage 3A shows that critic overoptimism is already severe on real posterior
states, worsens at the collapsed checkpoint, and is amplified by actor-selected
imagination rather than created by increasing horizon. See
[`STAGE3A_FINDINGS.md`](m4_reservoir_3goal_500k_seed0/stage3a_critic_provenance_calibration/STAGE3A_FINDINGS.md).

Stage 4A shows that counterfactual event labels repair completion-goal
selectivity from 33.3% to 97.4% and that a bounded success head repairs value
scale, but neither change automatically outperforms factual reward-only IR in
the real environment. See
[`STAGE4A_FINDINGS.md`](m4_reservoir_3goal_500k_seed0/stage4a_counterfactual_head_repair/STAGE4A_FINDINGS.md).

Stage 4B shows that a five-percent bounded-value correction raises sampled
success from 73.8% to 87.6% relative to reward-only IR and improves all three
sampled goals in all three adaptation replicates. It does not pass the complete
safety rule because deterministic worst-goal success falls by 13.3 percentage
points. See
[`STAGE4B_FINDINGS.md`](m4_reservoir_3goal_500k_seed0/stage4b_conservative_value/STAGE4B_FINDINGS.md).

Historical paths are intentionally not rewritten inside CSV/JSON extracts.
They identify the exact source host and run from which each result was derived.
