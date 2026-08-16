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

Stage 3B completed all 4,500 episodes without retry or pairing failure. Its
principal result is that reward-only IR transfers from synthetic actor
corruption to natural forgetting, including a 14% to 91% sampled recovery for
1.50M bury bones at horizon 15. See
[`STAGE3B_FINDINGS.md`](m4_reservoir_3goal_500k_seed0/stage3b_natural_forgetting_recovery/STAGE3B_FINDINGS.md).

Stage 3A shows that critic overoptimism is already severe on real posterior
states, worsens at the collapsed checkpoint, and is amplified by actor-selected
imagination rather than created by increasing horizon. See
[`STAGE3A_FINDINGS.md`](m4_reservoir_3goal_500k_seed0/stage3a_critic_provenance_calibration/STAGE3A_FINDINGS.md).

Historical paths are intentionally not rewritten inside CSV/JSON extracts.
They identify the exact source host and run from which each result was derived.
