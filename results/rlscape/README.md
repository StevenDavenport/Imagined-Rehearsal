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

Historical paths are intentionally not rewritten inside CSV/JSON extracts.
They identify the exact source host and run from which each result was derived.
