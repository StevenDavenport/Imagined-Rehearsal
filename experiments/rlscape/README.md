# RLScape continual imagined-rehearsal experiment registry

This directory is the canonical index for the RLScape research programme.
Stage identifiers in this registry name scientific experiments. They are not
the older engineering milestone numbers (`M0`--`M4`) that remain embedded in
historical log directories, configuration names, and checkpoint manifests.

## Naming contract

- A stage number is never reused for a different hypothesis.
- A letter suffix denotes a linked experiment that shares upstream artifacts
  but answers a distinct question.
- Historical artifact paths are immutable. Renaming a completed run would
  weaken provenance, so the registry maps its canonical stage to its old path.
- New run roots use `stage<id>_<slug>_<run-id>`.
- Every new experiment specification records its canonical stage ID, source
  checkpoint, Git revision, environment version, seed schedule, and spec hash.
- `planned`, `ready_to_run`, `running`, `complete`, and `superseded` describe execution state;
  `supported`, `rejected`, `mixed`, and `unresolved` describe the hypothesis.

The machine-readable companion is [`registry.json`](registry.json).

## Canonical evidence chain

| Stage | Canonical name | Status | Hypothesis result | Thesis mapping |
|---|---|---|---|---|
| **0** | Sequential Reservoir Baseline and Checkpoint IR | Complete | Mixed: reservoir enabled transient multi-goal recovery but did not stabilize learning; ordinary actor IR was conditional | Study I |
| **1A** | Same-State Goal-Conditioned Head Audit | Complete | Reward/continuation retained factual event recognition, but counterfactual goal selectivity was weak and the critic was overoptimistic | Study IIa |
| **1B** | Exact Imagination-Path Audit | Complete | Supported: 94--97% of the six-step IR target came from critic bootstrap | Study IIb |
| **2** | Controlled Actor Corruption and Recovery | Complete | Supported: critic-free reward-only IR recovered a damaged actor; standard bootstrapped IR did not | Study III |
| **3A** | Critic Provenance and Calibration Audit | Ready to run | Unresolved: determine whether critic failure originates on real posterior states, during bootstrap propagation, or through actor-selected imagination | Next diagnostic |
| **3B** | Natural Forgetting Recovery | Ready to run; run first | Unresolved: test reward-only IR on naturally forgotten checkpoint policies | Next intervention |
| **4** | Counterfactual Goal-Conditioning Repair | Reserved | Train explicit alternative-goal negatives and bounded goal-success values | Future repair |
| **5** | Gated Continual Imagined Rehearsal | Reserved | Combine reward-only/conservative IR with a prespecified gate and trust region | Future algorithm |
| **6** | Full Continual-Learning Validation | Reserved | Multi-seed/task-order validation with transfer, forgetting, and recurrence | Future validation |

Stages 3A and 3B are sibling experiments, not a forced sequence. Stage 3A
explains the failed teacher; Stage 3B tests the successful critic-free mechanism
on natural rather than synthetic damage. Their results jointly determine Stage
4 and Stage 5. The chosen operational order is **3B before 3A** because 3B is
the more direct intervention and requires no new diagnostic model path.

## Where each layer lives

| Layer | Location | Purpose |
|---|---|---|
| Registry and protocols | `experiments/rlscape/` | Canonical names, hypotheses, controls, claim boundaries, and status |
| Maintained launchers | `scripts/rlscape_stage*.py` | Canonically named experiment entry points |
| Historical launchers/configs | `scripts/rlscape_m*.py`, `rlscape_m*` configs | Compatibility with completed runs and manifests |
| Raw artifacts | `logs/rlscape/` on the execution host | Authoritative checkpoints, replay, episode logs, queues, and console logs |
| Compact audited results | `results/rlscape/` | Portable audit tables and figures copied from execution hosts |
| Research diary | `reports/rlscape_ir_research_diary/` | Chronological paper/thesis narrative and self-contained figure inputs |
| Historical plans | `experiments/rlscape/archive/` | Superseded protocols retained for provenance, not current status |

The completed reservoir run remains historically named
`m4_reservoir_3goal_500k_seed0`. Its canonical identity is Stage 0. Its nested
`head_audit`, `imagination_audit`, and `stage2_actor_recovery` directories are
the authoritative raw artifacts for Stages 1A, 1B, and 2 respectively. The
historical `M4` label must not be interpreted as canonical Stage 4.

## Maintained protocols

- [Stage 1A: Same-State Goal-Conditioned Head Audit](stage1a_same_state_head_audit.md)
- [Stage 1B: Exact Imagination-Path Audit](stage1b_imagination_path_audit.md)
- [Stage 2: Controlled Actor Corruption and Recovery](stage2_controlled_actor_recovery.md)
- [Stage 3A: Critic Provenance and Calibration Audit](stage3a_critic_provenance_calibration.md)
- [Stage 3B: Natural Forgetting Recovery](stage3b_natural_forgetting_recovery.md)

Stage 0's full methodology and results are preserved as Study I in the
[research diary](../../reports/rlscape_ir_research_diary/report.pdf). The
pre-result bridge and FIFO plans are in [`archive/`](archive/).

## Artifact rules for all new stages

Each run root should contain:

```text
experiment_spec.json     canonical hypothesis and immutable settings
queue.json               restart state for multi-unit workflows
commands/                exact child commands, when applicable
<unit>/summary.json      terminal unit result and integrity fields
pairing.json             paired-seed/reset audit
results.csv              compact complete result table
figures/                 generated visual summaries
stage<id>_complete       terminal marker written only after validation
```

Raw logs are never edited to match a later narrative. Corrections and derived
analyses receive a new spec version and record the hashes of their sources.
Reports may copy compact CSV/JSON extracts, but must identify the raw run root
and preserve historical paths inside those extracts.
