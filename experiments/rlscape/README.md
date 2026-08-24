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
| **3A** | Critic Provenance and Calibration Audit | Complete | Supported/refined: severe error is already present on real posterior states and actor-selected imagination amplifies it; horizon growth is not the primary source | Study V |
| **3B** | Natural Forgetting Recovery | Complete | Supported: reward-only IR improved both natural checkpoints and recovered 1.50M bury from 14% to 91% sampled; standard IR remained unsafe | Study IV |
| **4A** | Counterfactual Goal-Head Repair | Complete | Mixed: counterfactual labels repaired held-out goal semantics and bounded value repaired scale, but factual reward-only IR remained the strongest sampled teacher and bounded IR was still task-dependent | Study VI |
| **4B** | Conservative Bounded-Value Dose and Stability | Complete | Mixed: beta=.05 gave a replicated +13.8 pp sampled gain and +4.0 pp worst-goal gain over reward-only, but every mixed condition violated the deterministic worst-goal safety floor | Study VII |
| **5A** | Interpretable Gating for Imagined Rehearsal | Ready to run | Test entropy, MCPB JS, and consequence-aware gates against always-on IR with compute accounting | Future algorithm |
| **5B** | IR Necessity Discrimination | Ready to run | Test whether frozen gates adapt on a weak task and abstain on a competent task | Future algorithm |
| **5B.1** | Offline Performance-Gate Audit | Ready to run | Test whether prospective JS and imagined-performance signals separate later underperformance and paired IR need across tasks | Future algorithm |
| **5C** | Persistent Self-Terminating Recovery | Ready to run | Test recovery speed, natural gate shutoff, retained-task safety, and compute across episode-persistent IR | Future algorithm |
| **5D** | Hard-KL Update Safety | Reserved | Bound accepted actor drift with hard rollback after gating is understood | Future algorithm |
| **5E** | Continual Gated IR Integration | Reserved | Combine current-task learning, reservoir preservation, and gated persistent IR | Future algorithm |
| **6** | Full Continual-Learning Validation | Reserved | Multi-seed/task-order validation with transfer, forgetting, and recurrence | Future validation |

Stages 3A and 3B are sibling experiments, not a forced sequence. Stage 3B
established that the successful critic-free mechanism transfers from synthetic
to natural policy damage, while exposing task- and horizon-specific exceptions.
Stage 3A then localized the failed teacher: critic error is already severe on
real posterior states, the slow critic copies it, and actor-selected imagination
amplifies it without a long-horizon explosion. Their completed results jointly
motivate Stage 4A and Stage 5.

Stage 4A then intervened directly. Counterfactual labels raised held-out
completion-goal top-1 accuracy from 33.3% to 97.4%, and a separate bounded
success head removed gross value-scale bias. In real RLScape, however,
factual-only reward IR achieved the best sampled aggregate, while bounded IR
still damaged chop. The next stage must therefore test conservative value dose
and adaptation stability rather than assuming semantic calibration alone is a
sufficient actor-teaching objective.

Stage 4B made that test directly on the repaired Stage 4A checkpoint. Its
mixed objective is anchored exactly at reward-only (`beta=0`) and full bounded
IR (`beta=1`), with small prespecified doses, a correction-clipping control, and
an H=6 horizon control. Three replicates share explicit episode-step posterior
and action random keys as well as identical RLScape reset schedules. Advancement
depended on both macro improvement and worst-goal safety; a good mean could not
hide a damaged task. All 162 units and 6,075 episodes completed without a failed
attempt. `beta=.05` improved sampled macro success from .738 to .876 and
worst-goal success from .513 to .787, with positive macro change in all three
replicates. However, deterministic worst-goal success fell from .760 to .627.
No mixed condition passed the full prespecified safety rule, so reward-only
H=15 remains the approved Stage 5 base teacher while small value correction is
retained as a gated/regularised candidate.

Stage 5 is now an explicit algorithm ladder. Stage 5A prioritizes gating:
factual actor entropy, mean posterior-conditioned entropy, MCPB
Jensen--Shannon disagreement, and reward-conditioned action evidence are
audited before fixed interpretable gates are tested on held-out episodes.
Compute is measured against no-IR and always-on IR but does not constrain gate
selection. Stage 5B then tests the missing negative case: whether a frozen gate
distinguishes an IR-responsive weak task from a competent task on the same
checkpoint. Stage 5B.1 is an offline diagnostic pause: it asks whether JS is
calibrated to performance, whether its threshold transfers across tasks, and
whether imagined performance supplies the missing axis. Stage 5C carries actor and adaptation-optimizer state across
episodes and asks whether recovery becomes self-terminating. Hard-KL rollback
moves to Stage 5D; only after those mechanisms have separate evidence does
Stage 5E integrate them into continual training.

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
the authoritative raw artifacts for Stages 1A, 1B, and 2 respectively, and
`stage3a_critic_provenance_calibration` and
`stage3b_natural_forgetting_recovery` are the authoritative raw Stage 3A and
Stage 3B roots. Nested `stage4a_counterfactual_head_repair` and
`stage4b_conservative_value` are the authoritative raw Stage 4A and Stage 4B
roots. The historical `M4` label must not be interpreted as canonical Stage 4.

## Maintained protocols

- [Stage 1A: Same-State Goal-Conditioned Head Audit](stage1a_same_state_head_audit.md)
- [Stage 1B: Exact Imagination-Path Audit](stage1b_imagination_path_audit.md)
- [Stage 2: Controlled Actor Corruption and Recovery](stage2_controlled_actor_recovery.md)
- [Stage 3A: Critic Provenance and Calibration Audit](stage3a_critic_provenance_calibration.md)
- [Stage 3B: Natural Forgetting Recovery](stage3b_natural_forgetting_recovery.md)
- [Stage 4A: Counterfactual Goal-Head Repair](stage4a_counterfactual_head_repair.md)
- [Stage 4B: Conservative Bounded-Value Dose and Stability](stage4b_conservative_value_dose.md)
- [Stage 5A: Interpretable Gating for Imagined Rehearsal](stage5a_interpretable_gating.md)
- [Stage 5B: IR Necessity Discrimination](stage5b_ir_necessity_discrimination.md)
- [Stage 5B.1: Offline Performance-Gate Audit](stage5b1_performance_gate_audit.md)
- [Stage 5C: Persistent Self-Terminating Recovery](stage5c_persistent_recovery.md)

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
