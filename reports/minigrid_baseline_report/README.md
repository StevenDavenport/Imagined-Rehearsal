# MiniGrid continual-learning report

This package is the standalone report for the MiniGrid experiments on the
`crl` branch. It is deliberately separate from the RLScape comprehensive
research diary and contains the independent difficulty audition, the
visibility-first sequential baseline, and the counterfactual-head Phase-A
pilot, and the offline counterfactual twin-Q qualification.

Study I trained six tasks independently with a reduced DreamerV3 agent:

- six tasks x three seeds x 100,000 steps = 1.8 million training steps;
- four immutable checkpoints per run;
- 50 sampled and 50 deterministic episodes per checkpoint;
- 144 evaluation conditions and 7,200 evaluation episodes total.

Study II trained one goal-conditioned agent through
`go_to_door -> fetch -> go_to_object -> put_near`:

- three replay arms x three seeds x 800,000 steps = 7.2 million training steps;
- agent/optimizer snapshots every 25,000 steps and exact replay at each 200,000-step boundary;
- 1,512 paired sampled/deterministic evaluation conditions and 75,600 episodes;
- 45 same-state head audits and 45 critic-provenance audits;
- 30.17 hours on one RTX 4090, with all scientific units complete.

Study III reran the matched `uniform_reservoir`, seed-0 GoToDoor Phase-A case
with counterfactual reward and continuation supervision only:

- one task x one seed x 200,000 steps;
- nine checkpoints, 18 paired sampled/deterministic evaluation conditions,
  and 900 evaluation episodes;
- four read-only 0/200k head and critic-provenance diagnostics;
- final success of 88% sampled and 86% deterministic;
- near-zero wrong-goal completion reward and correct wrong-goal continuation;
- an unchanged critic that remained nearly flat across all six goal queries.

The pilot is a causal component experiment, not evidence about catastrophic
forgetting or multi-seed behavioral efficacy.

Study IV froze the seed-0 50:50-current/old checkpoint and replay immediately
after Fetch at 400k, then trained new two-goal discrete-action twin-Q critics:

- dose-matched factual-only and counterfactual arms x three Q seeds;
- 10,000 offline one-step TD updates per run, with EMA twin targets and the
  exact frozen-actor action expectation;
- no reward oracle and no bootstrap from the original Dreamer value heads;
- 256 balanced train episodes and 128 episode-disjoint held-out episodes;
- counterfactual training reduced factual return MAE by 19%, all-query Bellman
  MAE by 39%, and twin disagreement by 23%;
- the matching-minus-nonmatching value gap doubled and successful initial-state
  goal ranking rose from 50.0% to 75.5%.

This is a positive offline critic qualification, not an online actor or CIR
result. The three repetitions vary new Q initialization on one frozen Dreamer
checkpoint and replay split, and counterfactual queries have no oracle returns.

Contents:

- `report.tex`: canonical native LaTeX source;
- `figures/`: vector plot assets included by the LaTeX report;
- `build_report.py`: retained legacy ReportLab source used for the original report;
- `data/`: Study I immutable specification and compact analysis artifacts;
- `data/sequential_baseline/analysis/`: complete Study II aggregate tables;
- `data/sequential_baseline/diagnostics/`: compact summaries and audit specs
  for all 90 Study II diagnostics;
- `data/sequential_baseline/supervisor_status.json`: complete Study II command,
  timing, retry, and terminal-state provenance.
- `data/counterfactual_heads_phase_a_pilot/`: compact Study III specification,
  supervisor and pairing provenance, analysis tables, fixed state-bank
  selection, 0/200k diagnostic summaries, and source hashes.
- `data/twin_q_qualification/`: compact Study IV specification, final per-seed
  and aggregate metrics, frozen-Dreamer integrity hashes, and copied-file
  provenance.

Raw Study I-IV checkpoints, replay, JSONL logs, predictions, latent caches, and
critic rollout shards remain on the 4090 under
`/home/localadmin/experiment_logs/minigrid/`.

## Read-only cross-goal leakage audit

`scripts/minigrid_goal_conditioning_audit.py` reuses the completed Study II
same-state head predictions and critic-provenance rollouts. It holds each
posterior state fixed while sweeping the goal supplied to the reward,
continuation, critic, slow critic, and actor heads. It does not load an
optimizer, update a checkpoint, or write beneath `runs/` or `diagnostics/`.

Run it in the MiniGrid Conda environment on the machine that holds the raw
experiment directory:

```bash
export MINIGRID_RUN_ROOT="$HOME/experiment_logs/minigrid/sequential_baseline_12m_v1"
python scripts/minigrid_goal_conditioning_audit.py \
  --experiment-root "$MINIGRID_RUN_ROOT"
```

The audit verifies every input hash before and after analysis and writes its
derived tables to `analysis/goal_conditioning_audit/`. Start with
`FINDINGS.md` and `checkpoint_summary.csv`; the detailed CSVs retain both the
four-trained-goal scope and all six configured goal slots.

Regenerate with:

```bash
cd reports/minigrid_baseline_report
tectonic report.tex
```

The compiled PDF is written alongside the source as `report.pdf`, matching
the layout used by the other report packages in this repository.
