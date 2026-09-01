# MiniGrid continual-learning report

This package is the standalone report for the MiniGrid experiments on the
`crl` branch. It is deliberately separate from the RLScape comprehensive
research diary and now contains both the independent difficulty audition and
the visibility-first sequential baseline.

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

Contents:

- `build_report.py`: reproducible ReportLab source with embedded vector figures;
- `data/`: Study I immutable specification and compact analysis artifacts;
- `data/sequential_baseline/analysis/`: complete Study II aggregate tables;
- `data/sequential_baseline/diagnostics/`: compact summaries and audit specs
  for all 90 Study II diagnostics;
- `data/sequential_baseline/supervisor_status.json`: complete Study II command,
  timing, retry, and terminal-state provenance.

Raw Study I and Study II checkpoints, replay, JSONL logs, predictions, and
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
/home/zythax/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12 \
  reports/minigrid_baseline_report/build_report.py
```

The generated PDF is written alongside the source as `report.pdf`, matching
the layout used by the other report packages in this repository.
