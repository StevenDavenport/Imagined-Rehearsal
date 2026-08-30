# MiniGrid baseline report

This package is the standalone report for the initial fixed-mission MiniGrid
difficulty audition on the `crl` branch. It is deliberately separate from the
RLScape comprehensive research diary.

The study trained six tasks independently with a reduced DreamerV3 agent:

- six tasks x three seeds x 100,000 steps = 1.8 million training steps;
- four immutable checkpoints per run;
- 50 sampled and 50 deterministic episodes per checkpoint;
- 144 evaluation conditions and 7,200 evaluation episodes total.

Contents:

- `build_report.py`: reproducible ReportLab source with embedded vector figures;
- `data/`: immutable specification and compact analysis artifacts copied from
  `/home/localadmin/experiment_logs/minigrid/audition_12m_v1` on the 4090.

Regenerate with:

```bash
/home/zythax/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12 \
  reports/minigrid_baseline_report/build_report.py
```

The generated PDF is written alongside the source as `report.pdf`, matching
the layout used by the other report packages in this repository.
