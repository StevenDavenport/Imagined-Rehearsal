# RLScape imagined-rehearsal research diary

This package contains the expanded paper/thesis-chapter write-up of the
chronological RLScape continual-learning investigation:

1. **Stage 0 / Study I:** FIFO versus goal-stratified episode-reservoir
   training and checkpoint IR;
2. **Stages 1A--1B / Study II:** strong/weak checkpoint head and exact
   imagination-path audits;
3. **Stage 2 / Study III:** controlled actor corruption and recovery with
   horizon, entropy, and critic-bootstrap ablations;
4. **Stage 3B / Study IV:** reward-only rehearsal on naturally forgotten
   checkpoint policies;
5. **Stage 3A / Study V:** critic provenance on real posterior states and
   actor-versus-recorded imagined trajectories.

The canonical experiment names and remaining Stage 4--6 ladder are
maintained in the repository at `experiments/rlscape/README.md`.

The original 8 August report remains unchanged in the sibling
`rlscape_reservoir_report` directory and in the Documents archive.

- `report.tex`: LaTeX source.
- `report.pdf`: compiled report.
- `references.bib`: bibliography.
- `generate_figures.py`: reproducible plotting script.
- `figures/`: vector PDF and PNG figures, grouped by canonical stage for the
  diagnostic and recovery experiments.
- `data/`: compact audited CSV/JSON inputs grouped by canonical stage.

Regenerate from this directory with:

```bash
MPLCONFIGDIR=/tmp/rlscape-report-mpl python generate_figures.py
latexmk -pdf report.tex
```

The compact Study I tables were extracted on 2026-08-08. Stage II audit
summaries and Stage III result tables were added on 2026-08-13. Stage IV's 60
result rows, paired intervals, immutable specification, and figures were added
on 2026-08-15. Stage V's real-state calibration, horizon decomposition,
proposal contrasts, and integrity summaries were added after the completed GPU
audit. They make the report portable; the original JSONL logs remain the
authoritative raw artifacts.
