# RLScape imagined-rehearsal research diary

This package contains the expanded paper/thesis-chapter write-up of the
chronological RLScape continual-learning investigation:

1. **Stage 0 / Study I:** FIFO versus goal-stratified episode-reservoir
   training and checkpoint IR;
2. **Stages 1A--1B / Study II:** strong/weak checkpoint head and exact
   imagination-path audits;
3. **Stage 2 / Study III:** controlled actor corruption and recovery with
   horizon, entropy, and critic-bootstrap ablations.

The canonical experiment names and future Stage 3A--6 reservations are
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
summaries and Stage III result tables were added on 2026-08-13. They make the
report portable; the original JSONL logs remain the authoritative raw
artifacts.
