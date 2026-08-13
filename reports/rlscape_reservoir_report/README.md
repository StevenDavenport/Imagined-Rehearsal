# RLScape reservoir and actor-IR report

This package contains the paper-style write-up of the completed three-goal
sequential RLScape experiment.

- `report.tex`: LaTeX source.
- `report.pdf`: compiled report.
- `references.bib`: bibliography.
- `generate_figures.py`: reproducible plotting script.
- `figures/`: vector PDF and PNG versions of every data figure.
- `data/`: compact audited data tables used by the figures.

Regenerate from this directory with:

```bash
MPLCONFIGDIR=/tmp/rlscape-report-mpl python generate_figures.py
latexmk -pdf report.tex
```

The compact tables were extracted from the complete office-gpu run logs on
2026-08-08. They are intended to make the report portable; the original JSONL
logs remain the authoritative raw artifacts.
