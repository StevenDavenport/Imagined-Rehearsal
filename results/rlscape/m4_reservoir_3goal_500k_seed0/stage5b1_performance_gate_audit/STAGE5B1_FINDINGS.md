# Stage 5B.1 Offline Performance-Gate Audit

This is a retrospective diagnostic over prospectively recorded no-IR signals. It does not qualify a deployable gate.

Episodes: **90**. Paired classes: `{'ir_needed': 21, 'ir_unnecessary': 27, 'ir_harmful': 18, 'not_repaired': 24}`.

Core decision window: first **10** factual states.

## Feature separability

| Goal | Mode | JS AUC: underperform | Predicted-failure AUC | JS AUC: IR needed |
|---|---:|---:|---:|---:|
| bury_bones | sampled | 0.609 | 0.484 | nan |
| bury_bones | deterministic | 0.238 | 0.762 | 0.238 |
| chop_logs | sampled | 0.684 | 0.789 | nan |
| chop_logs | deterministic | 0.583 | 0.583 | 0.714 |
| kill_goblin | sampled | 0.427 | 0.693 | 0.610 |
| kill_goblin | deterministic | 0.625 | 0.875 | 1.000 |

## Interpretation boundary

- AUC measures ranking, not a safe operating threshold.
- Task-specific optimal thresholds are deliberately diagnostic and must not be deployed without held-out cross-task validation.
- The H=15 imagined success fraction measures whether the current actor already discovers success; zero predicted successes may indicate need, lack of a learnable direction, or both.
- A cross-validated pre/post imagined-performance improvement signal is not present in these historical traces and is the principal candidate for the next small intervention.

See `threshold_transfer.csv` and `figures/` for the task-dependence diagnosis.
