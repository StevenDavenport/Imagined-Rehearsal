# MiniGrid cross-goal leakage audit

This is a read-only analysis of the immutable same-state and imagination diagnostics produced by the completed sequential run. It does not modify or retrain any checkpoint.

- Inputs verified: 180 files (210.5 MiB)
- Final checkpoint: 800,000 environment steps
- Primary scope: the four trained goals; all six configured slots are also retained in the CSV tables.

## First task boundary (200,000 steps)

At this boundary only the first task has been trained. Similar predictions for the three future goal IDs are therefore especially direct evidence that a head is relying on shared state features rather than the requested goal.

| Arm | Completion reward factual | Reward nonmatch | Minimum goal value | Value goal range | All goals above factual return | Actor goal JS |
|---|---:|---:|---:|---:|---:|---:|
| current_old_50_50 | 0.154 | 0.162 | 0.675 | 0.013 | 0.940 | 0.001 |
| fifo_100k | 0.375 | 0.377 | 0.703 | 0.003 | 0.952 | 0.002 |
| uniform_reservoir | 0.385 | 0.387 | 0.645 | 0.006 | 0.939 | 0.001 |

## Final-checkpoint same-state results

| Arm | Reward factual | Reward nonmatch | Reward top-1 | Stop factual | Stop nonmatch | Stop top-1 |
|---|---:|---:|---:|---:|---:|---:|
| current_old_50_50 | 0.253 | 0.202 | 0.500 | 0.764 | 0.584 | 0.620 |
| fifo_100k | 0.000 | 0.000 | 0.035 | 0.036 | 0.128 | 0.180 |
| uniform_reservoir | 0.756 | 0.630 | 0.510 | 0.997 | 0.892 | 0.329 |

## Final-checkpoint critic and actor goal dependence

| Arm | Factual value bias | Minimum goal value | All goals above factual return | Value goal range | Reward-value corr. | Actor goal JS |
|---|---:|---:|---:|---:|---:|---:|
| current_old_50_50 | 1.055 | 0.940 | 0.979 | 0.373 | 0.061 | 0.039 |
| fifo_100k | 0.638 | 0.706 | 0.946 | 0.015 | -0.015 | 0.013 |
| uniform_reservoir | 2.515 | 2.458 | 0.994 | 0.312 | 0.003 | 0.037 |

## Final-checkpoint imagined-return decomposition

| Arm | Actor H15 full factual | Actor full nonmatch | Actor reward-only factual | Actor reward-only nonmatch | Recorded H15 full factual | Recorded reward-only factual |
|---|---:|---:|---:|---:|---:|---:|
| current_old_50_50 | 1.263 | 1.275 | 0.064 | 0.073 | 0.982 | 0.001 |
| fifo_100k | 0.726 | 0.725 | 0.255 | 0.255 | 0.702 | 0.004 |
| uniform_reservoir | 2.917 | 2.932 | 0.013 | 0.005 | 2.573 | 0.003 |

## Interpretation

The reward-leakage theory is supported most directly when completion reward remains high for nonmatching goals, critic optimism is broad across those same goal swaps, and centered reward-value correlation is positive on identical states. Flat, high reward and value across every goal can also support broad leakage even when centered correlation is small because there is little remaining cross-goal variance. The imagination table adds a temporal check: counterfactual reward-only return or reward sum that grows with horizon is a mechanism by which leakage can be amplified. When the full imagined return remains large but its reward-only component is small, the current numerical error is primarily carried by the critic bootstrap even if weak goal semantics helped create it upstream.

This audit is correlational. A weak same-checkpoint correlation does not rule out historical reward leakage because bootstrapped critic targets can retain earlier errors. The clean causal test remains a from-scratch counterfactual-head intervention with the ordinary critic objective left unchanged.
