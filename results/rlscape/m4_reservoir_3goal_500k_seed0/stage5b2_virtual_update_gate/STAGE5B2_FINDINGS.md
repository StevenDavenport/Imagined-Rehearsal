# Stage 5B.2 Virtual-Update Gate

This document is generated from the completed paired experiment. 

## Aggregate results

| Goal | Mode | Condition | Success | Tentative/step | Commit/step | Seconds/step |
|---|---|---:|---:|---:|---:|---:|
| kill_goblin | sampled | always_reward_only | 0.653 | 0.0000 | 0.9919 | 0.1294 |
| kill_goblin | sampled | no_ir | 0.480 | 0.0000 | 0.0000 | 0.1197 |
| kill_goblin | sampled | virtual_reward_dormant_100 | 0.487 | 0.2137 | 0.1445 | 0.1294 |
| kill_goblin | sampled | virtual_reward_dormant_25 | 0.653 | 0.3887 | 0.2488 | 0.1373 |
| kill_goblin | deterministic | always_reward_only | 0.533 | 0.0000 | 0.9924 | 0.1295 |
| kill_goblin | deterministic | no_ir | 0.347 | 0.0000 | 0.0000 | 0.1198 |
| kill_goblin | deterministic | virtual_reward_dormant_100 | 0.440 | 0.1886 | 0.1244 | 0.1283 |
| kill_goblin | deterministic | virtual_reward_dormant_25 | 0.453 | 0.4281 | 0.2823 | 0.1393 |
| bury_bones | sampled | always_reward_only | 0.660 | 0.0000 | 0.9922 | 0.1298 |
| bury_bones | sampled | no_ir | 0.373 | 0.0000 | 0.0000 | 0.1198 |
| bury_bones | sampled | virtual_reward_dormant_100 | 0.160 | 0.1454 | 0.0905 | 0.1264 |
| bury_bones | sampled | virtual_reward_dormant_25 | 0.000 | 0.3710 | 0.2292 | 0.1365 |
| bury_bones | deterministic | always_reward_only | 0.653 | 0.0000 | 0.9911 | 0.1294 |
| bury_bones | deterministic | no_ir | 0.373 | 0.0000 | 0.0000 | 0.1197 |
| bury_bones | deterministic | virtual_reward_dormant_100 | 0.187 | 0.1258 | 0.0762 | 0.1254 |
| bury_bones | deterministic | virtual_reward_dormant_25 | 0.200 | 0.3686 | 0.2253 | 0.1369 |
| chop_logs | sampled | always_reward_only | 0.807 | 0.0000 | 0.9891 | 0.1297 |
| chop_logs | sampled | no_ir | 0.833 | 0.0000 | 0.0000 | 0.1196 |
| chop_logs | sampled | virtual_reward_dormant_100 | 0.853 | 0.1774 | 0.1060 | 0.1274 |
| chop_logs | sampled | virtual_reward_dormant_25 | 0.867 | 0.3574 | 0.2167 | 0.1356 |
| chop_logs | deterministic | always_reward_only | 0.747 | 0.0000 | 0.9891 | 0.1301 |
| chop_logs | deterministic | no_ir | 0.813 | 0.0000 | 0.0000 | 0.1195 |
| chop_logs | deterministic | virtual_reward_dormant_100 | 0.787 | 0.1829 | 0.1158 | 0.1279 |
| chop_logs | deterministic | virtual_reward_dormant_25 | 0.907 | 0.3806 | 0.2304 | 0.1368 |

## Decision audit

- `virtual_reward_dormant_100`: 534 episodes received IR; 27 known repair opportunities received no update; 272 competent episodes received at least one update.
  Pooled AUC: initial predicted failure -> real underperformance `0.602`; first virtual improvement -> actual IR need `0.528`.
- `virtual_reward_dormant_25`: 645 episodes received IR; 6 known repair opportunities received no update; 346 competent episodes received at least one update.
  Pooled AUC: initial predicted failure -> real underperformance `0.659`; first virtual improvement -> actual IR need `0.470`.

Interpret all decisions prospectively, but necessity labels 
retrospectively. The latter are used only for audit and never by the gate.
