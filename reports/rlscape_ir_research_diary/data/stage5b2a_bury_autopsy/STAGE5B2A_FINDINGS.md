# Stage 5B.2a Bury-Bones Gate Autopsy

Status: **complete; interpretation pending**.

| Condition | Success | Updates/episode | Top-click fraction | Mean length |
|---|---:|---:|---:|---:|
| no_ir | 0.340 | 0.0 | 0.332 | 158.5 |
| no_ir_world_audit | 0.233 | 0.0 | 0.295 | 175.0 |
| always_reward_only | 0.607 | 124.2 | 0.358 | 124.2 |
| reward_cap_16 | 0.127 | 15.8 | 0.289 | 186.2 |
| reward_cap_46 | 0.113 | 44.9 | 0.304 | 187.6 |
| reward_cap_80 | 0.207 | 77.3 | 0.317 | 179.7 |
| reward_cap_127 | 0.613 | 91.4 | 0.355 | 119.8 |
| reward_random_46 | 0.540 | 30.9 | 0.409 | 134.0 |
| virtual_positive_reward_d25 | 0.000 | 48.6 | 0.350 | 200.0 |
| virtual_negative_reward_d25 | 0.173 | 16.1 | 0.303 | 179.9 |
| always_mixed_beta_0p005 | 0.000 | 200.0 | 0.334 | 200.0 |
| mixed_beta_0p005_cap_46 | 0.127 | 44.7 | 0.316 | 184.0 |
| mixed_beta_0p005_random_46 | 0.540 | 31.2 | 0.387 | 135.7 |
| virtual_positive_mixed_beta_0p005_d25 | 0.260 | 39.2 | 0.330 | 172.6 |

## Prespecified interpretation

- If both the `46`-update cap and matched random schedule collapse, the leading explanation is a partial-dose recovery valley.
- If random scheduling succeeds but virtual-positive selection collapses, the imagined score is selecting model-exploiting updates.
- If virtual-negative exceeds virtual-positive, the learned model ranks candidate updates in the wrong real-world direction.
- If exact mixed beta `0.005` avoids the reward-only attractor, a small independent value signal regularizes reward-model exploitation.
- No-IR world-audit behavior must match no-IR behavior exactly before its imagined diagnostics can be treated as non-intervening.

## Pairing audit

- Probe non-interference action match: `0.143114` over `8916` actions.

Figures and compact diagnostics are stored beside this file. Real outcomes are never visible to any gate or update scheduler during evaluation.
