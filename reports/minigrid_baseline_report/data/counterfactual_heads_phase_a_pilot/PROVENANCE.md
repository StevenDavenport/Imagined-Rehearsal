# Counterfactual-head Phase-A pilot evidence

This directory is a compact, read-only copy of the completed pilot used in the
MiniGrid report. The authoritative immutable experiment root remains:

```text
/home/localadmin/experiment_logs/minigrid/counterfactual_heads_phase_a_pilot_v1
```

Execution host: `office-4090`  
Experiment Git revision: `f5e6e256bed9d4e7cec59108a3f8405033ab71f1`  
Experiment spec digest: `7a8fd6d11a1823720a0982179602ca2e0e813aea659e7516be440ab7f80257a9`  
Supervisor terminal state: `complete`

The package contains the immutable experiment specification, supervisor and
evaluation-pairing provenance, all compact analysis tables, the fixed diagnostic
bank selection, and the 0/200k same-state head and critic-provenance summaries.
Raw checkpoints, replay, episode logs, prediction tensors, rollout rows, and
critic shards remain only in the authoritative root.

Principal source hashes, calculated on the execution host and verified after
copying, are:

| Artifact | SHA-256 |
|---|---|
| `experiment_spec.json` | `5de3e57445c87c9da08f237ba6457b8fffb2dd1e8fe61d5b98340d94edc8c055` |
| `supervisor_status.json` | `110d85afccf461ede83ec8b48b50fd1bef86305cca965a2aecd591b86470d66f` |
| `evaluation_pairing.json` | `b800817c9551fcb9591d4571c32fe950400db2bc96463a7f8825b84435a71210` |
| `analysis/summary.json` | `73ef169625b99e665b13624b2169e606a9d0e776778f5c2e127bded05e05eeab` |
| `analysis/evaluation_curves.csv` | `7d5cf411c5b82fd1c11083ded7a784c405305dd95114c3756a5a255c26af773e` |
| `diagnostics/head/.../step_000000200000/summary.json` | `c0524ed1c98af54bfb9129d7d8ff8e71ccaec83b380571dc7945297bd9d0dc18` |
| `diagnostics/head/.../step_000000200000/completion_confusion.csv` | `336666dd4995a762d53bf8b91c37be2e3adc42fd4ac215febcfac03149b4cf3b` |
| `diagnostics/critic_provenance/.../step_000000200000/summary.json` | `156855ba7c224d5800c885de91ffc83c9cca0b1f97d3e5a98c5d697d3b03fe13` |
| `diagnostics/critic_provenance/.../step_000000200000/aggregate_summary.csv` | `3cbd5fce4841d0b50236aa7dc585945cc6b27fbfab9b8f2e5686361b7d34e8ce` |

The pilot and matched baseline use their own final-replay diagnostic state
banks. Cross-run head comparisons in the report are therefore descriptive
matched-condition comparisons, while factual-versus-nonmatching goal contrasts
within each run hold posterior states exactly fixed.
