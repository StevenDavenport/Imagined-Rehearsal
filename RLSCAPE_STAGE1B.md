# RLScape Stage 1b: actor-IR imagination-path audit

Stage 1 established that factual reward and continuation prediction remain
strong, counterfactual goal selectivity is weak, and the critic becomes more
optimistic while behavior collapses. Stage 1b locates where that optimism
enters the exact actor-only imagined-rehearsal objective.

## Experimental control

The workflow uses the same immutable Stage 1 selection: 64 successful and 64
failed episodes for each trained goal, from the pinned 1.25M replay archive.
It compares the 1.25M strong checkpoint with the 1.5M weak checkpoint.

Four nonterminal decision states are selected deterministically from every
episode: the start, two intermediate states, and either the state immediately
before completion or a late failure state. This gives up to 1,536 posterior
anchors per checkpoint. Each checkpoint re-encodes the same frames and actions
through its own encoder and RSSM.

At every anchor, all five goal IDs are probed. For each goal, the audit uses:

- 128 posterior samples, matching evaluation-time actor IR;
- a six-step imagination horizon;
- the checkpoint's frozen actor, dynamics, reward, continuation, critic, slow
  critic, and normalizers;
- the same H+1 state convention, lambda=0.95 return, continuation weighting,
  advantage normalization, and entropy coefficient as actor IR;
- paired pseudorandom keys across goals and checkpoints.

This is up to 7,680 anchor--goal banks, 983,040 imagined particles, and about
5.9 million future transitions per checkpoint. No environment is launched and
no model parameter or optimizer state is updated.

## Recorded diagnostics

Raw per-particle, per-timestep shards retain:

- imagined reward and continuation;
- online and slow values;
- Dreamer lambda returns;
- reward-only Monte Carlo and lambda returns;
- bootstrap contribution;
- normalized advantage and continuation weight;
- sampled-action log probability and policy entropy;
- policy-gradient and entropy loss terms;
- imagined prior entropy and latent RMS;
- sampled actions.

The summary exposes both means and optimistic tails, reward/stop event rates,
return decomposition, entropy-to-policy-gradient magnitude, goal-confusion
matrices, and the timestep at which reward or termination enters imagination.

## Reliability

Each episode is written as an atomic shard. A restart validates and reuses
completed shards. Sampling keys derive from the episode, anchor, and experiment
seed rather than execution order, so resumed runs remain paired and
reproducible. Parameter SHA-256 digests are checked before and after each
checkpoint audit.

## Run

The completed Stage 1 audit must remain at `<experiment-root>/head_audit`.

```bash
xvfb-run -a -s "-screen 0 1280x1024x24" \
  python scripts/rlscape_stage1b_imagination.py \
  --experiment-root logs/rlscape/m4_reservoir_3goal_500k_seed0 \
  --horizon 6 \
  --particles 128 \
  --anchors-per-episode 4 \
  --stream-output
```

Rerunning the command resumes incomplete checkpoint audits and regenerates the
combined figures after both checkpoints complete.

Outputs are written to `<experiment-root>/imagination_audit/`:

- `strong/shards/` and `weak/shards/`: restartable raw episode shards;
- `rollout_summary.csv`: one row per anchor and probed goal;
- `goal_sweep.csv`: checkpoint goal-confusion summaries;
- `time_profile.csv`: reward, stop, and value by imagined timestep;
- `summary.json`: factual and counterfactual aggregate diagnostics;
- `figures/`: mean/tail return, reward/stop events, timing, and actor-objective
  decomposition;
- `comparison_summary.json` and `comparison_complete`: terminal artifacts.
