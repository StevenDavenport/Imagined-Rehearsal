# RLScape Stage 2: controlled actor corruption and recovery

**Status:** Complete. Reward-only IR recovered the corrupted actor strongly,
reaching 73.0% sampled mean success at `H=15` and 85.3% deterministic mean
success at `H=20`; standard and entropy-off bootstrapped IR remained weak or
destructive.

## Question

Can actor-only imagined rehearsal recover a deliberately damaged policy when
the world model, reward head, continuation head, critic, and goal conditioning
are held fixed? How does recovery change with imagination horizon, removal of
the entropy term, and removal of critic bootstrap?

Stage 2 is a controlled mechanism experiment. It is not another continual
training run and it does not claim to repair the counterfactual goal-semantics
failure identified by Stages 1 and 1b.

## Source checkpoint and corruption

The clean source is the pinned 1.25M checkpoint from the three-goal reservoir
experiment. It had sampled-policy success of 56%/49%/88% on kill goblin, bury
bones, and chop logs.

Only `pol/` parameters are corrupted. Encoder, RSSM, decoder, reward,
continuation, online value, slow value, normalizers, and all other model
parameters are copied exactly. Stage 2 generates zero-mask candidates at
strengths 0.1--0.6 and evaluates each for 20 paired sampled episodes per goal.
It selects the candidate closest to 35% aggregate clean-performance retention,
subject to an accepted 10--65% range and a penalty for uneven or apparently
beneficial corruption. No individual goal may retain more than 85% of its
clean success. The main assay will not start with an unqualified
corruption unless `--allow-unqualified-corruption` is explicitly supplied.

An already qualified actor-only checkpoint can instead be supplied with
`--corruption-checkpoint`.

## Recovery conditions

Temporary adapted parameters begin from the same corrupted actor at every
episode and are discarded at episode termination. The world model and critics
remain frozen in every condition. Adaptation occurs at the first state and
then every environment action, with one optimizer update, learning rate
4e-5, and 128 MCPB posterior samples.

The main matrix contains 15 conditions:

1. Clean actor, no adaptation.
2. Corrupted actor, no adaptation.
3. Standard bootstrapped actor IR at H in {3, 6, 15, 20}.
4. Standard actor IR with entropy coefficient zero at each horizon.
5. Reward-only actor IR at each horizon. This uses predicted rewards and
   continuation but sets the value bootstrap to zero. Posterior samples whose
   imagined trajectories contain zero reward contribute no policy-gradient
   signal. Entropy is also zero so it cannot dominate the sparse reward signal.
6. Frozen clean-reference policy distillation. At each trigger, the clean actor
   supplies its complete categorical action distribution for the current real
   posterior state and the temporary corrupted actor minimizes cross-entropy.
   This horizon-independent positive control establishes whether the
   corruption is mechanically reversible under the same actor optimizer.

Every condition is evaluated on all three goals with 100 sampled and 50
deterministic episodes by default: 90 paired units and 6,750 main evaluation
episodes. Calibration adds 420 episodes with the default six candidates.

## Interpretation

- Standard IR improves with longer H: short-horizon bootstrap was insufficient.
- Standard IR degrades with longer H: accumulated model error or exploitation
  is likely.
- Reward-only improves while standard IR fails: critic bootstrap is the main
  harmful signal.
- Entropy-off improves over standard IR: entropy gradients matter despite
  their small measured scalar loss.
- Reference distillation succeeds while every IR arm fails: actor adaptation
  and corruption are sound, but the learned rehearsal objective is not.
- Reference distillation fails: recalibrate corruption or optimizer/update
  strength before drawing conclusions about IR.

Recovery is reported both as raw success and as the fraction of the
clean-to-corrupted performance gap recovered. Cells where the clean-corrupt
gap is less than five percentage points are left undefined for normalized
recovery.

## Integrity and recovery behavior

The workflow:

- pins a complete milestone archive;
- stores a full immutable experiment specification and digest;
- records the corruption manifest and relative parameter L2;
- requires exact episode counts and unchanged persistent parameter digests;
- pairs goal/reset-seed identities across every condition;
- supervises the exact child process group with startup and stall timeouts;
- retries failed units up to three times;
- adopts a still-running child after launcher restart;
- reuses only completed units whose integrity artifacts are present;
- writes results atomically after every completed unit.

Outputs include calibration selection, per-condition results, baseline deltas,
optimizer diagnostics, pairing audits, PNG/PDF figures, queue state, individual
commands, and child console logs.

## Dry run

```bash
python scripts/rlscape_stage2_actor_recovery.py \
  --experiment-root logs/rlscape/m4_reservoir_3goal_500k_seed0 \
  --dry-run
```

## Full run

Run inside `tmux` and under Xvfb:

```bash
xvfb-run -a -s "-screen 0 1280x1024x24" \
  python scripts/rlscape_stage2_actor_recovery.py \
  --experiment-root logs/rlscape/m4_reservoir_3goal_500k_seed0
```

Add `--stream-output` only when full child output is wanted. Without it, each
unit writes `console.log` and the queue remains observable at:

```bash
python -m json.tool \
  logs/rlscape/m4_reservoir_3goal_500k_seed0/stage2_actor_recovery/queue.json \
  | tail -n 80
```

## Explicitly retained next repair

Counterfactual goal training is the next conditioning repair. When a state
completes goal A, the reward target must be zero and continuation target true
when the same latent is queried under goals B--E. The Stage 2 specification
records this future repair so it is not lost, but Stage 2 deliberately leaves
the heads unchanged to isolate actor recovery.
