# Stage 1b imagination-path audit findings

## Audit design

Stage 1b evaluates the exact actor-IR imagination objective without applying an
optimizer update. It compares the 1.25M strong checkpoint with the 1.5M weak
checkpoint using the immutable Stage 1 replay selection: 64 successful and 64
failed episodes for each of the three trained goals. Four deterministic,
nonterminal anchors were requested per episode; short episodes yielded 1,433
valid anchors. Every anchor was swept across all five goal IDs using 128 paired
posterior samples and horizon H=6. This produced 7,165 anchor--goal rollouts and
5,502,720 imagined action transitions per checkpoint. Parameter digests before
and after each audit match exactly.

The real sampled-policy success rates at the comparison checkpoints were
56%/49%/88% (kill/bury/chop) at 1.25M and at most 16% on every goal at 1.5M,
with a 9.7% mean at 1.5M.

## Main findings

1. **The six-step IR target is overwhelmingly critic bootstrap.** At the strong
   checkpoint, the mean factual lambda return is 1.450: only 0.072 comes from
   imagined reward and 1.378 from bootstrap (94.4%). At the weak checkpoint,
   the mean factual return rises to 1.868 even though real behavior collapses:
   0.039 comes from reward and 1.829 from bootstrap (97.3%). Ordinary anchors
   are more than 99% bootstrap-driven. Even directly before recorded
   completion, bootstrap still contributes 69.1% at the strong checkpoint and
   82.4% at the weak checkpoint.

2. **The weak checkpoint is more optimistic while imagining fewer successes.**
   From strong to weak, factual imagined reward-event incidence falls from
   6.76% to 3.52% of particles and reward-only return falls from 0.072 to
   0.039. Nevertheless, bootstrap rises by 0.451 and total lambda return rises
   by 0.419. Factual starting value rises from 1.315 to 1.668. This reproduces
   the inverse relationship from Stage 1: critic confidence increases as real
   competence deteriorates.

3. **Goal identity scarcely changes the imagined target.** Strong-checkpoint
   factual versus counterfactual returns are 1.450 versus 1.447; at the weak
   checkpoint they are 1.868 versus 1.865. Reward-event rates are similarly
   close: 6.76% versus 6.55% (strong) and 3.52% versus 3.09% (weak). On
   pre-completion anchors, where the model is most likely to imagine a reward,
   factual versus wrong-goal event rates are 34.0% versus 33.5% (strong) and
   21.2% versus 20.8% (weak). Thus the actor-IR pathway inherits the
   counterfactual semantic failure found in Stage 1.

4. **The replay state controls the return much more than the requested goal.**
   At 1.25M, mean returns from kill, bury, and chop replay anchors are roughly
   1.66--1.69, 1.91--2.00, and 0.83--0.84 across every probed goal. At 1.5M
   they become approximately 2.00--2.07, 2.22--2.29, and 1.38--1.40. The two
   never-trained goal columns, light-fire and catch-fish, are nearly identical:
   their mean absolute return difference is 0.0012 (strong) and 0.0024 (weak).
   Requested-goal top-1 identification from return is above the 20% chance
   level but degrades from 52.8% to 41.9%, and its mean factual-minus-wrong
   margin is only 0.0029/0.0032.

5. **Reward and stopping are detected as visual events, not goal-specific
   events.** For successful trajectories, factual reward-event incidence is
   12.5% (strong) and 6.7% (weak), versus 1.8% and 0.7% on failed trajectories.
   Events concentrate at the first imagined transition and near factual
   pre-completion states. Continuation produces almost the same pattern as the
   reward head. This is internally consistent with Stage 1: the heads recognize
   completion-like latent states but do not answer whether that completion
   belongs to the requested goal.

6. **Entropy does not dominate the measured scalar actor objective.** The mean
   absolute factual policy-gradient term is 0.0398 at the strong checkpoint and
   0.0196 at the weak checkpoint. The corresponding entropy terms are only
   0.000552 and 0.000180, roughly 1.4% and 0.9% of those aggregate magnitudes
   (mean per-rollout ratios are 2.2% and 1.6%). An entropy-off ablation remains
   cheap, but these data do not support entropy as the primary failure. This
   audit measures loss terms rather than parameter-gradient norms, so it cannot
   prove that their gradient ratio is identical.

7. **The actor itself becomes much more deterministic at the weak checkpoint.**
   Mean factual policy entropy falls from 1.915 to 0.628. At the strong
   checkpoint, chop has substantially higher policy entropy (3.36) than kill
   or bury (about 1.1); at the weak checkpoint all three are near 0.61--0.65.
   This is compatible with a broad policy collapse, but the frozen IR objective
   that would be used to repair it is simultaneously less informative: its
   mean absolute policy-gradient term roughly halves.

## Interpretation

Stage 1b locates the dominant IR failure upstream of gating and entropy. The
actor is optimized mainly against a severely optimistic critic bootstrap. That
bootstrap is largely determined by which replay trajectory supplied the
posterior state, not by the counterfactual goal requested for rehearsal. The
reward and continuation heads add a smaller but semantically incorrect signal:
completion-like states reward and terminate almost every goal.

This explains why ungated IR can damage a strong actor. It is not rehearsing a
clean goal-specific skill objective; it is fitting the actor to high-valued,
weakly goal-conditioned regions of its learned latent model. It also explains
why IR can occasionally improve a collapsed actor: even a biased signal can
move a poor policy, but the direction is not reliably aligned with real task
success.

## Consequence for Stage 2

Actor corruption and recovery remains valuable, but standard IR alone would
now be an ambiguous test because its teaching signal is known to be defective.
The controlled recovery assay should preserve a standard-IR arm while adding
diagnostic objective arms, at minimum:

- standard learned lambda-return IR;
- the same IR with entropy disabled;
- reward-only IR with critic bootstrap removed;
- a bounded or trusted-target positive control, so recovery failure can be
  attributed either to the actor-adaptation mechanism or to the learned target.

The larger algorithmic repair is to teach counterfactual semantics explicitly:
on a completion state for goal A, reward must be zero and continuation true for
goals B--E. Until reward, continuation, and value are functionally selective in
this same-state goal sweep, entropy gates or trust regions can limit damage but
cannot make the IR target correct.
