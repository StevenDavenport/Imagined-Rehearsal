# Continual Imagined Rehearsal: Research Plan

> [!IMPORTANT]
> This is the long-range conceptual plan for Continual Imagined Rehearsal
> (CIR), written before the completed RLScape mechanism experiments. It
> describes hypotheses and intended systems, not the current experimental
> status. In particular, its critic-first proposal is now challenged by the
> observed critic-bootstrap failure. The canonical executed evidence chain and
> next protocols live in
> [`experiments/rlscape/README.md`](experiments/rlscape/README.md).

## Central Thesis

> Real experience maintains a predictive model of the world. Behaviour is
> learned only in imagination, at the agent's current belief, and computation
> occurs only when either the model or behaviour needs correction.

CIR will investigate a complete separation between predictive learning and
behavioural learning:

```text
real transition sequences ------> world-model learning only
                                         |
current online posterior ---------------+----> imagined trajectories
                                                   |
                                              critic, then actor
```

The intended contribution is not a new continual replay algorithm. Imagined
Rehearsal (IR), its current-posterior timing, and its complete decoupling from
world-model training are the central ideas. Established continual world-model
memory, novelty detection, and adaptive-update mechanisms may be reused or
adapted as supporting components.

## Contribution Boundary

[DreamerV3](https://www.nature.com/articles/s41586-025-08744-2) already trains
an actor and critic using latent imagined trajectories. CIR therefore cannot
claim novelty merely because behaviour is trained in imagination. The proposed
difference is the learning contract:

- Dreamer normally starts behaviour-learning trajectories from posterior
  states sampled from replay and applies a fixed overall training ratio.
- CIR trains the world model from stored real sequences but does not allow
  those replay samples to train the actor or critic.
- CIR performs actor-critic learning through IR from the current online
  posterior, at the time the resulting behaviour is needed.
- Monte Carlo Posterior Batching (MCPB) supplies stochastic posterior and
  rollout diversity at that current belief.
- Predictive and behavioural computation can be scheduled by independent
  gates.

This makes Continual-Dreamer an important baseline and a possible source of
world-model retention machinery, rather than the conceptual foundation of the
method.

In paper terminology, the world-model objective should usually be called
**predictive** or **self-supervised world-model learning**. Observation and
latent-dynamics objectives are not ordinary labelled supervision, although
reward and continuation prediction do use direct targets. "Supervised world
model training" remains useful shorthand for the strict separation from the
actor-critic objective.

## Learning Contract

### Real experience

Real transitions are stored as contiguous sequences containing at least:

```text
observation, action, reward, is_first, is_terminal
```

A task or context identifier may also be required where identical states and
actions can have incompatible dynamics or rewards. These sequences train only:

- the encoder and decoder;
- the recurrent latent dynamics;
- the reward predictor; and
- the continuation predictor.

The initial stationary experiments will retain the repository's existing FIFO
buffer. Reservoir sampling, Continual-Dreamer, or ARROW-style memory should be
introduced only when lifetime model retention becomes an experimental
requirement.

### Imagined experience

At an eligible real interaction, CIR:

1. infers the current recurrent posterior;
2. applies MCPB to draw multiple stochastic states from its posterior logits;
3. imagines short trajectories through the frozen world model;
4. trains the critic from predicted rewards, continuation probabilities, and
   bootstrapped imagined returns;
5. recomputes values, returns, and advantages after the critic update; and
6. trains the actor only after the critic has received an opportunity to
   improve.

During CIR behaviour updates, gradients must not change the encoder, dynamics,
decoder, reward head, or continuation head.

## Critic-First Actor-Critic IR

### Motivation

The existing IR implementation assumes that the critic is trustworthy and
updates only the actor. That assumption becomes unsafe when IR is the sole
behaviour-learning mechanism. If actor and critic are both stale, an actor
update based on the pre-update critic can reinforce a poor value estimate.

CIR should treat each behaviour update as approximate policy iteration:

```text
policy evaluation          policy improvement
critic update       -----> actor update
```

The first IR opportunity after initialization, critic reset, or a future
world-model repair event will therefore be **critic-only**. An actor update is
allowed only after at least one critic update has occurred under the current
trusted model.

### Initial schedule

The smallest useful schedule is:

```text
IR event 1: imagine -> critic update -> no actor update
IR event 2: imagine -> critic update -> recompute targets -> actor update
IR event 3: imagine -> critic update -> recompute targets -> actor update
...
```

The actor must not consume advantages cached before the critic update. The
imagined features and actions may be reused for the first implementation, since
the actor has not yet changed, but critic predictions and lambda returns must be
recomputed.

The fixed schedule should later be ablated against:

- simultaneous actor-critic updates;
- one critic update per actor update;
- two or four critic updates per actor update;
- actor updates delayed until value loss or critic disagreement falls below a
  threshold; and
- a critic-only warm-up after every detected model-repair event.

The relevant precedent is the delayed-policy principle in
[TD3](https://proceedings.mlr.press/v80/fujimoto18a.html): update the critic more
often so that the policy is less likely to follow a noisy value estimate. More
recent high-update methods such as
[REDQ](https://iclr.cc/virtual/2021/poster/3320) and
[CrossQ](https://openreview.net/forum?id=PczQtTsTIX) also separate critic and
actor update frequencies. CIR should borrow this scheduling principle, not
their model-free Q-learning machinery. Two-timescale actor-critic optimization
also has a substantial theoretical history; for example,
[Wu et al. (2020)](https://arxiv.org/abs/2005.01350) analyse the critic on the
faster timescale and the actor on the slower one.

### Remaining critic caveats

Critic-first rehearsal reduces one source of error but does not make the
targets correct automatically. Imagined lambda returns still depend on:

- dynamics accuracy along the complete rollout;
- reward and continuation accuracy;
- the critic bootstrap at the rollout horizon; and
- adequate state coverage.

Repeated critic updates can propagate predicted reward information through a
stale bootstrap, while short rollouts limit model error. Target-value and
normalization behaviour should initially remain identical to Dreamer wherever
possible, then be ablated only if diagnostics identify them as a limitation.

## Independent Compute Gates

CIR ultimately has two separate decisions:

1. Does the predictive model require learning?
2. Given a trustworthy model, does behaviour require rehearsal?

The intended control logic is:

| Model state | Behaviour state | Learning action |
|---|---|---|
| Trusted | Adequate | Act without a gradient update |
| Trusted | Inadequate | Run critic-first actor-critic IR |
| Untrusted | Any | Train the world model; defer or shorten IR |
| Repaired | Inadequate | Run critic-only warm-up, then actor-critic IR |

### World-model gate

The model gate asks whether recent reality still agrees with the model. A cheap
initial score can combine:

```text
prior-posterior latent KL
+ reward prediction error
+ continuation prediction error
```

These signals avoid requiring a full image reconstruction solely for gating.
The score can be calibrated with an exponentially weighted mean and variance or
a simple change detector. A large standardized residual triggers a burst of
world-model-only updates from recent and retained sequences.

Training should stop when a held-out recent error has recovered or learning
progress has stalled. Raw surprise alone can waste compute on irreducibly
stochastic observations. Learning progress and a small validation stream are
therefore preferable to an unconditional error threshold. A low-frequency
audit update may be retained to prevent silent drift.

This mechanism is supporting machinery rather than the main novelty:

- [Dynamic Update-to-Data Ratio](https://arxiv.org/abs/2303.10144) adjusts
  DreamerV2 world-model training using held-out under/overfitting signals.
- [Novelty Detection in Reinforcement Learning with World Models](https://arxiv.org/abs/2310.08731)
  detects changes from disagreement between imagined and observed transitions.
- [Self-adapting Robotic Agents through Online Continual Reinforcement Learning with World Model Feedback](https://arxiv.org/abs/2603.04029)
  uses DreamerV3 prediction residuals to trigger its ordinary joint fine-tuning
  loop.

CIR differs from the last mechanism by decoupling model repair from behaviour
learning and directing behaviour learning through current-posterior IR.

### Behaviour gate

The behaviour gate asks whether a trustworthy model predicts that the current
actor-critic needs adaptation. Candidate signals include:

- disagreement among an ensemble of actors;
- critic or value-target disagreement;
- advantage magnitude;
- a recent return or predicted-value drop;
- policy entropy; and
- the expected policy KL or parameter update magnitude.

The first CIR experiments should keep behaviour rehearsal always on. Candidate
signals should be logged without controlling learning until their relationship
with the marginal benefit of an IR update is measured.

### Compute allocation

Once both gates are validated, they may control continuous budgets rather than
binary decisions:

- the model residual controls the number of predictive updates;
- posterior entropy or gradient variance controls MCPB batch size;
- critic uncertainty controls the critic-to-actor update ratio;
- actor disagreement controls the number of policy updates; and
- model distrust shortens or suppresses imagined trajectories.

All gate overhead, including forward passes used to calculate a decision, must
be included in compute measurements.

## Experimental Programme

### Experiment 0: actor-critic IR component validation

This deliberately stationary experiment is the first implementation target. It
does not require a continual benchmark, task switching, a new replay buffer, or
world-model gating.

Use an existing converged Dreamer checkpoint on one familiar task:

1. preserve its encoder, dynamics, decoder, reward head, and continuation head;
2. corrupt or reset both the actor and critic;
3. run MCPB rehearsal from the current posterior;
4. update the critic before allowing actor learning; and
5. measure behavioural recovery through real interaction.

The minimum comparison is:

| Variant | Initial critic | Critic IR | Actor IR | Update order |
|---|---|---:|---:|---|
| Original IR reference | Intact | No | Yes | Actor only |
| Failure control | Corrupted | No | Yes | Actor only |
| Joint actor-critic IR | Corrupted | Yes | Yes | Simultaneous |
| Critic-first CIR | Corrupted | Yes | Yes | Critic before actor |

This experiment establishes whether the world model can repair both components
without conflating the result with continual retention. It should record:

- return recovery versus real interactions;
- critic loss and value calibration;
- imagined returns versus subsequently observed returns;
- actor KL and entropy;
- number of critic and actor updates;
- imagined transitions; and
- wall-clock time.

### Experiment 0a: critic-only intermediary diagnosis

Before repeating actor-and-critic recovery, isolate critic rehearsal with a
healthy frozen actor. Use one fixed Cheetah Run checkpoint and a fixed dataset
of 12 healthy-policy episodes, split by episode into six rehearsal episodes and
six validation episodes. Corrupt only `val/` and synchronise `slowval/`; the
actor, encoder, RSSM, decoder, reward head, and continuation head must remain
bitwise unchanged. Start with a 20% zero-mask and qualify it by requiring a
clean-versus-corrupt value MAE of at least 0.25 clean-value standard deviations;
retry at 30% and 50% if required.

The experiment has two diagnostic parts:

1. Replay recorded healthy actions open-loop through the frozen world model at
   horizons 1, 3, 6, 12, 24, 48, and 96. Measure reward error, discounted-return
   error and correlation, prior-posterior KL, deterministic-state drift, and
   continuation predictions. Report return error separately for low, middle,
   and high realised-reward bands.
2. Repair only the critic from healthy-posterior anchors. Screen horizons 3, 6,
   12, 24, and 48 with 128 MCPB posterior samples, 128 critic updates, and three
   rehearsal seeds. Confirm the best two fidelity-qualified horizons plus the
   horizon-6 control using 512 posterior samples and 256 updates.

The primary repair quantity is closure of the initial corrupt-to-clean critic
MAE gap on held-out anchors. Also record calibration, rank correlation, value
spread, slow-critic error, optimizer loss, gradient norm, and update size at
updates 0, 1, 2, 4, 8, 16, 32, 64, 128, and 256. Critic snapshots and CSVs make
the run resumable. A namespace hash after each condition asserts that critic IR
has not modified the actor or world model.

Optionally load the existing actor-and-critic corrupted checkpoint solely as a
proposal-policy control. Compare its imagined return distribution, fraction of
predicted rewards above 0.5 and 0.8, and action diversity against the healthy
actor on identical posterior anchors. This distinguishes inadequate proposal
coverage from a broken critic target without allowing the damaged actor to
contaminate the critic-only repair experiment.

Interpretation is deliberately conditional. Poor recorded-action fidelity
redirects work to the world model. Adequate fidelity but less than 20% critic
gap closure points to critic targets or bootstrapping. At least 50% closure with
the healthy actor makes damaged-actor rollout coverage the next question.
Intermediate recovery calls for critic learning-rate, posterior-sample-count,
and update-count ablations before changing the architecture.

The implementation is `scripts/critic_intermediary_experiment.py`. Its stages
are `prepare`, `collect`, `probe`, `screen`, `confirm`, and `summarize`; `all`
runs them in order. All stages are CPU-compatible and reuse existing artifacts
when `--skip_existing` is supplied.

The screen is now complete. Its full results and interpretation are recorded in
[`EXPERIMENT_0A_CRITIC_DIAGNOSIS.md`](EXPERIMENT_0A_CRITIC_DIAGNOSIS.md). In
brief, world-model reward fidelity remained strong through horizon 48, whereas
the severely corrupted critic diverged under repeated self-bootstrapped
updates at every screened horizon. Confirmation was therefore cancelled. This
stress test motivates configurable IR-specific critic targets, but it does not
show that the much smaller, naturally occurring critic mismatch expected in CIR
will behave similarly.

### Experiment 0b: early Cheetah CIR pilot

Move directly to a small continual-learning run before broad ablations. Start
from the converged DMC Cheetah Run vision checkpoint and create an environmental
change without changing the observation, reward, termination, or task code:
task A applies the policy action unchanged, while task B multiplies every
continuous action by `0.7` immediately before the DMC step.

The fixed phase sequence is:

1. Run three frozen episodes on A to establish return, critic, actor, and model
   diagnostics.
2. Run two frozen episodes on B. Classify the shift as mild when mean B/A
   return exceeds `0.85`, severe below `0.15`, and usable in between, but do not
   silently retune or abort this first observation.
3. Repair only the world model for 128 replay updates using batches of eight
   sequences of length 32 and a fresh constant-rate `4e-5` model optimizer.
4. Run ten persistent CIR episodes on B. The world model is frozen inside each
   episode. Trigger MCPB-128 rehearsal at the first posterior and every 25
   environment steps, using horizon 12, one critic-to-actor update, and
   independent constant `1e-4` optimizers. The first trigger after model repair
   adds a critic-only warm-up update. After every episode, apply 32 model-only
   replay updates.
5. Evaluate frozen A and B policies after model repair and after B episodes 1,
   3, 5, and 10. Evaluation adds nothing to replay and performs no adaptation.

The replay remains the existing FIFO sequence store and contains experience
from A and B, but its gradients are restricted to `enc/`, `dyn/`, `dec/`,
`rew/`, and `con/`. Persistent IR may change `val/`, `slowval/`, and `pol/`, but
not the world-model namespaces. Full namespace hashes enforce both statements.
World-model updates happen only at episode boundaries so that an encoder or
RSSM change cannot invalidate the live recurrent carry mid-episode.
The pilot also disables cached replay latents (`replay_context: 0`), because
encoder/RSSM updates would make context entries written by an older model
stale; sampled length-32 sequences are therefore reconstructed directly.
Each independent replay batch starts from a fresh recurrent carry so unrelated
sampled windows cannot leak latent state into one another.

Every phase logs returns, A forgetting and B recovery, action saturation,
policy entropy and KL from the initial actor, online/slow critic disagreement,
value magnitude, realised versus predicted returns, reward coverage, and
open-loop reward/return fidelity at horizons 6, 12, 24, and 48. Critic IR also
logs the reward-only target and the extra contribution caused by bootstrapping;
the formula itself remains unchanged. Optimizer counts, imagined-update counts,
wall time, resident memory, raw trajectory latents, per-anchor probe rows, and
phase summaries are retained.

Any non-finite model, actor, critic, or diagnostic value stops and checkpoints
the run. The critic also stops after its held-out absolute value p99 exceeds ten
times the A baseline at two consecutive diagnostics. The phase and episode
state, replay, fresh optimizer states, and live agent are checkpointed so an
interrupted CPU run resumes rather than restarts.

This pilot deliberately omits A-return training, learned gates, replay changes,
seed sweeps, matched baselines, and critic bootstrap changes. Its job is to
expose the first real CIR failure clearly enough to choose the next experiment.
The reward-only critic target remains a documented fallback if bootstrapping is
again identified as the failure mechanism.

#### Actor-only follow-up

The completed critic-first pilot showed that world-model-only repair improved B
return before behaviour rehearsal, whereas repeated critic-first rehearsal
collapsed both B performance and A retention. The critic remained useful on
fixed competent trajectories but became uninformative on the actor's changing
live distribution, while its imagined targets were overwhelmingly determined
by its own bootstrap.

The immediate follow-up therefore repeats the same schedule with the online
critic, slow critic, value normalizers, and critic optimizer frozen. Actor IR
continues to use the same factored rollout, MCPB-128 sampling, horizon 12, and
constant-rate dedicated actor optimizer as the critic-first control. This is a
single-factor ablation: only the critic update and initial critic warm-up are
removed. Critic namespace hashes are checked after every rehearsal episode.

This run tests whether a healthy state-value critic can serve as comparatively
slow task knowledge during a dynamics change: the repaired world model supplies
the new action-to-state mapping, the frozen critic supplies state desirability,
and actor-only IR learns how to reach those states under the new dynamics.

#### World-model-only control

The actor-only run prevented the near-zero collapse caused by critic rehearsal,
but did not improve on the policy measured immediately after the initial model
repair and gradually lost A performance. The next control removes actor IR as
well as critic IR. Both behaviour components and their optimizer states are
frozen by namespace assertions.

All other experimental choices remain matched: three A baseline episodes, two
B shock episodes, 128 initial world-model updates, ten B collection episodes,
32 world-model updates after each collection episode, and frozen A/B evaluation
at milestones 1, 3, 5, and 10. Thus the model still receives 448 supervised
updates in total, but the run performs no imagined behaviour updates. This
isolates whether supervised adaptation of the latent dynamics is sufficient for
the action-gain change and whether continued latent representation learning by
itself causes A forgetting.

### Experiment 1: stationary CIR learning

Train on one stationary task without starting from a complete behaviour:

- real replay sequences train only the world model;
- current-posterior IR is the sole source of actor and critic gradients;
- world-model training and IR initially run at fixed, always-on schedules;
- the existing FIFO replay remains unchanged.

Compare:

- ordinary DreamerV3;
- DreamerV3 with matched update and imagination budgets;
- actor-only IR;
- simultaneous actor-critic IR; and
- critic-first actor-critic IR.

The purpose is to determine whether current-posterior behaviour learning has
enough state coverage to learn a competent policy at all. MCPB diversifies the
stochastic state at one belief, but it does not provide the historical-state
coverage of replay-started imagination. This is the central risk and should be
tested before any CRL infrastructure is built.

### Experiment 2: controlled environmental change

Use a simple `A -> B -> A` sequence with fixed update schedules and no learned
gates. Measure:

- immediate performance loss after each change;
- return area under the recovery curve;
- real steps to recover a fixed performance threshold;
- recurrence gain on the return to A;
- critic recovery relative to actor recovery; and
- world-model prediction error around each transition.

This is the first direct test of CIR's proposed advantage: a retained predictive
model can be converted back into useful behaviour rapidly at the current state.

### Experiment 3: adaptive world-model computation

Introduce only the world-model gate while keeping behaviour IR always on.
Compare:

- a fixed world-model update ratio;
- random updates with the same total number of gradients;
- periodic updates with the same budget;
- a known-boundary oracle;
- held-out-loss dynamic UTD; and
- CIR's prediction-driven repair schedule.

The gate succeeds only if it preserves performance while reducing optimizer
steps, FLOPs, or wall-clock time. Skipping gradients is not itself a research
result unless it improves the performance-compute frontier.

### Experiment 4: adaptive behaviour computation

Introduce the behaviour gate after the world-model gate is understood. Compare
always-on CIR with actor-disagreement, critic-uncertainty, advantage-based,
periodic, random, and oracle schedules under matched imagination budgets.

### Experiment 5: lifelong retention

Only now introduce longer task streams and specialised world-model memory.
Reservoir replay, Continual-Dreamer, or ARROW can be adopted as fixed
infrastructure. Stored sequences continue to train only the predictive model;
the actor and critic remain exclusively current-posterior CIR learners.

## Core Hypotheses

1. **Behavioural sufficiency:** Current-posterior MCPB provides enough imagined
   experience to train both critic and actor without replay-started behavioural
   learning.
2. **Critic-first stability:** Delaying the actor until the critic has adapted
   improves recovery and reduces harmful policy updates relative to a joint
   update.
3. **Transfer speed:** A retained world model allows CIR to recover behaviour
   faster after environmental change or task recurrence.
4. **Predictive efficiency:** Event-triggered model repair matches fixed-ratio
   model quality with fewer backward passes.
5. **Behavioural efficiency:** Gated CIR retains most of the benefit of
   always-on rehearsal with fewer imagined transitions.
6. **Trust boundary:** CIR becomes harmful when policy optimisation proceeds
   through an inaccurate model; a trust gate should predict and reduce this
   failure.

## Evaluation and Controls

Report more than final return:

- complete lifetime return curves;
- recovery AUC and steps to threshold;
- forward transfer, recurrence gain, and forgetting;
- world-model loss on recent and retained held-out sequences;
- predicted versus realised returns;
- actor and critic update counts;
- total imagined transitions;
- model and behaviour forward/backward FLOPs where practical;
- wall-clock training and acting throughput;
- replay memory and peak device memory; and
- performance with IR disabled during evaluation.

Required controls include:

- fixed, periodic, and random schedules matched to each learned gate's compute;
- ordinary and compute-matched Dreamer;
- actor-only, joint actor-critic, and critic-first IR;
- MCPB and single-posterior starts;
- multiple seeds and task orders;
- absolute performance alongside forgetting; and
- a known-boundary oracle separated clearly from task-agnostic methods.

## Main Failure Modes

### Insufficient state coverage

Current-posterior MCPB samples uncertainty at the current belief, not arbitrary
past beliefs. The critic may become overly local. If this is the observed
failure, the smallest extension is a short recent-posterior window rather than
lifelong behavioural replay.

### Model exploitation

An actor and critic trained together can become confident in inaccurate
imagined returns. Short horizons, MCPB, critic-first updates, conservative
learning rates, and the future model-trust gate are the initial safeguards.

### Circular critic targets

The critic is bootstrapped from its own stale prediction at the rollout
horizon. Critic-only warm-up and multiple evaluation steps may help, but should
not be assumed to eliminate the bias. Predicted returns must be checked against
subsequent real returns.

### Latent representation drift

Model-only updates can change the encoder and RSSM feature coordinates consumed
by the actor and critic. Behaviour can therefore deteriorate even when the
underlying task has not changed. This is a direct consequence of complete
decoupling, not merely an implementation nuisance. Log policy and value changes
on a fixed observation probe set before and after model updates. If drift is a
limiting factor, representation anchoring or a slow target encoder may become a
necessary world-model mechanism, but neither should be added before the failure
is observed. The critic-only warm-up after model repair is the first safeguard.

### Cold start

At initialization, an untrained world model cannot provide reliable behaviour
gradients. CIR may require a seed collection phase, a pretrained model, or an
exploratory policy until minimum model trust is reached. Experiment 0 avoids
this confound; Experiment 1 must expose and measure it. A damaged actor may also
generate imagined actions with insufficient coverage for critic recovery, so
policy entropy and imagined reward coverage must be logged.

### Contradictory tasks

If the same latent state and action imply different rewards or dynamics without
observable context, neither replay nor CIR can represent both truths reliably.
Task-context assumptions must be explicit in every benchmark.

## Minimal Implementation Strategy

The conceptual change is large, but the code changes should remain local and
configurable so that the existing Dreamer and original IR paths remain intact.

1. Extend the existing adaptation path in `dreamerv3/agent.py` with a critic
   optimizer rather than replacing the normal learner.
2. Factor the current imagined rollout so actor and critic losses consume the
   same MCPB trajectory while retaining stopped gradients into the world model.
3. Add a critic-only first-step state and a configurable critic-to-actor update
   cadence.
4. Recompute critic predictions, lambda returns, and advantages before every
   actor update.
5. Add critic corruption/reset tooling and focused metrics for Experiment 0.
6. Only after component validation, add a CIR mode in which real replay invokes
   model-only loss and current interaction invokes behaviour-only IR.
7. Leave FIFO retention, task runners, and gating unchanged until their
   corresponding experiment is reached.

The first coding milestone is therefore narrow:

> Implement MCPB critic rehearsal and critic-first actor rehearsal on a frozen,
> already accurate world model, then establish that both a damaged critic and
> actor can recover on one stationary task.

## Experiment 0 implementation contract

The stationary-task implementation uses episode-local evaluation adaptation.
Its first trigger performs one critic-only MCPB rollout, then samples a fresh
rollout, updates the critic, recomputes values and lambda-return advantages,
and updates the actor before the first real action. Later triggers perform only
the critic-to-actor pass. The world model, reward head, continuation head, and
normalization statistics remain frozen.

The legacy actor-only optimizer remains unchanged. Critic-first IR uses
separate constant-rate actor and critic optimizers so momentum cannot cross the
update boundary and the first updates do not inherit Dreamer's 1,000-step
training warm-up. The initial experiment fixes both rates to `1e-4`, an
imagination length of 6, one actor update per trigger, `every_k=1`, and MCPB
`start_batch=512`.

The first recovery suite uses one qualified 1.1M-step Walker Walk checkpoint,
zero-mask severities 0.2/0.5/0.8 with corruption seed 0, and five evaluation
seeds. It compares no adaptation, legacy actor-only IR, and critic-first IR.
The online and slow critics receive identical corruption.
