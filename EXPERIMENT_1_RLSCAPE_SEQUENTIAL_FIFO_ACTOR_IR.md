# Experiment 1: RLScape Sequential FIFO Learning with Temporary Actor IR

**Status:** Runner implemented and locally tested; GPU preflight and launch pending
**Priority:** High — intended to start as an unattended GPU run lasting as
long as required
**Environment:** `rl-scape==0.1.6`, Lumbridge v1 atomic goals
**Agent:** Goal-conditioned DreamerV3, nominal 50M profile
**Seed:** One exploratory seed initially (`seed=0`)
**Replay:** Fixed-capacity FIFO; no reservoir replay
**Action interface:** 28x18 flat left-click grid (504 categorical actions)
**Evidence level:** First continual-learning shakedown, not a seeded paper result

## 1. Executive summary

This experiment is the first persistent-agent traversal of the complete
RLScape task landscape:

```text
kill_goblin
    -> bury_bones
    -> chop_logs
    -> light_fire
    -> catch_fish
```

One goal-conditioned Dreamer agent, its optimizer state, world model, actor,
critics, replay, and global step survive every task boundary. The environment
goal changes in fixed blocks. The agent is not reinitialized between goals.

At every 250,000 training interactions, an exact persistent checkpoint is
pinned. Each pinned checkpoint is evaluated on the current and all previously
encountered goals. Every checkpoint/goal pairing is evaluated under two paired
conditions:

1. frozen policy, without Imagined Rehearsal (IR); and
2. temporary actor-only IR.

Actor IR is evaluation-local. It updates a disposable copy of the actor from
imagined trajectories and is discarded at the end of each evaluation episode.
It never changes the training trajectory. Consequently, this experiment needs
one sequential training run, not separate "baseline training" and "IR
training" runs. Both evaluation conditions start from the same persistent
checkpoint.

The experiment has three immediate purposes:

1. measure learning speed on each new task relative to the independent
   single-task agents;
2. measure forgetting of earlier tasks as the FIFO learner moves through the
   landscape; and
3. test whether a frozen retained world model and critic can temporarily
   restore behaviour by adapting only a copied actor.

This is deliberately the simplest continual baseline. It does not introduce
reservoir replay, critic adaptation, persistent actor rehearsal, learned compute
gates, language goals, or task-free inference.

The repository now contains a dedicated restart-safe sequential supervisor,
paired RLScape evaluator, exact checkpoint-plus-replay milestone archives, and
focused local tests. `embodied/run/cir.py` remains the earlier Cheetah
action-gain experiment and is not reused for this protocol. A live GPU
preflight is still required before the unattended run is authorized.

## 2. Scientific questions and hypotheses

### 2.1 Sequential learning

Can one shared goal-conditioned Dreamer agent learn each RLScape goal when
experience arrives in fixed task blocks rather than in independent runs or a
random multitask stream?

The within-phase learning curve for goal \(g\) is

```text
L_seq(g, x) = success rate after x real interactions with phase g.
```

It will be compared descriptively with the existing seed-0 independent-agent
curve:

```text
L_single(g, x).
```

The comparison measures forward transfer in the persistent learner. It does not
yet support a population-level claim because the first experiment uses one
seed.

### 2.2 Forgetting

Let \(R^{frozen}_{i,g}\) be frozen-policy success on goal \(g\), evaluated after
training phase \(i\). For a goal first trained in phase \(j\), forgetting after
phase \(i\) is:

```text
F(i, g) = max over k in [j, i] R_frozen(k, g) - R_frozen(i, g)
```

The phase-by-goal matrix reveals whether behaviour degrades gradually, at a
specific interfering task, or immediately after FIFO experience is replaced.

### 2.3 Temporary behavioural recovery

Let \(R^{IR}_{i,g}\) be success from the same checkpoint and paired evaluation
seeds after temporary actor-only imagined rehearsal. The direct IR effect is:

```text
Delta_IR(i, g) = R_IR(i, g) - R_frozen(i, g)
```

A positive value is only evidence for behavioural recovery if:

- the persistent checkpoint is identical between conditions;
- training replay and environment state do not change;
- the world model and both critics remain frozen during IR;
- the copied actor is discarded after every evaluation episode;
- the paired conditions see the same goal and initial environment seed or
  snapshot;
- no evaluation writes into the training checkpoint, replay, metrics, or
  optimizer state; and
- imagined reward/value predictions remain sufficiently serviceable to make
  the actor update interpretable.

### 2.4 Recurrence is deferred

This experiment stops training after `catch_fish`. It tests backward
transferability through evaluation-time frozen-versus-IR comparisons; it does
not train a return block.

A later experiment can fork one of the immutable checkpoints and train
`kill_goblin` again to measure reacquisition. Keeping that intervention
separate prevents return training from complicating the first actor-IR result.

## 3. Relationship to the single-task ladder

The running milestone-1 suite trains a fresh 50M agent for 200,000 interactions
on each goal. Its purpose is task learnability and systems calibration. The
sequential experiment begins only after that suite has:

- exited cleanly;
- produced a summary and exact 20-episode evaluation for every goal;
- left no RLScape server/client process holding the bridge ports;
- demonstrated that each task can produce factual success rewards; and
- shown credible learning on every goal selected for the sequential stream.

The independent agents provide descriptive controls for:

- first-success interaction count;
- success-rate learning curves;
- final-window online success;
- 200k held-out success;
- episode length; and
- action-location concentration.

They are not a statistically complete baseline because only one seed is being
used. The sequential experiment should therefore be described as a mechanistic
shakedown until later seeds reproduce it.

The `bury_bones` result already demonstrates why held-out evaluation is
important: it reached 100% over its final training window and 20/20 held-out
success. The `kill_goblin` run, by contrast, developed strong sampled training
performance but achieved only 1/20 under categorical-argmax evaluation. That
gap means evaluation action selection must be declared explicitly rather than
silently equating deterministic argmax with the only valid policy.

## 4. Proposed immutable training specification

These settings are fixed for the first run.

| Component | Proposed setting |
|---|---|
| RLScape package | `rl-scape==0.1.6` |
| Task order | `kill_goblin, bury_bones, chop_logs, light_fire, catch_fish` |
| Training seed | `0` |
| Model | DreamerV3 nominal 50M |
| RSSM | 32 stochastic categoricals x 32 classes; deter 4096; hidden 512 |
| Goal representation | Canonical five-way one-hot `goal_id` |
| Goal-agnostic modules | Image encoder, RSSM posterior/prior, decoder |
| Goal-conditioned modules | Reward, continuation, actor, online value, slow value |
| Action | Flat `Discrete(504)` click grid, 28 columns x 18 rows |
| Episode limit | 200 executed actions |
| Reward | Sparse +1 task success |
| Batch | 8 sequences x length 32 |
| Imagination horizon | 15 |
| Training ratio | 32 |
| Compute | CUDA, bfloat16 |
| Replay | FIFO, capacity 200,000 transitions, offline sampling |
| Checkpoints retained | 3 rolling recovery checkpoints plus 10 immutable milestones |
| Wall-clock save interval | 1,800 seconds |
| Logging interval | 300 seconds |
| Reservoir replay | Disabled |
| Persistent training IR | Disabled |
| Critic IR | Disabled |

### 4.1 Why FIFO remains at 200,000

Dreamer requires replay, so "no replay intervention" cannot mean no replay at
all. It means the ordinary bounded FIFO buffer remains unchanged.

A replay capacity larger than the entire continual stream would retain every
past transition and act like comprehensive rehearsal. That would obscure the
forgetting baseline and make a later reservoir comparison difficult to
interpret. At 200,000 transitions:

- old experience is present temporarily after a task switch;
- it is gradually displaced during the next phase;
- a full 200k phase can replace the previous phase's transitions; and
- memory remains compatible with the 62 GiB host.

At 240x160 RGB, 200,000 uncompressed frames alone occupy approximately
21.5 GiB before replay metadata and transient arrays. Only one learner and one
RLScape instance may run at a time.

### 4.2 Persistent state across phase boundaries

The following state must survive unchanged:

- encoder, RSSM, decoder, reward, continuation, actor, and both critics;
- all persistent optimizer states;
- return/value normalization state;
- global environment-step counter;
- RNG/checkpoint state owned by the agent;
- the complete FIFO replay and its sampling state; and
- checkpoint history required for exact recovery.

The new goal is supplied only through the RLScape fixed-goal environment and the
canonical `goal_id` observation. No network is replaced or resized at a task
boundary.

The safest current implementation route is a serial phase supervisor using one
training state directory. Each phase starts a fresh Dreamer process with a
different fixed-goal environment configuration and a cumulative step target.
`embodied.run.train` restores the agent, replay, and global step from the same
checkpoint directory. The supervisor archives the resolved configuration for
each phase because `dreamerv3/main.py` otherwise rewrites `config.yaml` in the
shared log directory.

### 4.3 Environment reset at task boundaries

The first protocol performs a full RLScape lifetime restart between training
phases. Within a phase, normal task resets are used between episodes.

This makes phase starts reproducible and prevents inventory, player location,
or server history from becoming an undeclared transfer channel. Behavioural and
representational transfer still occur through the persistent Dreamer state and
FIFO replay. This task-boundary lifetime reset is part of the fixed protocol. A
continuous RLScape lifetime would be a different scientifically valid
experiment.

## 5. Phase schedule and checkpoints

Every task receives 500,000 real training interactions. Exact, immutable
milestones are pinned at 250,000 and 500,000 phase-relative interactions:

```text
G1 kill_goblin :       0 ->   250,000 ->   500,000
G2 bury_bones  : 500,000 ->   750,000 -> 1,000,000
G3 chop_logs   :   1.00M ->     1.25M ->     1.50M
G4 light_fire  :   1.50M ->     1.75M ->     2.00M
G5 catch_fish  :   2.00M ->     2.25M ->     2.50M
```

Total persistent training is 2,500,000 interactions. Fixed budgets are used
even if a goal appears mastered early; adaptive stopping would give different
algorithms different experience and invalidate learning-speed comparisons.

### 5.1 Immutable milestone archives

The ten 250k milestones are permanent, self-contained experimental fork points.
They are not subject to the rolling `ckpt_keep` retention policy.

Each archive must preserve:

- all agent parameters;
- persistent actor/model/value optimizer states;
- slow critic and normalization state;
- exact global step and RNG/checkpoint state;
- the complete FIFO replay state and every replay chunk needed to restore it;
- resolved training configuration;
- task/phase position;
- git commit, package/runtime identity, and immutable experiment digest; and
- a file inventory with integrity hashes.

This is intentionally stronger than an evaluation-only model checkpoint. A
future ablation must be able to branch from a milestone, restore the exact
learner and replay distribution, and continue training without replaying the
preceding task stream.

Archives are first written to a temporary directory, validated by loading their
step and replay inventory, and then atomically renamed into place. Where the
filesystem supports safe reflinks or hardlinks for immutable replay chunks,
they may reduce copying, but archive correctness must not depend on mutable
source paths. No milestone is deleted automatically.

Ten full FIFO snapshots may consume a few hundred GiB. The measured host has
approximately 1.9 TiB free, so correctness and future forkability take priority
over aggressive deduplication.

### 5.2 Evaluation scheduling

Long evaluations must not endanger completion of persistent training. The
training process pins every milestone and adds its evaluation cells to a
durable queue. The default schedule completes the 2.5M training stream first,
then drains that queue serially.

Evaluating an immutable checkpoint later is scientifically equivalent to
evaluating it immediately: evaluation cannot modify the saved checkpoint or
later training. This ordering avoids introducing an extra RLScape lifetime
restart halfway through each task and ensures that a slow actor-IR sweep cannot
prevent later task phases from being trained.

### Required phase-boundary artifacts

For every 250k milestone, the supervisor must atomically record:

- phase index and goal;
- exact cumulative checkpoint step;
- exact checkpoint directory;
- start and finish timestamps;
- child exit code and restart history;
- per-phase real interaction count;
- replay capacity and current size;
- resolved config and its digest;
- git commit;
- Python and RLScape package versions;
- RLScape runtime validation result;
- goal ID observed in training episodes;
- number of completed episodes, successes, and deaths;
- first-success phase-relative step;
- final-100 online success;
- action-audit transition count;
- evaluation matrix cells requested and completed; and
- persistent parameter/replay hashes needed for contamination checks.

No phase or milestone advances merely because its child process exits. It
advances only when the exact target checkpoint exists, its step matches the
declared boundary, and its self-contained archive passes validation.

## 6. Evaluation protocol

### 6.1 Matrix

At 250k and 500k within phase \(i\), evaluate the current and all previously
trained goals:

| Checkpoint | kill | bury | chop | fire | fish |
|---|---:|---:|---:|---:|---:|
| kill @250k and @500k | frozen + IR | — | — | — | — |
| bury @250k and @500k | frozen + IR | frozen + IR | — | — | — |
| chop @250k and @500k | frozen + IR | frozen + IR | frozen + IR | — | — |
| fire @250k and @500k | frozen + IR | frozen + IR | frozen + IR | frozen + IR | — |
| fish @250k and @500k | frozen + IR | frozen + IR | frozen + IR | frozen + IR | frozen + IR |

This doubled triangular matrix contains 30 checkpoint/goal cells and 60 primary
sampled-policy conditions.

Future, not-yet-trained goals are excluded from actor-IR evaluation because
their goal-conditioned reward/value predictions have not received positive
real examples. Applying IR there could optimize model error rather than test
retained behavioural knowledge.

### 6.2 Paired environment state

Frozen and IR conditions for a matrix cell must use:

- the same immutable training checkpoint;
- the same goal;
- the same ordered reset seeds;
- the same RLScape runtime and action grid;
- the same episode limit; and
- matching initial snapshot/frame identities where RLScape exposes them.

Each condition runs in a disposable RLScape process from the queued immutable
milestone archive. Evaluation is serial and normally begins after persistent
training completes. Evaluation never shares the live training environment,
replay, or log directory.

The evaluator records initial frame hashes, snapshot IDs when available,
episode IDs, transition identities, and seeds. A mismatch does not get averaged
away; the pair is marked invalid and retried.

### 6.3 Policy action selection

RLScape uses a 504-way categorical pointing policy. A stochastic policy can
legitimately place probability mass on several useful target cells. Categorical
argmax can therefore understate a useful multimodal policy, as observed in the
`kill_goblin` learnability run.

Both evaluation action modes are retained:

- **sampled evaluation** measures the policy Dreamer actually trained;
- **deterministic argmax evaluation** measures the reliability of its modal
  action path.

Paired sampled evaluation is the primary frozen-versus-IR comparison.
Deterministic argmax is a larger secondary diagnostic. Both modes are labelled
separately and are never pooled into one success rate.

### 6.4 Episode count

Each matrix cell runs:

- 100 sampled episodes with the frozen actor;
- the same 100 environment seeds with temporary actor IR; and
- 50 deterministic episodes with the frozen actor; and
- the same 50 deterministic environment seeds with temporary actor IR.

Wilson intervals and paired episode outcomes are reported.

At the worst-case 200 actions per episode:

```text
sampled:       30 cells x 2 conditions x 100 episodes x 200 = 1,200,000 actions
deterministic: 30 cells x 2 conditions x  50 episodes x 200 =   600,000 actions
total maximum                                               = 1,800,000 actions
```

At 6.8 actions/second this is approximately 73.5 acting hours before startup and
actor-IR compute. Completion is not constrained to the user's six-day absence.
The actor-IR preflight must measure adaptation throughput so the status system
can report a realistic remaining-time estimate.

### 6.5 Evaluation outputs

Every matrix cell records:

- requested and completed episode count;
- success count and rate;
- Wilson confidence interval;
- action-length mean, median, and quantiles;
- deaths and termination reasons;
- progress maximum;
- categorical entropy;
- unique grid cells and top-cell concentration;
- actor-IR trigger/update/imagined-transition counts;
- actor KL from the persistent source actor;
- predicted imagined return before and after IR;
- persistent namespace hashes before and after evaluation;
- paired reset/snapshot identity validation; and
- wall time and inference/adaptation throughput.

Incomplete attempts use a new attempt directory. A completed matrix cell is
idempotently skipped after restart only if its immutable specification digest
and exact episode count match.

## 7. Temporary actor-only IR contract

### 7.1 Allowed state change

Within one IR evaluation episode:

```text
theta_actor_copy <- theta_actor_persistent
temporary_optimizer <- fresh optimizer state
theta_actor_copy <- actor-only imagined updates
```

At episode end:

```text
discard theta_actor_copy
discard temporary_optimizer
```

The next episode starts again from the persistent actor in the source
checkpoint. Adaptation does not accumulate across evaluation episodes.

### 7.2 Frozen state

IR must not change:

- persistent actor or its training optimizer;
- encoder;
- RSSM posterior or transition;
- decoder;
- reward head;
- continuation head;
- online critic;
- slow critic;
- normalization statistics;
- training replay;
- training checkpoint;
- training logger/counter; or
- training RLScape lifetime.

Before and after every IR condition, the evaluator hashes the persistent
world-model, critic, actor, optimizer, and normalizer namespaces. Any unexpected
change invalidates the cell and stops that evaluation condition.

### 7.3 Fixed first IR recipe

The existing evaluation path can perform current-posterior actor adaptation.
The first RLScape recipe is:

| Parameter | Proposed value |
|---|---:|
| Actor-only | yes |
| Critic updates | 0 |
| World-model updates | 0 |
| Imagined horizon | 6 |
| Actor updates per trigger | 1 |
| Trigger | first observation and every subsequent environment step |
| Posterior samples per trigger | 128 |
| Actor adaptation learning rate | `4e-5` |
| Entropy coefficient | `3e-4` |
| Episode-local reset | mandatory |

This uses Monte Carlo samples from the current posterior, not replay-reservoir
starts. It should be described as current-posterior batching unless the
implementation is extended to genuine multi-context posterior batching.

In the current evaluator's configuration semantics, "adapt every step" must be
encoded as `every_k=1`; `every_k=0` means no periodic triggers after the initial
observation. The immutable manifest records both the semantic rule and its
implementation value to prevent ambiguity.

The learning rate matches the persistent optimizer rather than the more
aggressive `1e-4` Cheetah pilot. Actor KL is logged, not used as a hidden
adaptive stopping rule in the first run.

### 7.4 Interpretation gate

An IR failure is not evidence against imagined rehearsal unless retained
model/value serviceability is checked. For every phase/goal cell, the protocol
should report:

- reward prediction on retained real sequences, where those sequences still
  exist in FIFO;
- continuation prediction;
- predicted versus realized evaluation return;
- critic value distribution and ranking on recorded posterior probes; and
- representation drift on fixed raw-history probes captured at task
  boundaries.

Full reservoir-backed probes are a later milestone. In the FIFO run, missing
old real sequences must be reported as missing rather than reconstructed or
silently substituted.

## 8. Primary metrics

### 8.1 Learning and transfer

- phase-relative steps to first success;
- moving success rate in fixed 100-episode and fixed-interaction windows;
- area under the within-phase success curve through 250k and 500k;
- phase-relative steps to 25%, 50%, 75%, and 90% success, where reached;
- difference and ratio versus the independent seed-0 agent on the same goal;
- episode length and success-conditioned episode length; and
- current-goal success at the 250k and 500k checkpoints.

### 8.2 Forgetting

- phase-by-goal frozen success matrix;
- per-goal final forgetting;
- maximum observed forgetting;
- average forgetting across learned goals;
- worst retained-goal success;
- episode-length degradation; and
- task boundary at which each degradation first appears.

### 8.3 IR recovery

- paired success difference;
- paired episode-length difference;
- normalized recovery of the frozen forgetting gap;
- number of episodes helped, harmed, or unchanged;
- actor KL and entropy change;
- imagined-return change;
- update and imagined-transition count; and
- recovery as a function of within-episode action index.

### 8.4 Systems

- exact real and evaluation interaction counts;
- replay occupancy;
- policy and training FPS;
- GPU memory and utilization;
- host RAM and disk;
- checkpoint save duration;
- RLScape startup duration;
- restart count and reason;
- no-progress time;
- orphan-process/port checks; and
- complete action-audit coverage.

## 9. Unattended execution and recovery requirements

Reliability is part of the experiment acceptance criterion. The supervisor must
be restartable from its top-level command and must never depend on an attached
terminal.

### 9.1 Process ownership

- Launch the complete supervisor under one `xvfb-run` inside a named tmux
  session.
- Start each Dreamer/evaluation child in its own process group.
- Track the child PID and all descendants.
- On timeout or interruption, send `SIGINT` to that child group first so
  `embodied.run.train` can save its exact recovery checkpoint.
- Allow a bounded checkpoint grace period before `SIGTERM`.
- Never use broad Java or Python kill patterns.
- Verify that RLScape ports are released before the next attempt.

### 9.2 Training recovery

`embodied.run.train` already saves an exact checkpoint when an exception
escapes and closes the Driver in `finally`. The new supervisor must:

1. read the exact checkpoint step before an attempt;
2. launch the same immutable phase command;
3. read the checkpoint step after exit;
4. resume if the target was not reached;
5. count total and same-step restarts separately;
6. archive every return code and console tail;
7. refuse to advance without the exact target checkpoint; and
8. preserve replay and optimizer state across the retry.

Implemented default limits:

- 20 total child restarts across the complete experiment;
- 4 consecutive same-checkpoint restarts;
- 15-second ordinary restart delay;
- 60-minute startup deadline before the first activity signal;
- 120-minute no-activity watchdog during training; and
- 240-minute no-activity watchdog during the much slower per-step IR sweep.

A repeated same-step failure is marked `blocked`; the supervisor does not
silently change batch size, model size, replay, task order, or training budget.

### 9.3 Evaluation recovery

Evaluation has no mutable training state and can safely retry from the immutable
boundary checkpoint:

- retry a failed cell in a new attempt directory;
- never append to a partial episode count;
- try missing cells again after the rest of the boundary matrix;
- preserve all failed artifacts and exception tails; and
- continue the remaining queue after bounded retries, record the missing cell,
  and revisit it after all other cells.

A persistently failing evaluation cell does not invalidate or block the
immutable training trajectory. It remains explicitly missing and prevents only
a false claim that the evaluation matrix is complete.

### 9.4 Heartbeat and monitoring

The top-level status file is rewritten atomically every 30 seconds while a
child is active and contains:

- overall state;
- phase and goal state;
- steps before and after each attempt;
- current child PID and owned process group;
- child start, heartbeat, and last-activity timestamps;
- current and total restart counts;
- completed/required paired evaluation units; and
- every exact child command and return code.

Training output is appended to `training/console.log`; evaluation output is
isolated by pair attempt and condition. The active-child record includes the
Linux process start tick and exact command, allowing a restarted supervisor to
adopt the correct live child without relying on a possibly reused PID.

### 9.5 Resource guards

The implemented supervisor currently checks:

- the source git commit as part of the immutable specification;
- `rl-scape==0.1.6`;
- a configurable free-disk floor, 300 GiB by default;
- the selected Python and Dreamer entry point;
- the exact checkpoint step; and
- both exact milestone archives before advancing a phase.

RLScape performs its own runtime-hash validation when each environment is
constructed. CUDA/GPU memory, port release, runtime hashes, and a real
checkpoint/replay restore remain explicit live-preflight checks. Nothing is
deleted automatically when a guard fails.

## 10. Artifact layout

Root:

```text
logs/rlscape/e1_fifo_seq_actor_ir_seed0/
```

Implemented core contents:

```text
experiment_spec.json
supervisor_status.json
active_child.json  # present only while a child is owned
evaluation_console.log
training/
  ckpt/
  replay/
  metrics.jsonl
  episodes.jsonl
  audit_env0.jsonl
  config.yaml
milestones/
  step_000000250000/
    checkpoint/
    replay/
    manifest.json
    archive_complete
  ...
evaluations/
  queue.json
  results.json
  results.csv
  step_.../
    GOAL/
      sampled/
        attempt_01/{frozen,actor_ir}/
      deterministic/
        attempt_01/{frozen,actor_ir}/
  ...
```

The training directory is the only mutable persistent learner state. Evaluation
directories are disposable consumers of immutable boundary checkpoints.

## 11. Current implementation support and gaps

### Already supported

- RLScape 0.1.6 adapter and runtime validation;
- canonical goal IDs in observations and audit records;
- five fixed-goal configurations;
- 28x18 public click-grid action wrapper;
- goal-conditioned reward, continuation, actor, online critic, and slow critic;
- goal-agnostic visual encoder/RSSM/decoder;
- 50M/bfloat16 GPU profile;
- exact final and exception recovery checkpoints in `embodied/run/train.py`;
- exact self-contained checkpoint-plus-active-FIFO milestone archives;
- exact-episode frozen evaluation;
- episode-level disposal of temporary evaluation parameters;
- actor-only adaptation path when critic adaptation is disabled;
- fixed-block cumulative sequential training supervisor;
- exact 250k milestone integration in the training loop;
- 30-cell, 60-pair frozen/actor-IR evaluation queue;
- explicit sampled and deterministic evaluation action selection;
- paired reset-seed and initial-frame digest assertions;
- before/after persistent parameter namespace digests;
- bounded retry/revisit behavior with fresh attempt directories;
- owned process groups, active-child adoption, heartbeat, and watchdogs; and
- JSON/CSV single-task and evaluation summaries.

### Remaining before launch

- commit and push the reviewed implementation;
- run the focused test suite on the GPU checkout;
- execute a short real two-goal phase-switch preflight;
- force and verify one real child recovery;
- execute paired sampled and deterministic actor-IR smoke evaluations;
- inspect GPU/RAM/disk use and RLScape port cleanup; and
- restart the top-level supervisor once during preflight to verify adoption or
  idempotent completion.

Phase-relative learning, forgetting, and IR-effect presentation tables are
postprocessing work after raw results exist; they are not allowed to mutate the
run or queue specification.

The existing `embodied/run/cir.py` is tied to DMC Cheetah action-scale phases and
does not implement this protocol. The new work should reuse its isolation and
diagnostic ideas, not its environment-specific state machine.

## 12. Required tests and preflight

### Unit tests

- phase specification digest rejects scientific-setting changes;
- target extension does not alter an otherwise identical run specification;
- completed phases are skipped idempotently;
- partial phases resume from the exact checkpoint;
- a phase cannot complete below its target step;
- each 250k milestone is pinned exactly once and survives rolling retention;
- a milestone restores agent, optimizer, normalizer, step, and exact replay;
- milestone validation rejects a missing or modified replay chunk;
- goal IDs remain canonical and constant within episodes;
- FIFO replay and optimizer state survive a goal switch;
- evaluation matrix construction is correct at every boundary;
- completed evaluation cells are skipped only after exact episode counts;
- failed cells receive new attempt directories;
- sampled and deterministic conditions cannot share an output directory;
- IR updates only the disposable actor namespace;
- critic, model, normalizer, and persistent optimizer hashes remain unchanged;
- temporary actor/optimizer state resets between episodes;
- paired reset identity mismatches invalidate a pair;
- heartbeat writes are atomic; and
- scoped process cleanup never targets unrelated processes.

### Fault-injection tests

- training child exits after real progress;
- training child exits before progress;
- RLScape startup fails;
- RLScape stalls after startup;
- checkpoint save receives an interrupt;
- evaluation exits after a partial episode count;
- supervisor itself is killed and restarted;
- stale PID/status files exist after a host reboot; and
- disk/RAM/runtime preflight fails.

### Live preflight

Before the unattended run:

1. run the focused unit suite;
2. run two RLScape goals for a small fixed budget through a real phase switch;
3. force one child restart and prove exact checkpoint/replay recovery;
4. run two frozen and two actor-IR paired episodes on both goals;
5. verify namespace hashes and temporary actor reset;
6. verify no orphan Java, Xvfb, or bridge ports;
7. inspect action/reward/goal audit coverage;
8. confirm peak GPU and host memory; and
9. restart the top-level supervisor and prove it skips completed work.

Only after this preflight passes should the long log root be created.

## 13. Runtime estimate

Measured RLScape acting throughput is approximately 6.8 interactions/second.
GPU training throughput is not the wall-clock bottleneck.

| Work | Approximate acting time |
|---|---:|
| Five 500k phases | 102 hours |
| 30-cell paired sampled + paired deterministic evaluation | up to 73.5 hours |

Startup, compilation, milestone archiving, recovery, and actor-IR updates add
overhead. Per-step IR with 128 posterior samples is likely to make the
evaluation sweep materially slower than environment acting alone. Completion is
not required within six days; the priority is a restart-safe, substantial
experiment. The live preflight must benchmark IR throughput and verify that
milestone archiving does not exhaust disk or stall training.

## 14. Stopping and claim boundaries

The experiment is complete only when:

- every declared training phase has an exact boundary checkpoint;
- every required matrix cell has its exact paired episode count or is explicitly
  recorded as a terminal failure under the chosen failure policy;
- no contamination assertion failed;
- action audits cover all non-reset training transitions;
- replay/phase/goal counts reconcile;
- summary matrices can be regenerated from raw artifacts; and
- all owned processes are closed.

With one task order and one seed, allowed conclusions are:

- whether the complete continual pipeline works;
- whether each goal was learned in this trajectory;
- where substantial forgetting appeared;
- whether temporary actor IR showed a paired effect worth repeating; and
- which failure modes require the next controlled experiment.

Not allowed:

- a general claim that actor IR prevents catastrophic forgetting;
- a population-level transfer claim;
- a reservoir-replay comparison;
- a claim that deterministic argmax is the only valid policy;
- a persistent CIR claim; or
- choosing hyperparameters after inspecting the final evaluation matrix and
  reporting the same run as confirmatory.

## 15. Final decision record

Confirmed:

1. fixed goal order:
   `kill_goblin -> bury_bones -> chop_logs -> light_fire -> catch_fish`;
2. 500,000 training interactions per goal;
3. immutable, self-contained fork checkpoints every 250,000 interactions;
4. 100 paired sampled episodes per frozen/IR matrix cell;
5. 50 paired deterministic episodes per frozen/IR matrix cell;
6. full RLScape lifetime restart at task boundaries;
7. bounded evaluation retries, then continue the queue and revisit missing
   cells at the end;
8. actor-only IR at every environment step, encoded as `every_k=1`;
9. one actor update per trigger, 128 current-posterior samples, horizon 6,
   learning rate `4e-5`, and entropy coefficient `3e-4`; and
10. frozen critic, world model, normalizers, persistent actor, and all
    persistent optimizer state during evaluation IR;
11. no return-training phase after `catch_fish`; and
12. return/reacquisition training is reserved for a separately designed
    experiment forked from the immutable milestone archives.

These choices become part of the immutable manifest. The unattended supervisor
must not change them in response to observed scores.
