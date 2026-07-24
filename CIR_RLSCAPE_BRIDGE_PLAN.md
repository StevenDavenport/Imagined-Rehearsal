# CIR–RLScape Bridge: Architecture and Experiment Plan

**Status (2026-07-23):** milestone-0 integration, the multigoal Dreamer
training smoke, and a clean one-seed/1k-action milestone-1 systems calibration
over all five goals are implemented and verified. Multi-seed single-task
learning curves, the joint upper bound, the sequential paired-evaluation
runner, and the reservoir extension remain future milestones.

**Pinned environment contract:** `rl-scape==0.1.1`.

**Scope boundary:** this plan replaces neither
[`CIR_RESEARCH_PLAN.md`](CIR_RESEARCH_PLAN.md) nor the Cheetah pilot. It defines
the smaller bridge experiment requested in [`CIR_RLScape.md`](CIR_RLScape.md).
The first scientific target is temporary, evaluation-local, actor-only imagined
rehearsal on top of an ordinarily trained goal-conditioned Dreamer agent. It is
not persistent actor CIR and it is not the earlier critic-first learning
contract.

## 1. Research objective and claim boundary

The programme asks whether retained predictive knowledge can be converted back
into useful goal-conditioned behaviour without collecting more training
experience:

> After sequential training has partially forgotten a goal, can a shared
> retained world model and a still-serviceable frozen critic support temporary
> actor-only imagined rehearsal that improves evaluation performance on that
> goal?

The actor copy may learn during one isolated evaluation episode. The training
agent, training replay, training environment lifetime, model, critic,
normalizers, and all persistent optimizer states must not change. The adapted
actor and its optimizer are discarded at the episode boundary.

This question deliberately separates four possible outcomes:

1. The task is not learnable from pixels and canonical mouse control.
2. The goal-conditioned architecture or shared capacity is inadequate.
3. Sequential training forgets, but retained model/value knowledge is no longer
   serviceable.
4. Serviceable imagined knowledge remains and temporary actor optimization can
   recover behaviour.

Only the fourth outcome is evidence for evaluation-time IR. A positive IR result
does not by itself establish persistent CIR, learned compute gates, task-free
context inference, critic repair, or semantic language conditioning.

The first training baseline is ordinary joint DreamerV3 training with the
existing FIFO replay. At every checkpoint in the task landscape, disposable
evaluations compare the same persistent learner with IR disabled and enabled.
Because evaluation-time IR cannot affect later training, both conditions share
one training trajectory and differ only inside paired evaluation clones.

Reservoir replay is a later retention intervention. It is not required for the
first sequential/IR experiment and it must not be introduced in the same causal
comparison as actor IR.

### Implementation checkpoint

The verified vertical slice now includes:

- a dedicated Gymnasium-to-Embodied adapter in
  `embodied/envs/rlscape.py`, pinned to `rl-scape==0.1.1`;
- public image, goal, reward, success, progress, death, and transition identity
  handling without exposing `privileged_state` to the agent;
- deterministic fixed, round-robin, and seeded-random goal schedules over all
  five v0.1.1 tasks;
- native factored `mode: Discrete(4)` and `position: Box(2)` actions, with
  no-op positions canonicalized before both execution and Dreamer replay/RSSM
  use;
- explicit goal conditioning for the ordinary Dreamer reward, continuation,
  actor, value, and slow-value heads while leaving the visual
  encoder/RSSM/decoder goal-agnostic;
- ordinary Dreamer continuation semantics, with success and death both encoded
  by `is_terminal`; separate success/death fields remain diagnostics only;
- `rlscape`, `rlscape_smoke`, and five fixed-goal named configurations;
- adapter, Driver, and FIFO replay contract tests.

Verification completed on 2026-07-23:

- `7 passed` in `embodied/tests/test_rlscape_adapter.py`;
- a live reset/step/close cycle traversed all five goals at `160x240`, with
  stable goal IDs `0..4` and server-confirmed canonical no-op actions;
- the original end-to-end `rlscape + rlscape_smoke` run completed six short
  episodes, crossed all five goals, and populated FIFO replay;
- after simplifying split termination to the ordinary goal-conditioned
  continuation head, a fresh 220-action run executed finite updates for
  `image`, `dyn`, `rep`, `rew`, `con`, `policy`, `value`, and `repval`, saved
  exactly at action 220, and contained no `comp/` namespace;
- a non-goal Dummy run still initializes and steps through the original
  unconditioned feature path, confirming that ordinary non-goal Dreamer
  behavior is preserved.

For a source checkout, RLScape needs its Maven and Java 8 toolchain. They can be
provided as environment variables:

```bash
export RL_SCAPE_MVN=/path/to/mvn
export RL_SCAPE_JAVA_HOME=/path/to/java/home
python dreamerv3/main.py \
  --configs rlscape rlscape_smoke \
  --logdir logs/rlscape/m0_smoke_seed0
```

`env.rlscape.mvn_path` and `env.rlscape.java_home` expose the same setup through
configuration. A prebuilt RLScape runtime avoids the Maven requirement.

This checkpoint deliberately does not claim task learning: the smoke is short,
and zero reward is expected. Its result is that real frames, mixed actions,
goal identity, FIFO replay, imagination, and gradients form one functioning
path. Milestone 1 is now the next scientific step.

### Milestone-1 systems calibration

The resumable `scripts/rlscape_m1_pilot.py` launcher now runs the five fixed-goal
configurations serially, owns per-run status/checkpoints, and produces JSON/CSV
summaries. The generic runner closes its Driver and logger deterministically,
saves an exact final checkpoint, and streams RLScape transition audits to disk
instead of retaining an unbounded in-memory list. Episode logs include the
executed action-mode distribution and position magnitude, with a coverage
assertion against the number of non-reset transitions.

The clean seed-0 calibration used 1,000 actions per goal, 200-action episodes,
FIFO replay, sparse success reward, no IR, and a deliberately light
`train_ratio=4` CPU configuration:

| Goal | Complete episodes | Successes | First success action | Logged actions | Finite losses |
|---|---:|---:|---:|---:|---|
| `kill_goblin` | 4 | 0 | — | 800/800 | yes |
| `bury_bones` | 4 | 0 | — | 800/800 | yes |
| `chop_logs` | 5 | 1 | 483 | 880/880 | yes |
| `light_fire` | 4 | 0 | — | 800/800 | yes |
| `catch_fish` | 4 | 0 | — | 800/800 | yes |

All five processes returned zero, handed the single RLScape bridge port to the
next task cleanly, updated every Dreamer loss family, and left no server/client
processes behind. Policy mode entropy was still essentially maximal when the
`chop_logs` completion occurred, so the isolated success is evidence that the
event/reward path is reachable, not evidence of learned behaviour. These are
systems-calibration results and do not satisfy milestone 1's multi-seed
learnability criterion.

The authoritative artifacts are:

```text
logs/rlscape/m1_pilot_seed0_1k_clean/summary.json
logs/rlscape/m1_pilot_seed0_1k_clean/summary.csv
```

An earlier `m1_pilot_seed0_1k` directory is superseded: it exposed and motivated
a fix for asynchronous episode-log scalar aliasing and process cleanup, but its
episode diagnostics must not be used scientifically.

The clean 1k calibration itself preceded removal of the experimental `comp/`
head. Its environment/throughput evidence remains useful, but its checkpoints
must not seed later learning runs. All longer single-task and continual runs
start from the simplified single-`con/` architecture verified by
`m0_single_con_smoke`.

The subsequent 10k-action, three-seed single-task screen completed 14 of 15
runs; `chop_logs` seed 0 stopped at action 5,719 after a Java client
`ArrayIndexOutOfBoundsException` caused an RLScape frame-barrier timeout. The
other runs were numerically finite and completed cleanly. Online success was
20.0% for `kill_goblin`, 0.7% for `bury_bones`, 42.4% for `chop_logs` including
the partial run, 0% for `light_fire`, and 0.7% for `catch_fish`. However, the
four-way action-mode entropy remained essentially maximal, and first-to-last
success did not rise. These trajectories demonstrate sparse-event reachability,
not competent learned policies.

The next exploratory upper-bound run therefore trains jointly on all five
goals. This is a deliberate deviation from first freezing a learnable-only
catalogue: uniformly random per-episode goals may let the easier reward-bearing
tasks teach shared mouse/interface dynamics that transfer to sparse tasks. It
does not replace the single-task evidence or authorize task-order selection.

`rlscape_m2_multitask` is implemented but not yet run. It uses:

- all five goals, sampled uniformly at each episode reset;
- 2,000,000 driver transitions and a 200-action episode cap (approximately
  1.99M executed actions when episodes usually reach the cap);
- the 50M Dreamer profile, batch size 8, replay sequences of length 32, and
  imagination length 15;
- training ratio 32 and a 5M-transition FIFO replay, which will not fill during
  this run;
- one RLScape environment because the current bridge is single-instance;
- CUDA with bfloat16 and no actor IR.

`rlscape_m2_eval` and `scripts/rlscape_m2_eval.py` provide a frozen,
deterministic final evaluation. They resolve the exact final checkpoint, disable
adaptation, run exactly 20 episodes separately on each fixed goal, close the
environment between goals, and produce combined JSON/CSV summaries. At the
measured acting throughput of approximately six actions per second, 2M
interactions takes roughly 93 hours; GPU acceleration primarily increases
training throughput rather than RLScape acting throughput.

The 50M profile should fit a 24GB GPU under bfloat16, but 240x160 activations
make this a hardware-dependent claim. The required preflight is a separate 2k
run at batch size 8; if it OOMs, rerun from a fresh log directory with
`--batch-size 4` while keeping the model, sequence length, train ratio, replay,
and interaction budget unchanged.

Long training runs save a recovery checkpoint when an exception escapes the
environment/driver. `scripts/rlscape_m2_train.py` supervises the child process
and resumes transient RLScape client or frame-barrier failures from that exact
state, with bounded total and no-progress restart counts.

## 2. Why the ladder is ordered this way

| Milestone | Question resolved | Required stopping result |
|---|---|---|
| 0 — integration smoke | Is the CIR process boundary faithful to RLScape v0.1.1? | Valid reset/action/close cycle with goal and transition identity audited. |
| 1 — five single-task runs | Which goals are learnable, at what horizon and cost, under canonical control? | Multi-seed learning evidence and calibrated fixed phase budgets. |
| 2 — joint multitask upper bound | Can one shared goal-conditioned model and behaviour system represent the learnable goals simultaneously? | Adequate per-goal success without sequential non-stationarity. |
| 3 — FIFO task landscape with paired IR evaluation | As one persistent learner moves through goal blocks, how quickly does it learn each current goal, how much does it forget previous goals, and can temporary actor IR restore performance? | Phase-by-goal frozen/IR matrices and transfer/forgetting curves on paired checkpoint/snapshot/seed tuples. |
| 4 — reservoir extension | After the first FIFO result is understood, how does fixed episode retention alter learning, forgetting, model serviceability, and the marginal value of actor IR? | FIFO-versus-reservoir comparison with IR behavior and compute held fixed. |
| 5 — conditional CIR extensions | Does the evaluation-local result justify persistent actor CIR or other rehearsal starts/gates? | Enter only after a repeatable isolated actor-IR benefit or a clearly diagnosed failure. |

The first continual sequence should traverse the full set of goals that meet the
single-task learnability criterion:

```text
G1 -> G2 -> G3 -> ... -> GK
```

Optionally append one or more return phases to measure recurrence. The
single-task study determines budgets and may determine order; it is not merely a
screen for a hardcoded three-task A/B/C subset.

At every checkpoint, evaluate the current goal and every previously encountered
goal. On the current goal, the frozen-versus-IR difference measures immediate
behavioural transfer at that point in training. On previous goals, it measures
forgetting and temporary recovery. Since IR is evaluation-local, it does not
change the persistent learner's subsequent learning curve; any “speed of
transfer” claim must be described as adapted evaluation performance or
within-episode recovery, not faster persistent parameter learning.

Persistent actor CIR becomes a design candidate only after milestone 3 produces
a repeatable paired benefit without training-state contamination.

## 3. Pre-implementation architecture map

### 3.1 Construction and configuration

- `dreamerv3/main.py:main()` merges named YAML configurations, constructs common
  run arguments, and selects `train`, `train_eval`, `eval_only`, `cir`, or the
  parallel runners.
- `dreamerv3/main.py:make_env()` parses `config.task` at the first underscore
  and dispatches to a fixed suite table. There is no `rlscape` suite.
- `dreamerv3/main.py:wrap_env()` normalizes and clips every continuous action,
  unifies dtypes, and validates spaces.
- `dreamerv3/main.py:make_agent()` creates and closes an environment solely to
  discover spaces, removes `log/*` observations and `reset`, and constructs the
  JAX Dreamer agent.
- `dreamerv3/main.py:make_replay()` always constructs
  `embodied.replay.Replay`; its optional selector mixture changes sampling, not
  retention.
- `dreamerv3/configs.yaml` has no RLScape, goal-conditioning, phase-stream,
  reservoir-episode, or clone-evaluation configuration.

### 3.2 Environment and transition flow

- `embodied/envs/from_gym.py:FromGym` understands dict observation/action spaces
  and Gym-style four/five-tuples, but imports legacy `gym`, not Gymnasium. It
  retains the latest `info` only as an object property and does not place reset
  metadata or step diagnostics in the returned observation.
- `embodied/core/wrappers.py:NormalizeAction`, `UnifyDtypes`, `CheckSpaces`, and
  `ClipAction` operate independently per action key and are compatible with a
  factored action dictionary.
- `embodied/core/driver.py:Driver._step()` sends the previous policy action to
  the environment, stacks the returned observation, separates `log/*`, calls
  the policy, and forms the recorded transition from observation, newly sampled
  action, policy outputs, and logs.
- `embodied/core/replay.py:Replay.add()` removes `log/*` fields. All remaining
  observation/action fields are stored automatically.
- The generic training loggers accept scalar `log/*` values and policy videos.
  They are not structured audit-record writers.

The consequence is that RLScape needs a dedicated CIR-side adapter. The generic
Gym wrapper cannot recover reset-only goal metadata after the fact, and it
cannot expose structured protocol diagnostics in a form that current runners
will safely preserve.

### 3.3 Agent and goal-dependent heads

- `dreamerv3/agent.py:Agent.__init__()` excludes only `is_first`, `is_last`,
  `is_terminal`, and `reward` from its encoder/decoder spaces. Any new ordinary
  observation key, including a naïvely added goal ID, would currently enter the
  encoder, RSSM posterior, and decoder.
- The encoder, RSSM, and decoder are `enc/`, `dyn/`, and `dec/`.
- `rew/` and `con/` are unconditioned MLP heads over RSSM features.
- `pol/` is one MLP trunk with a `DictHead`, and `val/` plus `slowval/` are
  unconditioned value heads.
- `Agent.policy()` infers the posterior and samples `pol/`.
- `Agent.loss()` trains the world model and ordinary replay-started imagined
  actor/critic jointly. Its continuation target is `~is_terminal`; it cannot
  distinguish task success from physical death.
- `Agent.model_loss()` is a live user change from the Cheetah programme. It
  supports model-only updates but has the same unconditioned reward and combined
  continuation semantics.
- `Agent._adapt_rollout()` supports one current posterior or MCPB stochastic
  resampling, freezes imagined features/reward/continuation, and produces the
  common rollout used by adaptation.
- `Agent._adapt_actor_loss()` already freezes value predictions and
  normalization updates. The `freeze_critic=True` branch uses the dedicated
  constant-rate actor optimizer.

The intended goal boundary therefore requires an explicit code path. It cannot
be achieved solely by changing the environment observation space.

### 3.4 Mixed actions

The canonical RLScape action is already natively representable:

- `dreamerv3/agent.py:Agent.__init__()` chooses `categorical` for discrete
  action keys and `bounded_normal` for continuous keys.
- `embodied/jax/heads.py:DictHead` builds a separate output distribution per
  key from a shared actor trunk.
- `dreamerv3/agent.py:imag_loss()` and `adapt_policy_loss()` sum log
  probabilities and entropies across action factors.
- `dreamerv3/rssm.py:RSSM` accepts an action dictionary, and
  `embodied/jax/nets.py:DictConcat` one-hot encodes discrete factors while
  concatenating continuous factors.
- `embodied/core/random.py:RandomAgent` and `Replay` preserve per-key actions.

Thus `mode: Discrete(4)` plus `position: Box(2)` should be the first
representation. There is no reason to flatten it into a large categorical
space.

One semantic correction is needed: RLScape canonicalizes the position to zero
for `noop`, whereas `Driver` records the policy-produced action rather than
`info["executed_action"]`. The action must be canonicalized before both replay
recording and RSSM use; doing this only inside the environment wrapper is too
late. The agent/action-adapter boundary should zero `position` whenever
`mode == 0` for real actions, random actions, replay inputs, and imagined
actions.

### 3.5 Replay

- `embodied/core/replay.py:Replay` creates one sampleable item for every sliding
  sequence start and evicts item IDs through `fifo`.
- Capacity is measured in sequence-start items, not complete episodes.
- Chunks can contain multiple episode boundaries; `is_first` resets the RSSM
  inside a sampled sequence.
- `Replay._annotate_batch()` protects sequence starts and boundaries.
- `Replay.save()/load()` persist retained chunks, but there is no episode
  admission counter, reservoir RNG state, retained-slot manifest, or goal
  composition.

Replacing `_remove()` with random item eviction would not implement the required
reservoir: it would sample sequence starts rather than completed episodes and
would bias admission by episode length. Episode-level reservoir replay should
be a distinct implementation selected by configuration.

### 3.6 Evaluation and checkpoint lifecycle

- `embodied/run/eval_only.py` creates a separate agent process, loads a
  checkpoint while leaving adaptation optimizers fresh, clones parameters at
  the first episode trigger, accumulates adaptation within the episode, and
  drops the clone at the boundary.
- `embodied/run/train_eval.py` has similar temporary parameter logic but shares
  the live agent. Policy, adaptation, and probe calls still advance action,
  adaptation, and probe counters and therefore change later RNG streams.
- `embodied/jax/agent.py:Agent.clone_params()` copies the complete parameter and
  optimizer tree; `extract_policy_params()` extracts encoder/RSSM/decoder/actor
  parameters for acting.
- `Agent.save()/load()` persist parameters and counters. The current temporary
  evaluator proves episode-local parameter disposal but does not prove
  namespace or counter invariance around an integrated training evaluation.
- `embodied/run/cir.py` is a fixed DMC Cheetah action-gain runner. It has useful
  namespace hashing, model-only training, fixed-trajectory diagnostics, and
  resumable state, but it performs persistent adaptation and has no task
  scheduler or RLScape clone ownership.

RLScape evaluation requires both a disposable environment and a disposable
agent. Reusing the training agent and restoring only its parameter arrays is not
sufficient because its counters determine JAX RNG seeds.

### 3.7 Tests

The live focused tests are:

- `embodied/tests/test_replay.py` for sliding replay capacity, sampling, and
  chunk save/restore;
- `embodied/tests/test_driver.py` for reset/action/episode alignment;
- `embodied/tests/test_cir_runner.py` for the Cheetah schedule, namespace
  prefixes, adaptation timing, and actor-only optimizer selection;
- `embodied/tests/test_critic_ir_tools.py` for corruption/diagnostic helpers.

There are no CIR-side RLScape adapter, goal-conditioning, reservoir-episode,
phase scheduler, clone-pairing, or complete evaluation-isolation tests.

## 4. Proposed data contract

### 4.1 Policy/replay transition

Each recorded transition should contain:

```text
image                    uint8[H,W,3]
mode                     int32 scalar in [0,4)
position                 float32[2] in [-1,1]
reward                   float32 scalar
is_first                 bool
is_last                  bool
is_terminal              bool  # any true Gymnasium termination
is_physical_terminal     bool  # player death, not goal success
goal_complete            bool  # authoritative success for current goal
goal_id                   int32 scalar in [0,5)
```

`goal_id` must be constant within an episode and mapped through a versioned
catalogue artifact:

```text
0 -> lumbridge_v1/kill_goblin@1
1 -> lumbridge_v1/bury_bones@1
2 -> lumbridge_v1/chop_logs@1
3 -> lumbridge_v1/light_fire@1
4 -> lumbridge_v1/catch_fish@1
```

The mapping is not a policy observation supplied by RLScape; it is the
experiment’s explicit task context, derived from reset metadata and repeated by
the CIR adapter on every transition.

### 4.2 Metrics and audit-only fields

| RLScape field | Replay | Scalar metrics | Dataset/evaluation audit |
|---|---:|---:|---:|
| `task_id`, schema version | Encoded `goal_id`; optional scalar schema version | Per-goal grouping | Full string and task definition |
| `task_success` | `goal_complete` | Success rate | Full value |
| `task_progress` | No initial learning input | Mean/max progress | Full value |
| `termination_reason` | Encoded through `is_terminal`; subtype excluded | Death/success counts | Full string |
| `truncation_reason` | `is_last`/`is_terminal` already express learning target | Timeout count | Full string |
| lifetime/episode/action/tick identity | No | Optional monotonicity failure count | Full identity envelope |
| `frame_barrier_tag` | No | Validation failure count | Full tag/hash |
| snapshot/evaluation IDs | No | Evaluation grouping | Full IDs |
| factual `events` | No | Typed event counts | Full structured events |
| `privileged_state` | Never | Only predeclared diagnostics | Restricted audit record, never policy/replay |
| executed action | Canonical policy action is stored | Bounds/canonicalization checks | Full executed action comparison |

A dedicated audit callback should write JSONL from adapter-emitted structured
metadata. These values should not be forced through `log/*`, whose current
training loggers assume scalar arrays, and must never enter the encoder by
accident.

### 4.3 Goal-conditioned head contract

Use fixed one-hot conditioning first:

```text
x_t = concat(deter_t, flatten(stoch_t), one_hot(goal_id, 5))
reward       = rew(x_t)
continuation = con(x_t)
actor        = pol(x_t)
value        = val(x_t)
slow value   = slowval(x_t)
```

The image encoder, RSSM posterior, RSSM prior/transition, and decoder remain
goal-agnostic:

```text
tokens_t = encoder(image_t)
(deter_t, stoch_t) = rssm(tokens_<=t, action_<t)
image_hat_t = decoder(deter_t, stoch_t)
```

The continuation head follows ordinary Dreamer semantics and targets
`~is_terminal`. In RLScape, authoritative success and player death are both
terminal, while a time-limit truncation is not. `goal_complete` and
`is_physical_terminal` remain available for metrics and audits but are excluded
from the encoder, RSSM, decoder, and learned head targets.

The goal must also travel in the recurrent policy carry so that
`policy_latent()`, actor-only adaptation, MCPB rollouts, critic probes, and
recorded-action probes use the same goal after the observation call. The
smallest compatible change is a fixed-shape `goal_id` entry initialized in the
policy RSSM carry and refreshed from every observation. It is context attached
to the carry, not part of RSSM dynamics state.

One shared network is retained for every head. There are no per-goal actors,
critics, or reward predictors.

## 5. Explicit design decisions

### 5.1 Dedicated CIR-side RLScape adapter

Add `embodied/envs/rlscape.py:RLScape` rather than modifying RLScape or relying
on `FromGym`.

Responsibilities:

- lazily import and version-check `rl_scape`;
- construct `RLScapeEnv` with explicit package/runtime settings;
- translate Gymnasium spaces/results to Embodied spaces/results;
- own reset kind, named goal selection, goal switching, and fixed reset seeds;
- cache and repeat the reset `task_id` as `goal_id`;
- expose physical termination and goal completion separately;
- canonicalize and validate mixed actions;
- emit structured transition-audit records through a side channel;
- forward snapshot creation and evaluation-clone creation to the underlying
  RLScape environment;
- close processes and evaluation runtimes idempotently.

The adapter must never place privileged state in policy observations.

### 5.2 Fixed one-hot goal encoding

Start with a five-way one-hot. It is auditable, has no language semantics, adds
no embedding optimizer namespace, and is sufficient to test conditioning.
A learned embedding is a later configuration-compatible alternative, not part
of the first vertical slice.

### 5.3 Ordinary goal-conditioned continuation

Retain Dreamer's single `con/` head and its `~is_terminal` target. For RLScape,
only its input changes from `concat(deter, stoch)` to
`concat(deter, stoch, one_hot(goal_id))`, just like reward, actor, and value.
This is the smallest early-version extension and leaves non-goal configurations
unchanged.

### 5.4 Canonical factored mouse control

Use:

```text
mode: categorical(4)
position: bounded_normal(2)
```

with position zeroed for no-op before replay/RSSM use. This matches the public
RLScape action without solution-specific abstractions.

Keep RLScape’s fixed grid as an explicit fallback:

```text
action_adapter.kind: fixed_grid
action_adapter.grid: [48, 32]
```

If used, represent mode, column, and row as separate categorical keys because
the current categorical head requires a common class count within one shaped
space. Do not encode the `4 * 48 * 32` product as one categorical action.

Switch to the grid only if canonical control fails the milestone-1 learnability
criterion after basic optimization/horizon checks. High-level options,
grounded entities, and task-specific action masks remain out of scope.

### 5.5 Episode reservoir as a separate replay

Add `embodied/core/episode_replay.py:EpisodeReplay` (or an equally isolated
module) with `retention="fifo"|"reservoir"` rather than changing the meaning of
the existing sliding replay.

For each worker:

1. stage transitions from `is_first` through `is_last`;
2. assert one constant `goal_id`;
3. increment lifetime `episodes_seen` exactly once on completion;
4. admit the episode with `min(1, M / episodes_seen)`;
5. if full and admitted, replace one of `M` episode slots uniformly.

Sampling first chooses a retained episode uniformly, then a valid sequence
start uniformly. The implementation must define and test how sequences shorter
than the training length are joined or padded; it must not silently discard
successful short episodes. The preferred rule is to concatenate independently
sampled retained episode fragments with `is_first=True` at every boundary until
the requested length is filled. RSSM reset semantics then remain explicit.

Persistence needs an atomic manifest containing:

- schema version and retention mode;
- capacity in episodes;
- `episodes_seen`;
- reservoir RNG state;
- retained slot-to-episode IDs;
- episode lengths and goal IDs;
- current per-worker partial episodes;
- sampling counters by goal.

Only retained episode files belong in the active manifest. Save/restore must
reproduce future admission decisions, not merely reload a statistically similar
buffer.

The complete first continual/IR experiment uses the existing FIFO replay.
Episode reservoir is implemented only afterward. Its first comparison keeps
the paired frozen/single/MCPB evaluation protocol unchanged so replay retention
and actor IR are not introduced simultaneously.

### 5.6 Pure phase scheduler

Add a runner-independent `GoalPhaseScheduler` with serializable state:

```text
phase_index
goal_id
phase_budget_actions
phase_actions_elapsed
global_actions_elapsed
order
budgets
```

Budgets count authoritative RLScape actions/logical transitions, not reset
observations. Switching occurs at an exact fixed budget and produces an explicit
truncation/boundary before the next task reset. Every method/seed/order uses the
same predeclared budgets calibrated from the single-task curves. Thresholds may
be metrics, never switching rules.

The runner should support all learnable RLScape goals in an arbitrary ordered
landscape, with optional recurrence phases. The scheduler accepts arbitrary
orders so default, reversed, and additional orders share one code path.

### 5.7 Runner-owned evaluation clones

Add an RLScape-specific continual runner rather than teaching generic
`Driver` to issue arbitrary snapshot RPCs in the first slice.

At an evaluation point:

1. finish or explicitly truncate the current training episode at a safe
   transition barrier;
2. capture one source snapshot from the live training environment;
3. serialize/hash the complete persistent training agent state, replay state,
   scheduler state, and relevant counters;
4. for every learned goal and evaluation seed, create a fresh RLScape evaluation
   clone from that snapshot and apply the same named-goal/reset seed;
5. create a disposable evaluation agent loaded from the same in-memory
   checkpoint state;
6. run exactly one condition (`frozen`, `single`, or `mcpb`);
7. close the clone and discard the evaluation agent in `finally`;
8. verify the persistent state hash and training environment identity did not
   change.

All three conditions for a tuple
`(checkpoint, source_snapshot, goal, clone_seed, reset_seed)` use independently
owned clones derived from the same source snapshot and the same seeds. They do
not run serially in one mutable clone.

### 5.8 Temporary actor lifecycle

The evaluation agent starts from the persistent checkpoint but initializes a
fresh evaluation-local actor optimizer. At the first eligible posterior:

- clone only the actor parameters into the adaptable namespace;
- keep encoder/RSSM/decoder/reward/continuation/value/slow-value and all
  normalization statistics frozen;
- perform configured actor updates through frozen imagined quantities;
- act using the adapted actor with the frozen posterior model;
- optionally repeat at a fixed cadence;
- record actor KL to the episode’s initial actor, entropy, update count, and
  imagined-transition count;
- enforce a configurable KL guard/early stop;
- discard actor and optimizer at `is_last`.

The existing `freeze_critic=True` adaptation branch is the nearest code path,
but the new evaluator should make actor-only semantics explicit instead of
depending on the older `train_critic` flag combination.

## 6. Gap matrix

| Requirement | Existing support | Missing work / contradiction | Recommended change | Likely files | Tests |
|---|---|---|---|---|---|
| Import v0.1.1 | Local RLScape checkout is at the 0.1.1 release commit | Default shell Python reports no installed `rl-scape`; requirements omit it | Pin and runtime-check `rl-scape==0.1.1` | `requirements.txt`, `embodied/envs/rlscape.py` | Version mismatch and missing-package errors |
| Reset named goal | RLScape reset metadata supplies task ID/definition | No suite adapter or reset-option path | Dedicated adapter with explicit goal/reset kind/seed | `dreamerv3/main.py`, `embodied/envs/rlscape.py` | Named reset and lifetime/task semantics |
| RGB observation | RLScape provides RGB-only policy data | Default 384×252 does not fit the current four-level decoder: width/16 is 24, above its max 16 | Use a protocol-bound RLScape resize divisible by 16; planned default 240×160 | Config and adapter | Exact space, dtype, decoder-compatible shape |
| Goal to policy/replay | Driver/replay preserve ordinary scalar fields | Goal exists only in reset `info`; naïve field enters RSSM | Repeat `goal_id`; explicitly exclude from encoder/decoder; carry to heads | Adapter, `dreamerv3/agent.py` | Reset-to-action-to-replay propagation |
| Goal consistency | Replay stores arbitrary fields | No invariant | Assert constant ID while staging/closing episode and when sampling | Adapter/replay | Reject mid-episode goal change |
| Shared physical dynamics | Current RSSM is shared | Encoder would condition on goal if naïvely added | Goal bypass around encoder/RSSM/decoder | `dreamerv3/agent.py` | RSSM output invariant to changed goal on identical pixels/history |
| Goal-conditioned reward | Shared reward MLP exists | It has no goal input | Concatenate fixed goal one-hot at head input | `dreamerv3/agent.py` | Same latent/different goal produces separate targets |
| Goal-conditioned actor/value | Shared dict actor and value/slow value exist | No goal input; carry lacks goal | Condition shared heads and persist current ID in policy carry | Agent and JAX wrapper as needed | Policy/value receive correct goal in real and imagined calls |
| Termination semantics | `is_last`, `is_terminal`, and `con/` exist | `con/` lacks goal context | Condition the ordinary `con/` input on goal and retain its `~is_terminal` target | Agent and config | Success/death/truncation truth table |
| Mixed action | Dict actor, random policy, RSSM, replay support mixed factors | No-op coordinates in replay can differ from executed canonical action | Canonical factored adapter; zero position for no-op everywhere | Adapter, agent action utility, random policy path | Bounds, dtypes, no-op canonicalization, real/imagined parity |
| Diagnostic metadata | Driver carries `log/*` | Structured/string logs conflict with scalar logger and are stripped from replay | Dedicated JSONL audit sink; small numeric metrics separately | Adapter and RLScape runners | Transition identity monotonicity and audit serialization |
| Single-task runner | Generic training loop works | Cannot request stable RLScape goal or success metrics | RLScape base plus five named configs and per-goal evaluator | Config, adapter, runner metrics | Fake-env learning-loop integration |
| Joint multitask | RLScape can switch goal per reset | No goal-aware heads or per-goal sampling/metrics | Random goal per episode over the learnable catalogue | Adapter/config/metrics | Seeded goal sequence and per-goal aggregation |
| Sequential phases | Fixed Cheetah runner demonstrates serialized state | It hardcodes action gains and recreates DMC envs | Pure scheduler plus RLScape continual runner | New scheduler/runner, `main.py`, `run/__init__.py` | Exact switch actions and resume |
| FIFO baseline | Existing replay is FIFO over sliding starts | It does not report goal composition | Preserve semantics; add composition instrumentation outside retention logic | `replay.py` or wrapper metrics | No retention change; sampled-goal counts |
| Episode reservoir | None | Sliding-item random eviction is statistically wrong | Separate complete-episode replay and manifest | `embodied/core/episode_replay.py`, `main.py` | Admission frequencies, uniform replacement, composition |
| Reservoir persistence | Chunk replay persists current retained sequence window | No `episodes_seen`, RNG, slots, or accepted-episode manifest | Atomic versioned manifest with exact RNG restoration | Episode replay | Save/restore and future-decision equivalence |
| Paired snapshots | RLScape exposes snapshots and isolated clones | Generic runners cannot own/call them | Runner directly owns raw training env and disposable clones | RLScape continual/eval runner | Same source ID/seeds across conditions |
| Temporary actor | Existing evaluator clones params per episode | Current integrated eval can change counters; dedicated actor path is implicit | Separate eval agent; explicit actor-only optimizer/namespace | JAX agent, eval session/runner | Actor changes, every other namespace/counter unchanged |
| Single vs MCPB | `_adapt_rollout()` supports `start_batch` | Goal is unavailable in current carry/rollout | Carry/repeat goal with posterior particles | `dreamerv3/agent.py` | Batch 1 and N use identical goal |
| Actor conservatism | LR/update count configurable; KL primitives exist | No reference-policy KL metric/guard in adaptation | Log exact factored KL and stop/skip past configured limit | Agent/evaluator | KL zero initially, guard behavior |
| Critic/model serviceability | Cheetah probes provide a starting pattern | Probes are unconditioned and DMC-specific | Per-goal reward/continuation, open-loop, return/ranking/calibration probes | Agent probes and RLScape metrics | Correct goal routed through each probe |
| Checkpoint isolation | Agent params/counters and replay are saveable | Training environment/scheduler/snapshot lineage not checkpointed; eval-only snapshot export is not durable | Persist scheduler and replay; record live snapshot lineage; keep evaluations in runner while source lifetime exists | Runner/checkpoint state | Resume phase/replay; no eval contamination |
| Process cleanup | RLScape close is idempotent; Driver has `finally` close in env server | No CIR wrapper clone ownership | Context-managed adapter/evaluation session with nested `finally` | Adapter/runner | Close on normal exit and injected exceptions |
| Deferred features | None required | Old research plan includes critic-first/persistent/gates | Keep disabled and absent from first bridge configs | Config/docs | Config rejects unsupported first-slice combinations |

## 7. Ordered implementation plan

Each step should be an independently reviewable commit. Stop at a failed
scientific or integration gate rather than building later infrastructure on an
unvalidated assumption.

### Commit 1 — adapter-only smoke slice

- Pin `rl-scape==0.1.1`.
- Add `embodied/envs/rlscape.py` and the `rlscape` suite dispatch.
- Translate RGB, canonical actions, goal metadata, completion/death diagnostics,
  and transition audits.
- Use one random/no-op policy process, one named task, and a clean `finally`
  close.
- Add fake-RLScape unit tests so CI does not need Java processes.

**Stop/rollback:** do not touch Dreamer heads until reset/action/identity
alignment is proven. If the public package lacks a required field observed in
the v0.1.1 local source, demonstrate that mismatch before proposing any
RLScape change.

### Commit 2 — goal-conditioned Dreamer heads

- Add goal-exclusion lists and fixed one-hot head conditioning.
- Carry the current goal through real policy, latent policy, imagination,
  report, and probes.
- Goal-condition the existing reward, continuation, actor, value, and slow-value
  heads; retain the ordinary `~is_terminal` continuation target.
- Preserve the existing non-goal head routes and Cheetah checkpoint shapes.
- Canonicalize no-op actions before policy output/replay/RSSM use.

**Stop/rollback:** identical pixels/history with changed goal must leave
encoder/RSSM features unchanged while changing only conditioned head inputs.
Do not proceed if old DMC tests/checkpoint construction regress.

### Commit 3 — milestone-0 executable and audit artifact

- Add a small smoke runner/config that resets one named task, executes valid
  canonical actions, writes reset/step audit JSONL, checks identity alignment,
  and closes all owned processes.
- Add an opt-in live integration test marker.

**Stopping criterion:** package version, frame identity, barrier tag,
goal/task mapping, action bounds/canonicalization, and process cleanup all pass.

### Commit 4 — five single-task configurations and metrics

- Add one named configuration per released goal.
- Run ordinary Dreamer training with FIFO replay and no IR.
- Record success rate, learning AUC, actions/ticks to success, episode horizon,
  acting/training throughput, wall time, and resource use.
- Use multiple fixed training/evaluation seeds and paired evaluation reset
  seeds.

**Stopping criterion:** set no continual order or phase budget until all five
results exist. Include every reliably learnable goal in the initial landscape.
If canonical control does not learn any task, run the predeclared fixed-grid
fallback as an action-representation diagnostic before adding semantic actions.

### Commit 5 — joint learnable-goal upper bound

- Freeze the learnable goal catalogue and any exclusions in an experiment
  manifest.
- Randomize goal per episode with seeded, auditable sampling.
- Add per-goal losses and performance.

**Stopping criterion:** if joint training cannot solve a goal that was reliably
solved alone, diagnose conditioning/capacity/interference before sequential
experiments. Sequential forgetting is uninterpretable without this upper bound.

### Commit 6 — clone-paired frozen evaluation foundation

- Add runner-owned source snapshot and disposable clone lifecycle.
- Add disposable evaluation agents loaded from one in-memory persistent state.
- Verify paired seeds and full training-state invariance.
- Establish frozen evaluation distributions for every learned goal.

**Stopping criterion:** no actor adaptation yet. Snapshot pairing and namespace,
counter, optimizer, replay, scheduler, and training-environment invariants must
pass first.

### Commit 7 — pure scheduler and sequential FIFO baseline

- Implement/test serializable exact-action phase scheduling.
- Add dense paired evaluation near boundaries and at fixed checkpoints.
- Persist the learner through the complete ordered goal landscape; task reset
  the environment without lifetime reset unless the manifest explicitly says
  otherwise.
- Produce the phase-by-goal performance matrix and transfer/forgetting metrics.

**Stopping criterion:** require at least one learned-then-degraded goal and
non-trivial absolute performance. If nothing was learned, “zero forgetting” is
not a success.

### Commit 8 — temporary actor-only single-start and MCPB IR

- Make actor-only semantics explicit.
- Add a fresh per-episode optimizer, factored actor KL, guard, update counts, and
  imagined-transition accounting.
- At every sequential checkpoint, compare frozen, start batch 1, and configured
  MCPB in fresh clones for the current and all previously encountered goals.
- Plot within-episode recovery and paired differences.

**Stopping criterion:** a positive result requires repeatable paired improvement
over frozen evaluation, passing model/critic serviceability diagnostics,
unchanged persistent state, and no unacceptable current-task or worst-task
regression.

### Commit 9 — later episode-reservoir extension

- Implement complete-episode admission, sampling, metrics, and exact
  save/restore.
- Validate reservoir statistics without RLScape.
- Rerun the identical goal landscape, phase budgets, compute, and paired
  evaluation protocol with reservoir retention.

**Stopping criterion:** empirical inclusion probability and uniform replacement
tests pass; resume produces the same future admission/sample sequence. Interpret
the FIFO-to-reservoir difference separately from the within-checkpoint
frozen-to-IR difference.

### Later, conditional work

Only after that stopping criterion:

- persistent actor CIR;
- more posterior/history start states if current-posterior coverage is limiting;
- critic adaptation with a non-circular validated target;
- learned compute gates;
- task-free context inference;
- natural-language goal semantics;
- representation anchoring if measured functional drift requires it.

## 8. Test plan

### Adapter and goal propagation

- Fake `RLScapeEnv.reset()` metadata becomes the expected stable `goal_id`.
- The goal appears on every transition and survives Driver → replay → sample.
- A changed goal inside an episode raises immediately.
- Task reset preserves lifetime; explicit lifetime reset requires a seed.
- No privileged state is present in `agent.obs_space` or replay.
- Version mismatch and an unknown task ID fail with actionable errors.

### Action representation

- `mode` dtype/range and `position` shape/dtype/bounds match RLScape.
- No-op position becomes exact zero before environment, replay, and RSSM.
- Non-noop positions round-trip through normalization without mutation.
- Random policy and imagined policy obey the same canonicalization.
- Optional fixed-grid cells map to their declared normalized centers.
- Mixed actor loss includes both factors without a flattened categorical.

### Conditioning and termination

- Encoder and RSSM features are invariant to goal ID on identical histories.
- Reward, continuation, actor, online value, and slow value all receive the
  goal.
- A success transition targets continuation false.
- Death targets continuation false.
- Time-limit truncation targets continuation true.
- `goal_complete` and `is_physical_terminal` remain diagnostic fields excluded
  from learned state and head targets.
- MCPB repeats one goal across all particles and imagined time steps.

### Replay

- FIFO baseline behavior remains identical to existing tests.
- Episode admission happens only at `is_last`.
- For capacity `M` and episode `n`, Monte Carlo inclusion frequencies match
  `min(1, M/n)` within a predeclared statistical tolerance.
- Replacement is uniform over slots.
- Goal and transition counts match retained episodes.
- Sampled minibatch goal composition is logged correctly.
- Short episodes are represented without invented cross-episode dynamics; every
  joined boundary has `is_first=True`.
- Save/restore preserves slots, `episodes_seen`, RNG, partial episodes,
  composition, and the next admission/sample outcomes.

### Scheduler

- Fixed action budgets switch at exact counts independent of episode success.
- Boundary truncation closes the replay episode before task change.
- Every configured goal block is represented as a distinct phase, including
  repeated goals used to measure recurrence.
- Resume restores phase and remaining budget exactly.
- Reversed orders and multiple seeds do not share mutable scheduler state.

### Snapshot-paired evaluation

- All conditions record the same source snapshot ID, goal, clone seed, and reset
  seed.
- Every condition owns a distinct evaluation ID/runtime.
- Clone closure occurs on success, policy exception, and adaptation exception.
- Evaluation data never enters training replay.
- Training environment lifetime/episode/action identity is unchanged.

### IR isolation

- At update 0, adapted/reference actor KL is zero.
- After an update, only the copied actor and its local optimizer may change.
- Encoder, RSSM, decoder, reward, continuation, online critic, slow critic,
  normalizers, persistent actor, and every persistent
  optimizer hash remain unchanged.
- Persistent action/adaptation/probe counters and RNG streams remain unchanged
  because evaluation uses a disposable agent.
- Actor state and optimizer are fresh again for the next episode/condition.
- KL guard, cadence, update count, horizon, start batch, and imagined-transition
  count are exact.

### End-to-end live smoke

An opt-in test against the external runtime:

- imports exactly 0.1.1;
- launches one environment;
- resets a named goal;
- checks task/frame identity;
- executes one action of every mode with legal coordinates;
- captures/restores a snapshot;
- creates and closes one evaluation clone;
- closes the source;
- verifies no owned process/runtime directory remains.

## 9. Proposed configuration surface

Names and initial defaults:

```yaml
env:
  rlscape:
    package_version: "0.1.1"
    launch: true
    server_dir: ""
    server_runtime_dir: ""
    client_dir: ""
    java_home: ""
    resize: [240, 160]          # width, height; both divisible by 16
    episode_length: 200         # ample slack for goals reachable in <10 actions
    reward_mode: sparse_success
    server_barrier: true
    sync_to_tick: true
    goal: kill_goblin
    goals: []                   # non-empty for joint/continual modes
    goal_switch: false
    reset_kind: task
    reset_seed: 0
    lifetime_seed: 0

goal:
  enabled: true
  key: goal_id
  count: 5
  encoding: onehot
  embedding_dim: 0              # only used when encoding=learned
  exclude_keys: [goal_complete, is_physical_terminal]

action_adapter:
  kind: canonical_factored
  canonicalize_noop_position: true
  grid: [48, 32]                # inactive fallback

replay:
  mode: fifo                    # fifo | episode_reservoir
  size: 5000000                 # M2 will not fill this FIFO capacity
  reservoir_episodes: 1000
  reservoir_seed: 0
  sample_unit: episode_then_sequence

continual:
  enabled: false
  order: []
  phase_action_budgets: []
  boundary_eval_offsets: [0]
  reset_kind: task

evaluation:
  goals: learned
  clone_seeds: [0, 1, 2, 3, 4]
  reset_seeds: [0, 1, 2, 3, 4]
  episodes_per_pair: 1
  snapshot_at_phase_boundaries: true
  conditions: [frozen]

actor_ir:
  enabled: false
  temporary: true
  actor_only: true
  cadence_actions: 0            # 0 means first posterior only
  updates_per_trigger: 1
  imagination_horizon: 15
  start_mode: single            # single | mcpb
  mcpb_batch: 64
  learning_rate: 1.0e-5
  entropy_scale: 3.0e-4
  max_kl: 0.02
  discard_at_episode_end: true
```

The `episode_length`, replay capacities, phase budgets, and ordered goal list
are experiment constants, not universal truths. Their final values must be
recorded after milestone-1 throughput and learning curves. They must then remain
fixed across the causal comparisons.

Named configuration target (the first seven entries are now implemented):

```text
rlscape
rlscape_smoke
rlscape_kill_goblin
rlscape_bury_bones
rlscape_chop_logs
rlscape_light_fire
rlscape_catch_fish
rlscape_joint_landscape
rlscape_sequential_fifo
rlscape_sequential_reservoir
rlscape_eval_ir_suite
```

## 10. Intended command matrix

The milestone-0 and milestone-1 commands now use the implemented configuration
surface. Later commands remain design targets until their runners and immutable
manifests exist.

### Milestone 0 — integration smoke

```bash
python dreamerv3/main.py \
  --configs rlscape rlscape_smoke \
  --seed 0 \
  --logdir logs/rlscape/m0_smoke_seed0
```

### Milestone 1 — all five single tasks

```bash
python dreamerv3/main.py \
  --configs rlscape rlscape_kill_goblin \
  --seed 0 \
  --logdir logs/rlscape/m1_kill_goblin_seed0

python dreamerv3/main.py \
  --configs rlscape rlscape_bury_bones \
  --seed 0 \
  --logdir logs/rlscape/m1_bury_bones_seed0

python dreamerv3/main.py \
  --configs rlscape rlscape_chop_logs \
  --seed 0 \
  --logdir logs/rlscape/m1_chop_logs_seed0

python dreamerv3/main.py \
  --configs rlscape rlscape_light_fire \
  --seed 0 \
  --logdir logs/rlscape/m1_light_fire_seed0

python dreamerv3/main.py \
  --configs rlscape rlscape_catch_fish \
  --seed 0 \
  --logdir logs/rlscape/m1_catch_fish_seed0
```

Repeat with the predeclared seed set. If required, the fixed-grid diagnostic
adds:

```bash
--action_adapter.kind fixed_grid
```

without changing task semantics.

### Milestone 2 — joint all-goal upper bound

```bash
python scripts/rlscape_m2_train.py \
  --seed 0 \
  --steps 2000000 \
  --logdir logs/rlscape/m2_multitask_2m_seed0

python scripts/rlscape_m2_eval.py \
  --checkpoint logs/rlscape/m2_multitask_2m_seed0 \
  --log-root logs/rlscape/m2_multitask_2m_seed0_eval20 \
  --seed 1000 \
  --episodes 20
```

### Milestone 3 — FIFO task landscape with paired actor-IR evaluation

```bash
python dreamerv3/main.py \
  --configs rlscape rlscape_sequential_fifo rlscape_eval_ir_suite \
  --seed 0 \
  --logdir logs/rlscape/m3_fifo_ir_suite_seed0
```

The sequential config contains the frozen ordered list of all learnable goals
and their action budgets. The evaluation suite compares frozen, single-start
IR, and MCPB in disposable paired clones at each checkpoint. Additional orders
receive separate immutable config names or manifest files.

The suite expands internally to:

```text
frozen: actor_ir.enabled=false
single: actor_ir.enabled=true, actor_ir.start_mode=single
mcpb:   actor_ir.enabled=true, actor_ir.start_mode=mcpb
```

Running them as one paired suite is intentional: an operator should not need to
coordinate snapshot IDs or seeds across independent commands.

### Milestone 4 — later reservoir extension

```bash
python dreamerv3/main.py \
  --configs rlscape rlscape_sequential_reservoir rlscape_eval_ir_suite \
  --seed 0 \
  --logdir logs/rlscape/m4_reservoir_ir_suite_seed0
```

The goal stream, phase budgets, optimization ratio, evaluation conditions,
snapshot/reset seeds, and all other compute settings match milestone 3. This is
the first replay-retention comparison and is not required for the first
iteration.

### Milestone 5 — conditional CIR extensions

No command is specified yet. Persistent actor CIR, learned gates, or alternative
rehearsal starts require a separate design after the FIFO paired-evaluation
result.

## 11. Required analysis outputs

Milestone 1:

- per-goal multi-seed success curves;
- learning AUC and actions/ticks to first and stable success;
- throughput and episode-length distributions;
- landscape inclusion, order, and phase-budget rationale.

Milestones 2–3:

- phase-by-goal success matrix;
- forward transfer;
- average and maximum forgetting;
- per-goal recurrence gain and actions to threshold where return phases exist;
- worst-goal performance;
- replay retained/sampled composition;
- model reward/continuation error by goal;
- open-loop model error and representation drift on fixed raw-history probes;
- critic ranking/calibration and predicted-vs-realized return by goal;
- actor entropy/KL;
- real/imagined transitions, wall time, throughput, RAM/device memory.

Milestone 3 paired evaluation:

- paired episode-level frozen-minus-IR differences;
- within-episode recovery curves;
- single-start versus MCPB effect;
- actor KL/entropy/update and imagined-transition counts;
- model/critic serviceability strata;
- current-task, previous-task, and worst-task effects;
- full persistent-state invariant report.

Milestone 4 additionally reports the FIFO-to-reservoir change in every learning,
forgetting, serviceability, and actor-IR measure.

Every result table must include absolute success/performance. Forgetting metrics
without demonstrated learning are not interpretable.

## 12. Consequential choices intentionally deferred

No user choice is needed before the first adapter and single-task slice. The
following decisions become consequential only after evidence exists:

- which goals meet the inclusion criterion and their order in the landscape;
- the fixed phase budgets and evaluation density;
- whether canonical control needs the fixed-grid fallback;
- whether milestone-4 uses pure reservoir or a later short-term/long-term
  mixture;
- whether a positive evaluation-local effect justifies persistent actor CIR;
- whether model/critic failure instead requires representation retention,
  broader imagination starts, or a redesigned critic target.

Those choices should be made from recorded milestone results, not hardcoded into
the initial integration.
