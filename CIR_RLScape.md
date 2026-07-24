We are beginning a new CIR experiment programme integrating the CIR repository with RLScape v0.1.1.

Repository in scope:
- CIR: /home/staff/steven/crl_ir

Environment dependency:
- Python package: rl-scape==0.1.1
- RLScape repository, if local inspection is useful: /home/staff/steven/RLScape

This turn is for repository inspection, architectural design, and a concrete implementation plan. Do not begin broad implementation yet. You may add or update one repository planning document after inspecting existing documentation, but do not duplicate an existing plan.

First inspect:
- repository instructions such as AGENTS.md;
- git status, branch, recent history, and all uncommitted changes;
- README.md, CIR_RESEARCH_PLAN.md, and existing experiment documentation;
- dreamerv3/agent.py;
- dreamerv3/configs.yaml;
- dreamerv3/main.py;
- embodied/run/cir.py;
- embodied/run/eval_only.py;
- embodied/core/replay.py;
- environment adapters and wrappers;
- relevant tests under embodied/tests or elsewhere.

Treat the live working tree as authoritative. Preserve all existing user changes.

# Scientific objective

The immediate objective is not the full persistent actor–critic CIR proposal.

We are building the simplest defensible bridge from working imagined rehearsal to continual goal-conditioned behaviour:

1. integrate RLScape correctly;
2. establish single-task learnability;
3. establish a joint multitask upper bound;
4. establish naive sequential learning and forgetting;
5. add fixed reservoir replay as the continual world-model substrate;
6. test temporary evaluation-time actor-only imagined rehearsal;
7. consider persistent actor CIR only if evaluation-time IR produces a positive result.

The central experiment asks:

Can a shared retained world model support temporary actor-only imagined rehearsal that restores partially forgotten goal-conditioned behaviour during evaluations of current and previous tasks, without additional real training experience?

Do not implement critic adaptation, learned compute gates, task-free context inference, semantic language rewards, or persistent CIR in the first vertical slice.

# RLScape v0.1.1 contract

RLScape currently provides:

- a controlled Lumbridge arena;
- five versioned atomic goals:
  - kill_goblin
  - bury_bones
  - chop_logs
  - light_fire
  - catch_fish
- RGB-only policy observations;
- goal ID and structured task definition in reset metadata, not observation_space;
- a canonical mixed action:
  - mode: Discrete(4), representing noop, move, left click, and right click;
  - position: Box(-1, 1, shape=(2,)), representing normalized absolute image coordinates;
- sparse +1 task-success reward or reward-free mode;
- task success, progress, factual events, death, and termination diagnostics;
- task resets and seeded lifetime resets;
- persistent or explicitly switched goal streams;
- content-addressed snapshots and restoration;
- isolated evaluation clones;
- protocol-v2 RGB frames bound to authoritative reset/action/restore transition identities;
- public package and external runtime installation.

Do not redesign RLScape unless a demonstrable integration blocker requires it. Prefer a CIR-side wrapper for goal observations and agent compatibility.

The environment is engineered but does not yet have learning evidence. None of the five goals has a multi-seed Dreamer learning curve.

# Task selection

Do not hardcode an A/B/C trio yet.

The first learning stage must test all five candidate goals separately. Select A, B, and C only after measuring:

- reliable single-task learning;
- comparable but controllable difficulty;
- shared reusable visual/dynamical structure;
- task-specific behavioural differences;
- potential for interference;
- reasonable episode horizon and throughput.

# Goal-conditioned model design

For version 0, the physical environment dynamics are shared across goals:

    p(s_{t+1} | s_t, a_t, g) = p(s_{t+1} | s_t, a_t)

The intended architecture is:

    z_t = q_phi(z_t | o_<=t, a_<t)
    r_hat_t = r_phi(z_t, g)
    pi_theta(a_t | z_t, g)
    V_psi(z_t, g)

Use one shared conditioned network for each goal-dependent head, not separate actor/critic/reward networks per task.

The initial intended conditioning boundary is:

Goal-agnostic:
- image encoder;
- RSSM posterior;
- RSSM transition/prior;
- decoder.

Goal-conditioned:
- reward predictor;
- actor;
- online critic;
- slow critic.

Continuation needs explicit design attention. RLScape termination includes both task success and physical failure. Prefer separating:

- physical continuation/death: c_env(z_t);
- goal completion: c_goal(z_t, g).

If the current Dreamer implementation only supports one continuation head, inspect how termination targets flow through replay and recommend one of:

1. split physical continuation and goal-completion prediction; or
2. temporarily goal-condition the combined continuation head.

Do not leave a goal-dependent termination target attached to a goal-agnostic predictor without acknowledging the contradiction.

Begin with a one-hot or learned task-ID embedding. Natural-language goals are not part of this experiment.

# Required data-flow audit

Determine exactly how goal g must travel through:

- reset metadata;
- environment wrapper observation/spec;
- driver and episode recording;
- replay storage;
- replay sampling;
- sequence batches;
- world-model reward/completion losses;
- policy calls during real action selection;
- critic and slow-critic calls;
- imagined rollouts;
- checkpoint save/restore;
- evaluation records and metrics.

The stored episode should contain at least:

    (observation, action, reward, continuation/terminal targets,
     is_first, is_last, is_terminal, goal/task ID)

Goal identity must remain constant and auditable within an episode.

Check whether environment info fields such as task_success, termination_reason, task_id, snapshot ID, transition identity, and barrier tag survive the current Embodied wrappers. Recommend which belong in replay, metrics only, or dataset audit records.

# Mixed-action audit

RLScape uses a Dict action containing discrete mode plus continuous position.

Inspect whether the current Dreamer/Embodied actor, distributions, replay, policy output, random policy, and environment wrappers support this mixed action natively.

If they do not, compare the smallest principled options:

- a factored actor with categorical mode and continuous coordinates;
- an explicit adapter using RLScape’s fixed grid action space;
- another minimal representation that does not encode task solutions.

Do not silently flatten the action into a huge categorical variable. Do not introduce grounded entities or high-level options in the initial implementation unless raw/fixed-grid control cannot establish single-task learning.

# Replay design

Stage 1 should preserve the existing FIFO baseline.

Stage 2 adds episode-level reservoir replay as fixed retention infrastructure. For capacity M and the nth completed episode:

    P(accept episode n) = min(1, M/n)

If accepted when full, replace one stored episode uniformly.

Requirements:

- admission occurs at complete-episode granularity;
- the observation stream index n persists across the lifetime;
- episode goal labels are retained;
- sampling is random over stored episodes/sequences;
- fixed task horizons should initially prevent severe episode-length bias;
- log retained episodes and transitions per goal;
- log sampled minibatch composition per goal;
- make FIFO versus reservoir a configuration choice;
- do not change replay semantics and IR behaviour in the same causal comparison.

Check whether the current replay architecture can implement statistically correct reservoir admission without breaking online insertion, persistence, checkpointing, or sequence sampling.

# Continual task protocol

Eventually the continual stream will be:

    A^T_A -> B^T_B -> C^T_C -> A^T_A-return

Tasks occur in blocks, not randomly every training episode.

Phase budgets must be fixed across methods and calibrated from multi-seed single-task curves. Do not allow each algorithm to switch tasks whenever it independently reaches a threshold.

The runner must eventually support:

- explicit task order;
- fixed interaction budget per phase;
- dense evaluation near phase boundaries;
- evaluation of every learned goal at each checkpoint;
- multiple task orders and seeds;
- persistent learner parameters across phases;
- declared task-reset versus lifetime-reset behaviour.

Plan this runner, but implement it only after the minimal single-task integration works.

Or, alternative to this, copy the experiment design of the continual-dreamer-paper. But this can be discussed when we get to this point of implimentation.

# Evaluation-time actor IR

The first IR comparison must be temporary and evaluation-local.

For the same persistent training checkpoint, goal, and cloned RLScape snapshot, compare:

1. frozen evaluation without adaptation;
2. temporary actor-only IR using one current-posterior start;
3. temporary actor-only IR using MCPB starts.

During evaluation-time IR:

May change:
- a copied actor;
- its temporary optimizer state within that evaluation episode.

Must remain frozen:
- persistent checkpoint parameters;
- encoder and RSSM;
- decoder;
- reward and continuation/completion heads;
- critic and slow critic;
- normalization state;
- training replay;
- training environment/lifetime;
- persistent optimizer states.

At episode end, discard the temporary actor and optimizer. The next evaluation must begin from the same persistent checkpoint.

Use isolated RLScape evaluation clones and paired snapshot seeds. Add assertions or namespace hashes proving that evaluation adaptation cannot contaminate training state.

Do not add critic-first adaptation. The previous Cheetah pilot showed circular critic bootstrap collapse. Actor-only updates must initially be conservative, configurable, and observable through actor KL, update count, and imagined-transition count.

# Experiment ladder

Design the repository changes around these runnable milestones:

Milestone 0 — integration smoke
- install/import RLScape v0.1.1;
- reset one named task;
- pass goal metadata to the policy;
- execute valid actions;
- preserve transition identity in diagnostics;
- close all processes cleanly.

Milestone 1 — single-task learning
- one configuration per released goal;
- multiple seeds;
- success rate, learning AUC, ticks/actions to success;
- determine whether canonical mouse control is learnable;
- no CIR and no reservoir replay.

Milestone 2 — joint multitask upper bound
- random goal per episode;
- shared goal-conditioned model and behaviour heads;
- verifies that conditioning and shared capacity can solve the selected tasks.

Milestone 3 — naive sequential baseline
- fixed A->B->C->A blocks;
- FIFO replay;
- phase-by-goal evaluation matrix;
- quantify forgetting, transfer, and recurrence.

Milestone 4 — retained-model baseline
- identical stream and compute;
- episode-level reservoir replay;
- no IR;
- determine whether the model and critic remain serviceable and whether actor headroom remains.

Milestone 5 — evaluation-time actor IR
- frozen versus single-start IR versus MCPB;
- paired evaluation clones and seeds;
- temporary actor only;
- no persistent parameter changes.

Persistent actor CIR is a later decision, not part of this implementation plan.

# Metrics and controls

Plan support for:

- single-task success and learning AUC;
- phase-by-goal performance matrix;
- forward transfer;
- average and maximum forgetting;
- A recurrence gain;
- steps to recover a threshold;
- worst-task performance;
- within-episode IR recovery curves;
- world-model loss/error by goal;
- predicted versus realized return;
- critic ranking/calibration on retained goals;
- actor KL and entropy;
- representation drift on fixed raw-history probes;
- real and imagined transition counts;
- replay composition by goal;
- wall time, acting throughput, memory, and device memory;
- performance with IR disabled.

Required controls eventually include:

- ordinary Dreamer;
- compute-matched Dreamer;
- single-task learners;
- joint random-goal multitask learner;
- naive sequential FIFO;
- sequential reservoir without IR;
- evaluation-time IR with single-posterior and MCPB starts.

# Historical lessons that must shape the implementation

The previous Cheetah action-gain pilot was inconclusive and contained serious confounds:

- critic-first rehearsal collapsed through circular self-bootstrap;
- persistent actor updates accumulated drift without beating model-only repair;
- current-posterior starts lost access to competent states after behavioural collapse;
- frozen actor/critic heads still changed functionally when encoder/RSSM representations moved;
- mixed unlabeled regimes created incompatible predictive targets;
- evaluations used too few episodes and unpaired environment seeds.

Therefore:

- keep the first IR stage actor-only and temporary;
- use explicit task labels;
- retain the shared world model through established replay;
- measure end-to-end functional drift, not only parameter hashes;
- use paired evaluation snapshots/seeds;
- never interpret IR until model and critic serviceability are measured;
- do not change world-model retention and behaviour adaptation in the same ablation.

# Deliverables for this turn

After inspecting the repository, provide:

1. Current architecture map
   - exact files, classes, and functions involved;
   - what already supports this design;
   - where the live repository differs from the old implementation map.

2. Gap matrix
   For every requirement above, list:
   - existing support;
   - missing work;
   - recommended change;
   - exact likely files;
   - tests needed.

3. Explicit design decisions
   Recommend, with reasoning:
   - CIR-side goal wrapper and goal encoding;
   - reward and continuation/completion conditioning;
   - mixed discrete/continuous action representation;
   - task metadata carried in replay;
   - reservoir location and persistence semantics;
   - phase scheduler design;
   - evaluation-clone ownership;
   - checkpoint and temporary-actor lifecycle.

4. Ordered implementation plan
   - smallest vertical slice first;
   - independently testable commits or milestones;
   - dependencies between steps;
   - stopping criteria and rollback points;
   - distinguish essential first-experiment changes from later CIR work.

5. Test plan
   Include unit and integration tests for:
   - goal propagation;
   - action conversion and bounds;
   - episode goal consistency;
   - correct reward/completion conditioning;
   - reservoir statistical admission;
   - replay save/restore;
   - phase switching;
   - snapshot-paired evaluation;
   - parameter/optimizer isolation during IR;
   - no training-state contamination;
   - namespace invariants.

6. Proposed configuration surface
   Show concrete configuration names and defaults for:
   - RLScape package/runtime;
   - goal/task selection;
   - action adapter;
   - goal embedding;
   - reset kind;
   - replay mode/capacity;
   - phase order and budgets;
   - evaluation goals/snapshots;
   - actor-IR cadence, updates, horizon, MCPB batch, and learning rate.

7. Experiment command matrix
   Give the intended commands/config combinations for milestones 0–5, even if later commands cannot run until implementation is complete.

8. Repository plan document
   Update the most appropriate existing research-plan document or create one clearly named RLScape/CIR bridge plan. Record assumptions, decisions, milestones, tests, and claim boundaries. Do not claim implementation that has not occurred.

Do not solve uncertainties by guessing about repository internals. Inspect the relevant path and cite it in the plan. Ask me only about choices that remain genuinely consequential after inspection.
