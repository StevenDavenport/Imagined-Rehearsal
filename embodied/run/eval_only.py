from collections import defaultdict
from functools import partial as bind
import json
import os
import pathlib
import pickle
import time

import elements
import embodied
import numpy as np

from .ir_gate import GATE_KINDS
from .ir_gate import cheap_decision
from .ir_gate import consequence_decision
from .ir_gate import virtual_update_decision


def _save_adapted_checkpoint(
    agent, params, destination, manifest, payload=None):
  """Atomically publish an evaluation-local parameter clone."""
  destination = pathlib.Path(destination).expanduser().resolve()
  if destination.exists():
    raise FileExistsError(destination)
  destination.parent.mkdir(parents=True, exist_ok=True)
  temporary = destination.with_name(
      f'.{destination.name}.tmp-{os.getpid()}')
  if temporary.exists():
    raise FileExistsError(temporary)
  temporary.mkdir()
  payload = payload if payload is not None else agent.export_params(params)
  with (temporary / 'agent.pkl').open('wb') as handle:
    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
  (temporary / 'persistent_ir_manifest.json').write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + '\n')
  (temporary / 'done').write_bytes(b'')
  temporary.replace(destination)


def eval_only(make_agent, make_env, make_logger, args):
  assert args.from_checkpoint

  agent = make_agent()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  trace_cfg = getattr(agent.config.eval_adapt, 'trace', None)
  trace_enabled = bool(getattr(trace_cfg, 'enabled', False))
  trace_file = None
  trace_episode = 0
  trace_episode_step = 0
  trace_return = 0.0
  if trace_enabled:
    trace_name = str(getattr(trace_cfg, 'filename', 'trace.jsonl'))
    trace_path = logdir / trace_name
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_file = trace_path.open('w')
    print('Trace file:', trace_path)
  step = logger.step
  usage = elements.Usage(**args.usage)
  agg = elements.Agg()
  epstats = elements.Agg()
  episodes = defaultdict(elements.Agg)
  should_log = elements.when.Clock(args.log_every)
  policy_fps = elements.FPS()
  recording = True

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    nonlocal trace_episode, trace_episode_step, trace_return
    if not recording:
      return
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      isimage = (value.dtype == np.uint8) and (value.ndim == 3)
      if isimage and worker == 0:
        episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

    if trace_file and worker == 0:
      if bool(np.asarray(tran['is_first']).item()):
        trace_episode_step = 0
        trace_return = 0.0
      reward = float(np.asarray(tran['reward']).item())
      trace_return += reward
      row = {
          'step': int(step),
          'episode_index': int(trace_episode),
          'episode_step': int(trace_episode_step),
          'reward': reward,
          'return_so_far': float(trace_return),
          'is_first': bool(np.asarray(tran['is_first']).item()),
          'is_last': bool(np.asarray(tran['is_last']).item()),
          'adapt_trigger': float(np.asarray(
              tran.get('log/eval_adapt/trigger', 0.0)).item()),
          'adapt_steps': float(np.asarray(
              tran.get('log/eval_adapt/steps', 0.0)).item()),
          'adapt_actor_steps': float(np.asarray(
              tran.get('log/eval_adapt/actor_steps', 0.0)).item()),
          'adapt_critic_steps': float(np.asarray(
              tran.get('log/eval_adapt/critic_steps', 0.0)).item()),
          'adapt_warmup': float(np.asarray(
              tran.get('log/eval_adapt/warmup', 0.0)).item()),
      }
      for key in (
          'log/eval_adapt/adapt_opt/loss',
          'log/eval_adapt/adapt_opt/grad_norm',
          'log/eval_adapt/adapt_opt/update_rms',
          'log/eval_adapt/adapt_actor_opt/loss',
          'log/eval_adapt/adapt_actor_opt/grad_norm',
          'log/eval_adapt/adapt_actor_opt/update_rms',
          'log/eval_adapt/adapt_critic_opt/loss',
          'log/eval_adapt/adapt_critic_opt/grad_norm',
          'log/eval_adapt/adapt_critic_opt/update_rms',
      ):
        if key in tran:
          row[key.replace('log/eval_adapt/', '').replace('/', '_')] = float(
              np.asarray(tran[key]).item())
      for key, value in tran.items():
        if key.startswith((
            'log/eval_adapt/gate/', 'log/eval_adapt/compute/')):
          row[key.replace('log/eval_adapt/', '').replace('/', '_')] = float(
              np.asarray(value).item())
      trace_file.write(json.dumps(row) + '\n')
      trace_episode_step += 1
      if row['is_last']:
        trace_episode += 1
        trace_episode_step = 0
        trace_return = 0.0

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=(not args.debug))
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(logfn)

  adapt_cfg = agent.config.eval_adapt
  persistence = str(getattr(adapt_cfg, 'persistence', 'episode')).lower()
  if persistence not in ('episode', 'run'):
    raise ValueError(
        'eval_adapt.persistence must be episode or run, got '
        f'{persistence!r}')
  output_checkpoint = str(getattr(adapt_cfg, 'output_checkpoint', ''))
  resume_adapt_state = bool(getattr(
      adapt_cfg, 'resume_adapt_state', False))
  if persistence == 'episode' and (output_checkpoint or resume_adapt_state):
    raise ValueError(
        'Persistent checkpoint settings require eval_adapt.persistence=run')
  adapt_objective = str(getattr(adapt_cfg, 'objective', 'standard'))
  if adapt_objective not in (
      'standard', 'reward_only', 'bounded', 'mixed_bounded', 'distill'):
    raise ValueError(
        f'Unknown eval_adapt.objective: {adapt_objective!r}')
  adapt_params = None
  adapt_policy_params = None
  reference_actor_params = None
  adapt_every_k = int(getattr(adapt_cfg, 'every_k', 0))
  gate_cfg = getattr(adapt_cfg, 'gate', None)
  gate_enabled = bool(getattr(gate_cfg, 'enabled', False))
  gate_audit = bool(getattr(gate_cfg, 'audit', False))
  gate_kind = str(getattr(gate_cfg, 'kind', 'none')).lower()
  if gate_kind not in GATE_KINDS:
    raise ValueError(f'Unknown eval_adapt.gate.kind: {gate_kind!r}')
  if gate_enabled and not bool(adapt_cfg.enabled):
    raise ValueError('eval_adapt.gate.enabled requires eval_adapt.enabled')
  if gate_enabled and gate_kind == 'none':
    raise ValueError('Enabled IR gate requires a non-none gate kind')
  if gate_enabled and adapt_objective == 'distill':
    raise ValueError('Stage 5A gates do not support distillation')
  virtual_gate = bool(gate_enabled and gate_kind == 'virtual_update')
  virtual_score = str(getattr(
      gate_cfg, 'virtual_score', 'reward_return')).lower()
  virtual_score_keys = {
      'reward_return': 'reward_return_mean',
      'success_rate': 'success_rate',
  }
  if virtual_score not in virtual_score_keys:
    raise ValueError(
        f'Unknown eval_adapt.gate.virtual_score: {virtual_score!r}')
  virtual_min_improvement = float(getattr(
      gate_cfg, 'virtual_min_improvement', 0.0))
  virtual_max_baseline = float(getattr(
      gate_cfg, 'virtual_max_baseline_score', float('inf')))
  virtual_dormant_every = int(getattr(
      gate_cfg, 'virtual_dormant_every', 25))
  virtual_active_every = int(getattr(
      gate_cfg, 'virtual_active_every', 1))
  virtual_deactivate_after = int(getattr(
      gate_cfg, 'virtual_deactivate_after', 3))
  if virtual_gate and bool(adapt_cfg.train_critic):
    raise ValueError('virtual_update currently requires a frozen critic')
  if virtual_gate and min(
      virtual_dormant_every, virtual_active_every,
      virtual_deactivate_after) < 1:
    raise ValueError('Virtual-update cadence and hysteresis must be positive')
  steps_since_adapt = 0
  steps_since_probe = 0
  virtual_active = False
  virtual_reject_streak = 0
  paired_rng = bool(getattr(adapt_cfg, 'paired_rng', False))
  paired_rng_seed = int(getattr(adapt_cfg, 'paired_rng_seed', 0))
  paired_rng_stride = int(getattr(adapt_cfg, 'paired_rng_stride', 1000))
  paired_episode = -1
  paired_step = 0
  if paired_rng_stride < 2:
    raise ValueError('eval_adapt.paired_rng_stride must be at least 2')
  if paired_rng and adapt_objective == 'distill':
    raise ValueError('Paired RNG is not implemented for distillation')
  if virtual_gate and not paired_rng:
    raise ValueError(
        'virtual_update requires eval_adapt.paired_rng=True so before and '
        'after scores use common random numbers')
  requested_mode = str(getattr(
      args, 'eval_policy_mode', 'deterministic')).lower()
  policy_modes = {'deterministic': 'eval', 'sampled': 'train'}
  if requested_mode not in policy_modes:
    raise ValueError(
        'run.eval_policy_mode must be deterministic or sampled, got '
        f'{requested_mode!r}')
  agent_policy_mode = policy_modes[requested_mode]

  def discard_adaptation():
    nonlocal adapt_params, adapt_policy_params, steps_since_adapt
    nonlocal steps_since_probe, virtual_active, virtual_reject_streak
    discard = getattr(agent, 'discard_params', None)
    if discard:
      discard(adapt_policy_params)
      discard(adapt_params)
    adapt_params = None
    adapt_policy_params = None
    steps_since_adapt = 0
    steps_since_probe = 0
    virtual_active = False
    virtual_reject_streak = 0

  def seed_kwargs(rng_index, offset):
    if not paired_rng:
      return {}
    return {
        'seed_index': int(rng_index) * max(1, int(adapt_cfg.steps)),
        'seed_base': paired_rng_seed + int(offset),
    }

  def put_scalar(outs, batch_shape, key, value):
    outs[key] = np.full(batch_shape, np.float32(value), np.float32)

  def maybe_virtual_adapt(
      carry, acts, outs, batch_shape, rng_index=None):
    """Tentatively update the actor and commit only paired improvements."""
    nonlocal adapt_params, adapt_policy_params, steps_since_adapt
    nonlocal steps_since_probe, virtual_active, virtual_reject_streak
    if adapt_params is None:
      adapt_params = agent.clone_params()

    was_active = bool(virtual_active)
    imag_length = int(getattr(adapt_cfg, 'imag_length', 1))
    start_batch = int(getattr(adapt_cfg, 'start_batch', 1))
    score_key = virtual_score_keys[virtual_score]

    before = time.perf_counter()
    before_rollout, before_features = agent.gate_features(
        carry, params=adapt_params, horizon=imag_length,
        start_batch=start_batch, consequence=True,
        **seed_kwargs(rng_index, 20))
    before_seconds = time.perf_counter() - before

    before = time.perf_counter()
    training_rollout, _ = agent.gate_features(
        carry, params=adapt_params, horizon=imag_length,
        start_batch=start_batch, consequence=True,
        **seed_kwargs(rng_index, 21))
    training_seconds = time.perf_counter() - before

    rollback = agent.snapshot_adapt_actor_state(adapt_params)
    before = time.perf_counter()
    tentative_params, tentative_carry, mets = agent.adapt_prepared(
        adapt_params, carry, training_rollout, adapt_cfg.steps,
        **seed_kwargs(rng_index, 22))
    update_seconds = time.perf_counter() - before

    before = time.perf_counter()
    after_rollout, after_features = agent.gate_features(
        carry, params=tentative_params, horizon=imag_length,
        start_batch=start_batch, consequence=True,
        **seed_kwargs(rng_index, 20))
    after_seconds = time.perf_counter() - before

    decision = virtual_update_decision(
        before_features[score_key], after_features[score_key],
        min_improvement=virtual_min_improvement,
        max_baseline_score=virtual_max_baseline)
    if decision.adapt:
      agent.discard_params(rollback)
      adapt_params = tentative_params
      carry = tentative_carry
      if adapt_policy_params is not None:
        agent.discard_params(adapt_policy_params)
      adapt_policy_params = agent.extract_policy_params(adapt_params)
      carry, acts, _ = agent.policy_latent(
          carry, params=adapt_policy_params, mode=agent_policy_mode,
          **({
              'seed_index': int(rng_index),
              'seed_base': paired_rng_seed,
          } if paired_rng else {}))
      virtual_active = True
      virtual_reject_streak = 0
      steps_since_adapt = 0
    else:
      adapt_params = agent.restore_adapt_actor_state(
          tentative_params, rollback)
      if was_active:
        virtual_reject_streak += 1
        if virtual_reject_streak >= virtual_deactivate_after:
          virtual_active = False
      else:
        virtual_reject_streak = 0

    put_scalar(outs, batch_shape, 'log/eval_adapt/trigger', decision.adapt)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/steps',
        float(adapt_cfg.steps) if decision.adapt else 0.0)
    put_scalar(outs, batch_shape, 'log/eval_adapt/gate/evaluated', True)
    put_scalar(outs, batch_shape, 'log/eval_adapt/gate/cheap_pass', True)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/gate/consequence_evaluated', True)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/gate/consequence_pass',
        decision.adapt)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/gate/final_pass',
        decision.adapt)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/gate/baseline_pass',
        decision.baseline_pass)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/gate/virtual_active_before',
        was_active)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/gate/virtual_active_after',
        virtual_active)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/gate/reject_streak',
        virtual_reject_streak)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/gate/score_before',
        decision.baseline_score)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/gate/score_after',
        decision.candidate_score)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/gate/score_improvement',
        decision.improvement)
    for prefix, features in (
        ('before', before_features), ('after', after_features)):
      for key, value in features.items():
        value = np.asarray(value)
        if value.size == 1:
          put_scalar(
              outs, batch_shape,
              f'log/eval_adapt/gate/{prefix}_{key}', value.item())
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/compute/cheap_seconds', 0.0)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/compute/consequence_seconds',
        before_seconds + training_seconds + after_seconds)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/compute/adapt_seconds',
        update_seconds)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/compute/before_probe_seconds',
        before_seconds)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/compute/after_probe_seconds',
        after_seconds)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/compute/training_rollout_seconds',
        training_seconds)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/compute/posterior_samples',
        3 * start_batch)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/compute/imagined_transitions',
        3 * start_batch * imag_length)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/compute/actor_updates',
        int(adapt_cfg.steps) if decision.adapt else 0)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/compute/tentative_actor_updates',
        int(adapt_cfg.steps))
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/compute/rejected_actor_updates',
        0 if decision.adapt else int(adapt_cfg.steps))
    for key, value in mets.items():
      value = np.asarray(value)
      if value.ndim == 0:
        put_scalar(outs, batch_shape, f'log/eval_adapt/{key}', value.item())
    agent.discard_params(before_rollout)
    agent.discard_params(training_rollout)
    agent.discard_params(after_rollout)
    steps_since_probe = 0
    return carry, acts

  def maybe_adapt(carry, acts, outs, batch_shape, rng_index=None):
    """Evaluate the optional gate and perform at most one reusable IR path."""
    nonlocal adapt_params, adapt_policy_params, steps_since_adapt
    if virtual_gate:
      return maybe_virtual_adapt(
          carry, acts, outs, batch_shape, rng_index)
    first_trigger = adapt_params is None
    prepared_rollout = None
    features = {}
    decision = None
    cheap_seconds = 0.0
    consequence_seconds = 0.0
    adapt_seconds = 0.0
    cheap_evaluated = bool(gate_enabled or gate_audit)
    consequence_evaluated = False
    imag_length = int(getattr(adapt_cfg, 'imag_length', 1))
    start_batch = int(getattr(adapt_cfg, 'start_batch', 1))

    if cheap_evaluated:
      before = time.perf_counter()
      _, features = agent.gate_features(
          carry, params=adapt_params,
          horizon=imag_length,
          start_batch=start_batch, consequence=False,
          **seed_kwargs(rng_index, 2))
      cheap_seconds = time.perf_counter() - before
      decision = cheap_decision(
          gate_kind if gate_enabled else 'none', features,
          entropy_threshold=float(getattr(
              gate_cfg, 'entropy_threshold', 1.0)),
          js_threshold=float(getattr(gate_cfg, 'js_threshold', 1.0)))
      need_consequence = bool(
          gate_audit or (gate_enabled and decision.needs_consequence))
      if need_consequence:
        before = time.perf_counter()
        prepared_rollout, consequence_features = agent.gate_features(
            carry, params=adapt_params,
            horizon=imag_length,
            start_batch=start_batch, consequence=True,
            **seed_kwargs(rng_index, 3))
        consequence_seconds = time.perf_counter() - before
        consequence_evaluated = True
        features.update(consequence_features)
        if gate_enabled and decision.needs_consequence:
          decision = consequence_decision(
              decision, features,
              min_success_rate=float(getattr(
                  gate_cfg, 'min_success_rate', 0.03125)),
              direction_threshold=float(getattr(
                  gate_cfg, 'direction_threshold', 1.0)))

    should_adapt = bool(adapt_cfg.enabled) and (
        not gate_enabled or bool(decision and decision.adapt))
    mets = {}
    if should_adapt:
      if adapt_params is None:
        adapt_params = agent.clone_params()
      before = time.perf_counter()
      if adapt_objective == 'distill':
        if reference_actor_params is None:
          raise RuntimeError('Reference actor parameters were not loaded')
        teacher_stats = agent.policy_stats(
            carry, actor_params=reference_actor_params)
        adapt_params, carry, mets = agent.distill_actor(
            adapt_params, carry, teacher_stats, adapt_cfg.steps)
      elif prepared_rollout is not None:
        adapt_params, carry, mets = agent.adapt_prepared(
            adapt_params, carry, prepared_rollout, adapt_cfg.steps,
            **seed_kwargs(rng_index, 4))
      else:
        adapt_params, carry, mets = agent.adapt(
            adapt_params, carry, adapt_cfg.steps,
            warmup=(first_trigger and bool(adapt_cfg.train_critic)),
            freeze_critic=not bool(adapt_cfg.train_critic),
            **seed_kwargs(rng_index, 3 if cheap_evaluated else 2))
      adapt_seconds = time.perf_counter() - before
      if adapt_policy_params is not None:
        discard = getattr(agent, 'discard_params', None)
        if discard:
          discard(adapt_policy_params)
      adapt_policy_params = agent.extract_policy_params(adapt_params)
      carry, acts, _ = agent.policy_latent(
          carry, params=adapt_policy_params, mode=agent_policy_mode,
          **({
              'seed_index': int(rng_index),
              'seed_base': paired_rng_seed,
          } if paired_rng else {}))
      steps_since_adapt = 0

    put_scalar(outs, batch_shape, 'log/eval_adapt/trigger', should_adapt)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/steps',
        float(adapt_cfg.steps) if should_adapt else 0.0)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/gate/evaluated', cheap_evaluated)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/gate/cheap_pass',
        bool(decision.cheap_pass) if decision else False)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/gate/consequence_evaluated',
        consequence_evaluated)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/gate/consequence_pass',
        bool(decision.consequence_pass)
        if decision and decision.consequence_pass is not None else False)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/gate/final_pass', should_adapt)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/compute/cheap_seconds',
        cheap_seconds)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/compute/consequence_seconds',
        consequence_seconds)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/compute/adapt_seconds',
        adapt_seconds)
    ordinary_rollouts = (
        int(adapt_cfg.steps)
        if should_adapt and prepared_rollout is None else 0)
    posterior_samples = start_batch * (
        int(cheap_evaluated) + int(consequence_evaluated) +
        ordinary_rollouts)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/compute/posterior_samples',
        posterior_samples)
    imagined = start_batch * imag_length * (
        int(consequence_evaluated) + ordinary_rollouts)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/compute/imagined_transitions',
        imagined)
    put_scalar(
        outs, batch_shape, 'log/eval_adapt/compute/actor_updates',
        int(adapt_cfg.steps) if should_adapt else 0)
    for key, value in features.items():
      value = np.asarray(value)
      if value.size == 1:
        put_scalar(
            outs, batch_shape, f'log/eval_adapt/gate/{key}', value.item())
    for key, value in mets.items():
      value = np.asarray(value)
      if value.ndim == 0:
        put_scalar(outs, batch_shape, f'log/eval_adapt/{key}', value.item())
    if prepared_rollout is not None:
      discard = getattr(agent, 'discard_params', None)
      if discard:
        discard(prepared_rollout)
    steps_since_adapt = 0
    return carry, acts

  def policy(carry, obs, **kwargs):
    nonlocal adapt_params, adapt_policy_params, steps_since_adapt
    nonlocal steps_since_probe, virtual_active, virtual_reject_streak
    nonlocal paired_episode, paired_step
    if paired_rng:
      if obs['is_first'].shape[0] != 1:
        raise ValueError('eval_adapt.paired_rng requires eval envs=1')
      if obs['is_first'].any():
        paired_episode += 1
        paired_step = 0
      else:
        paired_step += 1
      if paired_step >= paired_rng_stride:
        raise RuntimeError(
            'Episode exceeded eval_adapt.paired_rng_stride; increase it')
      rng_index = paired_episode * paired_rng_stride + paired_step
    else:
      rng_index = None
    carry, acts, outs = agent.policy(
        carry, obs, mode=agent_policy_mode, params=adapt_policy_params,
        **({
            'seed_index': int(rng_index),
            'seed_base': paired_rng_seed,
        } if paired_rng else {}))
    if adapt_cfg.enabled or gate_audit:
      if obs['is_first'].shape[0] != 1:
        raise ValueError('eval_adapt requires eval envs=1')
      if obs['is_last'].any():
        # The driver masks terminal actions. Adapting here would spend compute
        # on an unused action. Episode-local evaluation also throws its cloned
        # parameters away here; run-persistent evaluation deliberately keeps
        # the actor and optimizer state while resetting only cadence/carry.
        if persistence == 'episode':
          discard_adaptation()
        else:
          steps_since_adapt = 0
          steps_since_probe = 0
          virtual_active = False
          virtual_reject_streak = 0
      elif obs['is_first'].any():
        assert steps_since_adapt == 0, (
            'eval_adapt cadence leaked across episode boundary')
        if adapt_cfg.enabled and persistence == 'episode':
          assert adapt_params is None and adapt_policy_params is None, (
              'eval_adapt parameters leaked across episode boundary')
        steps_since_probe = 0
        virtual_active = False
        virtual_reject_streak = 0
        carry, acts = maybe_adapt(
            carry, acts, outs, obs['is_first'].shape, rng_index)
      else:
        if virtual_gate:
          steps_since_probe += 1
          cadence = (
              virtual_active_every if virtual_active
              else virtual_dormant_every)
          if steps_since_probe >= cadence:
            carry, acts = maybe_adapt(
                carry, acts, outs, obs['is_first'].shape, rng_index)
        else:
          steps_since_adapt += 1
          if adapt_every_k > 0 and steps_since_adapt >= adapt_every_k:
            carry, acts = maybe_adapt(
                carry, acts, outs, obs['is_first'].shape, rng_index)
    return carry, acts, outs

  def write_metrics():
    logger.add(agg.result())
    logger.add(epstats.result(), prefix='epstats')
    logger.add(usage.stats(), prefix='usage')
    logger.add({'fps/policy': policy_fps.result()})
    logger.add({'timer': elements.timer.stats()['summary']})
    logger.write()

  try:
    # Adaptation optimizer state is evaluation-local and older checkpoints do
    # not contain the critic-first optimizer namespaces. Always leave these
    # freshly initialized instead of requiring an exact checkpoint tree match.
    load_regex = (
        '^(?!model_opt/)' if resume_adapt_state else
        '^(?!(adapt_opt|adapt_actor_opt|adapt_critic_opt|model_opt)/)')
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=load_regex)))
    if adapt_cfg.enabled and adapt_objective == 'distill':
      reference_checkpoint = str(getattr(
          adapt_cfg, 'reference_checkpoint', ''))
      if not reference_checkpoint:
        raise ValueError(
            'eval_adapt.reference_checkpoint is required for distillation')
      # Temporarily load and copy only the clean actor parameters, and then
      # restore the evaluated (corrupted) actor.
      # Encoder/dynamics parameters are identical by Stage 2 construction.
      elements.checkpoint.load(reference_checkpoint, dict(
          agent=bind(agent.load, regex='^pol/')))
      reference_actor_params = agent.extract_actor_params()
      elements.checkpoint.load(args.from_checkpoint, dict(
          agent=bind(agent.load, regex='^pol/')))
    digests = getattr(agent, 'parameter_digests', None)
    integrity_before = digests() if digests else None

    print(f'Start {requested_mode} evaluation')
    driver.reset(agent.init_policy)
    warmup_episodes = int(getattr(args, 'eval_warmup_episodes', 0))
    if warmup_episodes < 0:
      raise ValueError('run.eval_warmup_episodes must be nonnegative')
    warmup_started = time.perf_counter()
    if warmup_episodes:
      recording = False
      driver(policy, episodes=warmup_episodes)
      recording = True
      # Warm-up exists only to remove compilation/start-up costs. It must not
      # become the first persistent recovery block.
      if persistence == 'run':
        discard_adaptation()
    warmup_seconds = time.perf_counter() - warmup_started
    evaluation_started = time.perf_counter()
    evaluation_start_step = int(step)
    eval_episodes = int(getattr(args, 'eval_episodes', 0))
    if eval_episodes:
      if eval_episodes < 0:
        raise ValueError('run.eval_episodes must be nonnegative')
      driver(policy, episodes=eval_episodes)
      write_metrics()
    else:
      while step < args.steps:
        driver(policy, steps=10)
        if should_log(step):
          write_metrics()
    evaluation_seconds = time.perf_counter() - evaluation_started
    evaluation_steps = int(step) - evaluation_start_step
    (logdir / 'eval_runtime.json').write_text(json.dumps({
        'evaluation_seconds': evaluation_seconds,
        'environment_steps': evaluation_steps,
        'steps_per_second': (
            evaluation_steps / evaluation_seconds
            if evaluation_seconds > 0 else float('nan')),
        'warmup_episodes': warmup_episodes,
        'warmup_seconds': warmup_seconds,
    }, indent=2, sort_keys=True) + '\n')
    integrity_after = digests() if digests else None
    adapted_payload = None
    if persistence == 'run' and output_checkpoint:
      adapted_payload = agent.export_params(adapt_params)
    checkpoint_digests = getattr(agent, 'checkpoint_digests', None)
    if (persistence == 'run' and adapted_payload is not None and
        checkpoint_digests):
      adapted_integrity = checkpoint_digests(adapted_payload)
    elif persistence == 'run' and digests and adapt_params is not None:
      adapted_integrity = digests(adapt_params)
    else:
      # Episode-local adaptation is discarded and has no persistent integrity
      # surface. Avoid gathering a full temporary checkpoint just to prove the
      # already-audited base agent remained unchanged.
      adapted_integrity = integrity_after
    frozen_groups = (
        'critic', 'bounded_value', 'reward_continuation', 'representation',
        'world_model', 'normalizers')
    persistent_violations = {
        key: (integrity_before[key], adapted_integrity[key])
        for key in frozen_groups
        if integrity_before is not None and adapted_integrity is not None and
        key in integrity_before and key in adapted_integrity and
        integrity_before[key] != adapted_integrity[key]}
    integrity = {
        'policy_mode': requested_mode,
        'adapt_enabled': bool(adapt_cfg.enabled),
        'adapt_objective': adapt_objective,
        'gate_enabled': gate_enabled,
        'gate_kind': gate_kind,
        'gate_audit': gate_audit,
        'virtual_score': virtual_score if virtual_gate else None,
        'virtual_min_improvement': (
            virtual_min_improvement if virtual_gate else None),
        'virtual_max_baseline_score': (
            virtual_max_baseline if virtual_gate else None),
        'virtual_dormant_every': (
            virtual_dormant_every if virtual_gate else None),
        'virtual_active_every': (
            virtual_active_every if virtual_gate else None),
        'virtual_deactivate_after': (
            virtual_deactivate_after if virtual_gate else None),
        'persistence': persistence,
        'resume_adapt_state': resume_adapt_state,
        'output_checkpoint': output_checkpoint,
        'paired_rng': paired_rng,
        'paired_rng_seed': paired_rng_seed if paired_rng else None,
        'paired_rng_stride': paired_rng_stride if paired_rng else None,
        'reference_checkpoint': str(getattr(
            adapt_cfg, 'reference_checkpoint', '')),
        'before': integrity_before,
        'after': integrity_after,
        'adapted': adapted_integrity,
        'persistent_frozen_component_violations': persistent_violations,
        'equal': integrity_before == integrity_after,
    }
    (logdir / 'eval_integrity.json').write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + '\n')
    if integrity_before is not None and not integrity['equal']:
      raise RuntimeError(
          'Persistent agent parameters changed during evaluation; see '
          f'{logdir / "eval_integrity.json"}')
    if persistent_violations:
      raise RuntimeError(
          'Persistent IR changed frozen components; see '
          f'{logdir / "eval_integrity.json"}')
    if output_checkpoint:
      _save_adapted_checkpoint(
          agent, adapt_params, output_checkpoint, {
              'format': 1,
              'source_checkpoint': str(args.from_checkpoint),
              'persistence': persistence,
              'resume_adapt_state': resume_adapt_state,
              'adapt_enabled': bool(adapt_cfg.enabled),
              'adapt_objective': adapt_objective,
              'gate_enabled': gate_enabled,
              'gate_kind': gate_kind,
              'episodes': eval_episodes,
              'integrity_before': integrity_before,
              'integrity_after': adapted_integrity,
              'frozen_component_violations': persistent_violations,
          }, payload=adapted_payload)
  finally:
    try:
      discard_adaptation()
      driver.close()
    finally:
      if trace_file:
        trace_file.close()
      if reference_actor_params is not None:
        agent.discard_params(reference_actor_params)
      logger.close()
