"""Resumable Cheetah A-to-B continual imagined rehearsal experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from functools import partial as bind

import elements
import embodied
import jax
import numpy as np


MODEL_PREFIXES = ('enc/', 'dyn/', 'dec/', 'rew/', 'con/')
BEHAVIOR_PREFIXES = ('pol/', 'val/', 'slowval/')
ACTOR_PREFIXES = ('pol/', 'adapt_opt/', 'adapt_actor_opt/')
CRITIC_PREFIXES = (
    'val/', 'slowval/', 'retnorm/', 'valnorm/', 'advnorm/',
    'adapt_critic_opt/')
SOURCE_REGEX = (
    '^(?!(adapt_opt|adapt_actor_opt|adapt_critic_opt|model_opt)/)')


class StopExperiment(RuntimeError):
  pass


def _scalar(value):
  value = np.asarray(value)
  return float(value.mean()) if value.size else float('nan')


def _finite_metrics(metrics):
  return all(np.isfinite(np.asarray(value)).all() for value in metrics.values())


def _digest(params, prefixes):
  result = hashlib.sha256()
  keys = sorted(key for key in params if key.startswith(prefixes))
  if not keys:
    raise ValueError(f'No parameters matched namespace prefixes {prefixes}')
  for key in keys:
    value = np.asarray(jax.device_get(params[key]))
    result.update(key.encode())
    result.update(str(value.dtype).encode())
    result.update(np.asarray(value.shape, np.int64).tobytes())
    result.update(value.tobytes())
  return result.hexdigest()


def _append_jsonl(path, row):
  def safe(value):
    if isinstance(value, (np.floating, float)):
      return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
      return int(value)
    if isinstance(value, dict):
      return {key: safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
      return [safe(item) for item in value]
    return value
  with path.open('a') as file:
    file.write(json.dumps(safe(row), sort_keys=True, allow_nan=False) + '\n')


def _append_csv(path, row):
  previous = []
  fields = []
  if path.exists():
    with path.open() as file:
      reader = csv.DictReader(file)
      fields = list(reader.fieldnames or ())
      previous = list(reader)
  new_fields = [key for key in row if key not in fields]
  fields.extend(new_fields)
  mode = 'w' if previous or new_fields else 'a'
  with path.open(mode) as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    if mode == 'w' or not path.exists():
      writer.writeheader()
    writer.writerows(previous)
    writer.writerow(row)


def _correlation(left, right):
  left = np.asarray(left, np.float64).reshape(-1)
  right = np.asarray(right, np.float64).reshape(-1)
  finite = np.isfinite(left) & np.isfinite(right)
  left, right = left[finite], right[finite]
  if len(left) < 2 or left.std() == 0 or right.std() == 0:
    return float('nan')
  return float(np.corrcoef(left, right)[0, 1])


def _discounted(reward, gamma):
  reward = np.asarray(reward, np.float64)
  return (reward * gamma ** np.arange(reward.shape[-1])).sum(-1)


def _categorical_kl(posterior, prior):
  def logsoftmax(value):
    value = np.asarray(value, np.float64)
    maximum = value.max(-1, keepdims=True)
    shifted = value - maximum
    return shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
  logp, logq = logsoftmax(posterior), logsoftmax(prior)
  return float((np.exp(logp) * (logp - logq)).sum(-1).mean())


def _leaf_list(value):
  value = np.asarray(value)
  return [jax.device_put(value[index]) for index in range(len(value))]


def _carry_from_trace(agent, trace, indices):
  carry = list(agent.init_policy(len(indices)))
  dyn = dict(carry[1])
  for key in ('deter', 'stoch', 'logit'):
    dyn[key] = _leaf_list(np.asarray(trace[key][indices], np.float32))
  carry[1] = dyn
  return tuple(carry)


def _normal_kl(current, reference):
  cm, cs = current
  rm, rs = reference
  cs = np.maximum(np.asarray(cs, np.float64), 1e-8)
  rs = np.maximum(np.asarray(rs, np.float64), 1e-8)
  cm, rm = np.asarray(cm, np.float64), np.asarray(rm, np.float64)
  return 0.5 * (
      np.square(cs / rs) + np.square(rm - cm) / np.square(rs) +
      2 * np.log(rs) - 2 * np.log(cs) - 1).sum(-1)


def _memory_mb():
  try:
    import psutil
    return psutil.Process().memory_info().rss / (1024 ** 2)
  except ImportError:
    return float('nan')


def _adapt_due(enabled, is_first, is_last, steps_since, interval):
  return bool(enabled and not is_last and (
      is_first or (interval > 0 and steps_since + 1 >= interval)))


def _probe_episode(agent, trace, actor_reference, cfg):
  horizons = tuple(int(x) for x in cfg.probe_horizons)
  maximum = max(horizons)
  indices = np.arange(
      0, max(0, len(trace['reward']) - maximum - 1),
      int(cfg.anchor_stride), dtype=np.int32)
  if not len(indices):
    return {'anchors': 0}, []

  values, slowvalues, value_entropies = [], [], []
  actor_entropies, actor_kls = [], []
  horizon_data = {horizon: {
      'reward_error': [], 'pred_return': [], 'real_return': [],
      'reward_above_0p5': [], 'reward_above_0p8': [],
      'posterior_kl': [], 'deter_relative_l2': [],
      'continuation_error': [],
  } for horizon in horizons}
  raw = []
  batch_size = int(cfg.probe_batch)

  for begin in range(0, len(indices), batch_size):
    batch = indices[begin:begin + batch_size]
    carry = _carry_from_trace(agent, trace, batch)
    critic = agent.critic_state(carry)
    current_actor = agent.policy_stats(carry)
    reference_actor = agent.policy_stats(carry, actor_reference)
    values.extend(np.asarray(critic['value']).reshape(-1).tolist())
    slowvalues.extend(np.asarray(critic['slowvalue']).reshape(-1).tolist())
    value_entropies.extend(
        np.asarray(critic['value_entropy']).reshape(-1).tolist())
    for key in agent.act_space:
      actor_entropies.extend(
          np.asarray(current_actor[f'{key}/entropy']).reshape(-1).tolist())
      mean_key, std_key = f'{key}/mean', f'{key}/stddev'
      if std_key in current_actor and std_key in reference_actor:
        actor_kls.extend(_normal_kl(
            (current_actor[mean_key], current_actor[std_key]),
            (reference_actor[mean_key], reference_actor[std_key]),
        ).reshape(-1).tolist())

    actions = {key: np.stack([
        trace[key][step:step + maximum] for step in batch])
        for key in agent.act_space}
    rollout = agent.critic_recorded_rollout(carry, actions)
    predicted = np.asarray(rollout['reward'])
    predicted_con = np.asarray(rollout['continuation'])
    prior_logit = np.asarray(rollout['logit'])
    prior_deter = np.asarray(rollout['deter'])
    for row_index, step in enumerate(batch):
      real_reward = trace['reward'][step + 1:step + maximum + 1]
      real_con = 1.0 - np.asarray(
          trace['is_terminal'][step + 1:step + maximum + 1], np.float32)
      for horizon in horizons:
        pred = predicted[row_index, :horizon]
        real = real_reward[:horizon]
        pred_return = _discounted(pred[None], cfg.gamma)[0]
        real_return = _discounted(real[None], cfg.gamma)[0]
        data = horizon_data[horizon]
        data['reward_error'].extend((pred - real).reshape(-1).tolist())
        data['pred_return'].append(pred_return)
        data['real_return'].append(real_return)
        data['reward_above_0p5'].append(float((pred >= 0.5).mean()))
        data['reward_above_0p8'].append(float((pred >= 0.8).mean()))
        data['posterior_kl'].append(_categorical_kl(
            trace['logit'][step + horizon],
            prior_logit[row_index, horizon - 1]))
        posterior_deter = np.asarray(
            trace['deter'][step + horizon], np.float64)
        difference = np.linalg.norm(
            np.asarray(prior_deter[row_index, horizon - 1], np.float64) -
            posterior_deter)
        data['deter_relative_l2'].append(float(
            difference / max(np.linalg.norm(posterior_deter), 1e-8)))
        data['continuation_error'].extend((
            predicted_con[row_index, :horizon] - real_con[:horizon]
        ).reshape(-1).tolist())
        raw.append(dict(
            anchor=int(step), horizon=horizon,
            predicted_return=float(pred_return),
            realised_return=float(real_return),
            reward_mae=float(np.abs(pred - real).mean())))

  values = np.asarray(values, np.float64)
  slowvalues = np.asarray(slowvalues, np.float64)
  summary = {
      'anchors': int(len(indices)),
      'critic/value_mean': float(values.mean()),
      'critic/value_std': float(values.std()),
      'critic/value_abs_p99': float(np.quantile(np.abs(values), 0.99)),
      'critic/slow_disagreement_mae': float(
          np.abs(values - slowvalues).mean()),
      'critic/value_entropy_mean': float(np.mean(value_entropies)),
      'actor/entropy_mean': float(np.mean(actor_entropies)),
      'actor/kl_from_initial_mean': (
          float(np.mean(actor_kls)) if actor_kls else float('nan')),
  }
  longest_real_return = np.asarray(horizon_data[maximum]['real_return'])
  summary.update({
      f'critic/value_vs_real_h{maximum}_mae': float(
          np.abs(values - longest_real_return).mean()),
      f'critic/value_vs_real_h{maximum}_correlation': _correlation(
          values, longest_real_return),
  })
  for horizon, data in horizon_data.items():
    errors = np.asarray(data['reward_error'])
    pred_returns = np.asarray(data['pred_return'])
    real_returns = np.asarray(data['real_return'])
    prefix = f'model/h{horizon}'
    summary.update({
        f'{prefix}_reward_mae': float(np.abs(errors).mean()),
        f'{prefix}_reward_bias': float(errors.mean()),
        f'{prefix}_return_mae': float(
            np.abs(pred_returns - real_returns).mean()),
        f'{prefix}_return_correlation': _correlation(
            pred_returns, real_returns),
        f'{prefix}_reward_above_0p5': float(
            np.mean(data['reward_above_0p5'])),
        f'{prefix}_reward_above_0p8': float(
            np.mean(data['reward_above_0p8'])),
        f'{prefix}_prior_posterior_kl': float(
            np.mean(data['posterior_kl'])),
        f'{prefix}_deter_relative_l2': float(
            np.mean(data['deter_relative_l2'])),
        f'{prefix}_continuation_mae': float(np.abs(
            np.asarray(data['continuation_error'])).mean()),
    })
  return summary, raw


def cir(make_agent, make_replay, make_env, make_stream, make_logger, args):
  """Run the fixed-schedule first CIR experiment described in the plan."""
  if not args.from_checkpoint:
    raise ValueError('CIR requires --run.from_checkpoint')
  cfg = args.cir
  if args.envs != 1:
    raise ValueError('CIR currently requires --run.envs 1')

  logdir = elements.Path(args.logdir)
  logdir.mkdir(parents=True, exist_ok=True)
  artifacts = logdir / 'cir_artifacts'
  artifacts.mkdir(parents=True, exist_ok=True)
  events_path = artifacts / 'events.jsonl'
  episodes_path = artifacts / 'episodes.csv'
  probes_path = artifacts / 'probes.jsonl'

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()
  step = logger.step

  # Always derive the fixed actor reference from the source checkpoint before
  # an experiment checkpoint can restore newer live parameters.
  elements.checkpoint.load(args.from_checkpoint, dict(
      agent=bind(agent.load, regex=SOURCE_REGEX)))
  actor_reference = agent.extract_actor_params()

  state = elements.Saveable(attrs=(
      'phase', 'baseline_completed', 'shock_completed',
      'model_repair_updates', 'cir_completed', 'eval_done',
      'baseline_scores', 'shock_scores', 'baseline_value_p99',
      'divergence_streak', 'warmup_pending', 'status',
      'train_env_steps', 'eval_env_steps', 'actor_updates',
      'critic_updates', 'imagined_transitions',
      'fixed_a_trajectory', 'fixed_b_trajectory'))
  state.phase = 'baseline_a'
  state.baseline_completed = 0
  state.shock_completed = 0
  state.model_repair_updates = 0
  state.cir_completed = 0
  state.eval_done = []
  state.baseline_scores = []
  state.shock_scores = []
  state.baseline_value_p99 = 0.0
  state.divergence_streak = 0
  state.warmup_pending = True
  state.status = 'running'
  state.train_env_steps = 0
  state.eval_env_steps = 0
  state.actor_updates = 0
  state.critic_updates = 0
  state.imagined_transitions = 0
  state.fixed_a_trajectory = ''
  state.fixed_b_trajectory = ''

  checkpoint = elements.Checkpoint(
      logdir / 'ckpt', keep=int(getattr(args, 'ckpt_keep', 1)), step=step)
  checkpoint.agent = agent
  checkpoint.replay = replay
  checkpoint.step = step
  checkpoint.state = state
  if checkpoint.exists():
    checkpoint.load()
  else:
    checkpoint.save()
  if state.status in ('stopped', 'complete'):
    print(
        f'CIR checkpoint status is {state.status}; '
        'inspect cir_artifacts/events.jsonl.')
    logger.close()
    return

  stream = None
  model_carry = None

  def event(kind, **values):
    row = dict(
        time=time.time(), environment_steps=int(step), phase=state.phase,
        kind=kind, model_updates=int(agent.n_model_updates),
        rehearsal_updates=int(agent.n_adapt), probes=int(agent.n_probes),
        train_env_steps=int(state.train_env_steps),
        eval_env_steps=int(state.eval_env_steps),
        actor_updates=int(state.actor_updates),
        critic_updates=int(state.critic_updates),
        imagined_transitions=int(state.imagined_transitions),
        actor_frozen=bool(cfg.freeze_actor),
        critic_frozen=bool(cfg.freeze_critic),
        memory_rss_mb=_memory_mb(),
        **values)
    _append_jsonl(events_path, row)
    scalars = {
        key: value for key, value in row.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))}
    if scalars:
      logger.add(scalars, prefix='cir')
      logger.write()

  def save_checkpoint():
    checkpoint.save()

  def run_episode(label, action_scale, collect, rehearse, warmup=False):
    trace_keys = tuple(dict.fromkeys((
        *agent.act_space.keys(), *agent.obs_space.keys(),
        'deter', 'stoch', 'logit')))
    trace = {key: [] for key in trace_keys}
    adapt_metrics = []
    adapt_elapsed = 0.0
    calls_since_adapt = 0
    first_trigger = bool(warmup)
    started = time.perf_counter()
    driver = embodied.Driver(
        [bind(make_env, 0, action_scale=float(action_scale))], parallel=False)

    def policy(carry, obs, **kwargs):
      nonlocal calls_since_adapt, first_trigger, adapt_elapsed
      carry, acts, outs = agent.policy(carry, obs, mode='eval')
      interval = int(cfg.adapt_every)
      trigger = _adapt_due(
          rehearse, bool(obs['is_first'][0]), bool(obs['is_last'][0]),
          calls_since_adapt, interval)
      if trigger:
        before = time.perf_counter()
        carry, metrics = agent.adapt_persistent(
            carry, steps=int(cfg.adapt_steps), warmup=first_trigger,
            freeze_critic=bool(cfg.freeze_critic))
        adapt_elapsed += time.perf_counter() - before
        first_trigger = False
        calls_since_adapt = 0
        if not _finite_metrics(metrics):
          raise StopExperiment('non-finite rehearsal metric')
        adapt_metrics.append({key: _scalar(value) for key, value in metrics.items()})
        actor_steps = int(round(_scalar(metrics.get('actor_steps', 0))))
        critic_steps = int(round(_scalar(metrics.get('critic_steps', 0))))
        state.actor_updates += actor_steps
        state.critic_updates += critic_steps
        rollouts = max(actor_steps, critic_steps)
        state.imagined_transitions += (
            rollouts * int(agent.config.eval_adapt.imag_length) *
            int(agent.config.eval_adapt.start_batch))
        carry, acts, _ = agent.policy_latent(carry)
      elif rehearse:
        calls_since_adapt += 1

      for key in agent.act_space:
        trace[key].append(np.asarray(acts[key][0]))
      for key in agent.obs_space:
        trace[key].append(np.asarray(obs[key][0]))
      dyn = carry[1]
      trace['deter'].append(np.asarray(dyn['deter'][0], np.float16))
      trace['stoch'].append(np.asarray(dyn['stoch'][0], np.uint8))
      trace['logit'].append(np.asarray(dyn['logit'][0], np.float16))
      return carry, acts, outs

    def transition(tran, worker):
      step.increment()
      if collect:
        state.train_env_steps += 1
        replay.add(tran, worker)
      else:
        state.eval_env_steps += 1

    driver.on_step(transition)
    try:
      driver.reset(agent.init_policy)
      driver(policy, episodes=1)
    finally:
      driver.close()

    trace = {key: np.stack(value) for key, value in trace.items()}
    score = float(np.asarray(trace['reward'], np.float64).sum())
    action_values = np.concatenate([
        np.asarray(trace[key], np.float64).reshape(len(trace[key]), -1)
        for key in agent.act_space], -1)
    summary = dict(
        label=label, action_scale=float(action_scale), score=score,
        length=int(len(trace['reward'])),
        action_abs_mean=float(np.abs(action_values).mean()),
        action_saturation=float((np.abs(action_values) >= 0.95).mean()),
        adapt_triggers=len(adapt_metrics),
        adapt_seconds=float(adapt_elapsed),
        episode_seconds=float(time.perf_counter() - started))
    if adapt_metrics:
      keys = sorted({key for row in adapt_metrics for key in row})
      for key in keys:
        values = [row[key] for row in adapt_metrics if key in row]
        summary[f'adapt/{key}'] = float(np.mean(values))
    if not _finite_metrics({k: v for k, v in summary.items()
                            if isinstance(v, (int, float))}):
      raise StopExperiment('non-finite episode diagnostic')
    return trace, summary

  def diagnose(trace, label):
    summary, rows = _probe_episode(agent, trace, actor_reference, cfg)
    summary = {'label': label, **summary}
    _append_jsonl(probes_path, summary)
    for row in rows:
      row['label'] = label
      _append_jsonl(artifacts / 'probe_anchors.jsonl', row)
    if not _finite_metrics({
        key: value for key, value in summary.items()
        if isinstance(value, (int, float)) and not math.isnan(value)}):
      raise StopExperiment('non-finite probe diagnostic')
    return summary

  def record_episode(trace, summary, probe):
    episode_id = sum((
        state.baseline_completed, state.shock_completed,
        state.cir_completed, len(state.eval_done)))
    path = artifacts / f'trajectory_{episode_id:03d}_{summary["label"]}.npz'
    np.savez_compressed(str(path), **trace)
    row = {
        key: value for key, value in {**summary, **probe}.items()
        if isinstance(value, (str, int, float))}
    row['trajectory'] = path.name
    _append_csv(episodes_path, row)
    event('episode', **row)
    return path

  def diagnose_fixed(filename, label):
    if not filename:
      return {}
    with np.load(str(artifacts / filename)) as data:
      trace = {key: data[key] for key in data.files}
    obs = {key: trace[key][None] for key in agent.obs_space}
    prevact = {}
    for key, space in agent.act_space.items():
      zero = np.zeros(space.shape, space.dtype)
      prevact[key] = np.concatenate([
          zero[None], trace[key][:-1]], axis=0)[None]
    carry = agent.init_observe(1)
    chunks = []
    chunk_size = int(cfg.reencode_chunk)
    length = len(trace['reward'])
    for begin in range(0, length, chunk_size):
      end = min(begin + chunk_size, length)
      valid = end - begin
      obs_chunk = {key: value[:, begin:end] for key, value in obs.items()}
      act_chunk = {
          key: value[:, begin:end] for key, value in prevact.items()}
      if valid < chunk_size:
        def pad(value):
          shape = (value.shape[0], chunk_size - valid, *value.shape[2:])
          return np.concatenate([value, np.zeros(shape, value.dtype)], axis=1)
        obs_chunk = {key: pad(value) for key, value in obs_chunk.items()}
        act_chunk = {key: pad(value) for key, value in act_chunk.items()}
      carry, feat = agent.posterior_chunk(
          carry, obs_chunk, act_chunk)
      chunks.append({
          key: np.asarray(value)[:, :valid] for key, value in feat.items()})
    for key in ('deter', 'stoch', 'logit'):
      trace[key] = np.concatenate([
          np.asarray(feat[key][0]) for feat in chunks], axis=0)
    summary = diagnose(trace, label)
    event('fixed_probe', **{
        key: value for key, value in summary.items()
        if isinstance(value, (str, int, float))})
    return summary

  def model_updates(count, label):
    nonlocal stream, model_carry
    if stream is None:
      stream = iter(agent.stream(make_stream(replay, 'train')))
      model_carry = agent.init_train(args.batch_size)
    behavior_before = _digest(agent.params, BEHAVIOR_PREFIXES)
    metrics = []
    started = time.perf_counter()
    for _ in range(int(count)):
      if not args.replay_context:
        # Independent replay windows must not inherit recurrent state from a
        # different randomly sampled sequence.
        model_carry = agent.init_train(args.batch_size)
      batch = next(stream)
      model_carry, outs, mets = agent.train_model(model_carry, batch)
      if 'replay' in outs:
        replay.update(outs['replay'])
      if not _finite_metrics(mets):
        raise StopExperiment('non-finite world-model metric')
      metrics.append({key: _scalar(value) for key, value in mets.items()})
    behavior_after = _digest(agent.params, BEHAVIOR_PREFIXES)
    if behavior_before != behavior_after:
      raise StopExperiment('model-only update changed behaviour namespace')
    result = dict(
        label=label, updates=int(count),
        seconds=float(time.perf_counter() - started),
        behavior_hash=behavior_after)
    keys = sorted({key for row in metrics for key in row})
    for key in keys:
      result[f'metric/{key}'] = float(np.mean([
          row[key] for row in metrics if key in row]))
    event('model_updates', **result)
    return result

  def official_eval(label, action_scale):
    model_before = _digest(agent.params, MODEL_PREFIXES)
    behavior_before = _digest(agent.params, BEHAVIOR_PREFIXES)
    trace, summary = run_episode(
        label, action_scale, collect=False, rehearse=False)
    probe = diagnose(trace, label)
    if model_before != _digest(agent.params, MODEL_PREFIXES):
      raise StopExperiment('frozen evaluation changed world-model namespace')
    if behavior_before != _digest(agent.params, BEHAVIOR_PREFIXES):
      raise StopExperiment('frozen evaluation changed behaviour namespace')
    record_episode(trace, summary, probe)
    return summary, probe

  def check_divergence(probe):
    current = float(probe.get('critic/value_abs_p99', 0.0))
    threshold = float(cfg.divergence_factor) * max(
        float(state.baseline_value_p99), 1e-6)
    state.divergence_streak = (
        state.divergence_streak + 1 if current > threshold else 0)
    event(
        'critic_guard', value_abs_p99=current, threshold=threshold,
        streak=state.divergence_streak)
    if state.divergence_streak >= int(cfg.divergence_patience):
      raise StopExperiment(
          f'critic value p99 exceeded {cfg.divergence_factor}x baseline at '
          f'{state.divergence_streak} consecutive diagnostics')

  try:
    while state.baseline_completed < int(cfg.baseline_episodes):
      index = state.baseline_completed + 1
      model_before = _digest(agent.params, MODEL_PREFIXES)
      behavior_before = _digest(agent.params, BEHAVIOR_PREFIXES)
      trace, summary = run_episode(
          f'baseline_a_{index}', cfg.action_scale_a, True, False)
      probe = diagnose(trace, summary['label'])
      if model_before != _digest(agent.params, MODEL_PREFIXES):
        raise StopExperiment('frozen A baseline changed world model')
      if behavior_before != _digest(agent.params, BEHAVIOR_PREFIXES):
        raise StopExperiment('frozen A baseline changed behaviour')
      path = record_episode(trace, summary, probe)
      if not state.fixed_a_trajectory:
        state.fixed_a_trajectory = path.name
      state.baseline_scores.append(summary['score'])
      state.baseline_value_p99 = max(
          state.baseline_value_p99,
          float(probe.get('critic/value_abs_p99', 0.0)))
      state.baseline_completed = index
      save_checkpoint()
    state.phase = 'shock_b'

    while state.shock_completed < int(cfg.shock_episodes):
      index = state.shock_completed + 1
      model_before = _digest(agent.params, MODEL_PREFIXES)
      behavior_before = _digest(agent.params, BEHAVIOR_PREFIXES)
      trace, summary = run_episode(
          f'shock_b_{index}', cfg.action_scale_b, True, False)
      probe = diagnose(trace, summary['label'])
      if model_before != _digest(agent.params, MODEL_PREFIXES):
        raise StopExperiment('frozen B shock changed world model')
      if behavior_before != _digest(agent.params, BEHAVIOR_PREFIXES):
        raise StopExperiment('frozen B shock changed behaviour')
      path = record_episode(trace, summary, probe)
      if not state.fixed_b_trajectory:
        state.fixed_b_trajectory = path.name
      state.shock_scores.append(summary['score'])
      state.shock_completed = index
      save_checkpoint()
    ratio = float(np.mean(state.shock_scores) / max(
        np.mean(state.baseline_scores), 1e-8))
    severity = 'mild' if ratio > cfg.mild_ratio else (
        'severe' if ratio < cfg.severe_ratio else 'usable')
    event('shift_strength', b_over_a=ratio, classification=severity)
    state.phase = 'model_repair_b'

    while state.model_repair_updates < int(cfg.initial_model_updates):
      count = min(
          int(cfg.model_checkpoint_interval),
          int(cfg.initial_model_updates) - state.model_repair_updates)
      model_updates(count, 'initial_b_repair')
      state.model_repair_updates += count
      save_checkpoint()
    if 'post_model_repair' not in state.eval_done:
      official_eval('post_model_repair_b', cfg.action_scale_b)
      official_eval('post_model_repair_a', cfg.action_scale_a)
      diagnose_fixed(state.fixed_b_trajectory, 'fixed_b_post_model_repair')
      diagnose_fixed(state.fixed_a_trajectory, 'fixed_a_post_model_repair')
      state.eval_done.append('post_model_repair')
      save_checkpoint()
    state.phase = 'cir_b'

    milestones = tuple(int(x) for x in cfg.eval_milestones)
    while state.cir_completed < int(cfg.cir_episodes):
      index = state.cir_completed + 1
      model_before = _digest(agent.params, MODEL_PREFIXES)
      actor_before = (
          _digest(agent.params, ACTOR_PREFIXES)
          if cfg.freeze_actor else None)
      critic_before = (
          _digest(agent.params, CRITIC_PREFIXES)
          if cfg.freeze_critic else None)
      label = (
          f'wm_b_collect_{index}' if cfg.freeze_actor
          else f'cir_b_train_{index}')
      trace, summary = run_episode(
          label, cfg.action_scale_b, True, not cfg.freeze_actor,
          warmup=state.warmup_pending)
      state.warmup_pending = False
      model_after = _digest(agent.params, MODEL_PREFIXES)
      if model_before != model_after:
        raise StopExperiment('IR changed frozen world-model namespace')
      if actor_before is not None and actor_before != _digest(
          agent.params, ACTOR_PREFIXES):
        raise StopExperiment('world-model-only episode changed frozen actor')
      if critic_before is not None and critic_before != _digest(
          agent.params, CRITIC_PREFIXES):
        raise StopExperiment('actor-only IR changed frozen critic namespace')
      probe = diagnose(trace, summary['label'])
      record_episode(trace, summary, probe)
      heldout_probe = diagnose_fixed(
          state.fixed_b_trajectory, f'fixed_b_guard_episode_{index}')
      check_divergence(heldout_probe)
      model_updates(int(cfg.episode_model_updates), f'b_episode_{index}')
      state.cir_completed = index
      if index in milestones and index not in state.eval_done:
        official_eval(f'milestone_{index}_b', cfg.action_scale_b)
        official_eval(f'milestone_{index}_a', cfg.action_scale_a)
        diagnose_fixed(
            state.fixed_b_trajectory, f'fixed_b_milestone_{index}')
        diagnose_fixed(
            state.fixed_a_trajectory, f'fixed_a_milestone_{index}')
        state.eval_done.append(index)
      save_checkpoint()

    state.phase = 'complete'
    state.status = 'complete'
    event('complete', baseline_mean=float(np.mean(state.baseline_scores)),
          shock_mean=float(np.mean(state.shock_scores)))
    save_checkpoint()
  except StopExperiment as error:
    state.status = 'stopped'
    event('stopped', reason=str(error))
    save_checkpoint()
    raise
  finally:
    logger.close()
