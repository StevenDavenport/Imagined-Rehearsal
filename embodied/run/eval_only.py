from collections import defaultdict
from functools import partial as bind
import json

import elements
import embodied
import numpy as np


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

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    nonlocal trace_episode, trace_episode_step, trace_return
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
  adapt_params = None
  adapt_policy_params = None
  adapt_every_k = int(getattr(adapt_cfg, 'every_k', 0))
  steps_since_adapt = 0
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
    discard = getattr(agent, 'discard_params', None)
    if discard:
      discard(adapt_policy_params)
      discard(adapt_params)
    adapt_params = None
    adapt_policy_params = None
    steps_since_adapt = 0

  def maybe_adapt(carry, acts, outs, batch_shape):
    nonlocal adapt_params, adapt_policy_params, steps_since_adapt
    first_trigger = adapt_params is None
    if adapt_params is None:
      adapt_params = agent.clone_params()
    adapt_params, carry, mets = agent.adapt(
        adapt_params, carry, adapt_cfg.steps,
        warmup=(first_trigger and bool(adapt_cfg.train_critic)),
        freeze_critic=not bool(adapt_cfg.train_critic))
    if adapt_policy_params is not None:
      discard = getattr(agent, 'discard_params', None)
      if discard:
        discard(adapt_policy_params)
    adapt_policy_params = agent.extract_policy_params(adapt_params)
    carry, acts, _ = agent.policy_latent(
        carry, params=adapt_policy_params, mode=agent_policy_mode)
    steps_since_adapt = 0
    outs['log/eval_adapt/trigger'] = np.full(batch_shape, 1.0, np.float32)
    outs['log/eval_adapt/steps'] = np.full(
        batch_shape, float(adapt_cfg.steps), np.float32)
    for key, value in mets.items():
      value = np.asarray(value)
      if value.ndim == 0:
        outs[f'log/eval_adapt/{key}'] = np.full(
            batch_shape, value.astype(np.float32), np.float32)
    return carry, acts

  def policy(carry, obs, **kwargs):
    nonlocal adapt_params, adapt_policy_params, steps_since_adapt
    carry, acts, outs = agent.policy(
        carry, obs, mode=agent_policy_mode, params=adapt_policy_params)
    if adapt_cfg.enabled:
      if obs['is_first'].shape[0] != 1:
        raise ValueError('eval_adapt requires eval envs=1')
      if obs['is_last'].any():
        # The driver masks terminal actions. Adapting here would spend compute
        # on an unused action and then immediately throw its parameters away.
        discard_adaptation()
      elif obs['is_first'].any():
        assert (
            adapt_params is None and
            adapt_policy_params is None and
            steps_since_adapt == 0
        ), 'eval_adapt state leaked across episode boundary'
        carry, acts = maybe_adapt(carry, acts, outs, obs['is_first'].shape)
      else:
        steps_since_adapt += 1
        if adapt_every_k > 0 and steps_since_adapt >= adapt_every_k:
          carry, acts = maybe_adapt(
              carry, acts, outs, obs['is_first'].shape)
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
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=(
            '^(?!(adapt_opt|adapt_actor_opt|adapt_critic_opt|model_opt)/)'))))
    digests = getattr(agent, 'parameter_digests', None)
    integrity_before = digests() if digests else None

    print(f'Start {requested_mode} evaluation')
    driver.reset(agent.init_policy)
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
    integrity_after = digests() if digests else None
    integrity = {
        'policy_mode': requested_mode,
        'adapt_enabled': bool(adapt_cfg.enabled),
        'before': integrity_before,
        'after': integrity_after,
        'equal': integrity_before == integrity_after,
    }
    (logdir / 'eval_integrity.json').write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + '\n')
    if integrity_before is not None and not integrity['equal']:
      raise RuntimeError(
          'Persistent agent parameters changed during evaluation; see '
          f'{logdir / "eval_integrity.json"}')
  finally:
    try:
      discard_adaptation()
      driver.close()
    finally:
      if trace_file:
        trace_file.close()
      logger.close()
