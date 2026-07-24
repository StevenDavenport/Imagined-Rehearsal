import collections
from functools import partial as bind

import elements
import embodied
import numpy as np


def train_eval(
    make_agent,
    make_replay_train,
    make_replay_eval,
    make_env_train,
    make_env_eval,
    make_stream,
    make_logger,
    args):

  agent = make_agent()
  replay_train = make_replay_train()
  replay_eval = make_replay_eval()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  agg = elements.Agg()
  train_episodes = collections.defaultdict(elements.Agg)
  train_epstats = elements.Agg()
  eval_episodes = collections.defaultdict(elements.Agg)
  eval_epstats = elements.Agg()
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = elements.when.Clock(args.log_every)
  should_report = elements.when.Clock(args.report_every)
  should_save = elements.when.Clock(args.save_every)

  @elements.timer.section('logfn')
  def logfn(tran, worker, mode):
    episodes = dict(train=train_episodes, eval=eval_episodes)[mode]
    epstats = dict(train=train_epstats, eval=eval_epstats)[mode]
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
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

  fns = [bind(make_env_train, i) for i in range(args.envs)]
  driver_train = embodied.Driver(fns, parallel=(not args.debug))
  driver_train.on_step(lambda tran, _: step.increment())
  driver_train.on_step(lambda tran, _: policy_fps.step())
  driver_train.on_step(replay_train.add)
  driver_train.on_step(bind(logfn, mode='train'))

  fns = [bind(make_env_eval, i) for i in range(args.eval_envs)]
  driver_eval = embodied.Driver(fns, parallel=(not args.debug))
  driver_eval.on_step(replay_eval.add)
  driver_eval.on_step(bind(logfn, mode='eval'))
  driver_eval.on_step(lambda tran, _: policy_fps.step())

  stream_train = iter(agent.stream(make_stream(replay_train, 'train')))
  stream_report = iter(agent.stream(make_stream(replay_train, 'report')))
  stream_eval = iter(agent.stream(make_stream(replay_eval, 'eval')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)
  carry_eval = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay_train) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay_train.update(outs['replay'])
      agg.add(mets, prefix='train')
  driver_train.on_step(trainfn)

  def reportfn(carry, stream):
    agg = elements.Agg()
    for _ in range(args.report_batches):
      batch = next(stream)
      carry, mets = agent.report(carry, batch)
      agg.add(mets)
    return carry, agg.result()

  cp = elements.Checkpoint(
      logdir / 'ckpt', keep=int(getattr(args, 'ckpt_keep', 1)))
  cp.step = step
  cp.agent = agent
  cp.replay_train = replay_train
  cp.replay_eval = replay_eval
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()
  should_save(step)  # Register that we just saved.

  print('Start training loop')
  train_policy = lambda *args: agent.policy(*args, mode='train')
  adapt_cfg = agent.config.eval_adapt
  adapt_params = None
  adapt_policy_params = None
  adapt_every_k = int(getattr(adapt_cfg, 'every_k', 0))
  steps_since_adapt = 0

  def maybe_adapt(carry, acts, outs, batch_shape):
    nonlocal adapt_params, adapt_policy_params, steps_since_adapt
    first_trigger = adapt_params is None
    if adapt_params is None:
      adapt_params = agent.clone_params()
    adapt_params, carry, mets = agent.adapt(
        adapt_params, carry, adapt_cfg.steps,
        warmup=(first_trigger and bool(adapt_cfg.train_critic)))
    adapt_policy_params = agent.extract_policy_params(adapt_params)
    carry, acts, _ = agent.policy_latent(carry, params=adapt_policy_params)
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

  def eval_policy(carry, obs, **kwargs):
    nonlocal adapt_params, adapt_policy_params, steps_since_adapt
    carry, acts, outs = agent.policy(
        carry, obs, mode='eval', params=adapt_policy_params)
    if adapt_cfg.enabled:
      if obs['is_first'].shape[0] != 1:
        raise ValueError('eval_adapt requires eval envs=1')
      if obs['is_first'].any():
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
      if obs['is_last'].any():
        adapt_params = None
        adapt_policy_params = None
        steps_since_adapt = 0
    return carry, acts, outs
  driver_train.reset(agent.init_policy)
  while step < args.steps:

    if should_report(step):
      print('Evaluation')
      driver_eval.reset(agent.init_policy)
      driver_eval(eval_policy, episodes=args.eval_eps)
      logger.add(eval_epstats.result(), prefix='epstats')
      if len(replay_train):
        carry_report, mets = reportfn(carry_report, stream_report)
        logger.add(mets, prefix='report')
      if len(replay_eval):
        carry_eval, mets = reportfn(carry_eval, stream_eval)
        logger.add(mets, prefix='eval')

    driver_train(train_policy, steps=10)

    if should_log(step):
      logger.add(agg.result())
      logger.add(train_epstats.result(), prefix='epstats')
      logger.add(replay_train.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()

  logger.close()
