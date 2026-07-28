import collections
from functools import partial as bind

import elements
import embodied
import numpy as np

from . import milestones


def _copy_scalar_logs(result):
  return {
      key: np.array(value, copy=True) for key, value in result.items()
      if key.startswith('log/') and np.asarray(value).ndim == 0
  }


def train(make_agent, make_replay, make_env, make_stream, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  @elements.timer.section('logfn')
  def logfn(tran, worker):
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
      episode_metrics = {
          'score': result.pop('score'),
          'length': result.pop('length'),
      }
      # The episode aggregate is also handed to epstats below, whose reducers
      # update NumPy arrays in place. Give asynchronous loggers independent
      # scalar ownership so later episodes cannot corrupt an already queued row.
      episode_metrics.update(_copy_scalar_logs(result))
      logger.add(episode_metrics, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(logfn)

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
  driver.on_step(trainfn)

  cp = elements.Checkpoint(
      logdir / 'ckpt', keep=int(getattr(args, 'ckpt_keep', 1)))
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()

  try:
    milestone_every = int(getattr(args, 'milestone_every', 0))
    milestone_root = str(getattr(args, 'milestone_dir', ''))
    if bool(milestone_every) != bool(milestone_root):
      raise ValueError(
          'run.milestone_every and run.milestone_dir must be set together')
    if milestone_every < 0:
      raise ValueError('run.milestone_every must be nonnegative')
    milestone_metadata = {
        'goal': str(getattr(args, 'milestone_goal', '')),
        'phase': int(getattr(args, 'milestone_phase', -1)),
        'spec_digest': str(getattr(args, 'milestone_spec_digest', '')),
    }
    next_milestone = None
    if milestone_every:
      current = int(step)
      for expected in range(milestone_every, current + 1, milestone_every):
        path = milestones.archive_path(milestone_root, expected)
        if path.exists():
          milestones.validate(path, expected_step=expected, full=False)
        elif expected == current:
          milestones.create(
              cp, replay, milestone_root, expected,
              metadata=milestone_metadata)
        else:
          raise milestones.MilestoneError(
              f'Missing past milestone {path}; step {current} can no longer '
              'reconstruct that exact training state')
      next_milestone = (
          (current // milestone_every) + 1) * milestone_every
  except BaseException:
    driver.close()
    logger.close()
    raise

  print('Start training loop')
  policy = lambda *args: agent.policy(*args, mode='train')
  driver.reset(agent.init_policy)
  try:
    while step < args.steps:

      driver(policy, steps=10)

      if (
          next_milestone is not None and
          next_milestone <= int(args.steps) and
          int(step) >= next_milestone
      ):
        if int(step) != next_milestone:
          raise milestones.MilestoneError(
              f'Training stepped past milestone {next_milestone}: {int(step)}')
        milestones.create(
            cp, replay, milestone_root, next_milestone,
            metadata=milestone_metadata)
        next_milestone += milestone_every

      if should_report(step) and len(replay):
        agg = elements.Agg()
        for _ in range(args.consec_report * args.report_batches):
          carry_report, mets = agent.report(carry_report, next(stream_report))
          agg.add(mets)
        logger.add(agg.result(), prefix='report')

      if should_log(step):
        logger.add(train_agg.result())
        logger.add(epstats.result(), prefix='epstats')
        logger.add(replay.stats(), prefix='replay')
        logger.add(usage.stats(), prefix='usage')
        logger.add({'fps/policy': policy_fps.result()})
        logger.add({'fps/train': train_fps.result()})
        logger.add({'timer': elements.timer.stats()['summary']})
        logger.write()

      if should_save(step):
        cp.save()

    # Wall-clock checkpoint intervals can be longer than calibration runs.
    # Persist the exact terminal action count so a larger follow-on budget does
    # not resume from an older mid-run checkpoint and duplicate experience.
    cp.save()

  except BaseException:
    # RLScape owns an external client/server bridge that can fail independently
    # of the learner. Preserve the last fully processed transition and optimizer
    # state so a supervisor can safely resume a long run.
    try:
      cp.save()
    except Exception as error:
      print(f'Could not save recovery checkpoint: {error}')
    raise

  finally:
    driver.close()
    logger.close()
