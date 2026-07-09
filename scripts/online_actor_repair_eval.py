#!/usr/bin/env python3
"""Evaluate actor-only online repair from real environment interactions.

This mirrors the existing eval-only rehearsal path, but replaces imagined
adaptation with actor-only updates on short real trajectories. It intentionally
keeps Dreamer training code unchanged and reuses the existing world-model
encoder/posterior path plus actor/value losses wherever possible.
"""

from __future__ import annotations

import os
import argparse
import pathlib
import sys
from collections import defaultdict
import math
from functools import partial as bind

folder = pathlib.Path(__file__).resolve().parent.parent / 'dreamerv3'
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))

import elements
import embodied
import numpy as np
import portal
import ruamel.yaml as yaml
import jax
import jax.numpy as jnp
import ninjax as nj

from dreamerv3 import main as dv3_main
from dreamerv3.agent import adapt_policy_loss, sample
from embodied.jax import internal, transform


f32 = jnp.float32
sg = jax.lax.stop_gradient


# Remove the leading env batch dimension from one-step driver outputs before
# storing them in the online update buffer. The buffers should hold per-step
# objects, and the [B, T, ...] layout gets rebuilt only when forming an update
# window.
def _strip_batch_dim(x):
  x = np.asarray(x)
  if x.ndim > 0 and x.shape[0] == 1:
    x = x[0]
  return x.copy()


# Copy nested numpy/JAX outputs into standalone host buffers so the online
# update window is independent of any device-backed arrays held by the driver.
def _copy_tree(tree):
  return jax.tree.map(_strip_batch_dim, tree)


# Build dummy zero-actions matching the environment action spaces. These are
# used to pad the final action slot so the real-trajectory policy loss aligns
# with Dreamer's existing [B, T+1] action convention.
def _zeros_like_act_space(act_space):
  return {k: np.zeros(v.shape, v.dtype) for k, v in act_space.items()}


class OnlineActorRepairUpdater:
  # Wrap an actor-only real-trajectory update in the same JAX transform style
  # used by the existing rehearsal path. This lets us reuse Dreamer modules,
  # sharding, and optimizer state without modifying core code.
  def __init__(self, agent):
    self.agent = agent
    self.model = agent.model
    tm, ts = agent.train_mirrored, agent.train_sharded
    tp = agent.train_params_sharding
    allo_sharding = {k: v for k, v in tp.items() if k in agent.policy_keys}
    dona_sharding = {k: v for k, v in tp.items() if k not in agent.policy_keys}
    _, ar = agent.partition_rules

    def update_fn(obs, prevact, act, seed=None):
      def lossfn(obs, prevact, act):
        loss, metrics = self._loss(obs, prevact, act)
        return loss, metrics
      metrics, _ = self.model.adapt_opt(
          lossfn, obs, prevact, act, has_aux=True)
      return (metrics,)

    self._update = transform.apply(
        nj.pure(update_fn), agent.train_mesh,
        (dona_sharding, allo_sharding, tm, ts, ts, ts),
        (tp, tm), ar,
        return_params=True,
        donate_params=True,
        use_shardmap=agent.jaxcfg.use_shardmap)

  # Rebuild posterior latents from real observations and compute an actor-only
  # policy loss using real rewards plus frozen value targets. Reward and
  # continuation heads are intentionally not used here.
  def _loss(self, obs, prevact, act):
    batch = obs['is_first'].shape[0]
    enc_carry = self.model.enc.initial(batch)
    dyn_carry = self.model.dyn.initial(batch)
    reset = obs['is_first']
    _, _, tokens = self.model.enc(enc_carry, obs, reset, training=False)
    _, _, repfeat = self.model.dyn.observe(
        dyn_carry, tokens, prevact, reset, training=False)

    inp = sg(self.model.feat2tensor(repfeat))
    rew = sg(obs['reward'])
    con = f32(~obs['is_terminal'])
    if self.model.config.contdisc:
      con *= 1 - 1 / self.model.config.horizon
    policy = self.model.pol(inp, 2)
    value = self.model.val(sg(inp), 2)
    slowvalue = self.model.slowval(sg(inp), 2)

    loss_cfg = self.model.config.imag_loss.copy()
    loss_cfg['actent'] = self.model.config.eval_adapt.actent
    loss, metrics = adapt_policy_loss(
        act, rew, con, policy, value, slowvalue,
        self.model.retnorm, self.model.valnorm, self.model.advnorm,
        contdisc=self.model.config.contdisc,
        horizon=self.model.config.horizon,
        **loss_cfg)
    if self.model.config.eval_adapt.lr != self.model.config.opt.lr:
      scale = self.model.config.eval_adapt.lr / self.model.config.opt.lr
      loss *= scale
    return loss, metrics

  # Run one actor-only optimizer update on a short real trajectory window and
  # return updated params plus scalar optimizer diagnostics.
  def update(self, params, obs, prevact, act):
    obs = internal.device_put(obs, self.agent.train_sharded)
    prevact = internal.device_put(prevact, self.agent.train_sharded)
    act = internal.device_put(act, self.agent.train_sharded)
    allo = {k: v for k, v in params.items() if k in self.agent.policy_keys}
    dona = {k: v for k, v in params.items() if k not in self.agent.policy_keys}
    seed = self.agent._seeds(self.agent.n_adapt, self.agent.train_mirrored)
    self.agent.n_adapt.increment()
    with self.agent.train_lock:
      params, metrics = self._update(dona, allo, seed, obs, prevact, act)
    metrics = self.agent._take_outs(internal.fetch_async(metrics))
    return params, metrics


# Stack a batch of sampled real observation windows into Dreamer's expected
# [B, T, ...] layout for actor-only online repair.
def _stack_obs(agent, obs_windows):
  keys = agent.obs_space.keys()
  return {
      k: np.stack([
          np.stack([np.asarray(obs[k]) for obs in window], 0)
          for window in obs_windows
      ], 0)
      for k in keys
  }


# Build the previous-action sequence aligned with each sampled observation
# window. The first slot uses zeros or the action just before the window.
def _stack_prevact(agent, act_list, starts, obs_count):
  zeros = _zeros_like_act_space(agent.act_space)
  out = {}
  for k in agent.act_space:
    batch = []
    for start in starts:
      seq = []
      if start == 0:
        seq.append(np.asarray(zeros[k]))
      else:
        seq.append(np.asarray(act_list[start - 1][k]))
      selected = act_list[start: start + max(0, obs_count - 1)]
      seq.extend(np.asarray(act[k]) for act in selected)
      batch.append(np.stack(seq, 0))
    out[k] = np.stack(batch, 0)
  return out


# Build the action sequence consumed by Dreamer's actor loss helper. The final
# slot is padded with zeros so the log-prob term matches the existing T-1 loss
# support used by adapt_policy_loss().
def _stack_policy_act(agent, act_list, starts, obs_count):
  zeros = _zeros_like_act_space(agent.act_space)
  out = {}
  for k in agent.act_space:
    batch = []
    for start in starts:
      selected = act_list[start: start + max(0, obs_count - 1)]
      seq = [np.asarray(act[k]) for act in selected]
      seq.append(np.asarray(zeros[k]))
      batch.append(np.stack(seq, 0))
    out[k] = np.stack(batch, 0)
  return out


# Sample a minibatch of short real trajectory windows from the current episode
# buffer. This is the key change that makes the online baseline less flimsy
# than a single highly correlated sliding-window update.
def _sample_update_batch(agent, obs_buffer, act_buffer, window, batch_size, rng):
  assert len(obs_buffer) >= window
  n_starts = len(obs_buffer) - window + 1
  batch_size = min(batch_size, n_starts)
  starts = rng.choice(n_starts, size=batch_size, replace=False)
  starts = np.sort(starts).tolist()
  obs_windows = [obs_buffer[s: s + window] for s in starts]
  obs = _stack_obs(agent, obs_windows)
  prevact = _stack_prevact(agent, act_buffer, starts, window)
  act = _stack_policy_act(agent, act_buffer, starts, window)
  return obs, prevact, act, starts


# Standalone eval loop for real-interaction actor repair. This mirrors the
# existing eval_only structure but swaps imagined updates for short real-window
# actor updates during the episode.
def online_actor_repair_eval(config, persist_actor=False):
  args = elements.Config(
      **config.run,
      replica=config.replica,
      replicas=config.replicas,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      report_length=config.report_length,
      consec_train=config.consec_train,
      consec_report=config.consec_report,
      replay_context=config.replay_context,
  )
  assert args.from_checkpoint, 'Need run.from_checkpoint'
  if args.envs != 1:
    raise ValueError('online actor repair requires run.envs=1')

  agent = dv3_main.make_agent(config)
  logger = dv3_main.make_logger(config)
  updater = OnlineActorRepairUpdater(agent)

  logdir = elements.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  agg = elements.Agg()
  epstats = elements.Agg()
  episodes = defaultdict(elements.Agg)
  should_log = elements.when.Clock(args.log_every)
  policy_fps = elements.FPS()

  elements.checkpoint.load(args.from_checkpoint, dict(
      agent=bind(agent.load, regex='^(?!adapt_opt/)')))

  online_params = None
  online_policy_params = None
  online_every_k = int(getattr(agent.config.eval_adapt, 'every_k', 1))
  online_window = int(getattr(agent.config.eval_adapt, 'imag_length', 1)) + 1
  online_batch_size = max(1, int(getattr(agent.config.eval_adapt, 'start_batch', 1)))
  online_opt_steps = max(1, int(getattr(agent.config.eval_adapt, 'steps', 1)))
  online_warmup = online_window
  steps_since_update = 0
  obs_buffer = []
  act_buffer = []
  rng = np.random.default_rng(int(config.seed))

  # Keep logging behavior close to eval_only so output metrics remain familiar
  # and comparable with the rehearsal experiments.
  @elements.timer.section('logfn')
  def logfn(tran, worker):
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

  # Trigger a small batch of actor updates from sampled real trajectory
  # windows within the current episode buffer, then expose comparable scalar
  # diagnostics through the logger.
  def maybe_online_update(batch_shape):
    nonlocal online_params, online_policy_params, steps_since_update
    if len(obs_buffer) < online_warmup:
      return {}
    if online_params is None:
      online_params = agent.clone_params()
    last_mets = {}
    sampled_batch = 0
    sampled_starts = []
    for _ in range(online_opt_steps):
      obs, prevact, act, starts = _sample_update_batch(
          agent, obs_buffer, act_buffer, online_window, online_batch_size, rng)
      sampled_batch = obs['is_first'].shape[0]
      sampled_starts = starts
      online_params, last_mets = updater.update(online_params, obs, prevact, act)
    online_policy_params = agent.extract_policy_params(online_params)
    steps_since_update = 0
    outs = {
        'log/online_repair/trigger': np.full(batch_shape, 1.0, np.float32),
        'log/online_repair/window': np.full(batch_shape, float(online_window), np.float32),
        'log/online_repair/batch_size': np.full(batch_shape, float(sampled_batch), np.float32),
        'log/online_repair/opt_steps': np.full(batch_shape, float(online_opt_steps), np.float32),
        'log/online_repair/buffer_size': np.full(batch_shape, float(len(obs_buffer)), np.float32),
        'log/online_repair/start_min': np.full(batch_shape, float(min(sampled_starts) if sampled_starts else 0), np.float32),
        'log/online_repair/start_max': np.full(batch_shape, float(max(sampled_starts) if sampled_starts else 0), np.float32),
        'log/online_repair/persist_actor': np.full(batch_shape, float(persist_actor), np.float32),
    }
    for key, value in last_mets.items():
      value = np.asarray(value)
      if value.ndim == 0:
        outs[f'log/online_repair/{key}'] = np.full(
            batch_shape, value.astype(np.float32), np.float32)
    return outs

  # Policy wrapper used by the driver. It resets temporary actor params at
  # episode start, accumulates real transitions, periodically updates the
  # actor, and then acts with the latest repaired policy parameters.
  def policy(carry, obs, **kwargs):
    nonlocal online_params, online_policy_params, steps_since_update
    if obs['is_first'].any():
      obs_buffer.clear()
      act_buffer.clear()
      steps_since_update = 0
      if not persist_actor:
        online_params = None
        online_policy_params = None
    obs_buffer.append(_copy_tree(obs))
    outs = {}
    if (not obs['is_first'].any() and online_every_k > 0 and
        steps_since_update >= online_every_k - 1):
      outs.update(maybe_online_update(obs['is_first'].shape))
    carry, acts, pol_outs = agent.policy(
        carry, obs, mode='eval', params=online_policy_params)
    outs.update(pol_outs)
    act_buffer.append(_copy_tree(acts))
    steps_since_update += 1
    return carry, acts, outs

  # Reuse the normal environment factory and logger plumbing so this runner
  # behaves like the rest of the evaluation tooling.
  fns = [bind(dv3_main.make_env, config, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=(not args.debug))
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(logfn)

  driver.reset(agent.init_policy)
  print('Start online actor repair evaluation')
  while step < args.steps:
    driver(policy, steps=10)
    if should_log(step):
      logger.add(agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

  logger.close()


# Parse the standard Dreamer config stack and hand off to the standalone
# online-repair evaluator without touching the existing run scripts.
def main(argv=None):
  from dreamerv3.agent import Agent
  [elements.print(line) for line in Agent.banner]

  argv = list(argv or sys.argv[1:])
  persist_actor = False
  cleaned = []
  i = 0
  while i < len(argv):
    if argv[i] == '--online_repair_persist_actor':
      if i + 1 >= len(argv):
        raise ValueError('--online_repair_persist_actor requires True/False')
      persist_actor = str(argv[i + 1]).lower() in ('1', 'true', 'yes', 'y')
      i += 2
      continue
    cleaned.append(argv[i])
    i += 1

  configs = elements.Path(folder / 'configs.yaml').read()
  configs = yaml.YAML(typ='safe').load(configs)
  parsed, other = elements.Flags(configs=['defaults']).parse_known(cleaned)
  config = elements.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(other)
  config = config.update(logdir=(
      config.logdir.format(timestamp=elements.timestamp())))

  if 'JOB_COMPLETION_INDEX' in os.environ:
    config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))
  print('Replica:', config.replica, '/', config.replicas)

  logdir = elements.Path(config.logdir)
  logdir.mkdir()
  config.save(logdir / 'config.yaml')

  def init():
    elements.timer.global_timer.enabled = config.logger.timer

  portal.setup(
      errfile=config.errfile and logdir / 'error',
      clientkw=dict(logging_color='cyan'),
      serverkw=dict(logging_color='cyan'),
      initfns=[init],
      ipv6=config.ipv6,
  )

  online_actor_repair_eval(config, persist_actor=persist_actor)


if __name__ == '__main__':
  raise SystemExit(main())
