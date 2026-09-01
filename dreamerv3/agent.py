import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import rssm

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


def select_policy_action(policy, mode, seed=None):
  if mode == 'train':
    if seed is not None:
      keys = sorted(policy)
      seeds = jax.random.split(seed, len(keys))
      return {
          key: policy[key].sample(action_seed)
          for key, action_seed in zip(keys, seeds)}
    return sample(policy)
  if mode == 'eval':
    return jax.tree.map(lambda dist: dist.pred(), policy)
  raise ValueError(f'Unknown policy mode {mode!r}; expected train or eval')


def categorical_entropy(prob):
  """Categorical entropy with numerically safe zero-probability handling."""
  prob = jnp.asarray(prob, f32)
  return -(prob * jnp.log(jnp.maximum(prob, 1e-8))).sum(-1)


def categorical_js(prob, axis=1):
  """Jensen--Shannon disagreement along an ensemble/sample axis."""
  prob = jnp.asarray(prob, f32)
  mixture = prob.mean(axis)
  # Round-off can make the entropy difference a few ulps negative even though
  # Jensen--Shannon divergence is nonnegative by definition.
  return jnp.maximum(
      categorical_entropy(mixture) - categorical_entropy(prob).mean(axis),
      0.0)


def normalized_categorical_entropy(prob):
  classes = int(prob.shape[-1])
  scale = jnp.log(jnp.asarray(max(classes, 2), f32))
  return jnp.clip(categorical_entropy(prob) / scale, 0.0, 1.0)


def categorical_js_pair(left, right):
  pair = jnp.stack([left, right], 1)
  return jnp.clip(
      categorical_js(pair, axis=1) / jnp.log(jnp.asarray(2.0, f32)),
      0.0, 1.0)


class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config
    self.goal_cfg = config.goal_conditioning
    self.goal_enabled = bool(self.goal_cfg.enabled)
    self.goal_key = str(self.goal_cfg.key)
    self.goal_count = int(self.goal_cfg.count)
    self.goal_exclude_keys = tuple(self.goal_cfg.exclude_keys)
    self.counterfactual_heads = config.counterfactual_heads
    self.counterfactual_heads_enabled = bool(
        self.counterfactual_heads.enabled)
    if self.goal_enabled:
      assert self.goal_count > 0, self.goal_count
      for key in (self.goal_key, *self.goal_exclude_keys):
        assert key in obs_space, (key, tuple(obs_space))
    if self.counterfactual_heads_enabled:
      if not self.goal_enabled or self.goal_count < 2:
        raise ValueError(
            'Counterfactual head training requires at least two goals')
      if float(self.counterfactual_heads.weight) < 0:
        raise ValueError('Counterfactual head loss weight must be nonnegative')

    exclude = {'is_first', 'is_last', 'is_terminal', 'reward'}
    if self.goal_enabled:
      exclude.update((self.goal_key, *self.goal_exclude_keys))
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    self.enc = {
        'simple': rssm.Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc')
    self.dyn = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
    self.dec = {
        'simple': rssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol')

    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)
    self.bounded_value_enabled = bool(config.bounded_value.enabled)
    self.bval = None
    if self.bounded_value_enabled:
      bounded_cfg = config.bounded_value.copy()
      bounded_cfg.pop('enabled')
      self.bval = embodied.jax.MLPHead(
          binary, **bounded_cfg, name='bval')

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    self.modules = [
        self.dyn, self.enc, self.dec, self.rew, self.con, self.pol, self.val]
    self.model_modules = [self.dyn, self.enc, self.dec, self.rew, self.con]
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')
    model_opt = config.opt.copy()
    model_opt.update(
        lr=config.model_lr, schedule='const', warmup=0, anneal=0)
    self.model_opt = embodied.jax.Optimizer(
        self.model_modules, self._make_opt(**model_opt), summary_depth=1,
        name='model_opt')
    self.adapt_opt = embodied.jax.Optimizer(
        self.pol, self._make_opt(**config.opt), summary_depth=1,
        name='adapt_opt')
    adapt_actor_opt = config.opt.copy()
    adapt_actor_opt.update(
        lr=config.eval_adapt.lr, schedule='const', warmup=0, anneal=0,
        update_rms_clip=float(getattr(
            config.eval_adapt, 'update_rms_clip', 0.0)))
    self.adapt_actor_opt = embodied.jax.Optimizer(
        self.pol, self._make_opt(**adapt_actor_opt), summary_depth=1,
        name='adapt_actor_opt')
    adapt_critic_opt = config.opt.copy()
    adapt_critic_opt.update(
        lr=config.eval_adapt.critic_lr, schedule='const', warmup=0, anneal=0)
    self.adapt_critic_opt = embodied.jax.Optimizer(
        self.val, self._make_opt(**adapt_critic_opt), summary_depth=1,
        name='adapt_critic_opt')
    self.head_repair_enabled = bool(config.head_repair.enabled)
    self.repair_head_opt = None
    self.repair_value_opt = None
    if self.head_repair_enabled:
      repair_opt = config.opt.copy()
      repair_opt.update(
          lr=config.head_repair.learning_rate, schedule='const',
          warmup=0, anneal=0)
      self.repair_head_opt = embodied.jax.Optimizer(
          [self.rew, self.con], self._make_opt(**repair_opt),
          summary_depth=1, name='repair_head_opt')
      if not self.bounded_value_enabled:
        raise ValueError(
            'head_repair requires agent.bounded_value.enabled=True')
      self.repair_value_opt = embodied.jax.Optimizer(
          self.bval, self._make_opt(**repair_opt),
          summary_depth=1, name='repair_value_opt')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    self.scales = scales
    self.model_scales = {
        k: v for k, v in scales.items()
        if k not in ('policy', 'value', 'repval')}

  @property
  def policy_keys(self):
    return '^(enc|dyn|dec|pol)/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    dyn_carry = self.dyn.initial(batch_size)
    dyn_carry = {**dyn_carry, 'logit': jnp.zeros_like(dyn_carry['stoch'])}
    if self.goal_enabled:
      dyn_carry[self.goal_key] = jnp.zeros((batch_size,), jnp.int32)
    return (
        self.enc.initial(batch_size),
        dyn_carry,
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_train(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_report(self, batch_size):
    return self.init_train(batch_size)

  def policy(self, carry, obs, mode='train'):
    return self._policy(carry, obs, mode=mode, action_seed=None)

  def policy_paired(self, carry, obs, action_seed, mode='train'):
    """Policy step with an explicit action-sampling key for paired assays."""
    return self._policy(carry, obs, mode=mode, action_seed=action_seed)

  def _policy(self, carry, obs, mode='train', action_seed=None):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    # Keep posterior logits in carry so adaptation can resample start states.
    dyn_carry = {**dyn_carry, 'logit': feat['logit']}
    goal = obs[self.goal_key] if self.goal_enabled else None
    if self.goal_enabled:
      dyn_carry[self.goal_key] = goal
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
    policy = self.pol(self._head_input(feat, goal), bdims=1)
    act = self._canonical_action(
        select_policy_action(policy, mode, action_seed))
    out = {}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    carry = (enc_carry, dyn_carry, dec_carry, act)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def policy_latent(self, carry, mode='train'):
    return self._policy_latent(carry, mode=mode, action_seed=None)

  def policy_latent_paired(self, carry, action_seed, mode='train'):
    """Latent policy step sharing the paired action key with policy_paired."""
    return self._policy_latent(carry, mode=mode, action_seed=action_seed)

  def _policy_latent(self, carry, mode='train', action_seed=None):
    enc_carry, dyn_carry, dec_carry, _ = carry
    policy = self.pol(self._head_input(dyn_carry), bdims=1)
    act = self._canonical_action(
        select_policy_action(policy, mode, action_seed))
    out = {}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(carry=carry, act=act)))
    carry = (enc_carry, dyn_carry, dec_carry, act)
    return carry, act, out

  def policy_stats(self, carry):
    """Return distribution parameters for behavioural drift diagnostics."""
    _, dyn_carry, _, _ = carry
    policy = self.pol(self._head_input(dyn_carry), bdims=1)
    stats = {}
    for key, dist in policy.items():
      stats[f'{key}/mean'] = dist.pred()
      stats[f'{key}/entropy'] = dist.entropy()
      base = getattr(dist, 'output', dist)
      if hasattr(base, 'logits'):
        stats[f'{key}/logits'] = base.logits
      if hasattr(base, 'stddev'):
        stats[f'{key}/stddev'] = base.stddev
    return stats

  def gate_features(self, carry, horizon=15, start_batch=128,
                    consequence=False):
    """Measure uncertainty and optionally prepare a reusable IR rollout.

    The cheap features use only posterior resampling plus actor forwards. The
    consequence branch additionally runs the ordinary IR imagination and
    returns it unchanged so a passing gate can reuse it for the actor update.
    All statistics are normalized to approximately [0, 1].
    """
    _, dyn_carry, _, _ = carry
    horizon = int(horizon)
    start_batch = max(1, int(start_batch))
    batch = int(dyn_carry['deter'].shape[0])
    goal = self._carry_goal(dyn_carry)

    factual = self.pol(self._head_input(dyn_carry, goal), 1)
    if 'logit' not in dyn_carry:
      raise KeyError('Stage 5A gating requires posterior logits in carry')
    logit = jnp.repeat(dyn_carry['logit'], start_batch, axis=0)
    sampled = {
        'deter': jnp.repeat(dyn_carry['deter'], start_batch, axis=0),
        'stoch': self.dyn.sample_posterior(logit),
    }
    sampled_goal = (
        jnp.repeat(goal, start_batch, axis=0)
        if self.goal_enabled else None)
    posterior_policy = self.pol(
        self._head_input(sampled, sampled_goal), 1)

    actual_entropy = []
    posterior_entropy = []
    posterior_js = []
    for key in sorted(posterior_policy):
      factual_base = getattr(factual[key], 'output', factual[key])
      posterior_base = getattr(
          posterior_policy[key], 'output', posterior_policy[key])
      if not (hasattr(factual_base, 'logits') and
              hasattr(posterior_base, 'logits')):
        raise ValueError(
            'Stage 5A gates require categorical actor distributions: '
            f'{key}')
      factual_prob = jax.nn.softmax(factual_base.logits, -1)
      posterior_prob = jax.nn.softmax(posterior_base.logits, -1)
      classes = int(posterior_prob.shape[-1])
      posterior_prob = posterior_prob.reshape(
          (batch, start_batch, classes))
      actual_entropy.append(normalized_categorical_entropy(factual_prob))
      posterior_entropy.append(
          normalized_categorical_entropy(posterior_prob).mean(1))
      posterior_js.append(
          jnp.clip(
              categorical_js(posterior_prob, axis=1) /
              jnp.log(jnp.asarray(max(classes, 2), f32)), 0.0, 1.0))

    metrics = {
        'actor_entropy': jnp.stack(actual_entropy).mean(),
        'posterior_entropy': jnp.stack(posterior_entropy).mean(),
        'js_disagreement': jnp.stack(posterior_js).mean(),
        'success_rate': jnp.asarray(0.0, f32),
        'success_action_concentration': jnp.asarray(0.0, f32),
        'success_action_divergence': jnp.asarray(0.0, f32),
        'reward_return_mean': jnp.asarray(0.0, f32),
        'reward_return_std': jnp.asarray(0.0, f32),
        'reward_return_sem': jnp.asarray(0.0, f32),
    }
    if not consequence:
      return {}, metrics

    rollout = self._adapt_rollout(
        carry, horizon=horizon, start_batch=start_batch)
    reward_threshold = float(getattr(
        self.config.eval_adapt.gate, 'reward_threshold', 0.5))
    success = (rollout['rew'].max(-1) >= reward_threshold).astype(f32)
    success = success.reshape((batch, start_batch))
    metrics['success_rate'] = success.mean()
    disc = 1 if self.config.contdisc else 1 - 1 / self.config.horizon
    zeros = jnp.zeros_like(rollout['rew'])
    reward_return = lambda_return(
        zeros, 1 - rollout['con'], rollout['rew'], zeros, zeros,
        disc, self.config.imag_loss.lam)[:, 0]
    metrics['reward_return_mean'] = reward_return.mean()
    metrics['reward_return_std'] = reward_return.std()
    metrics['reward_return_sem'] = (
        reward_return.std() /
        jnp.sqrt(jnp.asarray(max(int(reward_return.size), 1), f32)))

    # Hallucinated-reward diagnostics. These deliberately describe the raw
    # frozen reward-model predictions rather than the optimized objective so
    # an evaluator can detect an actor learning to exploit a spurious reward
    # attractor. All are scalars and therefore remain cheap to serialize in
    # the per-step Stage 5 trace.
    reward = rollout['rew']
    reward_peak = reward.max(-1)
    reward_peak_step = jnp.argmax(reward, -1)
    reward_event = reward_peak >= reward_threshold
    reward_event_count = reward_event.astype(f32).sum()
    first_event = jnp.argmax(
        (reward >= reward_threshold).astype(i32), -1)
    metrics.update({
        'reward_step_mean': reward.mean(),
        'reward_step_std': reward.std(),
        'reward_step_max': reward.max(),
        'reward_sum_mean': reward.sum(-1).mean(),
        'reward_peak_mean': reward_peak.mean(),
        'reward_peak_p95': jnp.quantile(reward_peak, 0.95),
        'reward_peak_timestep_mean': reward_peak_step.astype(f32).mean(),
        'reward_event_step_rate': (
            (reward >= reward_threshold).astype(f32).mean()),
        'reward_first_event_timestep_mean': (
            (first_event.astype(f32) * reward_event.astype(f32)).sum() /
            jnp.maximum(reward_event_count, 1.0)),
    })

    concentrations = []
    divergences = []
    first_policy = self.pol(
        jax.tree.map(lambda x: x[:, 0], rollout['inp']), 1)
    for key in sorted(first_policy):
      base = getattr(first_policy[key], 'output', first_policy[key])
      if not hasattr(base, 'logits'):
        raise ValueError(
            'Stage 5A consequence gates require categorical actions: '
            f'{key}')
      prob = jax.nn.softmax(base.logits, -1)
      classes = int(prob.shape[-1])
      prob = prob.reshape((batch, start_batch, classes))
      mixture = prob.mean(1)
      action = jnp.asarray(rollout['imgact'][key][:, 0], i32)
      action = action.reshape((batch, start_batch))
      weighted = jax.nn.one_hot(action, classes, dtype=f32) * success[..., None]
      count = success.sum(1, keepdims=True)
      empirical = weighted.sum(1) / jnp.maximum(count, 1.0)
      empirical = jnp.where(count > 0, empirical, mixture)
      concentrations.append(1.0 - normalized_categorical_entropy(empirical))
      divergences.append(categorical_js_pair(mixture, empirical))

      # Record both the overall imagined action attractor and the action at
      # the state immediately preceding the largest predicted reward. The
      # latter is diagnostic only: it does not alter the gate or actor loss.
      imagined_action = jnp.asarray(rollout['imgact'][key], i32)
      all_action_prob = jax.nn.one_hot(
          imagined_action.reshape((-1,)), classes, dtype=f32).mean(0)
      preceding_step = jnp.maximum(reward_peak_step - 1, 0)
      preceding_action = jnp.take_along_axis(
          imagined_action, preceding_step[:, None], axis=1)[:, 0]
      reward_action_count = (
          jax.nn.one_hot(preceding_action, classes, dtype=f32) *
          reward_event[:, None]).sum(0)
      reward_action_prob = reward_action_count / jnp.maximum(
          reward_event_count, 1.0)
      safe_key = str(key).replace('/', '_')
      metrics.update({
          f'imagined_action_{safe_key}_mode': jnp.argmax(
              all_action_prob).astype(f32),
          f'imagined_action_{safe_key}_concentration': all_action_prob.max(),
          f'imagined_action_{safe_key}_entropy': (
              normalized_categorical_entropy(all_action_prob)),
          f'reward_preceding_action_{safe_key}_mode': jnp.argmax(
              reward_action_count).astype(f32),
          f'reward_preceding_action_{safe_key}_concentration': jnp.where(
              reward_event_count > 0, reward_action_prob.max(), 0.0),
      })
    metrics['success_action_concentration'] = jnp.stack(
        concentrations).mean()
    metrics['success_action_divergence'] = jnp.stack(divergences).mean()
    return rollout, metrics

  def posterior_chunk(self, carry, obs, prevact):
    """Re-encode a recorded sequence chunk while preserving recurrent state."""
    enc_carry, dyn_carry = carry
    reset = obs['is_first']
    enc_carry, _, tokens = self.enc(
        enc_carry, obs, reset, training=False)
    dyn_carry, _, feat = self.dyn.observe(
        dyn_carry, tokens, self._canonical_action(prevact), reset,
        training=False)
    return (enc_carry, dyn_carry), feat

  def head_audit_chunk(self, carry, obs, prevact, actor_distribution=False):
    """Evaluate all goal-conditioned heads on one recorded sequence.

    The observations are encoded with the checkpoint under audit. Each
    posterior state is then repeated across every configured goal ID, keeping
    the latent fixed so differences can only come from goal conditioning in
    the heads. This function is inference-only and does not update parameters,
    normalizers, recurrent entries, or optimizer state.
    """
    enc_carry, dyn_carry = carry
    reset = obs['is_first']
    enc_carry, _, tokens = self.enc(
        enc_carry, obs, reset, training=False)
    dyn_carry, _, feat = self.dyn.observe(
        dyn_carry, tokens, self._canonical_action(prevact), reset,
        training=False)

    if not self.goal_enabled:
      raise ValueError('head_audit_chunk requires goal conditioning')
    goals = jnp.arange(self.goal_count, dtype=jnp.int32)
    goals = jnp.broadcast_to(
        goals, (*feat['deter'].shape[:2], self.goal_count))
    swept = {
        key: jnp.broadcast_to(
            value[:, :, None],
            (*value.shape[:2], self.goal_count, *value.shape[2:]))
        for key, value in feat.items()
    }
    inp = sg(self._head_input(swept, goals))
    reward = self.rew(inp, 3)
    continuation = self.con(inp, 3)
    continuation_prob = continuation.prob(1)
    epsilon = jnp.finfo(continuation_prob.dtype).eps
    continuation_safe = jnp.clip(
        continuation_prob, epsilon, 1 - epsilon)
    continuation_entropy = -(
        continuation_safe * jnp.log(continuation_safe) +
        (1 - continuation_safe) * jnp.log(1 - continuation_safe))
    value = self.val(inp, 3)
    slowvalue = self.slowval(inp, 3)
    voffset, vscale = self.valnorm.stats()

    logits = feat['logit']
    probs = jax.nn.softmax(logits, -1)
    posterior_entropy = -jnp.sum(
        probs * jnp.log(jnp.maximum(probs, 1e-8)), -1).mean(-1)
    result = {
        'reward': reward.pred(),
        'reward_entropy': reward.entropy(),
        'continuation': continuation_prob,
        # Binary intentionally implements logp/prob but not entropy in
        # embodied.jax.outs. Compute the exact Bernoulli entropy here.
        'continuation_entropy': continuation_entropy,
        'value': value.pred() * vscale + voffset,
        'slowvalue': slowvalue.pred() * vscale + voffset,
        'value_entropy': value.entropy(),
        'slowvalue_entropy': slowvalue.entropy(),
        'posterior_entropy': posterior_entropy,
        'deter_rms': jnp.sqrt(jnp.mean(jnp.square(feat['deter']), -1)),
        'stoch_rms': jnp.sqrt(
            jnp.mean(jnp.square(feat['stoch']), (-2, -1))),
    }
    if actor_distribution:
      policy = self.pol(inp, 3)
      if len(policy) != 1:
        raise ValueError(
            'Actor-distribution head audit requires one action branch, got '
            f'{tuple(policy)}')
      distribution = next(iter(policy.values()))
      distribution = getattr(distribution, 'output', distribution)
      distribution = getattr(distribution, 'dist', distribution)
      if not hasattr(distribution, 'logits'):
        raise ValueError(
            'Actor-distribution head audit requires a categorical policy')
      probability = jax.nn.softmax(distribution.logits, -1)
      result['actor_probability'] = probability
      result['actor_entropy'] = -jnp.sum(
          probability * jnp.log(jnp.maximum(probability, 1e-8)), -1)
    if getattr(self, 'bounded_value_enabled', False):
      result['bounded_value'] = self.bval(inp, 3).prob(1)
    return (enc_carry, dyn_carry), result

  def repair_reward_continuation(self, batch, counterfactual=False):
    """Update cloned reward/continuation heads on frozen posterior features."""
    if not self.head_repair_enabled:
      raise ValueError('Head repair is disabled')
    goals = jnp.asarray(batch['actual_goal'], jnp.int32)
    query_count = self.goal_count
    if counterfactual:
      queries = jnp.broadcast_to(
          jnp.arange(query_count, dtype=jnp.int32),
          (goals.shape[0], query_count))
    else:
      queries = jnp.broadcast_to(goals[:, None], (goals.shape[0], query_count))
    feat = {
        key: jnp.broadcast_to(
            batch[key][:, None],
            (batch[key].shape[0], query_count, *batch[key].shape[1:]))
        for key in ('deter', 'stoch')}
    inp = sg(self._head_input(feat, queries))
    matching = queries == goals[:, None]
    reward = jnp.asarray(batch['reward'], f32)[:, None]
    if counterfactual:
      reward_target = reward * f32(matching)
      physical = jnp.asarray(batch['physical_terminal'], bool)[:, None]
      complete = jnp.asarray(batch['goal_complete'], bool)[:, None]
      terminal = physical | (complete & matching)
      continuation_target = f32(~terminal)
    else:
      reward_target = jnp.broadcast_to(reward, matching.shape)
      terminal = jnp.asarray(batch['is_terminal'], bool)[:, None]
      continuation_target = jnp.broadcast_to(f32(~terminal), matching.shape)
    if self.config.contdisc:
      continuation_target *= 1 - 1 / float(self.config.horizon)

    def lossfn(_):
      reward_dist = self.rew(inp, 2)
      continuation_dist = self.con(inp, 2)
      reward_loss = reward_dist.loss(reward_target)
      continuation_loss = continuation_dist.loss(continuation_target)
      loss = reward_loss.mean() + continuation_loss.mean()
      metrics = {
          'reward_loss': reward_loss.mean(),
          'continuation_loss': continuation_loss.mean(),
          'reward_prediction': reward_dist.pred().mean(),
          'continuation_prediction': continuation_dist.prob(1).mean(),
          'reward_target': reward_target.mean(),
          'continuation_target': continuation_target.mean(),
      }
      event = jnp.asarray(batch['goal_complete'], bool)
      event_mask = event[:, None]
      prediction = reward_dist.pred()
      metrics['event_reward_matching'] = (
          (prediction * f32(event_mask & matching)).sum() /
          jnp.maximum((event_mask & matching).sum(), 1))
      metrics['event_reward_nonmatching'] = (
          (prediction * f32(event_mask & ~matching)).sum() /
          jnp.maximum((event_mask & ~matching).sum(), 1))
      return loss, metrics

    metrics, mets = self.repair_head_opt(
        lossfn, batch, has_aux=True)
    metrics.update(mets)
    return (metrics,)

  def repair_bounded_value(self, batch):
    """Fit a Bernoulli success-value head to factual Monte Carlo returns."""
    if not self.head_repair_enabled or not self.bounded_value_enabled:
      raise ValueError('Bounded-value repair is disabled')
    goal = jnp.asarray(batch['actual_goal'], jnp.int32)
    feat = {key: batch[key] for key in ('deter', 'stoch')}
    inp = sg(self._head_input(feat, goal))
    target = jnp.clip(jnp.asarray(batch['value_target'], f32), 0, 1)

    def lossfn(_):
      dist = self.bval(inp, 1)
      prediction = dist.prob(1)
      losses = dist.loss(target)
      loss = losses.mean()
      return loss, {
          'bounded_value_loss': loss,
          'bounded_value_prediction': prediction.mean(),
          'bounded_value_target': target.mean(),
          'bounded_value_mae': jnp.abs(prediction - target).mean(),
          'bounded_value_min': prediction.min(),
          'bounded_value_max': prediction.max(),
      }

    metrics, mets = self.repair_value_opt(lossfn, batch, has_aux=True)
    metrics.update(mets)
    return (metrics,)

  def imagination_audit(self, start, goal, horizon=6, start_batch=128):
    """Reproduce the actor-IR imagination objective without updating it.

    ``start`` contains a batch of posterior RSSM states and ``goal`` contains
    one goal ID per state. Posterior sampling, policy sampling, the H+1 state
    convention, lambda returns, normalizer use, and entropy coefficient match
    the actor-only evaluation adaptation path exactly. Outputs retain the
    MCPB posterior-sample axis so rare optimistic trajectories remain visible.
    """
    horizon = int(horizon)
    start_batch = int(start_batch)
    if horizon < 1 or start_batch < 1:
      raise ValueError((horizon, start_batch))
    batch = start['deter'].shape[0]
    logit = jnp.repeat(start['logit'], start_batch, axis=0)
    imagined_start = {
        'deter': jnp.repeat(start['deter'], start_batch, axis=0),
        'stoch': self.dyn.sample_posterior(logit),
    }
    goal = jnp.repeat(jnp.asarray(goal, jnp.int32), start_batch, axis=0)
    policyfn = lambda feat: self._canonical_action(
        sample(self.pol(self._head_input(feat, goal), 1)))
    _, future_feat, imgact = self.dyn.imagine(
        imagined_start, policyfn, horizon, training=True)
    first = dict(
        deter=imagined_start['deter'][:, None],
        stoch=imagined_start['stoch'][:, None],
        logit=jnp.zeros_like(future_feat['logit'][:, :1]),
    )
    imgfeat = concat([first, future_feat], 1)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgact, lastact], 1)

    return self._imagination_audit_outputs(
        imgfeat, imgact, goal, batch, start_batch)

  def recorded_imagination_audit(
      self, start, goal, actions, horizon=6, start_batch=128):
    """Audit the IR return under a recorded-action proposal control.

    The MCPB posterior sampling and all learned heads match
    :meth:`imagination_audit`, but the first ``horizon`` actions come from the
    replay episode rather than the checkpoint actor. This isolates optimism
    generally present in the learned model/value system from optimism induced
    by actor-selected imagined trajectories.
    """
    horizon = int(horizon)
    start_batch = int(start_batch)
    if horizon < 1 or start_batch < 1:
      raise ValueError((horizon, start_batch))
    batch = start['deter'].shape[0]
    logit = jnp.repeat(start['logit'], start_batch, axis=0)
    imagined_start = {
        'deter': jnp.repeat(start['deter'], start_batch, axis=0),
        'stoch': self.dyn.sample_posterior(logit),
    }
    goal = jnp.repeat(jnp.asarray(goal, jnp.int32), start_batch, axis=0)
    actions = self._canonical_action(jax.tree.map(
        lambda value: jnp.repeat(value, start_batch, axis=0), actions))
    _, future_feat, imgact = self.dyn.imagine(
        imagined_start, actions, horizon, training=False)
    first = dict(
        deter=imagined_start['deter'][:, None],
        stoch=imagined_start['stoch'][:, None],
        logit=jnp.zeros_like(future_feat['logit'][:, :1]),
    )
    imgfeat = concat([first, future_feat], 1)
    policyfn = lambda feat: self._canonical_action(
        sample(self.pol(self._head_input(feat, goal), 1)))
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgact, lastact], 1)
    return self._imagination_audit_outputs(
        imgfeat, imgact, goal, batch, start_batch)

  def _imagination_audit_outputs(
      self, imgfeat, imgact, goal, batch, start_batch):
    """Evaluate the common learned IR target on an imagined trajectory."""
    horizon = imgfeat['deter'].shape[1] - 1

    imggoal = self._repeat_goal(goal, horizon + 1)
    inp = sg(self._head_input(imgfeat, imggoal))
    reward = sg(self.rew(inp, 2).pred())
    continuation = sg(self._continuation(inp, 2))
    policy = self.pol(inp, 2)
    value_dist = self.val(inp, 2)
    slowvalue_dist = self.slowval(inp, 2)
    voffset, vscale = self.valnorm.stats()
    value = value_dist.pred() * vscale + voffset
    slowvalue = slowvalue_dist.pred() * vscale + voffset
    target_value = (
        slowvalue if bool(self.config.imag_loss.slowtar) else value)
    discount = (
        1.0 if bool(self.config.contdisc)
        else 1 - 1 / float(self.config.horizon))
    last = jnp.zeros_like(continuation)
    terminal = 1 - continuation
    lam = float(self.config.imag_loss.lam)
    returns = lambda_return(
        last, terminal, reward, target_value, target_value, discount, lam)
    zeros = jnp.zeros_like(reward)
    reward_only_mc = lambda_return(
        zeros, terminal, reward, zeros, zeros, discount, 1.0)
    reward_only_lambda = lambda_return(
        zeros, terminal, reward, zeros, zeros, discount, lam)
    bootstrap = returns - reward_only_lambda

    weight = jnp.cumprod(discount * continuation, 1) / discount
    roffset, rscale = self.retnorm(returns, update=False)
    advantage = (returns - target_value[:, :-1]) / rscale
    aoffset, ascale = self.advnorm(advantage, update=False)
    advantage_normalized = (advantage - aoffset) / ascale
    logpi = sum([
        dist.logp(sg(imgact[key]))[:, :-1]
        for key, dist in policy.items()])
    entropies = {
        key: dist.entropy()[:, :-1] for key, dist in policy.items()}
    policy_entropy = sum(entropies.values())
    reinforce_loss = (
        sg(weight[:, :-1]) * -logpi * sg(advantage_normalized))
    entropy_loss = (
        sg(weight[:, :-1]) *
        -float(self.config.eval_adapt.actent) * policy_entropy)

    future_feat = jax.tree.map(lambda x: x[:, 1:], imgfeat)
    prior_prob = jax.nn.softmax(future_feat['logit'], -1)
    prior_entropy = -jnp.sum(
        prior_prob * jnp.log(jnp.maximum(prior_prob, 1e-8)), -1).mean(-1)
    result = {
        'reward': reward,
        'continuation': continuation,
        'value': value,
        'slowvalue': slowvalue,
        'target_value': target_value,
        'discount': jnp.ones_like(reward) * discount,
        'lambda': jnp.ones_like(reward) * lam,
        'return': returns,
        'reward_only_mc': reward_only_mc,
        'reward_only_lambda': reward_only_lambda,
        'bootstrap_contribution': bootstrap,
        'advantage': advantage,
        'advantage_normalized': advantage_normalized,
        'weight': weight[:, :-1],
        'logpi': logpi,
        'policy_entropy': policy_entropy,
        'reinforce_loss': reinforce_loss,
        'entropy_loss': entropy_loss,
        'prior_entropy': prior_entropy,
        'deter_rms': jnp.sqrt(
            jnp.mean(jnp.square(future_feat['deter']), -1)),
        'stoch_rms': jnp.sqrt(
            jnp.mean(jnp.square(future_feat['stoch']), (-2, -1))),
        'action': jax.tree.map(lambda x: x[:, :-1], imgact),
    }
    reshape = lambda x: x.reshape((batch, start_batch, *x.shape[1:]))
    return jax.tree.map(reshape, result)

  def critic_state(self, carry):
    """Return frozen-head diagnostics for one posterior policy carry."""
    _, dyn_carry, _, _ = carry
    inp = sg(self._head_input(dyn_carry))
    voffset, vscale = self.valnorm.stats()
    value = self.val(inp, 1)
    slowvalue = self.slowval(inp, 1)
    result = {
        'reward': self.rew(inp, 1).pred(),
        'continuation': self._continuation(inp, 1),
        'value': value.pred() * vscale + voffset,
        'slowvalue': slowvalue.pred() * vscale + voffset,
        'value_entropy': value.entropy(),
        'slowvalue_entropy': slowvalue.entropy(),
        'deter': dyn_carry['deter'],
        'stoch': dyn_carry['stoch'],
        'logit': dyn_carry['logit'],
    }
    return result

  def critic_rollout(self, carry, horizon=6, start_batch=1):
    """Generate a fixed-policy imagination bank without modifying parameters."""
    _, dyn_carry, _, _ = carry
    start_batch = max(1, int(start_batch))
    start = {
        'deter': dyn_carry['deter'],
        'stoch': dyn_carry['stoch'],
    }
    if start_batch > 1:
      logit = jnp.repeat(dyn_carry['logit'], start_batch, axis=0)
      start['deter'] = jnp.repeat(start['deter'], start_batch, axis=0)
      start['stoch'] = self.dyn.sample_posterior(logit)
    goal = self._carry_goal(dyn_carry)
    if self.goal_enabled and start_batch > 1:
      goal = jnp.repeat(goal, start_batch, axis=0)
    policyfn = lambda feat: self._canonical_action(
        sample(self.pol(self._head_input(feat, goal), 1)))
    _, imgfeat, imgact = self.dyn.imagine(
        start, policyfn, int(horizon), training=False)
    imggoal = self._repeat_goal(goal, int(horizon))
    inp = sg(self._head_input(imgfeat, imggoal))
    voffset, vscale = self.valnorm.stats()
    result = {
        'reward': self.rew(inp, 2).pred(),
        'continuation': self._continuation(inp, 2),
        'value': self.val(inp, 2).pred() * vscale + voffset,
        'slowvalue': self.slowval(inp, 2).pred() * vscale + voffset,
        'value_entropy': self.val(inp, 2).entropy(),
        'slowvalue_entropy': self.slowval(inp, 2).entropy(),
        'deter': imgfeat['deter'],
        'stoch': imgfeat['stoch'],
        'logit': imgfeat['logit'],
        'action': imgact,
    }
    return result

  def critic_recorded_rollout(self, carry, actions):
    """Imagine a supplied action sequence for open-loop fidelity probes."""
    _, dyn_carry, _, _ = carry
    start = {'deter': dyn_carry['deter'], 'stoch': dyn_carry['stoch']}
    horizon = jax.tree.leaves(actions)[0].shape[1]
    _, imgfeat, imgact = self.dyn.imagine(
        start, self._canonical_action(actions), int(horizon), training=False)
    goal = self._carry_goal(dyn_carry)
    imggoal = self._repeat_goal(goal, int(horizon))
    inp = sg(self._head_input(imgfeat, imggoal))
    voffset, vscale = self.valnorm.stats()
    result = {
        'reward': self.rew(inp, 2).pred(),
        'continuation': self._continuation(inp, 2),
        'value': self.val(inp, 2).pred() * vscale + voffset,
        'slowvalue': self.slowval(inp, 2).pred() * vscale + voffset,
        'value_entropy': self.val(inp, 2).entropy(),
        'slowvalue_entropy': self.slowval(inp, 2).entropy(),
        'deter': imgfeat['deter'],
        'stoch': imgfeat['stoch'],
        'logit': imgfeat['logit'],
        'action': imgact,
    }
    return result

  def train(self, carry, data):
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True)
    metrics.update(mets)
    self.slowval.update()
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    # if self.config.replay.fracs.priority > 0:
    #   outs['replay']['priority'] = losses['model']
    carry = (*carry, self._canonical_action(
        {k: data[k][:, -1] for k in self.act_space}))
    if nj.creating():
      def dummy(carry):
        _, dyn_carry, _, _ = carry
        policy = self.pol(self._head_input(dyn_carry), bdims=1)
        loss = sum([
            v.logp(jnp.zeros_like(v.pred())).mean()
            for v in policy.values()])
        return loss, (carry, {})
      self.adapt_opt(dummy, carry, has_aux=True)
      def dummy_actor(carry):
        _, dyn_carry, _, _ = carry
        policy = self.pol(self._head_input(dyn_carry), bdims=1)
        loss = sum([
            0.0 * v.logp(jnp.zeros_like(v.pred())).mean()
            for v in policy.values()])
        return loss, (carry, {})
      def dummy_critic(carry):
        _, dyn_carry, _, _ = carry
        value = self.val(self._head_input(dyn_carry), bdims=1)
        return 0.0 * value.pred().mean(), (carry, {})
      self.adapt_actor_opt(dummy_actor, carry, has_aux=True)
      self.adapt_critic_opt(dummy_critic, carry, has_aux=True)
      def dummy_model(carry, obs, prevact, training):
        loss, aux = self.model_loss(carry, obs, prevact, training)
        return 0.0 * loss, aux
      self.model_opt(
          dummy_model, carry[:3], obs, prevact,
          training=True, has_aux=True)
      if self.bounded_value_enabled:
        _, dyn_carry, _, _ = carry
        # The bounded head is intentionally outside the ordinary Dreamer
        # optimizer. Materialize it nonetheless so exact audit/evaluation
        # checkpoint loads have a stable parameter tree.
        self.bval(self._head_input(dyn_carry), bdims=1).pred()
      if self.head_repair_enabled:
        B = obs['is_first'].shape[0]
        repair_batch = {
            'deter': jnp.zeros((B, self.dyn.deter), f32),
            'stoch': jnp.zeros(
                (B, self.dyn.stoch, self.dyn.classes), f32),
            'actual_goal': jnp.zeros((B,), i32),
            'reward': jnp.zeros((B,), f32),
            'is_terminal': jnp.zeros((B,), bool),
            'physical_terminal': jnp.zeros((B,), bool),
            'goal_complete': jnp.zeros((B,), bool),
            'value_target': jnp.zeros((B,), f32),
        }
        repair_inp = self._head_input(
            {'deter': repair_batch['deter'],
             'stoch': repair_batch['stoch']},
            repair_batch['actual_goal'])
        def dummy_repair(_):
          loss = (
              self.rew(repair_inp, 1).loss(
                  repair_batch['reward']).mean() +
              self.con(repair_inp, 1).loss(
                  f32(~repair_batch['is_terminal'])).mean())
          return 0.0 * loss, {}
        def dummy_bounded(_):
          loss = self.bval(repair_inp, 1).loss(
              repair_batch['value_target']).mean()
          return 0.0 * loss, {}
        self.repair_head_opt(
            dummy_repair, repair_batch, has_aux=True)
        self.repair_value_opt(
            dummy_bounded, repair_batch, has_aux=True)
    return carry, outs, metrics

  def train_model(self, carry, data):
    """Train only encoder, RSSM, decoder, reward, and continuation heads."""
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    metrics, (carry, entries, outs, mets) = self.model_opt(
        self.model_loss, carry, obs, prevact, training=True, has_aux=True)
    metrics.update(mets)
    outputs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outputs['replay'] = updates
    carry = (*carry, self._canonical_action(
        {k: data[k][:, -1] for k in self.act_space}))
    return carry, outputs, metrics

  def model_loss(self, carry, obs, prevact, training):
    """World-model objective without imagination or behaviour gradients."""
    enc_carry, dyn_carry, dec_carry = carry
    prevact = self._canonical_action(prevact)
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    head_losses, head_metrics = self._reward_continuation_losses(
        repfeat, obs)
    losses.update(head_losses)
    metrics.update(head_metrics)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))

    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)
    if self.goal_enabled and bool(getattr(
        self.config, 'goal_loss_metrics', False)):
      goal_ids = jnp.asarray(obs[self.goal_key], i32)
      for goal_id in range(self.goal_count):
        mask = f32(goal_ids == goal_id)
        count = jnp.maximum(mask.sum(), 1.0)
        metrics[f'batch_goal/{goal_id}'] = mask.mean()
        for key, value in losses.items():
          metrics[f'loss_by_goal/{goal_id}/{key}'] = (
              (value * mask).sum() / count)
    assert set(losses) == set(self.model_scales), (
        sorted(losses), sorted(self.model_scales))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum(v.mean() * self.model_scales[k] for k, v in losses.items())
    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def loss(self, carry, obs, prevact, training):
    enc_carry, dyn_carry, dec_carry = carry
    prevact = self._canonical_action(prevact)
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    head_losses, head_metrics = self._reward_continuation_losses(
        repfeat, obs)
    losses.update(head_losses)
    metrics.update(head_metrics)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)
    start_goal = (
        obs[self.goal_key][:, -K:].reshape((B * K,))
        if self.goal_enabled else None)
    policyfn = lambda feat: self._canonical_action(
        sample(self.pol(self._head_input(feat, start_goal), 1)))
    _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
    first = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
    imggoal = self._repeat_goal(start_goal, H + 1)
    inp = self._head_input(imgfeat, imggoal)
    los, imgloss_out, mets = imag_loss(
        imgact,
        self.rew(inp, 2).pred(),
        self._continuation(inp, 2),
        self.pol(inp, 2),
        self.val(inp, 2),
        self.slowval(inp, 2),
        self.retnorm, self.valnorm, self.advnorm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)

    # Replay
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      replay_goal = obs[self.goal_key][:, -K:] if self.goal_enabled else None
      inp = self._head_input(feat, replay_goal)
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    if self.goal_enabled and bool(getattr(
        self.config, 'goal_loss_metrics', False)):
      goal_ids = jnp.asarray(obs[self.goal_key], i32)
      for goal_id in range(self.goal_count):
        for key, value in losses.items():
          selected_goals = goal_ids[:, -value.shape[1]:]
          mask = f32(selected_goals == goal_id)
          metrics[f'loss_by_goal/{goal_id}/{key}'] = (
              (value * mask).sum() / jnp.maximum(mask.sum(), 1.0))
        metrics[f'batch_goal/{goal_id}'] = f32(
            goal_ids == goal_id).mean()
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry, dec_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, training=False)
    mets.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(obs['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Video preds
    for key in self.dec.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      B, T, H, W, C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
      metrics[f'openloop/{key}'] = grid

    carry = (*new_carry, self._canonical_action(
        {k: data[k][:, -1] for k in self.act_space}))
    return carry, metrics

  def adapt(self, carry, critic_only=False, freeze_critic=False):
    if not self.config.eval_adapt.enabled:
      return carry, {}

    # The frozen-critic branch deliberately uses the same dedicated actor
    # optimizer and factored rollout as critic-first IR. This makes it a clean
    # ablation: the critic update is the only removed operation. The legacy
    # actor-only path remains available when both flags are false.
    if self.config.eval_adapt.train_critic or freeze_critic:
      rollout = self._adapt_rollout(carry)
      metrics = {}

      if not freeze_critic:
        def critic_lossfn(carry):
          loss, metrics = self._adapt_critic_loss(rollout)
          return loss, (carry, metrics)

        metrics, (carry, mets) = self.adapt_critic_opt(
            critic_lossfn, carry, has_aux=True)
        metrics.update(prefix(mets, 'adapt_critic'))
        self.slowval.update()
      if critic_only:
        if freeze_critic:
          raise ValueError('critic_only adaptation cannot update a frozen critic')
        return carry, metrics

      def actor_lossfn(carry):
        objective = getattr(
            self, '_adapt_actor_objective', self._adapt_actor_loss)
        loss, metrics = objective(rollout)
        return loss, (carry, metrics)

      actor_metrics, (carry, mets) = self.adapt_actor_opt(
          actor_lossfn, carry, has_aux=True)
      metrics.update(actor_metrics)
      metrics.update(prefix(mets, 'adapt_actor'))
      metrics.update(self._adapt_viewer_metrics(rollout['imgfeat']))
      return carry, metrics

    if critic_only:
      raise ValueError('critic_only adaptation requires train_critic=True')

    def lossfn(carry):
      loss, metrics = self._adapt_loss(carry)
      return loss, (carry, metrics)

    metrics, (carry, mets) = self.adapt_opt(lossfn, carry, has_aux=True)
    metrics.update(prefix(mets, 'adapt'))
    return carry, metrics

  def adapt_prepared(self, carry, rollout):
    """Apply actor-only IR to a rollout already computed by a gate.

    Stage 5A uses this path to ensure that consequence gating does not pay for
    one imagination to decide and a second identical imagination to update.
    """
    if bool(self.config.eval_adapt.train_critic):
      raise ValueError('Prepared Stage 5A adaptation requires a frozen critic')

    def actor_lossfn(carry):
      loss, metrics = self._adapt_actor_objective(rollout)
      return loss, (carry, metrics)

    metrics, (carry, mets) = self.adapt_actor_opt(
        actor_lossfn, carry, has_aux=True)
    metrics.update(prefix(mets, 'adapt_actor'))
    metrics.update(self._adapt_viewer_metrics(rollout['imgfeat']))
    return carry, metrics

  def adapt_critic_horizon(self, carry, horizon=6, start_batch=1):
    """Critic-only update with explicit static rollout controls for ablations."""
    rollout = self._adapt_rollout(
        carry, horizon=int(horizon), start_batch=int(start_batch))

    def critic_lossfn(carry):
      loss, metrics = self._adapt_critic_loss(rollout)
      return loss, (carry, metrics)

    metrics, (carry, mets) = self.adapt_critic_opt(
        critic_lossfn, carry, has_aux=True)
    metrics.update(prefix(mets, 'adapt_critic'))
    self.slowval.update()
    return carry, metrics

  def _adapt_rollout(self, carry, horizon=None, start_batch=None):
    _, dyn_carry, _, _ = carry
    H = int(horizon or self.config.eval_adapt.imag_length)
    start_batch = max(1, int(
        start_batch or self.config.eval_adapt.start_batch))

    start = {
        'deter': dyn_carry['deter'],
        'stoch': dyn_carry['stoch'],
    }
    if start_batch > 1:
      if 'logit' not in dyn_carry:
        raise KeyError(
            'Missing posterior logits for eval_adapt.start_batch > 1.')
      logit = jnp.repeat(dyn_carry['logit'], start_batch, axis=0)
      start['deter'] = jnp.repeat(start['deter'], start_batch, axis=0)
      start['stoch'] = self.dyn.sample_posterior(logit)

    goal = self._carry_goal(dyn_carry)
    if self.goal_enabled and start_batch > 1:
      goal = jnp.repeat(goal, start_batch, axis=0)
    policyfn = lambda feat: self._canonical_action(
        sample(self.pol(self._head_input(feat, goal), 1)))
    _, imgfeat, imgact = self.dyn.imagine(start, policyfn, H, training=True)
    first = dict(
        deter=start['deter'][:, None],
        stoch=start['stoch'][:, None],
        logit=jnp.zeros_like(imgfeat['logit'][:, :1]),
    )
    imgfeat = concat([first, imgfeat], 1)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgact, lastact], 1)

    imggoal = self._repeat_goal(goal, H + 1)
    inp = sg(self._head_input(imgfeat, imggoal))
    rew = sg(self.rew(inp, 2).pred())
    con = sg(self._continuation(inp, 2))
    return {
        'imgfeat': imgfeat,
        'imgact': sg(imgact),
        'inp': inp,
        'rew': rew,
        'con': con,
    }

  def _adapt_critic_loss(self, rollout):
    policy = self.pol(rollout['inp'], 2)
    value = self.val(rollout['inp'], 2)
    slowvalue = self.slowval(rollout['inp'], 2)
    imag_loss_cfg = self.config.imag_loss.copy()
    imag_loss_cfg['actent'] = self.config.eval_adapt.actent
    losses, outs, metrics = imag_loss(
        rollout['imgact'], rollout['rew'], rollout['con'],
        policy, value, slowvalue,
        self.retnorm, self.valnorm, self.advnorm,
        update=False,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        **imag_loss_cfg)
    disc = 1 if self.config.contdisc else 1 - 1 / self.config.horizon
    zeros = jnp.zeros_like(rollout['rew'])
    reward_only_mc = lambda_return(
        zeros, 1 - rollout['con'], rollout['rew'], zeros, zeros,
        disc, 1.0)
    reward_only_lambda = lambda_return(
        zeros, 1 - rollout['con'], rollout['rew'], zeros, zeros,
        disc, imag_loss_cfg['lam'])
    bootstrapped = outs['ret']
    contribution = bootstrapped - reward_only_lambda
    voffset, vscale = self.valnorm.stats()
    value_pred = value.pred() * vscale + voffset
    metrics.update({
        'target_reward_only_mean': reward_only_mc.mean(),
        'target_reward_only_std': reward_only_mc.std(),
        'target_reward_only_lambda_mean': reward_only_lambda.mean(),
        'target_reward_only_lambda_std': reward_only_lambda.std(),
        'target_bootstrapped_mean': bootstrapped.mean(),
        'target_bootstrapped_std': bootstrapped.std(),
        'target_bootstrap_contribution_mean': contribution.mean(),
        'target_bootstrap_contribution_abs': jnp.abs(contribution).mean(),
        'target_bootstrap_to_reward_ratio': (
            jnp.abs(contribution).mean() /
            (jnp.abs(reward_only_lambda).mean() + 1e-8)),
        'value_abs_p99': jnp.quantile(jnp.abs(value_pred), 0.99),
        'value_target_mae': jnp.abs(
            value_pred[:, :-1] - bootstrapped).mean(),
    })
    loss = losses['value'].mean()
    metrics['loss'] = loss
    return loss, metrics

  def _adapt_actor_loss(self, rollout):
    policy = self.pol(rollout['inp'], 2)
    value = self.val(rollout['inp'], 2)
    slowvalue = self.slowval(rollout['inp'], 2)
    imag_loss_cfg = self.config.imag_loss.copy()
    imag_loss_cfg['actent'] = self.config.eval_adapt.actent
    return adapt_policy_loss(
        rollout['imgact'], rollout['rew'], rollout['con'],
        policy, value, slowvalue,
        self.retnorm, self.valnorm, self.advnorm,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        **imag_loss_cfg)

  def _adapt_reward_only_actor_loss(self, rollout):
    """Actor IR using imagined rewards without critic bootstrapping.

    This deliberately does not subtract a learned value baseline or the
    historical return-normalizer offset. Posterior samples whose imagined
    trajectories contain zero reward therefore contribute no policy-gradient
    signal (apart from optional entropy), while samples that reach predicted
    reward reinforce their sampled actions.
    """
    policy = self.pol(rollout['inp'], 2)
    imag_loss_cfg = self.config.imag_loss.copy()
    disc = 1 if self.config.contdisc else 1 - 1 / self.config.horizon
    zeros = jnp.zeros_like(rollout['rew'])
    ret = lambda_return(
        zeros, 1 - rollout['con'], rollout['rew'], zeros, zeros,
        disc, imag_loss_cfg['lam'])
    _, rscale = self.retnorm(ret, update=False)
    advantage = ret / jnp.maximum(rscale, 1e-8)
    weight = jnp.cumprod(disc * rollout['con'], 1) / disc
    logpi = sum([
        dist.logp(sg(rollout['imgact'][key]))[:, :-1]
        for key, dist in policy.items()])
    entropies = {
        key: dist.entropy()[:, :-1] for key, dist in policy.items()}
    entropy = sum(entropies.values())
    actent = float(self.config.eval_adapt.actent)
    loss = sg(weight[:, :-1]) * -(
        logpi * sg(advantage) + actent * entropy)
    metrics = {
        'adv': advantage.mean(),
        'adv_std': advantage.std(),
        'rew': rollout['rew'].mean(),
        'con': rollout['con'].mean(),
        'ret': ret.mean(),
        'reward_only': ret.mean(),
        'reward_particle_rate': (ret.max(-1) >= .5).mean(),
        'weight': weight[:, :-1].mean(),
    }
    for key, value in entropies.items():
      metrics[f'ent/{key}'] = value.mean()
    return loss.mean(), metrics

  def _adapt_bounded_actor_loss(self, rollout):
    """Actor IR with learned reward and a bounded success bootstrap."""
    if not self.bounded_value_enabled:
      raise ValueError(
          'bounded IR requires agent.bounded_value.enabled=True')
    policy = self.pol(rollout['inp'], 2)
    bounded = self.bval(rollout['inp'], 2).prob(1)
    disc = 1 if self.config.contdisc else 1 - 1 / self.config.horizon
    ret = lambda_return(
        jnp.zeros_like(rollout['con']), 1 - rollout['con'], rollout['rew'],
        bounded, bounded, disc, self.config.imag_loss.lam)
    _, rscale = self.retnorm(ret, update=False)
    advantage = (ret - bounded[:, :-1]) / jnp.maximum(rscale, 1e-8)
    weight = jnp.cumprod(disc * rollout['con'], 1) / disc
    logpi = sum([
        dist.logp(sg(rollout['imgact'][key]))[:, :-1]
        for key, dist in policy.items()])
    entropies = {
        key: dist.entropy()[:, :-1] for key, dist in policy.items()}
    entropy = sum(entropies.values())
    actent = float(self.config.eval_adapt.actent)
    loss = sg(weight[:, :-1]) * -(
        logpi * sg(advantage) + actent * entropy)
    return loss.mean(), {
        'adv': advantage.mean(),
        'adv_std': advantage.std(),
        'rew': rollout['rew'].mean(),
        'con': rollout['con'].mean(),
        'ret': ret.mean(),
        'bounded_value': bounded.mean(),
        'bounded_value_min': bounded.min(),
        'bounded_value_max': bounded.max(),
        'weight': weight[:, :-1].mean(),
    }

  def _adapt_mixed_bounded_actor_loss(self, rollout):
    """Interpolate reward-only IR toward bounded-value IR.

    ``mix_beta=0`` is exactly reward-only IR and ``mix_beta=1`` is exactly
    bounded-value IR when correction clipping is disabled. Intermediate doses
    add only the bounded head's incremental advantage, avoiding an accidental
    second copy of the imagined reward signal.
    """
    if not self.bounded_value_enabled:
      raise ValueError(
          'mixed bounded IR requires agent.bounded_value.enabled=True')
    policy = self.pol(rollout['inp'], 2)
    bounded = self.bval(rollout['inp'], 2).prob(1)
    disc = 1 if self.config.contdisc else 1 - 1 / self.config.horizon
    zeros = jnp.zeros_like(rollout['rew'])
    reward_ret = lambda_return(
        zeros, 1 - rollout['con'], rollout['rew'], zeros, zeros,
        disc, self.config.imag_loss.lam)
    bounded_ret = lambda_return(
        zeros, 1 - rollout['con'], rollout['rew'], bounded, bounded,
        disc, self.config.imag_loss.lam)
    _, rscale = self.retnorm(reward_ret, update=False)
    rscale = jnp.maximum(rscale, 1e-8)
    reward_advantage = reward_ret / rscale
    bounded_advantage = (bounded_ret - bounded[:, :-1]) / rscale
    value_correction = bounded_advantage - reward_advantage
    clip = float(getattr(
        self.config.eval_adapt, 'value_correction_clip', 0.0))
    if clip > 0:
      value_correction_used = jnp.clip(value_correction, -clip, clip)
    else:
      value_correction_used = value_correction
    beta = float(getattr(self.config.eval_adapt, 'mix_beta', 0.0))
    advantage = reward_advantage + beta * value_correction_used
    weight = jnp.cumprod(disc * rollout['con'], 1) / disc
    logpi = sum([
        dist.logp(sg(rollout['imgact'][key]))[:, :-1]
        for key, dist in policy.items()])
    entropies = {
        key: dist.entropy()[:, :-1] for key, dist in policy.items()}
    entropy = sum(entropies.values())
    actent = float(self.config.eval_adapt.actent)
    loss = sg(weight[:, :-1]) * -(
        logpi * sg(advantage) + actent * entropy)
    metrics = {
        'adv': advantage.mean(),
        'adv_std': advantage.std(),
        'rew': rollout['rew'].mean(),
        'con': rollout['con'].mean(),
        'ret': reward_ret.mean(),
        'reward_only': reward_ret.mean(),
        'bounded_ret': bounded_ret.mean(),
        'bounded_value': bounded.mean(),
        'bounded_value_min': bounded.min(),
        'bounded_value_max': bounded.max(),
        'value_correction': value_correction.mean(),
        'value_correction_abs': jnp.abs(value_correction).mean(),
        'value_correction_used_abs': jnp.abs(value_correction_used).mean(),
        'value_correction_clip_rate': (
            (jnp.abs(value_correction) > clip).mean()
            if clip > 0 else jnp.asarray(0.0, f32)),
        'mix_beta': jnp.asarray(beta, f32),
        'weight': weight[:, :-1].mean(),
    }
    for key, value in entropies.items():
      metrics[f'ent/{key}'] = value.mean()
    return loss.mean(), metrics

  def _adapt_actor_objective(self, rollout):
    objective = str(getattr(
        self.config.eval_adapt, 'objective', 'standard'))
    if objective == 'standard':
      return self._adapt_actor_loss(rollout)
    if objective == 'reward_only':
      return self._adapt_reward_only_actor_loss(rollout)
    if objective == 'bounded':
      return self._adapt_bounded_actor_loss(rollout)
    if objective == 'mixed_bounded':
      return self._adapt_mixed_bounded_actor_loss(rollout)
    if objective == 'distill':
      raise ValueError(
          'Reference distillation uses distill_actor(), not imagination')
    raise ValueError(f'Unknown eval_adapt.objective: {objective!r}')

  def distill_actor(self, carry, teacher_stats):
    """Fit the temporary actor to a reference categorical distribution."""

    def lossfn(carry):
      _, dyn_carry, _, _ = carry
      goal = self._carry_goal(dyn_carry)
      inp = sg(self._head_input(dyn_carry, goal))
      policy = self.pol(inp, 1)
      losses = {}
      agreements = {}
      for key, dist in policy.items():
        logits_key = f'{key}/logits'
        if logits_key not in teacher_stats:
          raise ValueError(
              f'Reference distillation requires categorical logits: '
              f'{logits_key}')
        student = getattr(dist, 'output', dist)
        if not hasattr(student, 'logits'):
          raise ValueError(
              f'Reference distillation only supports categorical actions: '
              f'{key}')
        teacher_logits = sg(teacher_stats[logits_key])
        teacher_prob = jax.nn.softmax(teacher_logits, -1)
        student_logprob = jax.nn.log_softmax(student.logits, -1)
        losses[key] = -(teacher_prob * student_logprob).sum(-1)
        agreements[key] = (
            jnp.argmax(student.logits, -1) ==
            jnp.argmax(teacher_logits, -1)).astype(f32)
      loss = sum(losses.values()).mean()
      metrics = {'loss': loss}
      for key, dist in policy.items():
        metrics[f'nll/{key}'] = losses[key].mean()
        metrics[f'ent/{key}'] = dist.entropy().mean()
        metrics[f'agreement/{key}'] = agreements[key].mean()
      return loss, (carry, metrics)

    metrics, (carry, mets) = self.adapt_actor_opt(
        lossfn, carry, has_aux=True)
    metrics.update(prefix(mets, 'adapt_actor'))
    return carry, metrics

  def _adapt_loss(self, carry):
    rollout = self._adapt_rollout(carry)
    loss, metrics = self._adapt_actor_objective(rollout)
    if self.config.eval_adapt.lr != self.config.opt.lr:
      scale = self.config.eval_adapt.lr / self.config.opt.lr
      loss *= scale
    metrics.update(self._adapt_viewer_metrics(rollout['imgfeat']))
    return loss, metrics

  def _adapt_viewer_metrics(self, imgfeat):
    metrics = {}
    viewer = getattr(self.config.eval_adapt, 'viewer', None)
    viewer_enabled = bool(getattr(viewer, 'enabled', False))
    if viewer_enabled and self.dec.imgkeys:
      traj_index = int(getattr(viewer, 'traj_index', 0))
      traj_index = max(0, min(traj_index, imgfeat['deter'].shape[0] - 1))
      feat = jax.tree.map(lambda x: x[traj_index:traj_index + 1], imgfeat)
      reset = jnp.zeros(feat['deter'].shape[:2], bool)
      _, _, recons = self.dec({}, feat, reset, training=False)
      key = sorted(self.dec.imgkeys)[0]
      frames = jnp.clip(recons[key].pred()[0] * 255, 0, 255).astype(jnp.uint8)
      metrics['viewer_imag_frames'] = frames
    return metrics

  def _head_input(self, feat, goal=None):
    """Concatenate public goal identity onto behaviour-head features."""
    inp = self.feat2tensor(feat)
    if not self.goal_enabled:
      return inp
    if goal is None:
      goal = self._carry_goal(feat)
    goal = jnp.asarray(goal, jnp.int32)
    onehot = jax.nn.one_hot(goal, self.goal_count, dtype=nn.COMPUTE_DTYPE)
    assert onehot.shape[:-1] == inp.shape[:-1], (
        onehot.shape, inp.shape)
    return jnp.concatenate([inp, onehot], -1)

  def _reward_continuation_losses(self, repfeat, obs):
    """Return factual head losses plus optional off-goal oracle supervision.

    The factual objective is byte-for-byte equivalent to ordinary Dreamer.
    Counterfactual examples query every nonmatching goal on the same frozen
    posterior feature. Their reward is zero; their episode continues after a
    completion for another goal, but still stops at a physical terminal. The
    stop-gradient is deliberate: the intervention updates only the reward and
    continuation heads, not the encoder or sequence model.
    """
    goal = obs[self.goal_key] if self.goal_enabled else None
    headinp = self._head_input(repfeat, goal)
    reward_dist = self.rew(
        sg(headinp, skip=self.config.reward_grad), 2)
    continuation_dist = self.con(headinp, 2)
    factual_reward = reward_dist.loss(obs['reward'])
    continuation_target = f32(~obs['is_terminal'])
    if self.config.contdisc:
      continuation_target *= 1 - 1 / self.config.horizon
    factual_continuation = continuation_dist.loss(continuation_target)
    losses = {
        'rew': factual_reward,
        'con': factual_continuation,
    }
    metrics = {}
    if not self.counterfactual_heads_enabled:
      return losses, metrics

    actual_goal = jnp.asarray(goal, i32)
    query_goal = jnp.broadcast_to(
        jnp.arange(self.goal_count, dtype=i32),
        (*actual_goal.shape, self.goal_count))
    swept_feat = {
        key: jnp.broadcast_to(
            value[:, :, None],
            (*value.shape[:2], self.goal_count, *value.shape[2:]))
        for key, value in repfeat.items() if key in ('deter', 'stoch')
    }
    counterfactual_inp = sg(self._head_input(swept_feat, query_goal))
    matching = query_goal == actual_goal[..., None]
    nonmatching = ~matching

    reward_target = jnp.asarray(obs['reward'], f32)[..., None] * f32(matching)
    physical = jnp.asarray(obs['is_physical_terminal'], bool)[..., None]
    complete = jnp.asarray(obs['goal_complete'], bool)[..., None]
    terminal = physical | (complete & matching)
    counterfactual_continuation_target = f32(~terminal)
    if self.config.contdisc:
      counterfactual_continuation_target *= 1 - 1 / self.config.horizon

    swept_reward_dist = self.rew(counterfactual_inp, 3)
    swept_continuation_dist = self.con(counterfactual_inp, 3)
    swept_reward_loss = swept_reward_dist.loss(reward_target)
    swept_continuation_loss = swept_continuation_dist.loss(
        counterfactual_continuation_target)
    mask = f32(nonmatching)
    denominator = jnp.maximum(mask.sum(-1), 1.0)
    counterfactual_reward = (swept_reward_loss * mask).sum(-1) / denominator
    counterfactual_continuation = (
        swept_continuation_loss * mask).sum(-1) / denominator
    weight = f32(self.counterfactual_heads.weight)
    losses['rew'] = factual_reward + weight * counterfactual_reward
    losses['con'] = factual_continuation + weight * counterfactual_continuation

    def masked_mean(value, selected):
      selected = f32(selected)
      return (value * selected).sum() / jnp.maximum(selected.sum(), 1.0)

    reward_prediction = swept_reward_dist.pred()
    continuation_prediction = swept_continuation_dist.prob(1)
    completion = complete & jnp.ones_like(nonmatching)
    metrics.update({
        'counterfactual/reward_loss': counterfactual_reward.mean(),
        'counterfactual/continuation_loss': (
            counterfactual_continuation.mean()),
        'counterfactual/event_reward_factual': masked_mean(
            reward_prediction, completion & matching),
        'counterfactual/event_reward_nonmatching': masked_mean(
            reward_prediction, completion & nonmatching),
        'counterfactual/event_continuation_factual': masked_mean(
            continuation_prediction, completion & matching),
        'counterfactual/event_continuation_nonmatching': masked_mean(
            continuation_prediction, completion & nonmatching),
    })
    return losses, metrics

  def _carry_goal(self, carry):
    if not self.goal_enabled:
      return None
    if self.goal_key in carry:
      return carry[self.goal_key]
    return jnp.zeros(carry['deter'].shape[:-1], jnp.int32)

  def _repeat_goal(self, goal, length):
    if not self.goal_enabled:
      return None
    goal = jnp.asarray(goal, jnp.int32)
    return jnp.broadcast_to(goal[:, None], (goal.shape[0], int(length)))

  def _continuation(self, inp, bdims):
    """Predict ordinary Dreamer continuation from latent state and goal."""
    return self.con(inp, bdims).prob(1)

  def _canonical_action(self, action):
    """Remove semantically ignored position values for RLScape no-ops."""
    if not bool(self.goal_cfg.canonicalize_noop_position):
      return action
    assert 'mode' in action and 'position' in action, tuple(action)
    mode = jnp.asarray(action['mode'])
    position = jnp.asarray(action['position'])
    mask = (mode == 0)[..., None]
    position = jnp.where(mask, jnp.zeros_like(position), position)
    return {**action, 'position': position}

  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry))
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
    return carry, obs, self._canonical_action(prevact), stepid

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
      update_rms_clip: float = 0.0,
  ):
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    chain.append(embodied.jax.opt.clip_by_rms(update_rms_clip))
    return optax.chain(*chain)


def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret
  return losses, outs, metrics


def adapt_policy_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    **kwargs,
):
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = sg(value.pred()) * vscale + voffset
  slowval = sg(slowvalue.pred()) * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update=False)
  adv = (sg(ret) - sg(tarval[:, :-1])) / rscale
  aoffset, ascale = advnorm(adv, update=False)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))

  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ((ret - roffset) / rscale).mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()

  return policy_loss.mean(), metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)
