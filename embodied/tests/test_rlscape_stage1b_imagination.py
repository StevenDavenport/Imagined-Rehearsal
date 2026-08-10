from types import SimpleNamespace
import csv
import importlib
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import ninjax as nj
import pytest
import ruamel.yaml as yaml

from dreamerv3.agent import Agent
from dreamerv3.agent import lambda_return
from scripts import rlscape_stage1b_imagination as workflow


imagination_audit = importlib.import_module('embodied.run.imagination_audit')


class _ChunkEncoder:

  def __call__(self, carry, obs, reset, training):
    del reset, training
    return carry, {}, obs['tokens']


class _ChunkDynamics:

  def observe(self, carry, tokens, action, reset, training):
    del action, reset, training
    assert set(carry) == {'deter', 'stoch'}
    feat = {
        'deter': tokens,
        'stoch': jnp.zeros((*tokens.shape[:-1], 1, 2)),
        'logit': jnp.zeros((*tokens.shape[:-1], 1, 2)),
    }
    return carry, {}, feat


def test_posterior_chunk_keeps_rssm_carry_free_of_goal_metadata():
  agent = object.__new__(Agent)
  agent.goal_enabled = True
  agent.goal_key = 'goal_id'
  agent.enc = _ChunkEncoder()
  agent.dyn = _ChunkDynamics()
  agent._canonical_action = lambda action: action
  carry = ({}, {
      'deter': jnp.zeros((1, 2)),
      'stoch': jnp.zeros((1, 1, 2)),
  })
  obs = {
      'tokens': jnp.zeros((1, 3, 2)),
      'is_first': jnp.zeros((1, 3), bool),
      'goal_id': jnp.zeros((1, 3), jnp.int32),
  }
  prevact = {'action': jnp.zeros((1, 3), jnp.int32)}

  carry, _ = agent.posterior_chunk(carry, obs, prevact)
  carry, _ = agent.posterior_chunk(carry, obs, prevact)

  assert set(carry[1]) == {'deter', 'stoch'}


def _episode(length=12, success=True):
  complete = np.zeros(length, bool)
  if success:
    complete[-1] = True
  return {
      'reward': complete.astype(np.float32),
      'is_last': np.arange(length) == length - 1,
      'is_terminal': complete,
      'goal_complete': complete,
  }


def test_anchor_selection_spans_episode_and_excludes_terminal_state():
  success = imagination_audit.select_anchors(_episode(success=True), 4)
  failure = imagination_audit.select_anchors(_episode(success=False), 4)

  assert [item['timestep'] for item in success] == [0, 3, 7, 10]
  assert success[-1]['kind'] == 'pre_completion'
  assert [item['timestep'] for item in failure] == [0, 3, 7, 10]
  assert failure[-1]['kind'] == 'late_failure'
  assert all(item['timestep'] < 11 for item in success + failure)


class _ScalarDistribution:

  def __init__(self, value):
    self.value = value

  def pred(self):
    return self.value

  def entropy(self):
    return jnp.zeros_like(self.value)


class _ScalarHead:

  def __init__(self, value):
    self.value = value

  def __call__(self, inp, bdims):
    del bdims
    return _ScalarDistribution(jnp.zeros(inp.shape[:-1]) + self.value)


class _BinaryDistribution:

  def __init__(self, probability):
    self.probability = probability

  def prob(self, event):
    assert event == 1
    return self.probability


class _BinaryHead:

  def __call__(self, inp, bdims):
    del bdims
    return _BinaryDistribution(jnp.zeros(inp.shape[:-1]) + .9)


class _PolicyDistribution:

  def __init__(self, shape):
    self.shape = shape

  def sample(self, seed):
    del seed
    return jnp.zeros(self.shape, jnp.int32)

  def logp(self, action):
    return jnp.zeros_like(action, jnp.float32) - .5

  def entropy(self):
    return jnp.ones(self.shape, jnp.float32) * 2


class _PolicyHead:

  def __call__(self, inp, bdims):
    del bdims
    return {'action': _PolicyDistribution(inp.shape[:-1])}


class _Dynamics:

  def sample_posterior(self, logit):
    index = jnp.argmax(logit, -1)
    return jax.nn.one_hot(index, logit.shape[-1])

  def imagine(self, start, policyfn, horizon, training):
    assert training is True
    action = policyfn(start)
    feat = {
        'deter': jnp.repeat(start['deter'][:, None], horizon, 1),
        'stoch': jnp.repeat(start['stoch'][:, None], horizon, 1),
        'logit': jnp.zeros((*start['stoch'].shape, horizon)),
    }
    # Move the synthetic class axis back to its expected final position.
    feat['logit'] = jnp.moveaxis(feat['logit'], -1, 1)
    actions = {
        key: jnp.repeat(value[:, None], horizon, 1)
        for key, value in action.items()}
    return start, feat, actions


class _Normalize:

  def stats(self):
    return 0.0, 1.0

  def __call__(self, value, update):
    assert update is False
    return 0.0, 1.0


def test_model_imagination_audit_matches_ir_lambda_return_convention():
  agent = object.__new__(Agent)
  agent.goal_enabled = True
  agent.goal_count = 5
  agent.goal_cfg = SimpleNamespace(canonicalize_noop_position=False)
  agent.config = SimpleNamespace(
      contdisc=True, horizon=333,
      imag_loss=SimpleNamespace(slowtar=False, lam=.95),
      eval_adapt=SimpleNamespace(actent=3e-4))
  agent.dyn = _Dynamics()
  agent.pol = _PolicyHead()
  agent.rew = _ScalarHead(.1)
  agent.con = _BinaryHead()
  agent.val = _ScalarHead(.4)
  agent.slowval = _ScalarHead(.3)
  agent.valnorm = _Normalize()
  agent.retnorm = _Normalize()
  agent.advnorm = _Normalize()
  agent._head_input = lambda feat, goal=None: jnp.concatenate([
      feat['deter'],
      feat['stoch'].reshape((*feat['stoch'].shape[:-2], -1)),
      jax.nn.one_hot(goal, 5),
  ], -1)

  _, output = nj.pure(agent.imagination_audit)(
      {},
      {
          'deter': jnp.ones((1, 2)),
          'stoch': jnp.ones((1, 1, 2)),
          'logit': jnp.zeros((1, 1, 2)),
      },
      jnp.asarray([2]), horizon=3, start_batch=4,
      seed=jax.random.PRNGKey(0))

  assert output['reward'].shape == (1, 4, 4)
  assert output['return'].shape == (1, 4, 3)
  assert output['action']['action'].shape == (1, 4, 3)
  reward = output['reward'].reshape(4, 4)
  continuation = output['continuation'].reshape(4, 4)
  value = output['value'].reshape(4, 4)
  expected = lambda_return(
      jnp.zeros_like(continuation), 1 - continuation, reward,
      value, value, 1.0, .95)
  np.testing.assert_allclose(output['return'][0], expected)
  assert (np.asarray(output['entropy_loss']) < 0).all()


def _shard(path, episode_id, actual_goal, success):
  anchors, goals, particles, horizon = 2, 5, 3, 3
  shape_state = (anchors, goals, particles, horizon + 1)
  shape_step = (anchors, goals, particles, horizon)
  reward = np.zeros(shape_state, np.float32)
  reward[:, :, :, 1] = .6
  continuation = np.ones(shape_state, np.float32) * .9
  arrays = {
      'spec_digest': np.asarray('abc'),
      'episode_id': np.asarray(episode_id),
      'actual_goal': np.asarray(actual_goal),
      'episode_success': np.asarray(success),
      'anchor_ordinal': np.arange(anchors),
      'anchor_timestep': np.arange(anchors),
      'anchor_kind': np.asarray(['start', 'late_failure']),
      'anchor_fraction': np.asarray([0, .9]),
      'recorded_reward_horizon': np.asarray([0, 0]),
      'recorded_completion_horizon': np.asarray([0, 0]),
      'reward': reward,
      'continuation': continuation,
      'value': np.ones(shape_state),
      'slowvalue': np.ones(shape_state),
      'return': np.ones(shape_step) * .8,
      'reward_only_lambda': np.ones(shape_step) * .3,
      'bootstrap_contribution': np.ones(shape_step) * .5,
      'advantage': np.ones(shape_step) * .1,
      'advantage_normalized': np.ones(shape_step) * .2,
      'weight': np.ones(shape_step),
      'policy_entropy': np.ones(shape_step) * 2,
      'reinforce_loss': np.ones(shape_step) * .4,
      'entropy_loss': np.ones(shape_step) * -.01,
      'prior_entropy': np.ones(shape_step),
      'deter_rms': np.ones(shape_step),
      'stoch_rms': np.ones(shape_step),
  }
  np.savez_compressed(path, **arrays)


def test_shard_summary_retains_factual_and_counterfactual_rollouts(tmp_path):
  first, second = tmp_path / 'a.npz', tmp_path / 'b.npz'
  _shard(first, 1, 0, True)
  _shard(second, 2, 1, False)

  rows, summary = imagination_audit.summarize_shards(
      [first, second], ['kill', 'bury', 'chop', 'light', 'fish'], 3)

  assert len(rows) == 20
  assert summary['episodes'] == 2
  assert summary['anchors'] == 4
  assert summary['rollouts'] == 20
  assert summary['factual']['return_mean'] == pytest.approx(.8)
  assert summary['counterfactual']['reward_event_particle_rate'] == 1
  profiles = imagination_audit.summarize_time_profiles(
      [first, second], ['kill', 'bury', 'chop', 'light', 'fish'], 3)
  assert len(profiles) == 2 * 5 * 4
  assert profiles[0]['reward_event'] in (0, 1)


def test_stage1b_command_uses_exact_actor_ir_budget(tmp_path):
  command = workflow.build_command(
      python=pathlib.Path('/env/bin/python'), repo_root=pathlib.Path('/repo'),
      logdir=tmp_path / 'audit', checkpoint=pathlib.Path('/checkpoint'),
      selection=tmp_path / 'selection.json', seed=0, horizon=6,
      particles=128, anchors_per_episode=4)

  configs = command[command.index('--configs') + 1:command.index('--seed')]
  assert configs[-1] == 'rlscape_m3_imagination_audit'
  assert command[command.index('--imagination_audit.horizon') + 1] == '6'
  assert command[command.index('--imagination_audit.particles') + 1] == '128'
  assert command[command.index(
      '--imagination_audit.anchors_per_episode') + 1] == '4'


def test_stage1b_config_is_read_only_50m_model():
  path = pathlib.Path(__file__).parents[2] / 'dreamerv3' / 'configs.yaml'
  configs = yaml.YAML(typ='safe').load(path.read_text())
  audit = configs['rlscape_m3_imagination_audit']

  assert audit['script'] == 'imagination_audit'
  assert audit['.*\\.rssm']['deter'] == 4096
  assert audit['eval_adapt.enabled'] is False
  assert configs['defaults']['imagination_audit']['horizon'] == 6
  assert configs['defaults']['imagination_audit']['particles'] == 128


def test_stage1b_figures_cover_all_failure_channels(tmp_path):
  fields = (
      'actual_goal', 'probed_goal', 'return_mean', 'return_p95',
      'reward_event_particle_rate', 'stop_particle_rate',
      'reward_only_mean', 'bootstrap_mean', 'reinforce_loss_abs',
      'entropy_loss_abs')
  for label, scale in (('strong', 1.0), ('weak', 1.2)):
    directory = tmp_path / label
    directory.mkdir()
    rows = []
    for actual in range(3):
      for probed in range(5):
        values = [actual, probed, .4 * scale, .8 * scale, .1, .2,
                  .1, .3 * scale, .02, .001]
        rows.append(dict(zip(fields, values)))
    with (directory / 'goal_sweep.csv').open('w', newline='') as handle:
      writer = csv.DictWriter(handle, fieldnames=fields)
      writer.writeheader()
      writer.writerows(rows)
    time_fields = (
        'actual_goal', 'probed_goal', 'step', 'reward_event', 'stop')
    time_rows = []
    for actual in range(3):
      for probed in range(5):
        for step in range(7):
          time_rows.append(dict(zip(
              time_fields, (actual, probed, step, .1 * scale, .2))))
    with (directory / 'time_profile.csv').open('w', newline='') as handle:
      writer = csv.DictWriter(handle, fieldnames=time_fields)
      writer.writeheader()
      writer.writerows(time_rows)

  workflow.make_figures(tmp_path)

  names = (
      'imagined_return', 'imagined_return_p95', 'imagined_reward_events',
      'imagined_stop_events', 'ir_objective_decomposition')
  names = (*names, 'imagination_time_profile')
  for name in names:
    assert (tmp_path / 'figures' / f'{name}.png').is_file()
    assert (tmp_path / 'figures' / f'{name}.pdf').is_file()
