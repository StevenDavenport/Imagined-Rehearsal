from types import SimpleNamespace
import importlib
import json
import pathlib

import jax.numpy as jnp
import numpy as np

from dreamerv3.agent import Agent
from scripts import rlscape_stage4a_counterfactual_repair as stage4a


head_audit = importlib.import_module('embodied.run.head_audit')


def _episode(path, goal, success, episode_id):
  del episode_id
  np.savez_compressed(
      path,
      image=np.zeros((3, 2, 2, 3), np.uint8),
      action=np.zeros(3, np.int32),
      reward=np.asarray([0, 0, int(success)], np.float32),
      is_first=np.asarray([1, 0, 0], bool),
      is_last=np.asarray([0, 0, 1], bool),
      is_terminal=np.asarray([0, 0, int(success)], bool),
      is_physical_terminal=np.zeros(3, bool),
      goal_complete=np.asarray([0, 0, int(success)], bool),
      goal_id=np.full(3, goal, np.int32),
  )


def _replay(path):
  slots = {'0': [], '1': [], '2': []}
  episode_id = 0
  for goal in range(3):
    for success in (False, False, True, True):
      filename = f'episode_{episode_id:04d}.npz'
      _episode(path / filename, goal, success, episode_id)
      slots[str(goal)].append({
          'episode_id': episode_id, 'goal_id': goal, 'length': 3,
          'filename': filename})
      episode_id += 1
  (path / 'manifest.json').write_text(json.dumps({
      'format': 1, 'kind': 'episode_replay', 'slots': slots}))


def test_split_banks_are_balanced_exact_deterministic_and_disjoint(tmp_path):
  replay = tmp_path / 'replay'
  replay.mkdir()
  _replay(replay)
  train_path = tmp_path / 'train.json'
  heldout_path = tmp_path / 'heldout.json'

  train, heldout = head_audit.create_split_selections(
      replay, train_path, heldout_path,
      train_successes_per_goal=1, train_failures_per_goal=1,
      heldout_successes_per_goal=1, heldout_failures_per_goal=1, seed=17)

  assert len(train['episodes']) == len(heldout['episodes']) == 6
  assert not (
      {row['episode_id'] for row in train['episodes']} &
      {row['episode_id'] for row in heldout['episodes']})
  for selection in (train, heldout):
    for goal in range(3):
      rows = [row for row in selection['episodes'] if row['goal_id'] == goal]
      assert sorted(row['success'] for row in rows) == [False, True]
  assert head_audit.load_selection(train_path)['split'] == 'repair_train'
  assert head_audit.load_selection(heldout_path)['split'] == 'heldout_audit'


def test_split_banks_fail_instead_of_silently_underfilling(tmp_path):
  replay = tmp_path / 'replay'
  replay.mkdir()
  _replay(replay)
  with np.testing.assert_raises_regex(RuntimeError, 'Insufficient goal'):
    head_audit.create_split_selections(
        replay, tmp_path / 'train.json', tmp_path / 'heldout.json',
        train_successes_per_goal=2, train_failures_per_goal=2,
        heldout_successes_per_goal=1, heldout_failures_per_goal=1)


class _Distribution:

  def __init__(self, shape, binary=False):
    self.shape = shape
    self.binary = binary
    self.target = None

  def loss(self, target):
    self.target = target
    return jnp.zeros(self.shape, jnp.float32)

  def pred(self):
    return jnp.zeros(self.shape, jnp.float32)

  def prob(self, value):
    del value
    return jnp.ones(self.shape, jnp.float32) * .5


class _Head:

  def __init__(self, binary=False):
    self.binary = binary
    self.calls = []

  def __call__(self, inp, bdims):
    del bdims
    dist = _Distribution(inp.shape[:-1], self.binary)
    self.calls.append(dist)
    return dist


class _Optimizer:

  def __call__(self, lossfn, batch, has_aux=False):
    assert has_aux
    loss, aux = lossfn(batch)
    return {'repair_opt/loss': loss}, aux


def _repair_agent():
  agent = object.__new__(Agent)
  agent.head_repair_enabled = True
  agent.goal_enabled = True
  agent.goal_count = 3
  agent.config = SimpleNamespace(contdisc=False, horizon=200)
  agent.rew = _Head()
  agent.con = _Head(binary=True)
  agent.repair_head_opt = _Optimizer()
  agent._head_input = lambda feat, goal: jnp.concatenate([
      feat['deter'], feat['stoch'].reshape((feat['stoch'].shape[0],
                                           feat['stoch'].shape[1], -1)),
      goal[..., None]], -1)
  return agent


def _repair_batch():
  return {
      'deter': jnp.zeros((2, 2), jnp.float32),
      'stoch': jnp.zeros((2, 1, 2), jnp.float32),
      'actual_goal': jnp.asarray([0, 1], jnp.int32),
      'reward': jnp.asarray([1, 0], jnp.float32),
      'is_terminal': jnp.asarray([1, 1], bool),
      'physical_terminal': jnp.asarray([0, 1], bool),
      'goal_complete': jnp.asarray([1, 0], bool),
  }


def test_counterfactual_targets_zero_alternative_reward_and_keep_it_alive():
  agent = _repair_agent()
  agent.repair_reward_continuation(_repair_batch(), counterfactual=True)

  np.testing.assert_array_equal(
      agent.rew.calls[-1].target, [[1, 0, 0], [0, 0, 0]])
  np.testing.assert_array_equal(
      agent.con.calls[-1].target, [[0, 1, 1], [0, 0, 0]])


def test_factual_control_repeats_targets_to_match_counterfactual_dose():
  agent = _repair_agent()
  agent.repair_reward_continuation(_repair_batch(), counterfactual=False)

  np.testing.assert_array_equal(
      agent.rew.calls[-1].target, [[1, 1, 1], [0, 0, 0]])
  np.testing.assert_array_equal(
      agent.con.calls[-1].target, np.zeros((2, 3), np.float32))


def test_stage4a_condition_matrix_and_bounded_config_layering(tmp_path):
  conditions = stage4a.build_eval_conditions()
  assert [condition['name'] for condition in conditions] == [
      'original_frozen', 'original_standard_h6',
      'original_reward_only_h15', 'factual_reward_only_h15',
      'counterfactual_reward_only_h15',
      'counterfactual_bounded_reward_only_h15',
      'counterfactual_bounded_h6', 'counterfactual_bounded_h15']
  bounded = [condition for condition in conditions if condition['bounded']]
  assert [condition['objective'] for condition in bounded] == [
      'reward_only', 'bounded', 'bounded']

  args = SimpleNamespace(
      python=pathlib.Path('/env/python'), repo_root=pathlib.Path('/repo'),
      seed=0, heldout_selection=tmp_path / 'heldout.json', chunk_length=32)
  command = stage4a.build_audit_command(
      args, logdir=tmp_path / 'audit', checkpoint=tmp_path / 'checkpoint',
      bounded=True)
  configs = command[command.index('--configs') + 1:command.index('--seed')]
  assert configs[-1] == 'rlscape_stage4a_bounded_value'


class _PolicyDistribution:

  def __init__(self, shape):
    self.shape = shape

  def logp(self, action):
    return jnp.zeros_like(action, jnp.float32) - .5

  def entropy(self):
    return jnp.ones(self.shape, jnp.float32)


class _BoundedDistribution:

  def __init__(self, shape):
    self.shape = shape

  def prob(self, value):
    del value
    return jnp.ones(self.shape, jnp.float32) * .4


class _ReturnNorm:

  def __call__(self, value, update):
    del value
    assert update is False
    return 0.0, 1.0


def test_bounded_ir_uses_bounded_bootstrap_without_historical_critic():
  agent = object.__new__(Agent)
  agent.bounded_value_enabled = True
  agent.config = SimpleNamespace(
      contdisc=True, horizon=333, imag_loss=SimpleNamespace(lam=.95),
      eval_adapt=SimpleNamespace(actent=0.0))
  agent.pol = lambda inp, bdims: {
      'action': _PolicyDistribution(inp.shape[:-1])}
  agent.bval = lambda inp, bdims: _BoundedDistribution(inp.shape[:-1])
  agent.retnorm = _ReturnNorm()
  rollout = {
      'inp': jnp.zeros((2, 4, 3), jnp.float32),
      'imgact': {'action': jnp.zeros((2, 4), jnp.int32)},
      'rew': jnp.zeros((2, 4), jnp.float32),
      'con': jnp.ones((2, 4), jnp.float32),
  }

  zero_loss, zero_metrics = agent._adapt_bounded_actor_loss(rollout)
  np.testing.assert_allclose(zero_loss, 0, atol=1e-6)
  np.testing.assert_allclose(zero_metrics['bounded_value'], .4)
  rollout['rew'] = rollout['rew'].at[:, 1].set(1)
  reward_loss, reward_metrics = agent._adapt_bounded_actor_loss(rollout)
  assert float(reward_loss) > 0
  assert float(reward_metrics['ret']) > .4
  assert not hasattr(agent, 'val')
