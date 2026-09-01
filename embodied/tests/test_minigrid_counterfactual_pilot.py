import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import ruamel.yaml as yaml

from dreamerv3.agent import Agent
from scripts import minigrid_counterfactual_pilot as pilot


class _Distribution:

  def __init__(self, inp):
    self.inp = inp
    self.target = None

  def loss(self, target):
    self.target = target
    return jnp.zeros(target.shape, jnp.float32)

  def pred(self):
    return jnp.zeros(self.target.shape, jnp.float32)

  def prob(self, value):
    del value
    return jnp.ones(self.target.shape, jnp.float32) * 0.5


class _Head:

  def __init__(self):
    self.calls = []

  def __call__(self, inp, bdims):
    del bdims
    dist = _Distribution(inp)
    self.calls.append(dist)
    return dist


def _agent():
  agent = object.__new__(Agent)
  agent.goal_enabled = True
  agent.goal_key = 'goal_id'
  agent.goal_count = 3
  agent.counterfactual_heads_enabled = True
  agent.counterfactual_heads = SimpleNamespace(weight=1.0)
  agent.config = SimpleNamespace(
      reward_grad=True, contdisc=False, horizon=100)
  agent.rew = _Head()
  agent.con = _Head()

  def head_input(feat, goal):
    deter = feat['deter']
    stoch = feat['stoch'].reshape((*feat['stoch'].shape[:-2], -1))
    return jnp.concatenate([deter, stoch, goal[..., None]], -1)

  agent._head_input = head_input
  return agent


def test_counterfactual_training_targets_only_nonmatching_goals():
  agent = _agent()
  repfeat = {
      'deter': jnp.zeros((1, 2, 2), jnp.float32),
      'stoch': jnp.zeros((1, 2, 1, 2), jnp.float32),
  }
  obs = {
      'goal_id': jnp.asarray([[0, 1]], jnp.int32),
      'reward': jnp.asarray([[1, 0]], jnp.float32),
      'is_terminal': jnp.asarray([[1, 1]], bool),
      'is_physical_terminal': jnp.asarray([[0, 1]], bool),
      'goal_complete': jnp.asarray([[1, 0]], bool),
  }

  losses, metrics = agent._reward_continuation_losses(repfeat, obs)

  np.testing.assert_array_equal(
      agent.rew.calls[0].target, [[1, 0]])
  np.testing.assert_array_equal(
      agent.rew.calls[1].target, [[[1, 0, 0], [0, 0, 0]]])
  np.testing.assert_array_equal(
      agent.con.calls[1].target, [[[0, 1, 1], [0, 0, 0]]])
  assert losses['rew'].shape == losses['con'].shape == (1, 2)
  assert set(metrics) == {
      'counterfactual/reward_loss',
      'counterfactual/continuation_loss',
      'counterfactual/event_reward_factual',
      'counterfactual/event_reward_nonmatching',
      'counterfactual/event_continuation_factual',
      'counterfactual/event_continuation_nonmatching',
  }


def test_pilot_dry_run_is_single_seed_single_env_phase_a(tmp_path):
  assert pilot.main([
      '--experiment-root', str(tmp_path), '--dry-run', '--min-free-gb', '0',
  ]) == 0
  spec = json.loads((tmp_path / 'experiment_spec.json').read_text())
  status = json.loads((tmp_path / 'supervisor_status.json').read_text())

  assert spec['name'] == pilot.EXPERIMENT_NAME
  assert spec['tasks'] == ['go_to_door']
  assert spec['seeds'] == [0]
  assert list(spec['arms']) == ['uniform_reservoir']
  assert spec['total_steps_per_run'] == 200_000
  assert spec['total_training_steps'] == 200_000
  assert spec['snapshot_steps'] == list(range(0, 200_001, 25_000))
  assert spec['extra_train_configs'] == ['minigrid_counterfactual_heads']
  state = status['states']['uniform_reservoir__seed_0000']
  command = state['phases']['phase_00_go_to_door']['planned_command']
  assert 'minigrid_counterfactual_heads' in command
  assert command[command.index('--run.steps') + 1] == '200000'
  assert len(state['evaluations']) == 18


def test_counterfactual_config_is_opt_in_and_keeps_one_training_env():
  configs = yaml.YAML(typ='safe').load(
      (pilot.ROOT / 'dreamerv3' / 'configs.yaml').read_text())

  assert configs['defaults']['agent']['counterfactual_heads'] == {
      'enabled': False, 'weight': 1.0}
  assert configs['minigrid_counterfactual_heads'][
      'agent.counterfactual_heads.enabled'] is True
  assert configs['minigrid_sequential']['run']['envs'] == 1
