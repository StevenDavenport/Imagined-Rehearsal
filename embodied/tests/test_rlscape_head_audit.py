from types import SimpleNamespace
import importlib
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import ruamel.yaml as yaml

from dreamerv3.agent import Agent
from scripts import rlscape_head_audit as workflow


head_audit = importlib.import_module('embodied.run.head_audit')


def _episode(path, goal, success):
  length = 3
  np.savez_compressed(
      path,
      image=np.zeros((length, 4, 4, 3), np.uint8),
      action=np.zeros(length, np.int32),
      reward=np.asarray([0, 0, int(success)], np.float32),
      is_first=np.asarray([1, 0, 0], bool),
      is_last=np.asarray([0, 0, 1], bool),
      is_terminal=np.asarray([0, 0, int(success)], bool),
      is_physical_terminal=np.zeros(length, bool),
      goal_complete=np.asarray([0, 0, int(success)], bool),
      goal_id=np.full(length, goal, np.int32),
  )


def _replay(tmp_path):
  slots = {'0': [], '1': [], '2': []}
  episode_id = 0
  for goal in range(3):
    for success in (False, False, True, True):
      filename = f'episode_{episode_id:04d}.npz'
      _episode(tmp_path / filename, goal, success)
      slots[str(goal)].append({
          'episode_id': episode_id,
          'goal_id': goal,
          'length': 3,
          'filename': filename,
      })
      episode_id += 1
  (tmp_path / 'manifest.json').write_text(json.dumps({
      'format': 1,
      'kind': 'episode_replay',
      'slots': slots,
  }))


def test_selection_is_balanced_deterministic_and_manifest_bound(tmp_path):
  replay = tmp_path / 'replay'
  replay.mkdir()
  _replay(replay)
  first_path = tmp_path / 'first.json'
  second_path = tmp_path / 'second.json'

  first = head_audit.create_selection(
      replay, first_path, successes_per_goal=1, failures_per_goal=1, seed=7)
  second = head_audit.create_selection(
      replay, second_path, successes_per_goal=1, failures_per_goal=1, seed=7)

  assert first['counts'] == second['counts']
  assert [x['episode_id'] for x in first['episodes']] == [
      x['episode_id'] for x in second['episodes']]
  assert len(first['episodes']) == 6
  for goal in range(3):
    selected = [x for x in first['episodes'] if x['goal_id'] == goal]
    assert sorted(x['success'] for x in selected) == [False, True]
  assert head_audit.load_selection(first_path)['episodes'] == first['episodes']


def test_discounted_returns_reports_both_alignment_conventions():
  inclusive, exclusive = head_audit.discounted_returns(
      np.asarray([0, 0, 1], np.float32),
      np.asarray([0, 0, 1], bool),
      .9)

  np.testing.assert_allclose(inclusive, [.81, .9, 1])
  np.testing.assert_allclose(exclusive, [.81, .9, 0])


def test_binary_and_value_metrics_handle_perfect_predictions():
  binary = head_audit.binary_metrics([0, 0, 1, 1], [0, 0, 1, 1])
  value = head_audit.regression_metrics([0, .5, 1], [0, .5, 1])

  assert binary['auroc'] == 1
  assert binary['average_precision'] == 1
  assert binary['brier_clipped'] == 0
  assert value['mae'] == 0
  assert value['rmse'] == 0
  assert value['pearson'] == 1


def test_summary_separates_goal_completion_from_physical_termination():
  states, goals = 6, 5
  actual = np.repeat(np.arange(3), 2)
  complete = np.tile([False, True], 3)
  reward_prediction = np.zeros((states, goals), np.float32)
  reward_prediction[np.arange(states), actual] = complete
  continuation = np.full((states, goals), .997, np.float32)
  continuation[np.arange(states), actual] *= ~complete
  value = np.zeros((states, goals), np.float32)
  arrays = {
      'actual_goal': actual,
      'reward_target': complete.astype(np.float32),
      'is_terminal': complete,
      'is_physical_terminal': np.zeros(states, bool),
      'goal_complete': complete,
      'discount': np.asarray(.997, np.float32),
      'reward_prediction': reward_prediction,
      'continuation_prediction': continuation,
      'value_prediction': value,
      'slowvalue_prediction': value,
      'is_first': np.tile([True, False], 3),
      'episode_success': np.ones(states, bool),
      'return_inclusive': complete.astype(np.float32),
      'return_exclusive': np.zeros(states, np.float32),
      'episode_id': np.repeat(np.arange(3), 2),
      'posterior_entropy': np.ones(states),
      'deter_rms': np.ones(states),
      'stoch_rms': np.ones(states),
      'reward_entropy': np.zeros((states, goals)),
      'continuation_entropy': np.zeros((states, goals)),
      'value_entropy': np.zeros((states, goals)),
      'slowvalue_entropy': np.zeros((states, goals)),
  }

  summary = head_audit.summarize_predictions(
      arrays, ['kill', 'bury', 'chop', 'light', 'fish'])

  assert len(summary['head_cross_goal']) == 15
  assert len(summary['initial_value_cross_goal']) == 15
  assert summary['completion_selectivity']['reward_top1_accuracy'] == 1
  assert summary['continuation_stop_by_state'][
      'goal_completion']['predicted_stop_mean'] == 1
  assert summary['continuation_stop_by_state'][
      'physical_terminal']['count'] == 0


class _Encoder:

  def __call__(self, carry, obs, reset, training):
    del obs, reset, training
    return carry, {}, jnp.zeros((1, 2, 1))


class _Dynamics:

  def observe(self, carry, tokens, prevact, reset, training):
    del tokens, prevact, reset, training
    feat = {
        'deter': jnp.ones((1, 2, 2)),
        'stoch': jnp.ones((1, 2, 1, 2)),
        'logit': jnp.zeros((1, 2, 1, 2)),
    }
    return carry, {}, feat


class _Distribution:

  def __init__(self, prediction):
    self.prediction = prediction

  def pred(self):
    return self.prediction

  def entropy(self):
    return jnp.zeros_like(self.prediction)

  def prob(self, value):
    del value
    return self.prediction


class _GoalHead:

  def __call__(self, inp, bdims):
    del bdims
    # The final three coordinates are the goal one-hot vector.
    goal_value = (
        inp[..., -3] + 2 * inp[..., -2] + 3 * inp[..., -1])
    return _Distribution(goal_value)


class _BinaryDistribution:

  def __init__(self, logit):
    self.logit = logit

  def prob(self, value):
    assert value == 1
    return jax.nn.sigmoid(self.logit)

  def entropy(self):
    raise NotImplementedError


class _BinaryGoalHead:

  def __call__(self, inp, bdims):
    del bdims
    logit = inp[..., -3] + 2 * inp[..., -2] + 3 * inp[..., -1]
    return _BinaryDistribution(logit)


class _Norm:

  def stats(self):
    return 0.0, 1.0


def test_model_head_audit_holds_latent_fixed_and_sweeps_goals():
  agent = object.__new__(Agent)
  agent.goal_enabled = True
  agent.goal_count = 3
  agent.goal_key = 'goal_id'
  agent.goal_cfg = SimpleNamespace(canonicalize_noop_position=False)
  agent.enc = _Encoder()
  agent.dyn = _Dynamics()
  agent.rew = _GoalHead()
  agent.con = _BinaryGoalHead()
  agent.val = _GoalHead()
  agent.slowval = _GoalHead()
  agent.valnorm = _Norm()
  agent.feat2tensor = lambda feat: jnp.concatenate([
      feat['deter'], feat['stoch'].reshape((*feat['stoch'].shape[:-2], -1))
  ], -1)

  _, output = agent.head_audit_chunk(
      ({}, {}),
      {'is_first': jnp.zeros((1, 2), bool)},
      {'action': jnp.zeros((1, 2), jnp.int32)})

  assert output['reward'].shape == (1, 2, 3)
  np.testing.assert_array_equal(output['reward'][0, 0], [1, 2, 3])
  np.testing.assert_array_equal(output['value'][0, 1], [1, 2, 3])
  np.testing.assert_allclose(output['posterior_entropy'], np.log(2))
  assert np.isfinite(output['continuation_entropy']).all()
  assert (output['continuation_entropy'] > 0).all()


def test_workflow_command_is_offline_checkpoint_audit(tmp_path):
  command = workflow.build_command(
      python=pathlib.Path('/env/bin/python'), repo_root=pathlib.Path('/repo'),
      logdir=tmp_path / 'audit', checkpoint=pathlib.Path('/archive/checkpoint'),
      selection=tmp_path / 'selection.json', seed=0, chunk_length=32)

  configs = command[command.index('--configs') + 1:command.index('--seed')]
  assert configs == [
      'rlscape', 'rlscape_click_grid', 'rlscape_chop_logs',
      'rlscape_m3_head_audit']
  assert command[command.index('--run.from_checkpoint') + 1] == (
      '/archive/checkpoint')
  assert '--run.eval_episodes' not in command


def test_head_audit_config_is_large_model_and_read_only():
  path = pathlib.Path(__file__).parents[2] / 'dreamerv3' / 'configs.yaml'
  configs = yaml.YAML(typ='safe').load(path.read_text())
  audit = configs['rlscape_m3_head_audit']

  assert audit['script'] == 'head_audit'
  assert audit['.*\\.rssm']['deter'] == 4096
  assert audit['.*\\.rssm']['classes'] == 32
  assert audit['eval_adapt.enabled'] is False
  assert audit['jax']['precompile'] is False
