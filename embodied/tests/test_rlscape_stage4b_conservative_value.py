from types import SimpleNamespace
import json
import pathlib
import sys

import jax.numpy as jnp
import numpy as np

from dreamerv3.agent import Agent
from embodied.jax import opt
from scripts import rlscape_stage4b_conservative_value as stage4b


class _PolicyDistribution:

  def __init__(self, shape):
    self.shape = shape

  def logp(self, action):
    return jnp.zeros_like(action, jnp.float32) - .5

  def entropy(self):
    return jnp.ones(self.shape, jnp.float32)


class _BoundedDistribution:

  def __init__(self, values):
    self.values = values

  def prob(self, value):
    del value
    return self.values


class _ReturnNorm:

  def __call__(self, value, update):
    del value
    assert update is False
    return 0.0, 1.0


class _ImagLoss(dict):

  __getattr__ = dict.__getitem__


def _agent(beta, clip=0.0):
  agent = object.__new__(Agent)
  agent.bounded_value_enabled = True
  agent.config = SimpleNamespace(
      contdisc=True, horizon=333, imag_loss=_ImagLoss(lam=.95),
      eval_adapt=SimpleNamespace(
          actent=0.0, mix_beta=beta, value_correction_clip=clip))
  agent.pol = lambda inp, bdims: {
      'action': _PolicyDistribution(inp.shape[:-1])}
  agent.bval = lambda inp, bdims: _BoundedDistribution(
      jnp.clip(inp[..., 0], .05, .95))
  agent.retnorm = _ReturnNorm()
  return agent


def _rollout():
  values = jnp.asarray([.1, .8, .2, .9], jnp.float32)
  return {
      'inp': jnp.broadcast_to(values[None, :, None], (2, 4, 1)),
      'imgact': {'action': jnp.zeros((2, 4), jnp.int32)},
      'rew': jnp.asarray([[0, 1, 0, 0], [0, 0, 0, 0]], jnp.float32),
      'con': jnp.ones((2, 4), jnp.float32) * .98,
  }


def test_mixed_objective_has_exact_reward_and_bounded_endpoints():
  rollout = _rollout()
  reward_agent = _agent(0.0)
  bounded_agent = _agent(1.0)

  reward_loss, _ = reward_agent._adapt_reward_only_actor_loss(rollout)
  mixed_zero_loss, _ = reward_agent._adapt_mixed_bounded_actor_loss(rollout)
  bounded_loss, _ = bounded_agent._adapt_bounded_actor_loss(rollout)
  mixed_one_loss, _ = bounded_agent._adapt_mixed_bounded_actor_loss(rollout)

  np.testing.assert_allclose(mixed_zero_loss, reward_loss, rtol=1e-6)
  np.testing.assert_allclose(mixed_one_loss, bounded_loss, rtol=1e-6)


def test_mixed_objective_clips_only_incremental_value_correction():
  _, metrics = _agent(.25, clip=.05)._adapt_mixed_bounded_actor_loss(
      _rollout())

  assert float(metrics['value_correction_abs']) > .05
  assert float(metrics['value_correction_used_abs']) <= .05 + 1e-6
  assert float(metrics['value_correction_clip_rate']) > 0


def test_stage4b_matrix_is_prespecified_and_reward_endpoint_is_beta_zero():
  conditions = stage4b.build_conditions()
  assert [condition['name'] for condition in conditions] == [
      'frozen', 'reward_only_h15', 'mixed_b005_h15', 'mixed_b010_h15',
      'mixed_b025_h15', 'mixed_b025_clip050_h15',
      'mixed_b025_ucap15e6_h15', 'mixed_b025_h6', 'mixed_b100_h15']
  reward = conditions[1]
  anchor = conditions[-1]
  assert reward['objective'] == anchor['objective'] == 'mixed_bounded'
  assert reward['mix_beta'] == 0
  assert anchor['mix_beta'] == 1
  assert all(condition['actent'] == 0 for condition in conditions)


def test_stage4b_command_encodes_separate_paired_environment_and_agent_seeds(
    tmp_path):
  args = SimpleNamespace(
      python=pathlib.Path('/env/python'), repo_root=pathlib.Path('/repo'),
      checkpoint=pathlib.Path('/checkpoint'), eval_seed_base=4000,
      agent_seed_base=5000, episode_length=200, grid_columns=28,
      grid_rows=18, adapt_steps=1, adapt_lr=4e-5, adapt_every_k=1,
      start_batch=128, mvn_path='', java_home='', paired_rng_stride=1000)
  condition = stage4b.build_conditions()[3]

  command = stage4b.build_command(
      args, logdir=tmp_path, condition=condition, goal='chop_logs',
      policy_mode='sampled', episodes=50, replicate=2)

  assert command[command.index('--seed') + 1] == '4002'
  assert command[command.index('--eval_adapt.paired_rng_seed') + 1] == '5002'
  assert command[command.index('--eval_adapt.mix_beta') + 1] == '0.1'
  assert command[command.index('--eval_adapt.paired_rng') + 1] == 'True'
  capped = stage4b.build_command(
      args, logdir=tmp_path, condition=stage4b.build_conditions()[6],
      goal='chop_logs', policy_mode='sampled', episodes=50, replicate=2)
  assert capped[capped.index('--eval_adapt.update_rms_clip') + 1] == '1.5e-05'


def test_final_update_rms_clip_bounds_optimizer_step():
  transform = opt.clip_by_rms(1e-3)
  updates = {'a': jnp.ones((100,), jnp.float32)}
  clipped, _ = transform.update(updates, transform.init(updates), updates)

  rms = jnp.sqrt(jnp.mean(jnp.square(clipped['a'])))
  np.testing.assert_allclose(rms, 1e-3, rtol=1e-5)


def test_stage4b_aggregate_reports_mean_worst_goal_and_makes_figures(tmp_path):
  rows = []
  for mode in stage4b.POLICY_MODES:
    for condition in stage4b.build_conditions():
      for goal_index, goal in enumerate(stage4b.GOALS):
        for replicate in range(2):
          rate = .8 - .2 * goal_index
          rows.append({
              'policy_mode': mode, 'condition': condition['name'],
              'goal': goal, 'replicate': replicate,
              'episodes': 10, 'successes': int(rate * 10),
              'success_rate': rate, 'mix_beta': condition['mix_beta'],
              'value_correction_clip': condition['value_correction_clip'],
              'update_rms_clip': condition['update_rms_clip'],
              'horizon': condition['horizon'],
              'paired_delta_vs_reward_only': 0.0,
              'paired_delta_vs_frozen': 0.0,
              'paired_improved_vs_reward_only': 0,
              'paired_harmed_vs_reward_only': 0,
          })

  goals, conditions = stage4b.aggregate(rows)
  selected = next(
      row for row in conditions
      if row['policy_mode'] == 'sampled' and row['condition'] == 'frozen')
  assert np.isclose(selected['mean_goal_success'], .6)
  assert np.isclose(selected['worst_goal_success'], .4)
  assert len(goals) == 2 * 9 * 3
  stage4b.make_figures(tmp_path, rows, goals, conditions)
  assert (tmp_path / 'figures' / 'value_dose_sampled.pdf').is_file()
  assert (tmp_path / 'figures' / 'mean_worst_goal.pdf').is_file()
  assert (tmp_path / 'figures' / 'paired_delta_heatmap.pdf').is_file()
  assert (tmp_path / 'figures' / 'replicate_stability.pdf').is_file()


def test_stage4b_dry_run_writes_and_enforces_immutable_spec(tmp_path):
  checkpoint = (
      tmp_path / 'stage4a' / 'checkpoints' / 'counterfactual_bounded')
  checkpoint.mkdir(parents=True)
  (checkpoint / 'done').write_bytes(b'')
  (checkpoint / 'agent.pkl').write_bytes(b'agent')
  (checkpoint / 'repair_manifest.json').write_text(json.dumps({
      'condition': 'counterfactual_bounded',
      'counterfactual_reward_continuation': True,
      'bounded_value': True, 'value_updates': 2000,
      'frozen_component_violations': {},
  }))
  output = tmp_path / 'stage4b'
  common = [
      '--stage4a-root', str(tmp_path / 'stage4a'),
      '--output-root', str(output), '--python', sys.executable, '--dry-run']

  assert stage4b.main(common) == 0
  spec = json.loads((output / 'experiment_spec.json').read_text())
  assert spec['canonical_stage'] == 'stage4b'
  assert spec['replicates'] == 3
  assert len(spec['conditions']) == 9
  assert stage4b.main(common) == 0
  try:
    stage4b.main([*common, '--sampled-episodes', '51'])
  except RuntimeError as exc:
    assert 'different specification' in str(exc)
  else:
    raise AssertionError('Changed Stage 4B spec was silently accepted')
