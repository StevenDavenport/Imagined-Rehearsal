from types import SimpleNamespace
import pathlib
import sys

import jax.numpy as jnp
import numpy as np

from dreamerv3.agent import Agent
from embodied.jax import outs
from scripts import rlscape_stage2_actor_recovery as stage2


def test_stage2_condition_matrix_has_requested_horizon_ablations():
  conditions = stage2.build_conditions((3, 6, 15, 20))
  assert len(conditions) == 15
  assert [item['name'] for item in conditions[:2]] == [
      'clean_no_adapt', 'corrupt_no_adapt']
  for family in ('standard_ir', 'entropy_off_ir', 'reward_only_ir'):
    selected = [item for item in conditions if item['family'] == family]
    assert [item['horizon'] for item in selected] == [3, 6, 15, 20]
  reward_only = [
      item for item in conditions if item['family'] == 'reward_only_ir']
  assert all(item['objective'] == 'reward_only' for item in reward_only)
  assert all(item['actent'] == 0 for item in reward_only)
  assert conditions[-1]['objective'] == 'distill'
  assert conditions[-1]['horizon'] == 0


def test_stage2_command_encodes_objective_horizon_and_reference():
  conditions = {item['name']: item for item in stage2.build_conditions()}
  common = dict(
      python=pathlib.Path('/env/python'), repo_root=pathlib.Path('/repo'),
      logdir=pathlib.Path('/logs/run'), checkpoint=pathlib.Path('/corrupt'),
      reference_checkpoint=pathlib.Path('/clean'), goal='chop_logs',
      policy_mode='sampled', episodes=100, seed=2000,
      episode_length=200, grid_columns=28, grid_rows=18,
      adapt_steps=1, adapt_lr=4e-5, adapt_every_k=1, start_batch=128)

  reward = stage2.build_command(
      condition=conditions['reward_only_ir_h20'], **common)
  assert reward[reward.index('--eval_adapt.objective') + 1] == 'reward_only'
  assert reward[reward.index('--eval_adapt.imag_length') + 1] == '20'
  assert reward[reward.index('--eval_adapt.actent') + 1] == '0.0'
  assert '--eval_adapt.reference_checkpoint' not in reward

  distill = stage2.build_command(
      condition=conditions['reference_distill'], **common)
  assert distill[distill.index('--eval_adapt.objective') + 1] == 'distill'
  assert distill[
      distill.index('--eval_adapt.reference_checkpoint') + 1] == '/clean'


def test_stage2_corruption_selection_prefers_qualified_target():
  rows = []
  clean = dict(zip(stage2.GOALS, (.6, .5, .9)))
  candidates = {
      'mild': (.5, .4, .8),
      'target': (.2, .2, .3),
      'severe': (.0, .0, .1),
  }
  for goal, score in clean.items():
    rows.append({'condition': 'clean', 'goal': goal, 'success_rate': score})
  for name, scores in candidates.items():
    for goal, score in zip(stage2.GOALS, scores):
      rows.append({'condition': name, 'goal': goal, 'success_rate': score})

  result = stage2.select_corruption(
      rows, list(candidates), target_retention=.35,
      retention_min=.1, retention_max=.65)

  assert result['selected']['name'] == 'target'
  assert result['selected']['qualified'] is True


def test_paired_intervals_preserve_episode_level_direction():
  clean = np.ones(20)
  corrupt_values = np.zeros(20)
  recovered = np.asarray([1] * 15 + [0] * 5, np.float64)

  result = stage2._paired_intervals(
      clean, corrupt_values, recovered, seed=7, samples=2000)

  assert result['paired_delta_vs_corrupt'] == .75
  assert result['paired_discordant_improved'] == 15
  assert result['paired_discordant_harmed'] == 0
  assert result['paired_delta_vs_corrupt_ci_low'] < .75
  assert result['paired_delta_vs_corrupt_ci_high'] >= .75


class _PolicyDistribution:

  def __init__(self, shape):
    self.shape = shape

  def logp(self, action):
    return jnp.zeros_like(action, jnp.float32) - .5

  def entropy(self):
    return jnp.ones(self.shape, jnp.float32)


class _Policy:

  def __call__(self, inp, bdims):
    del bdims
    return {'action': _PolicyDistribution(inp.shape[:-1])}


class _ReturnNorm:

  def __call__(self, value, update):
    del value
    assert update is False
    return 0.0, 1.0


def _reward_only_agent():
  agent = object.__new__(Agent)
  agent.config = SimpleNamespace(
      contdisc=True, horizon=333,
      imag_loss={'lam': .95},
      eval_adapt=SimpleNamespace(actent=0.0))
  agent.pol = _Policy()
  agent.retnorm = _ReturnNorm()
  return agent


def test_reward_only_ir_removes_value_bootstrap_and_zero_reward_gradient():
  agent = _reward_only_agent()
  batch, horizon = 4, 3
  rollout = {
      'inp': jnp.zeros((batch, horizon + 1, 2)),
      'imgact': {'action': jnp.zeros((batch, horizon + 1), jnp.int32)},
      'rew': jnp.zeros((batch, horizon + 1)),
      'con': jnp.ones((batch, horizon + 1)) * .9,
  }
  zero_loss, zero_metrics = agent._adapt_reward_only_actor_loss(rollout)
  np.testing.assert_allclose(zero_loss, 0.0)
  np.testing.assert_allclose(zero_metrics['ret'], 0.0)

  rollout['rew'] = rollout['rew'].at[:, 1].set(1.0)
  reward_loss, reward_metrics = agent._adapt_reward_only_actor_loss(rollout)
  assert float(reward_loss) > 0
  assert float(reward_metrics['ret']) > 0
  assert 'value' not in rollout


class _CaptureOptimizer:

  def __call__(self, lossfn, carry, has_aux):
    assert has_aux is True
    loss, (carry, metrics) = lossfn(carry)
    return {'optimizer_loss': loss}, (carry, metrics)


def test_reference_distillation_uses_full_teacher_distribution():
  agent = object.__new__(Agent)
  student_logits = jnp.asarray([[3.0, 0.0, -1.0]])
  agent.pol = lambda inp, bdims: {
      'action': outs.Categorical(student_logits)}
  agent._carry_goal = lambda carry: jnp.asarray([0])
  agent._head_input = lambda carry, goal: carry['deter']
  agent.adapt_actor_opt = _CaptureOptimizer()
  carry = (
      None, {'deter': jnp.zeros((1, 2)), 'stoch': jnp.zeros((1, 1, 2))},
      None, {'action': jnp.zeros((1,), jnp.int32)})

  _, mismatched = agent.distill_actor(
      carry, {'action/logits': jnp.asarray([[-1.0, 0.0, 3.0]])})
  _, matched = agent.distill_actor(
      carry, {'action/logits': student_logits})

  assert float(mismatched['adapt_actor/agreement/action']) == 0.0
  assert float(matched['adapt_actor/agreement/action']) == 1.0
  assert float(mismatched['adapt_actor/loss']) > float(
      matched['adapt_actor/loss'])


def test_parse_horizons_rejects_duplicates_and_nonpositive_values():
  assert stage2.parse_ints('3,6,15,20') == (3, 6, 15, 20)
  for value in ('3,3', '0,3', ''):
    try:
      stage2.parse_ints(value)
    except ValueError:
      pass
    else:
      raise AssertionError(value)


def test_stage2_dry_run_writes_immutable_plan(tmp_path, monkeypatch):
  archive = tmp_path / 'archive'
  (archive / 'checkpoint').mkdir(parents=True)
  monkeypatch.setattr(
      stage2, 'resolve_milestone', lambda experiment_root, step: archive)
  output = tmp_path / 'stage2'

  result = stage2.main([
      '--experiment-root', str(tmp_path / 'experiment'),
      '--output-root', str(output),
      '--python', sys.executable,
      '--dry-run',
  ])

  assert result == 0
  spec = stage2.json.loads((output / 'stage2_spec.json').read_text())
  queue = stage2.json.loads((output / 'queue.json').read_text())
  assert spec['horizons'] == [3, 6, 15, 20]
  assert len(spec['conditions']) == 15
  assert spec['future_conditioning_repair'] == 'counterfactual_goal_training'
  assert queue['status'] == 'planned'
