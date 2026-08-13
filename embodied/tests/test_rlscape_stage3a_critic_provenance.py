import importlib
import pathlib

import jax.numpy as jnp
import numpy as np
import ruamel.yaml as yaml

from dreamerv3.agent import lambda_return
from scripts import rlscape_stage3a_critic_provenance as stage3a


critic_provenance = importlib.import_module('embodied.run.critic_provenance')


def test_stage3a_command_pins_horizons_and_read_only_runner():
  command = stage3a.build_command(
      python=pathlib.Path('/env/python'), repo_root=pathlib.Path('/repo'),
      logdir=pathlib.Path('/logs/audit'),
      checkpoint=pathlib.Path('/checkpoint'),
      selection=pathlib.Path('/selection.json'), seed=0,
      horizons=(1, 3, 6, 15, 20), posterior_samples=128,
      anchors_per_episode=4, chunk_length=32)
  assert 'rlscape_stage3a_critic_provenance' in command
  assert command[command.index('--critic_provenance.horizons') + 1] == (
      '1,3,6,15,20')
  assert command[
      command.index('--critic_provenance.posterior_samples') + 1] == '128'
  assert command[command.index('--critic_provenance.seed_base') + 1] == (
      str(stage3a.DEFAULT_POSTERIOR_SEED_BASE))


def test_stage3a_parse_ints_rejects_duplicates():
  assert stage3a.parse_ints('1,3,6') == (1, 3, 6)
  try:
    stage3a.parse_ints('1,3,3')
  except ValueError:
    pass
  else:
    raise AssertionError('duplicate horizons must be rejected')


def test_stage3a_numpy_prefix_return_matches_dreamer_recurrence():
  reward = np.asarray([[0, .1, .2, .3]], np.float32)
  continuation = np.asarray([[1, .9, .8, .7]], np.float32)
  bootstrap = np.asarray([[.4, .5, .6, .7]], np.float32)
  actual = critic_provenance.lambda_return_numpy(
      reward, continuation, bootstrap, .997, .95)
  expected = lambda_return(
      jnp.zeros_like(jnp.asarray(continuation)),
      1 - jnp.asarray(continuation), jnp.asarray(reward),
      jnp.asarray(bootstrap), jnp.asarray(bootstrap), .997, .95)
  np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_stage3a_config_is_read_only_50m_model():
  path = pathlib.Path(__file__).parents[2] / 'dreamerv3' / 'configs.yaml'
  configs = yaml.YAML(typ='safe').load(path.read_text())
  audit = configs['rlscape_stage3a_critic_provenance']
  assert audit['script'] == 'critic_provenance'
  assert audit['.*\\.rssm']['deter'] == 4096
  assert audit['eval_adapt.enabled'] is False
  assert configs['defaults']['critic_provenance']['horizons'] == [
      1, 3, 6, 15, 20]
  assert configs['defaults']['critic_provenance']['posterior_samples'] == 128


def test_stage3a_summary_decomposes_bootstrap_and_proposals():
  posterior_samples, horizon = 4, 3
  shape = (1, posterior_samples, horizon + 1)
  reward = np.zeros(shape, np.float32)
  reward[:, :, 1:] = .2
  result = {
      'reward': reward,
      'continuation': np.ones(shape, np.float32),
      'value': np.full(shape, 2.0, np.float32),
      'slowvalue': np.full(shape, 1.5, np.float32),
      'target_value': np.full(shape, 1.0, np.float32),
      'discount': np.ones(shape, np.float32),
      'lambda': np.zeros(shape, np.float32),
      'return': np.full((1, posterior_samples, horizon), 1.2, np.float32),
      'reward_only_lambda': np.full(
          (1, posterior_samples, horizon), .2, np.float32),
      'bootstrap_contribution': np.full(
          (1, posterior_samples, horizon), 1.0, np.float32),
      'prior_entropy': np.full(
          (1, posterior_samples, horizon), .4, np.float32),
      'deter_rms': np.full(
          (1, posterior_samples, horizon), .5, np.float32),
      'stoch_rms': np.full(
          (1, posterior_samples, horizon), .6, np.float32),
      'action': {'action': np.asarray(
          [[[0, 1, 2], [0, 1, 2], [1, 1, 2], [1, 2, 2]]])},
  }
  row = critic_provenance.summarize_result(
      result, proposal='recorded', horizon=horizon, episode_id=7,
      actual_goal=0, probed_goal=0, episode_success=True,
      anchor={'ordinal': 0, 'timestep': 4, 'kind': 'start', 'fraction': .2},
      realized_rewards=np.asarray([0, 0, 1]), realized_completion=True,
      recorded_control_available=True)
  assert np.isclose(row['bootstrap_mean'], 1.0)
  assert np.isclose(row['reward_only_mean'], .2)
  assert np.isclose(row['return_bias_vs_realized'], .2)
  assert row['proposal'] == 'recorded'
  assert row['first_action_unique_fraction'] == .5


def test_stage3a_actor_proposal_does_not_claim_recorded_return_calibration():
  posterior_samples, horizon = 2, 1
  shape = (1, posterior_samples, horizon + 1)
  result = {
      'reward': np.zeros(shape, np.float32),
      'continuation': np.ones(shape, np.float32),
      'value': np.zeros(shape, np.float32),
      'slowvalue': np.zeros(shape, np.float32),
      'target_value': np.zeros(shape, np.float32),
      'discount': np.ones(shape, np.float32),
      'lambda': np.zeros(shape, np.float32),
      'prior_entropy': np.zeros(
          (1, posterior_samples, horizon), np.float32),
      'deter_rms': np.zeros(
          (1, posterior_samples, horizon), np.float32),
      'stoch_rms': np.zeros(
          (1, posterior_samples, horizon), np.float32),
      'action': {'action': np.zeros(
          (1, posterior_samples, horizon), np.int32)},
  }
  row = critic_provenance.summarize_result(
      result, proposal='actor', horizon=horizon, episode_id=0,
      actual_goal=0, probed_goal=0, episode_success=False,
      anchor={'ordinal': 0, 'timestep': 0, 'kind': 'start', 'fraction': 0},
      realized_rewards=None, realized_completion=None,
      recorded_control_available=True)
  assert np.isnan(row['return_bias_vs_realized'])


def test_stage3a_actor_recorded_contrast_is_signed_actor_minus_control():
  common = dict(
      checkpoint='strong', horizon=6, relationship='factual',
      actual_goal=0, actual_goal_name='kill_goblin', rows=10,
      reward_only_mean=.2, prior_entropy_mean=.5,
      control_scope='paired_recorded')
  rows = [
      {**common, 'proposal': 'actor', 'return_mean': 1.5,
       'bootstrap_mean': 1.3},
      {**common, 'proposal': 'recorded', 'return_mean': 1.1,
       'bootstrap_mean': .9},
  ]
  contrast = stage3a.actor_recorded_contrast(rows)[0]
  assert np.isclose(contrast['actor_minus_recorded_return'], .4)
  assert np.isclose(contrast['actor_minus_recorded_bootstrap'], .4)
