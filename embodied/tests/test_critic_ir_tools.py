import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))

import corrupt_actor_checkpoint as corrupt  # noqa: E402
import eval_suite  # noqa: E402
import critic_intermediary_experiment as intermediary  # noqa: E402


def test_actor_critic_corruption_syncs_slow_value(tmp_path):
  source = tmp_path / 'source'
  outdir = tmp_path / 'corrupted'
  source.mkdir()
  params = {
      'pol/mlp/kernel': np.ones((8, 8), np.float32),
      'pol/head/action/kernel': np.ones((8, 2), np.float32),
      'val/mlp/kernel': np.ones((8, 8), np.float32),
      'val/head/logits/kernel': np.ones((8, 5), np.float32),
      'slowval/mlp/kernel': np.full((8, 8), 2.0, np.float32),
      'slowval/head/logits/kernel': np.full((8, 5), 2.0, np.float32),
      'slowval_count': np.array(17, np.int32),
      'dyn/kernel': np.full((8, 8), 3.0, np.float32),
  }
  data = {'params': params, 'counters': {'updates': 1}}
  spec = {
      'name': 'both_zero_mask',
      'method': 'zero_mask',
      'scope': 'all',
      'strength': 0.5,
      'severity': 'moderate',
  }

  manifest = corrupt.save_variant(
      source, outdir, corrupt.clone_agent_checkpoint(data), spec, 0, 'both')

  with (outdir / 'agent.pkl').open('rb') as f:
    result = pickle.load(f)['params']
  assert manifest['target'] == 'both'
  assert manifest['actor_key_count'] == 2
  assert manifest['critic_key_count'] == 2
  assert manifest['actor_relative_l2'] > 0
  assert manifest['critic_relative_l2'] > 0
  assert np.array_equal(result['slowval/mlp/kernel'], result['val/mlp/kernel'])
  assert np.array_equal(
      result['slowval/head/logits/kernel'], result['val/head/logits/kernel'])
  assert np.array_equal(result['slowval_count'], params['slowval_count'])
  assert np.array_equal(result['dyn/kernel'], params['dyn/kernel'])
  assert json.loads((outdir / 'manifest.json').read_text()) == manifest


def test_actor_only_corruption_remains_backward_compatible(tmp_path):
  params = {
      'pol/mlp/kernel': np.ones((8, 8), np.float32),
      'val/mlp/kernel': np.ones((8, 8), np.float32),
      'slowval/mlp/kernel': np.full((8, 8), 2.0, np.float32),
  }
  data = {'params': params, 'counters': {}}
  manifest = corrupt.save_variant(
      tmp_path, tmp_path / 'actor', corrupt.clone_agent_checkpoint(data),
      {'name': 'actor', 'method': 'zero_mask', 'scope': 'all',
       'strength': 0.5},
      0, 'actor')

  with (tmp_path / 'actor' / 'agent.pkl').open('rb') as f:
    result = pickle.load(f)['params']
  assert manifest['target'] == 'actor'
  assert manifest['critic_key_count'] == 0
  assert np.array_equal(result['val/mlp/kernel'], params['val/mlp/kernel'])
  assert np.array_equal(
      result['slowval/mlp/kernel'], params['slowval/mlp/kernel'])


def test_eval_suite_propagates_critic_configuration(tmp_path):
  suite = tmp_path / 'suite.json'
  suite.write_text(json.dumps([{
      'name': 'critic_ir',
      'eval_adapt': {'enabled': True, 'train_critic': True},
  }]))
  args = SimpleNamespace(
      suite_json=str(suite),
      default_adapt_train_critic=False,
      default_adapt_steps=1,
      default_adapt_imag_length=6,
      default_adapt_lr=1e-4,
      default_adapt_critic_lr=2e-4,
      default_adapt_actent=3e-4,
      default_adapt_start_batch=512,
  )

  experiments = eval_suite.build_experiments(args)

  assert experiments == [{
      'name': 'critic_ir',
      'eval_adapt': {
          'enabled': True,
          'train_critic': True,
          'steps': 1,
          'imag_length': 6,
          'lr': 1e-4,
          'critic_lr': 2e-4,
          'every_k': 0,
          'actent': 3e-4,
          'start_batch': 512,
      },
  }]


def test_intermediary_error_metrics_and_reward_bands():
  target = np.asarray([0.0, 1.0, 2.0, 3.0])
  metrics = intermediary.error_metrics(target + 1.0, target)
  bands, names = intermediary.quantile_bands(target)

  assert metrics['mae'] == 1.0
  assert metrics['rmse'] == 1.0
  assert metrics['bias'] == 1.0
  assert np.isclose(metrics['pearson'], 1.0)
  assert set(names.values()) == {'low', 'middle', 'high'}
  assert bands.tolist() == [0, 1, 2, 2]


def test_intermediary_defaults_match_staged_cpu_plan(tmp_path):
  args = intermediary.parse_args([
      'screen', '--outdir', str(tmp_path),
      '--clean_checkpoint', str(tmp_path / 'checkpoint')])

  assert args.jax_platform == 'cpu'
  assert args.episodes == 12
  assert args.train_episodes == 6
  assert args.fidelity_horizons == [1, 3, 6, 12, 24, 48, 96]
  assert args.screen_horizons == [3, 6, 12, 24, 48]
  assert args.rehearsal_seeds == [0, 1, 2]
  assert (args.screen_particles, args.screen_updates) == (128, 128)
  assert (args.confirm_particles, args.confirm_updates) == (512, 256)
