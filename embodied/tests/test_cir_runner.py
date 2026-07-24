import csv
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import elements
import numpy as np

from dreamerv3.agent import Agent as DreamerAgent
from embodied.envs.dmc import DMC
from embodied.run import cir as run_cir
from embodied.run.cir import (
    ACTOR_PREFIXES, CRITIC_PREFIXES, SOURCE_REGEX, _adapt_due, _append_csv,
    _normal_kl)


class _FakeDMWrapper:

  def __init__(self):
    self.act_space = {
        'reset': elements.Space(bool),
        'action': elements.Space(np.float32, (2,), -1.0, 1.0),
    }
    self.obs_space = {
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }
    self.last_action = None

  def step(self, action):
    self.last_action = action
    return dict(
        reward=np.float32(0), is_first=False, is_last=False,
        is_terminal=False)


class _FakeOptimizer:

  def __init__(self):
    self.calls = 0

  def __call__(self, lossfn, carry, has_aux):
    del has_aux
    self.calls += 1
    _, aux = lossfn(carry)
    return {'optimizer_metric': np.float32(1)}, aux


class _FakeSlowValue:

  def __init__(self):
    self.updates = 0

  def update(self):
    self.updates += 1


class CIRRunnerTest(unittest.TestCase):

  def test_action_gain_scales_continuous_action_without_mutating_input(self):
    env = object.__new__(DMC)
    env._env = _FakeDMWrapper()
    env._dmenv = SimpleNamespace(physics=SimpleNamespace(
        render=lambda *args, **kwargs: np.zeros((4, 4, 3), np.uint8)))
    env._size = (4, 4)
    env._proprio = True
    env._image = True
    env._camera = 0
    env._action_scale = 0.7
    action = {'reset': np.asarray(False),
              'action': np.asarray([1.0, -0.5], np.float32)}

    env.step(action)

    np.testing.assert_allclose(env._env.last_action['action'], [0.7, -0.35])
    np.testing.assert_allclose(action['action'], [1.0, -0.5])

  def test_initial_actor_kl_is_zero(self):
    mean = np.asarray([[0.2, -0.1]], np.float32)
    std = np.asarray([[0.4, 0.7]], np.float32)
    np.testing.assert_allclose(
        _normal_kl((mean, std), (mean, std)), 0.0, atol=1e-7)

  def test_checkpoint_filter_leaves_new_optimizers_fresh(self):
    self.assertTrue(re.match(SOURCE_REGEX, 'dyn/kernel'))
    self.assertTrue(re.match(SOURCE_REGEX, 'pol/kernel'))
    for key in (
        'adapt_opt/state', 'adapt_actor_opt/state',
        'adapt_critic_opt/state', 'model_opt/state'):
      self.assertFalse(re.match(SOURCE_REGEX, key), key)

  def test_frozen_critic_namespace_excludes_actor_and_world_model(self):
    for key in (
        'val/head/kernel', 'slowval/head/kernel', 'retnorm/hi/value',
        'adapt_critic_opt/state/1/0'):
      self.assertTrue(key.startswith(CRITIC_PREFIXES), key)
    for key in ('pol/head/kernel', 'dyn/kernel', 'adapt_actor_opt/state/1/0'):
      self.assertFalse(key.startswith(CRITIC_PREFIXES), key)

  def test_frozen_actor_namespace_excludes_critic_and_world_model(self):
    for key in (
        'pol/head/kernel', 'adapt_opt/state/1/0',
        'adapt_actor_opt/state/1/0'):
      self.assertTrue(key.startswith(ACTOR_PREFIXES), key)
    for key in ('val/head/kernel', 'dyn/kernel', 'adapt_critic_opt/state/1/0'):
      self.assertFalse(key.startswith(ACTOR_PREFIXES), key)

  def test_frozen_critic_uses_dedicated_actor_path_only(self):
    actor_opt = _FakeOptimizer()
    critic_opt = _FakeOptimizer()
    slowval = _FakeSlowValue()
    fake = SimpleNamespace(
        config=SimpleNamespace(eval_adapt=SimpleNamespace(
            enabled=True, train_critic=False)),
        adapt_actor_opt=actor_opt,
        adapt_critic_opt=critic_opt,
        slowval=slowval,
        _adapt_rollout=lambda carry: {'carry': carry, 'imgfeat': None},
        _adapt_actor_loss=lambda rollout: (
            np.float32(0), {'loss': np.float32(len(rollout))}),
        _adapt_viewer_metrics=lambda imgfeat: {})

    _, metrics = DreamerAgent.adapt(
        fake, carry='carry', freeze_critic=True)

    self.assertEqual(actor_opt.calls, 1)
    self.assertEqual(critic_opt.calls, 0)
    self.assertEqual(slowval.updates, 0)
    self.assertIn('adapt_actor/loss', metrics)

  def test_rehearsal_interval_is_exact_and_skips_terminal_state(self):
    self.assertTrue(_adapt_due(True, True, False, 0, 25))
    self.assertFalse(_adapt_due(True, False, False, 23, 25))
    self.assertTrue(_adapt_due(True, False, False, 24, 25))
    self.assertFalse(_adapt_due(True, False, True, 24, 25))

  def test_csv_schema_expands_without_misaligning_old_rows(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'episodes.csv'
      _append_csv(path, {'label': 'a', 'score': 1.0})
      _append_csv(path, {'label': 'b', 'score': 2.0, 'adapt/loss': 3.0})
      with path.open() as file:
        rows = list(csv.DictReader(file))
      self.assertEqual(rows[0]['label'], 'a')
      self.assertEqual(rows[0]['score'], '1.0')
      self.assertEqual(rows[0]['adapt/loss'], '')
      self.assertEqual(rows[1]['adapt/loss'], '3.0')

  def test_run_module_exports_cir_entrypoint(self):
    self.assertTrue(callable(run_cir))


if __name__ == '__main__':
  unittest.main()
