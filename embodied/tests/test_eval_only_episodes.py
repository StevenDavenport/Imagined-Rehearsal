import importlib
from types import SimpleNamespace

import elements
import numpy as np

from embodied.envs.dummy import Dummy


class EvalAgent:

  def __init__(self, act_space):
    self.act_space = act_space
    self.config = SimpleNamespace(
        eval_adapt=SimpleNamespace(enabled=False, every_k=0))

  def init_policy(self, batch_size):
    return ()

  def policy(self, carry, obs, mode='eval', params=None):
    del obs, mode, params
    batch_size = 1
    acts = {
        key: np.zeros((batch_size, *space.shape), space.dtype)
        for key, space in self.act_space.items() if key != 'reset'
    }
    return carry, acts, {}

  def load(self, data, regex=None):
    del data, regex


class EvalEnv(Dummy):

  def __init__(self):
    super().__init__('disc', size=(8, 8), length=2)
    self.closed = False

  def close(self):
    self.closed = True


class EvalLogger:

  def __init__(self):
    self.step = elements.Counter()
    self.episodes = []
    self.closed = False

  def add(self, values, prefix=None):
    if prefix == 'episode':
      self.episodes.append(dict(values))

  def write(self):
    pass

  def close(self):
    self.closed = True


def test_eval_only_can_stop_at_an_exact_episode_count(tmp_path, monkeypatch):
  module = importlib.import_module('embodied.run.eval_only')
  monkeypatch.setattr(module.elements.checkpoint, 'load', lambda *args: None)
  env = EvalEnv()
  agent = EvalAgent(env.act_space)
  logger = EvalLogger()
  args = SimpleNamespace(
      from_checkpoint='unused',
      logdir=str(tmp_path),
      envs=1,
      debug=True,
      usage={},
      log_every=1000,
      eval_episodes=3,
      steps=999,
  )

  module.eval_only(
      lambda: agent, lambda index: env, lambda: logger, args)

  assert len(logger.episodes) == 3
  assert int(logger.step) == 9
  assert env.closed
  assert logger.closed
