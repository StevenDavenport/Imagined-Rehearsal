import importlib
import json
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


class AdaptiveEvalAgent(EvalAgent):

  def __init__(self, act_space):
    super().__init__(act_space)
    self.config.eval_adapt = SimpleNamespace(
        enabled=True, train_critic=False, every_k=1, steps=1)
    self.policy_modes = []
    self.latent_modes = []
    self.freeze_critic = []
    self.discarded = 0

  def policy(self, carry, obs, mode='eval', params=None):
    self.policy_modes.append(mode)
    return super().policy(carry, obs, mode=mode, params=params)

  def clone_params(self):
    return {'pol/x': np.ones(1)}

  def adapt(
      self, params, carry, steps=1, warmup=False, freeze_critic=False):
    del steps, warmup
    self.freeze_critic.append(freeze_critic)
    return params, carry, {'actor_steps': np.float32(1)}

  def extract_policy_params(self, params):
    return dict(params)

  def policy_latent(self, carry, params=None, mode='train'):
    del params
    self.latent_modes.append(mode)
    acts = {
        key: np.zeros((1, *space.shape), space.dtype)
        for key, space in self.act_space.items() if key != 'reset'
    }
    return carry, acts, {}

  def discard_params(self, params):
    if params is not None:
      self.discarded += 1

  def parameter_digests(self):
    return {'all': {'sha256': 'constant', 'arrays': 1}}


class EvalEnv(Dummy):

  def __init__(self):
    super().__init__('disc', size=(8, 8), length=2)
    self.closed = False

  def close(self):
    self.closed = True


class PairedAdaptiveEvalAgent(AdaptiveEvalAgent):

  def __init__(self, act_space):
    super().__init__(act_space)
    self.config.eval_adapt.paired_rng = True
    self.config.eval_adapt.paired_rng_seed = 700
    self.config.eval_adapt.paired_rng_stride = 1000
    self.policy_seeds = []
    self.adapt_seeds = []
    self.latent_seeds = []

  def policy(
      self, carry, obs, mode='eval', params=None,
      seed_index=None, seed_base=None):
    self.policy_seeds.append((seed_index, seed_base))
    return super().policy(carry, obs, mode=mode, params=params)

  def adapt(
      self, params, carry, steps=1, warmup=False, freeze_critic=False,
      seed_index=None, seed_base=None):
    self.adapt_seeds.append((seed_index, seed_base))
    return super().adapt(
        params, carry, steps=steps, warmup=warmup,
        freeze_critic=freeze_critic)

  def policy_latent(
      self, carry, params=None, mode='train',
      seed_index=None, seed_base=None):
    self.latent_seeds.append((seed_index, seed_base))
    return super().policy_latent(carry, params=params, mode=mode)


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


def test_eval_only_actor_ir_is_sampled_actor_only_and_episode_local(
    tmp_path, monkeypatch):
  module = importlib.import_module('embodied.run.eval_only')
  monkeypatch.setattr(module.elements.checkpoint, 'load', lambda *args: None)
  env = EvalEnv()
  agent = AdaptiveEvalAgent(env.act_space)
  logger = EvalLogger()
  args = SimpleNamespace(
      from_checkpoint='unused',
      logdir=str(tmp_path),
      envs=1,
      debug=True,
      usage={},
      log_every=1000,
      eval_episodes=2,
      eval_policy_mode='sampled',
      steps=999,
  )

  module.eval_only(
      lambda: agent, lambda index: env, lambda: logger, args)

  assert set(agent.policy_modes) == {'train'}
  assert set(agent.latent_modes) == {'train'}
  assert agent.freeze_critic and all(agent.freeze_critic)
  # Dummy length 2 yields an initial and one nonterminal adaptation per episode;
  # terminal observations must not trigger another update.
  assert len(agent.freeze_critic) == 4
  assert agent.discarded >= 4
  integrity = (tmp_path / 'eval_integrity.json').read_text()
  assert '"equal": true' in integrity


def test_eval_only_paired_rng_restarts_by_episode_and_separates_streams(
    tmp_path, monkeypatch):
  module = importlib.import_module('embodied.run.eval_only')
  monkeypatch.setattr(module.elements.checkpoint, 'load', lambda *args: None)
  env = EvalEnv()
  agent = PairedAdaptiveEvalAgent(env.act_space)
  logger = EvalLogger()
  args = SimpleNamespace(
      from_checkpoint='unused', logdir=str(tmp_path), envs=1, debug=True,
      usage={}, log_every=1000, eval_episodes=2,
      eval_policy_mode='sampled', steps=999)

  module.eval_only(
      lambda: agent, lambda index: env, lambda: logger, args)

  assert agent.policy_seeds[:3] == [(0, 700), (1, 700), (2, 700)]
  assert agent.policy_seeds[3:6] == [
      (1000, 700), (1001, 700), (1002, 700)]
  assert agent.adapt_seeds == [
      (0, 702), (1, 702), (1000, 702), (1001, 702)]
  assert agent.latent_seeds == [
      (0, 700), (1, 700), (1000, 700), (1001, 700)]
  integrity = json.loads((tmp_path / 'eval_integrity.json').read_text())
  assert integrity['paired_rng'] is True
  assert integrity['paired_rng_seed'] == 700
