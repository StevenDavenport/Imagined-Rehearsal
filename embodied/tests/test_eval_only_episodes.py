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
    self.clone_calls = 0

  def policy(self, carry, obs, mode='eval', params=None):
    self.policy_modes.append(mode)
    return super().policy(carry, obs, mode=mode, params=params)

  def clone_params(self):
    self.clone_calls += 1
    return {'pol/x': np.ones(1)}

  def export_params(self, params=None):
    return {'params': params or {'pol/x': np.zeros(1)}, 'counters': {}}

  def checkpoint_digests(self, checkpoint):
    del checkpoint
    return self.parameter_digests()

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

  def parameter_digests(self, params=None):
    del params
    return {'all': {'sha256': 'constant', 'arrays': 1}}


class EvalEnv(Dummy):

  def __init__(self, length=2):
    super().__init__('disc', size=(8, 8), length=length)
    self.closed = False

  def close(self):
    self.closed = True


class PairedAdaptiveEvalAgent(AdaptiveEvalAgent):

  def __init__(self, act_space):
    super().__init__(act_space)
    self.config.eval_adapt.paired_rng = True
    self.config.eval_adapt.paired_rng_seed = 700
    self.config.eval_adapt.paired_rng_stride = 1000
    self.config.eval_adapt.trace = SimpleNamespace(
        enabled=True, filename='stage5_trace.jsonl')
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


class GatedAdaptiveEvalAgent(AdaptiveEvalAgent):

  def __init__(self, act_space, entropy):
    super().__init__(act_space)
    self.config.eval_adapt.imag_length = 15
    self.config.eval_adapt.start_batch = 128
    self.config.eval_adapt.objective = 'reward_only'
    self.config.eval_adapt.actent = 0.0
    self.config.eval_adapt.gate = SimpleNamespace(
        enabled=True, kind='entropy', audit=False,
        entropy_threshold=.5, js_threshold=.5,
        min_success_rate=.03, direction_threshold=.2)
    self.entropy = entropy
    self.gate_calls = 0
    self.prepared_calls = 0

  def gate_features(
      self, carry, params=None, horizon=15, start_batch=128,
      consequence=False, **kwargs):
    del carry, params, horizon, start_batch, kwargs
    self.gate_calls += 1
    return ({'prepared': np.ones(1)} if consequence else {}), {
        'actor_entropy': np.float32(self.entropy),
        'posterior_entropy': np.float32(self.entropy),
        'js_disagreement': np.float32(.1),
        'success_rate': np.float32(.5 if consequence else 0),
        'success_action_concentration': np.float32(.5 if consequence else 0),
        'success_action_divergence': np.float32(.5 if consequence else 0),
    }

  def adapt_prepared(self, params, carry, rollout, steps=1, **kwargs):
    del rollout, kwargs
    self.prepared_calls += 1
    return super().adapt(params, carry, steps=steps, freeze_critic=True)


class VirtualAdaptiveEvalAgent(GatedAdaptiveEvalAgent):

  def __init__(self, act_space, delta, dormant_every=3):
    super().__init__(act_space, entropy=.1)
    self.config.eval_adapt.paired_rng = True
    self.config.eval_adapt.paired_rng_seed = 700
    self.config.eval_adapt.paired_rng_stride = 1000
    self.config.eval_adapt.trace = SimpleNamespace(
        enabled=True, filename='stage5_trace.jsonl')
    self.config.eval_adapt.gate.kind = 'virtual_update'
    self.config.eval_adapt.gate.virtual_score = 'reward_return'
    self.config.eval_adapt.gate.virtual_min_improvement = 0.0
    self.config.eval_adapt.gate.virtual_max_baseline_score = 1e30
    self.config.eval_adapt.gate.virtual_dormant_every = dormant_every
    self.config.eval_adapt.gate.virtual_active_every = 1
    self.config.eval_adapt.gate.virtual_deactivate_after = 2
    self.delta = float(delta)

  def policy(
      self, carry, obs, mode='eval', params=None,
      seed_index=None, seed_base=None):
    del seed_index, seed_base
    return super().policy(carry, obs, mode=mode, params=params)

  def policy_latent(
      self, carry, params=None, mode='train',
      seed_index=None, seed_base=None):
    del seed_index, seed_base
    return super().policy_latent(carry, params=params, mode=mode)

  def clone_params(self):
    self.clone_calls += 1
    return {
        'pol/x': np.zeros(1, np.float32),
        'adapt_actor_opt/x': np.zeros(1, np.float32),
    }

  def snapshot_adapt_actor_state(self, params):
    return {key: value.copy() for key, value in params.items()}

  def restore_adapt_actor_state(self, params, snapshot):
    del params
    return snapshot

  def gate_features(
      self, carry, params=None, horizon=15, start_batch=128,
      consequence=False, **kwargs):
    del carry, horizon, start_batch, kwargs
    self.gate_calls += 1
    score = float(params['pol/x'][0])
    return ({'prepared': np.ones(1)} if consequence else {}), {
        'actor_entropy': np.float32(.1),
        'posterior_entropy': np.float32(.1),
        'js_disagreement': np.float32(.1),
        'success_rate': np.float32(score > 0),
        'success_action_concentration': np.float32(0),
        'success_action_divergence': np.float32(0),
        'reward_return_mean': np.float32(score),
        'reward_return_std': np.float32(0),
        'reward_return_sem': np.float32(0),
    }

  def adapt_prepared(self, params, carry, rollout, steps=1, **kwargs):
    del rollout, kwargs
    self.prepared_calls += 1
    candidate = {key: value.copy() for key, value in params.items()}
    candidate['pol/x'] += self.delta * steps
    return candidate, carry, {'actor_steps': np.float32(steps)}


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


def test_eval_only_warmup_is_timed_but_not_recorded(tmp_path, monkeypatch):
  module = importlib.import_module('embodied.run.eval_only')
  monkeypatch.setattr(module.elements.checkpoint, 'load', lambda *args: None)
  env = EvalEnv()
  agent = EvalAgent(env.act_space)
  logger = EvalLogger()
  args = SimpleNamespace(
      from_checkpoint='unused', logdir=str(tmp_path), envs=1, debug=True,
      usage={}, log_every=1000, eval_episodes=3, eval_warmup_episodes=1,
      steps=999)

  module.eval_only(lambda: agent, lambda index: env, lambda: logger, args)

  assert len(logger.episodes) == 3
  assert int(logger.step) == 12
  runtime = json.loads((tmp_path / 'eval_runtime.json').read_text())
  assert runtime['warmup_episodes'] == 1
  assert runtime['environment_steps'] == 9


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
  assert agent.clone_calls == 2
  assert agent.discarded >= 4
  integrity = (tmp_path / 'eval_integrity.json').read_text()
  assert '"equal": true' in integrity


def test_eval_only_run_persistence_reuses_adapted_params_across_episodes(
    tmp_path, monkeypatch):
  module = importlib.import_module('embodied.run.eval_only')
  monkeypatch.setattr(module.elements.checkpoint, 'load', lambda *args: None)
  env = EvalEnv()
  agent = AdaptiveEvalAgent(env.act_space)
  agent.config.eval_adapt.persistence = 'run'
  agent.config.eval_adapt.output_checkpoint = str(tmp_path / 'adapted')
  logger = EvalLogger()
  args = SimpleNamespace(
      from_checkpoint='unused', logdir=str(tmp_path), envs=1, debug=True,
      usage={}, log_every=1000, eval_episodes=2,
      eval_policy_mode='sampled', steps=999)

  module.eval_only(lambda: agent, lambda index: env, lambda: logger, args)

  assert agent.clone_calls == 1
  assert len(agent.freeze_critic) == 4
  integrity = json.loads((tmp_path / 'eval_integrity.json').read_text())
  assert integrity['persistence'] == 'run'
  assert (tmp_path / 'adapted' / 'agent.pkl').is_file()
  assert (tmp_path / 'adapted' / 'done').is_file()
  manifest = json.loads((
      tmp_path / 'adapted' / 'persistent_ir_manifest.json').read_text())
  assert manifest['episodes'] == 2
  assert manifest['frozen_component_violations'] == {}


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


def test_eval_only_entropy_gate_skips_updates_below_threshold(
    tmp_path, monkeypatch):
  module = importlib.import_module('embodied.run.eval_only')
  monkeypatch.setattr(module.elements.checkpoint, 'load', lambda *args: None)
  env = EvalEnv()
  agent = GatedAdaptiveEvalAgent(env.act_space, entropy=.2)
  logger = EvalLogger()
  args = SimpleNamespace(
      from_checkpoint='unused', logdir=str(tmp_path), envs=1, debug=True,
      usage={}, log_every=1000, eval_episodes=2,
      eval_policy_mode='sampled', steps=999)

  module.eval_only(lambda: agent, lambda index: env, lambda: logger, args)

  assert agent.gate_calls == 4
  assert not agent.freeze_critic
  integrity = json.loads((tmp_path / 'eval_integrity.json').read_text())
  assert integrity['gate_enabled'] is True
  assert integrity['gate_kind'] == 'entropy'


def test_eval_only_virtual_gate_commits_improvements_every_active_step(
    tmp_path, monkeypatch):
  module = importlib.import_module('embodied.run.eval_only')
  monkeypatch.setattr(module.elements.checkpoint, 'load', lambda *args: None)
  env = EvalEnv(length=5)
  agent = VirtualAdaptiveEvalAgent(env.act_space, delta=1)
  logger = EvalLogger()
  args = SimpleNamespace(
      from_checkpoint='unused', logdir=str(tmp_path), envs=1, debug=True,
      usage={}, log_every=1000, eval_episodes=1,
      eval_policy_mode='sampled', steps=999)

  module.eval_only(lambda: agent, lambda index: env, lambda: logger, args)

  traces = [json.loads(line) for line in (
      tmp_path / 'stage5_trace.jsonl').read_text().splitlines()]
  probes = [row for row in traces if 'gate_score_improvement' in row]
  assert len(probes) == 5
  assert all(row['adapt_trigger'] == 1 for row in probes)
  assert all(np.isclose(row['gate_score_improvement'], 1) for row in probes)
  assert agent.prepared_calls == 5


def test_eval_only_virtual_gate_rolls_back_and_uses_dormant_cadence(
    tmp_path, monkeypatch):
  module = importlib.import_module('embodied.run.eval_only')
  monkeypatch.setattr(module.elements.checkpoint, 'load', lambda *args: None)
  env = EvalEnv(length=7)
  agent = VirtualAdaptiveEvalAgent(
      env.act_space, delta=-1, dormant_every=3)
  logger = EvalLogger()
  args = SimpleNamespace(
      from_checkpoint='unused', logdir=str(tmp_path), envs=1, debug=True,
      usage={}, log_every=1000, eval_episodes=1,
      eval_policy_mode='sampled', steps=999)

  module.eval_only(lambda: agent, lambda index: env, lambda: logger, args)

  traces = [json.loads(line) for line in (
      tmp_path / 'stage5_trace.jsonl').read_text().splitlines()]
  probes = [row for row in traces if 'gate_score_improvement' in row]
  assert [row['episode_step'] for row in probes] == [0, 3, 6]
  assert all(row['adapt_trigger'] == 0 for row in probes)
  assert all(np.isclose(row['gate_score_before'], 0) for row in probes)
  assert all(np.isclose(row['gate_score_after'], -1) for row in probes)
