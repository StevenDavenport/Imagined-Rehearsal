#!/usr/bin/env python3
"""Frozen offline twin-Q qualification for two-goal MiniGrid CRL.

This experiment asks whether ordinary one-step counterfactual relabelling can
train a goal-selective critic without an oracle value target. The Dreamer
encoder, RSSM, actor, reward/continuation heads, and existing critics remain
immutable. Two newly initialized discrete-action Q networks bootstrap only
from EMA copies of themselves under the frozen actor.

The factual control repeats each observed goal label so it receives exactly
the same number of labelled examples per update as the counterfactual arm.
The counterfactual arm queries both selected goals for every transition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import pickle
import subprocess
import sys
import time
from functools import partial as bind
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np

from checkpoint_utils import resolve_checkpoint_path


elements = embodied = jax = jnp = optax = yaml = dv3_main = None

EXPERIMENT_NAME = 'minigrid_offline_twin_q_qualification_v1'
FORMAT = 1
ARMS = ('factual_only', 'counterfactual')
GOAL_NAMES = {
    0: 'go_to_door',
    1: 'fetch',
    2: 'go_to_object',
    3: 'put_near',
    4: 'unlock_pickup',
    5: 'pickup_dist',
}


def load_runtime_dependencies() -> None:
  """Keep specification and target helpers lightweight for unit tests."""
  global elements, embodied, jax, jnp, optax, yaml, dv3_main
  if elements is not None:
    return
  import elements as elements_module
  import embodied as embodied_module
  import jax as jax_module
  import jax.numpy as jnp_module
  import optax as optax_module
  import ruamel.yaml as yaml_module
  from dreamerv3 import main as main_module
  elements = elements_module
  embodied = embodied_module
  jax = jax_module
  jnp = jnp_module
  optax = optax_module
  yaml = yaml_module
  dv3_main = main_module


def parse_ints(text: str) -> tuple[int, ...]:
  values = tuple(int(value.strip()) for value in text.split(',') if value.strip())
  if not values:
    raise argparse.ArgumentTypeError('Expected at least one integer')
  return values


def json_dump(path: pathlib.Path, value: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
  temporary.replace(path)


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
  if not rows:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  keys = list(rows[0])
  for row in rows[1:]:
    keys.extend(key for key in row if key not in keys)
  temporary = path.with_suffix(path.suffix + '.tmp')
  with temporary.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def sha256_file(path: pathlib.Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def git_revision() -> str:
  try:
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
  except (OSError, subprocess.CalledProcessError):
    return 'unknown'


def elapsed_text(seconds: float) -> str:
  seconds = max(0, int(seconds))
  hours, remainder = divmod(seconds, 3600)
  minutes, seconds = divmod(remainder, 60)
  return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def counterfactual_targets(
    actual_goal: np.ndarray,
    query_goal: np.ndarray,
    reward: np.ndarray,
    goal_complete: np.ndarray,
    is_physical_terminal: np.ndarray,
    is_terminal: np.ndarray,
    is_last: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Return event-local reward and continuation targets for queried goals.

  Reaching the recorded goal terminates only that goal's relabelled task.
  Physical termination and non-goal episode endings terminate every query.
  This deliberately does not use a reward oracle or an existing value head.
  """
  actual_goal = np.asarray(actual_goal)
  query_goal = np.asarray(query_goal)
  reward = np.asarray(reward, np.float32)
  goal_complete = np.asarray(goal_complete, bool)
  physical = np.asarray(is_physical_terminal, bool)
  terminal = np.asarray(is_terminal, bool)
  last = np.asarray(is_last, bool)
  matching = query_goal == actual_goal
  reward_target = np.where(matching, reward, 0.0).astype(np.float32)
  non_goal_end = (terminal | last) & ~goal_complete
  stop = physical | non_goal_end | (goal_complete & matching)
  return reward_target, (~stop).astype(np.float32)


def factual_return_to_go(
    reward: np.ndarray, is_terminal: np.ndarray, is_last: np.ndarray,
    gamma: float,
) -> np.ndarray:
  """Return from state t for rewards observed at t+1 onward."""
  reward = np.asarray(reward, np.float32)
  terminal = np.asarray(is_terminal, bool)
  last = np.asarray(is_last, bool)
  result = np.zeros(len(reward), np.float32)
  future = 0.0
  for step in range(len(reward) - 2, -1, -1):
    next_step = step + 1
    future = float(reward[next_step]) + float(gamma) * float(
        not (terminal[next_step] or last[next_step])) * future
    result[step] = future
  return result


def distance_to_next_reward(reward: np.ndarray) -> np.ndarray:
  """Number of actions from state t to its next positive reward, or -1."""
  reward = np.asarray(reward)
  result = np.full(len(reward), -1, np.int32)
  next_reward = -1
  for step in range(len(reward) - 1, -1, -1):
    if reward[step] > 0:
      next_reward = step
    if next_reward > step:
      result[step] = next_reward - step
  return result


def build_spec(args, checkpoint: pathlib.Path) -> dict[str, Any]:
  goals = list(args.goals)
  if len(goals) != 2 or len(set(goals)) != 2:
    raise ValueError('This qualification requires exactly two distinct goals')
  unknown = [goal for goal in goals if goal not in GOAL_NAMES]
  if unknown:
    raise ValueError(f'Unknown MiniGrid goal IDs: {unknown}')
  checkpoint_marker = checkpoint / 'done'
  return {
      'format': FORMAT,
      'name': EXPERIMENT_NAME,
      'git_revision': git_revision(),
      'source': {
          'checkpoint': str(checkpoint),
          'checkpoint_done_sha256': sha256_file(checkpoint_marker),
          'replay': str(pathlib.Path(args.replay_root).expanduser().resolve()),
      },
      'goals': goals,
      'goal_names': [GOAL_NAMES[goal] for goal in goals],
      'arms': list(ARMS),
      'q_seeds': list(args.q_seeds),
      'selection': {
          'seed': args.selection_seed,
          'train_successes_per_goal': args.train_successes_per_goal,
          'train_failures_per_goal': args.train_failures_per_goal,
          'heldout_successes_per_goal': args.heldout_successes_per_goal,
          'heldout_failures_per_goal': args.heldout_failures_per_goal,
      },
      'critic': {
          'kind': 'independent_discrete_action_twin_q',
          'hidden_size': args.hidden_size,
          'hidden_layers': 2,
          'batch_size': args.batch_size,
          'updates': args.updates,
          'learning_rate': args.learning_rate,
          'target_ema_tau': args.target_tau,
          'gamma': args.gamma,
          'report_every': args.report_every,
          'checkpoint_every': args.checkpoint_every,
          'output_initialization_scale': 1e-3,
      },
      'invariants': {
          'dreamer_parameters_frozen': True,
          'actor_frozen': True,
          'existing_value_used_as_bootstrap': False,
          'reward_oracle_used': False,
          'target': (
              'r_g + gamma * c_g * E_frozen_actor['
              'min(Q1_target,Q2_target)]'),
          'factual_control_dose_matched': True,
          'counterfactual_scope': 'observed event labels for both selected goals',
      },
  }


def write_immutable_spec(path: pathlib.Path, spec: dict[str, Any]) -> None:
  if path.exists():
    existing = json.loads(path.read_text())
    if existing != spec:
      raise RuntimeError(
          f'Experiment specification differs from existing immutable file: {path}')
  else:
    json_dump(path, spec)


def update_status(outdir: pathlib.Path, **updates: Any) -> None:
  path = outdir / 'supervisor_status.json'
  status = json.loads(path.read_text()) if path.exists() else {
      'format': FORMAT, 'name': EXPERIMENT_NAME, 'state': 'created'}
  status.update(updates)
  status['updated_at_unix'] = time.time()
  json_dump(path, status)


def prepare(args, outdir: pathlib.Path, checkpoint: pathlib.Path) -> None:
  from embodied.run.head_audit import create_split_selections
  outdir.mkdir(parents=True, exist_ok=True)
  spec = build_spec(args, checkpoint)
  write_immutable_spec(outdir / 'experiment_spec.json', spec)
  selection_dir = outdir / 'selections'
  train_path = selection_dir / 'train.json'
  heldout_path = selection_dir / 'heldout.json'
  if not train_path.exists() and not heldout_path.exists():
    create_split_selections(
        args.replay_root, train_path, heldout_path, goals=args.goals,
        train_successes_per_goal=args.train_successes_per_goal,
        train_failures_per_goal=args.train_failures_per_goal,
        heldout_successes_per_goal=args.heldout_successes_per_goal,
        heldout_failures_per_goal=args.heldout_failures_per_goal,
        seed=args.selection_seed)
  elif not (train_path.exists() and heldout_path.exists()):
    raise RuntimeError('Only one selection file exists; refusing a mixed bank')
  update_status(outdir, state='prepared')
  print(f'Prepared immutable experiment at {outdir}', flush=True)


def make_agent_config(args, outdir: pathlib.Path):
  (outdir / 'encoder_runtime').mkdir(parents=True, exist_ok=True)
  raw = yaml.YAML(typ='safe').load(
      elements.Path(ROOT / 'dreamerv3' / 'configs.yaml').read())
  config = elements.Config(raw['defaults'])
  for name in ('minigrid', 'minigrid_size12m', 'minigrid_head_audit'):
    config = config.update(raw[name])
  return config.update(
      task='minigrid_go_to_door', seed=0,
      logdir=str(outdir / 'encoder_runtime'),
      jax=config.jax.update(
          platform=args.jax_platform, prealloc=False, precompile=False),
      run=config.run.update(envs=1, debug=True))


def _pad(array: np.ndarray, amount: int, *, first: bool = False) -> np.ndarray:
  if amount <= 0:
    return array
  padding = np.zeros((amount, *array.shape[1:]), array.dtype)
  if first:
    padding[...] = True
  return np.concatenate([array, padding], 0)


def _previous_actions(data: dict[str, np.ndarray], act_space: dict) -> dict:
  result = {}
  for key in act_space:
    action = np.asarray(data[key])
    previous = np.zeros_like(action)
    previous[1:] = action[:-1]
    result[key] = previous
  return result


def encode_episode(
    agent, record: dict[str, Any], goals: tuple[int, ...],
    chunk_length: int, gamma: float,
) -> dict[str, np.ndarray]:
  path = pathlib.Path(record['path'])
  with np.load(path) as archive:
    data = {key: np.array(archive[key], copy=True) for key in archive.files}
  length = len(data['reward'])
  if not np.all(np.asarray(data['goal_id']) == int(record['goal_id'])):
    raise RuntimeError(f'Goal changed within episode: {path}')
  previous = _previous_actions(data, agent.act_space)
  carry = agent.init_observe(1)
  outputs = {key: [] for key in (
      'deter', 'stoch', 'actor_probability', 'value', 'slowvalue')}
  for start in range(0, length, chunk_length):
    stop = min(length, start + chunk_length)
    valid = stop - start
    pad = chunk_length - valid
    obs = {
        key: _pad(np.asarray(data[key][start:stop]), pad,
                  first=(key == 'is_first'))[None]
        for key in agent.obs_space}
    prevact = {
        key: _pad(np.asarray(value[start:stop]), pad)[None]
        for key, value in previous.items()}
    carry, chunk = agent.head_audit_chunk(
        carry, obs, prevact, actor_distribution=True, latent=True)
    for key in outputs:
      outputs[key].append(np.asarray(chunk[key])[0, :valid])
  outputs = {key: np.concatenate(value, 0) for key, value in outputs.items()}
  goal_index = np.asarray(goals, np.int32)
  state_index = np.arange(max(length - 1, 0), dtype=np.int32)
  valid = (
      ~np.asarray(data['is_last'][:-1], bool) &
      ~np.asarray(data['is_first'][1:], bool))
  state_index = state_index[valid]
  next_index = state_index + 1
  returns = factual_return_to_go(
      data['reward'], data['is_terminal'], data['is_last'], gamma)
  distance = distance_to_next_reward(data['reward'])
  actual_local = goals.index(int(record['goal_id']))
  return {
      'deter': outputs['deter'].astype(np.float16),
      'stoch': outputs['stoch'].astype(np.uint8),
      'actor_probability': outputs['actor_probability'][:, goal_index].astype(
          np.float16),
      'source_value': outputs['value'][:, goal_index].astype(np.float32),
      'source_slowvalue': outputs['slowvalue'][:, goal_index].astype(np.float32),
      'state_index': state_index,
      'next_state_index': next_index,
      'action': np.asarray(data['action'][state_index], np.int32),
      'actual_goal': np.full(len(state_index), actual_local, np.int8),
      'reward': np.asarray(data['reward'][next_index], np.float32),
      'goal_complete': np.asarray(data['goal_complete'][next_index], bool),
      'is_physical_terminal': np.asarray(
          data['is_physical_terminal'][next_index], bool),
      'is_terminal': np.asarray(data['is_terminal'][next_index], bool),
      'is_last': np.asarray(data['is_last'][next_index], bool),
      'return_to_go': returns[state_index],
      'distance_to_reward': distance[state_index],
      'is_initial': np.asarray(data['is_first'][state_index], bool),
      'episode_success': np.full(
          len(state_index), bool(record['success']), bool),
      'episode_id': np.full(
          len(state_index), int(record['episode_id']), np.int64),
      'timestep': state_index,
  }


def encode_split(
    agent, selection_path: pathlib.Path, cache_dir: pathlib.Path,
    args,
) -> None:
  from embodied.run.head_audit import load_selection
  selection = load_selection(selection_path)
  cache_dir.mkdir(parents=True, exist_ok=True)
  rows = []
  total = len(selection['episodes'])
  for number, record in enumerate(selection['episodes'], 1):
    filename = f'episode_{int(record["episode_id"]):08d}.npz'
    target = cache_dir / filename
    if not target.exists():
      arrays = encode_episode(
          agent, record, args.goals, args.chunk_length, args.gamma)
      temporary = target.with_suffix(target.suffix + '.tmp')
      with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
      temporary.replace(target)
    with np.load(target) as arrays:
      states = len(arrays['deter'])
      transitions = len(arrays['state_index'])
    rows.append({
        'episode_id': int(record['episode_id']),
        'goal_id': int(record['goal_id']),
        'success': bool(record['success']),
        'file': filename, 'states': states, 'transitions': transitions,
    })
    print(
        f'Encoded {cache_dir.name} episode {number}/{total}: '
        f'goal={record["goal_id"]} success={record["success"]} '
        f'states={states}', flush=True)
  json_dump(cache_dir / 'manifest.json', {
      'format': FORMAT, 'kind': 'minigrid_twin_q_latent_cache',
      'selection': str(selection_path), 'episodes': rows,
      'states': sum(row['states'] for row in rows),
      'transitions': sum(row['transitions'] for row in rows),
  })


def encode(args, outdir: pathlib.Path, checkpoint: pathlib.Path) -> None:
  load_runtime_dependencies()
  config = make_agent_config(args, outdir)
  agent = dv3_main.make_agent(config)
  elements.checkpoint.load(str(checkpoint), {'agent': agent.load})
  before = agent.parameter_digests()
  cache_root = outdir / 'cache'
  encode_split(
      agent, outdir / 'selections' / 'train.json', cache_root / 'train', args)
  encode_split(
      agent, outdir / 'selections' / 'heldout.json', cache_root / 'heldout', args)
  after = agent.parameter_digests()
  integrity = {
      'before': before, 'after': after,
      'unchanged': before == after,
  }
  json_dump(outdir / 'encoder_integrity.json', integrity)
  if not integrity['unchanged']:
    raise RuntimeError('Dreamer parameters changed during latent encoding')
  update_status(outdir, state='encoded', encoder_integrity=True)


STATE_KEYS = (
    'deter', 'stoch', 'actor_probability', 'source_value', 'source_slowvalue')
TRANSITION_KEYS = (
    'action', 'actual_goal', 'reward', 'goal_complete',
    'is_physical_terminal', 'is_terminal', 'is_last', 'return_to_go',
    'distance_to_reward', 'is_initial', 'episode_success', 'episode_id',
    'timestep')


def load_cache(cache_dir: pathlib.Path) -> dict[str, np.ndarray]:
  manifest = json.loads((cache_dir / 'manifest.json').read_text())
  state_values = {key: [] for key in STATE_KEYS}
  transition_values = {key: [] for key in TRANSITION_KEYS}
  state_indices, next_indices = [], []
  offset = 0
  for row in manifest['episodes']:
    with np.load(cache_dir / row['file']) as archive:
      for key in STATE_KEYS:
        state_values[key].append(np.array(archive[key], copy=True))
      for key in TRANSITION_KEYS:
        transition_values[key].append(np.array(archive[key], copy=True))
      state_indices.append(np.asarray(archive['state_index'], np.int32) + offset)
      next_indices.append(
          np.asarray(archive['next_state_index'], np.int32) + offset)
      offset += len(archive['deter'])
  data = {
      key: np.concatenate(values, 0)
      for key, values in {**state_values, **transition_values}.items()}
  data['state_index'] = np.concatenate(state_indices, 0)
  data['next_state_index'] = np.concatenate(next_indices, 0)
  return data


def init_q_params(key, input_size: int, hidden_size: int, action_count: int):
  keys = jax.random.split(key, 3)
  return {
      'w1': jax.random.normal(keys[0], (input_size, hidden_size)) * np.sqrt(
          2.0 / input_size),
      'b1': jnp.zeros((hidden_size,), jnp.float32),
      'w2': jax.random.normal(keys[1], (hidden_size, hidden_size)) * np.sqrt(
          2.0 / hidden_size),
      'b2': jnp.zeros((hidden_size,), jnp.float32),
      'wout': jax.random.normal(
          keys[2], (hidden_size, action_count)) * 1e-3,
      'bout': jnp.zeros((action_count,), jnp.float32),
  }


def q_values(params, feature, goal, goal_count: int):
  goal_onehot = jax.nn.one_hot(goal, goal_count, dtype=jnp.float32)
  inp = jnp.concatenate([feature.astype(jnp.float32), goal_onehot], -1)
  hidden = jax.nn.gelu(inp @ params['w1'] + params['b1'])
  hidden = jax.nn.gelu(hidden @ params['w2'] + params['b2'])
  return hidden @ params['wout'] + params['bout']


def device_cache(data: dict[str, np.ndarray]) -> dict[str, Any]:
  return {key: jax.device_put(value) for key, value in data.items()}


def state_feature(data, index):
  deter = data['deter'][index].astype(jnp.float32)
  stoch = data['stoch'][index].reshape((len(index), -1)).astype(jnp.float32)
  return jnp.concatenate([deter, stoch], -1)


def queried_goals(actual, goal_count: int, counterfactual: bool):
  if counterfactual:
    return jnp.broadcast_to(
        jnp.arange(goal_count, dtype=jnp.int32), (len(actual), goal_count))
  return jnp.broadcast_to(actual[:, None], (len(actual), goal_count))


def jax_targets(data, transition, query):
  actual = data['actual_goal'][transition, None]
  matching = query == actual
  complete = data['goal_complete'][transition, None]
  physical = data['is_physical_terminal'][transition, None]
  terminal = data['is_terminal'][transition, None]
  last = data['is_last'][transition, None]
  reward = jnp.where(matching, data['reward'][transition, None], 0.0)
  non_goal_end = (terminal | last) & ~complete
  continuation = ~(physical | non_goal_end | (complete & matching))
  return reward.astype(jnp.float32), continuation.astype(jnp.float32)


def make_train_step(
    optimizer, batch_size: int, goal_count: int, gamma: float,
    target_tau: float, counterfactual: bool,
):
  def step(params, target_params, opt_state, key, data):
    key, sample_key = jax.random.split(key)
    count = len(data['action'])
    transition = jax.random.randint(
        sample_key, (batch_size,), 0, count, dtype=jnp.int32)
    current = data['state_index'][transition]
    following = data['next_state_index'][transition]
    current_feature = state_feature(data, current)
    next_feature = state_feature(data, following)
    actual = data['actual_goal'][transition]
    query = queried_goals(actual, goal_count, counterfactual)
    repeated_current = jnp.broadcast_to(
        current_feature[:, None],
        (batch_size, goal_count, current_feature.shape[-1])).reshape(
            (batch_size * goal_count, -1))
    repeated_next = jnp.broadcast_to(
        next_feature[:, None],
        (batch_size, goal_count, next_feature.shape[-1])).reshape(
            (batch_size * goal_count, -1))
    flat_query = query.reshape(-1)
    action = jnp.broadcast_to(
        data['action'][transition, None], query.shape).reshape(-1)
    reward, continuation = jax_targets(data, transition, query)
    actor_all = data['actor_probability'][following]
    actor_index = jnp.broadcast_to(query[..., None], query.shape + (
        actor_all.shape[-1],))
    actor = jnp.take_along_axis(actor_all, actor_index, axis=1)

    target_q1 = q_values(
        target_params[0], repeated_next, flat_query, goal_count).reshape(
            (batch_size, goal_count, -1))
    target_q2 = q_values(
        target_params[1], repeated_next, flat_query, goal_count).reshape(
            (batch_size, goal_count, -1))
    next_value = (actor * jnp.minimum(target_q1, target_q2)).sum(-1)
    target = jax.lax.stop_gradient(
        reward + gamma * continuation * next_value)

    def lossfn(candidate):
      q1 = q_values(
          candidate[0], repeated_current, flat_query, goal_count).reshape(
              (batch_size, goal_count, -1))
      q2 = q_values(
          candidate[1], repeated_current, flat_query, goal_count).reshape(
              (batch_size, goal_count, -1))
      chosen1 = jnp.take_along_axis(q1, action.reshape(
          (batch_size, goal_count, 1)), -1)[..., 0]
      chosen2 = jnp.take_along_axis(q2, action.reshape(
          (batch_size, goal_count, 1)), -1)[..., 0]
      loss1 = optax.huber_loss(chosen1, target, delta=1.0)
      loss2 = optax.huber_loss(chosen2, target, delta=1.0)
      loss = (loss1 + loss2).mean()
      metrics = {
          'loss': loss,
          'target_mean': target.mean(),
          'q1_mean': chosen1.mean(),
          'q2_mean': chosen2.mean(),
          'q_disagreement': jnp.abs(chosen1 - chosen2).mean(),
      }
      return loss, metrics

    (_, metrics), grads = jax.value_and_grad(lossfn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    target_params = jax.tree.map(
        lambda target_value, value:
        (1.0 - target_tau) * target_value + target_tau * value,
        target_params, params)
    return params, target_params, opt_state, key, metrics
  return jax.jit(step)


def make_predict(goal_count: int):
  def predict(params, deter, stoch, actor):
    feature = jnp.concatenate([
        deter.astype(jnp.float32),
        stoch.reshape((len(stoch), -1)).astype(jnp.float32)], -1)
    batch = len(feature)
    query = jnp.broadcast_to(
        jnp.arange(goal_count, dtype=jnp.int32), (batch, goal_count))
    repeated = jnp.broadcast_to(
        feature[:, None], (batch, goal_count, feature.shape[-1])).reshape(
            (batch * goal_count, -1))
    flat_query = query.reshape(-1)
    q1 = q_values(params[0], repeated, flat_query, goal_count).reshape(
        (batch, goal_count, -1))
    q2 = q_values(params[1], repeated, flat_query, goal_count).reshape(
        (batch, goal_count, -1))
    value = (actor * jnp.minimum(q1, q2)).sum(-1)
    return value, q1, q2
  return jax.jit(predict)


def predict_all(params, data, batch_size: int, goal_count: int):
  predict = make_predict(goal_count)
  values, q1s, q2s = [], [], []
  count = len(data['deter'])
  for start in range(0, count, batch_size):
    stop = min(count, start + batch_size)
    value, q1, q2 = predict(
        params, data['deter'][start:stop], data['stoch'][start:stop],
        data['actor_probability'][start:stop])
    values.append(np.asarray(jax.device_get(value)))
    q1s.append(np.asarray(jax.device_get(q1)))
    q2s.append(np.asarray(jax.device_get(q2)))
  return (
      np.concatenate(values, 0), np.concatenate(q1s, 0),
      np.concatenate(q2s, 0))


def _mean(values: np.ndarray) -> float:
  values = np.asarray(values, np.float64)
  return float(values.mean()) if len(values) else float('nan')


def _mae(prediction: np.ndarray, target: np.ndarray) -> float:
  return _mean(np.abs(np.asarray(prediction) - np.asarray(target)))


def evaluation_metrics(
    params, heldout, gamma: float, goal_count: int,
    evaluation_batch_size: int,
) -> dict[str, float]:
  value, q1, q2 = predict_all(
      params, heldout, evaluation_batch_size, goal_count)
  transition = np.arange(len(heldout['action']))
  current = heldout['state_index']
  following = heldout['next_state_index']
  actual = heldout['actual_goal'].astype(np.int64)
  other = 1 - actual
  factual_value = value[current, actual]
  other_value = value[current, other]
  source = heldout['source_slowvalue'][current]
  source_factual = source[np.arange(len(current)), actual]
  source_other = source[np.arange(len(current)), other]
  all_query = np.broadcast_to(np.arange(goal_count), (len(current), goal_count))
  reward, continuation = counterfactual_targets(
      heldout['actual_goal'][:, None], all_query,
      heldout['reward'][:, None], heldout['goal_complete'][:, None],
      heldout['is_physical_terminal'][:, None],
      heldout['is_terminal'][:, None], heldout['is_last'][:, None])
  bellman_target = reward + gamma * continuation * value[following]
  action = heldout['action'].astype(np.int64)
  qmin = np.minimum(q1[current], q2[current])
  chosen = np.take_along_axis(
      qmin, np.broadcast_to(action[:, None, None],
                            (len(action), goal_count, 1)), axis=2)[..., 0]
  initial = heldout['is_initial'].astype(bool)
  successful = heldout['episode_success'].astype(bool)
  initial_success = initial & successful
  metrics = {
      'factual_return_mae': _mae(factual_value, heldout['return_to_go']),
      'factual_return_bias': _mean(factual_value - heldout['return_to_go']),
      'factual_value_mean': _mean(factual_value),
      'nonmatching_value_mean': _mean(other_value),
      'nonmatching_value_p95': float(np.quantile(other_value, .95)),
      'matching_minus_nonmatching_mean': _mean(factual_value - other_value),
      'both_goals_above_0p5_fraction': _mean((value[current] > .5).all(-1)),
      'successful_initial_goal_rank_accuracy': _mean(
          factual_value[initial_success] > other_value[initial_success]),
      'bellman_mae_all_queries': _mae(chosen, bellman_target),
      'bellman_mae_factual': _mae(
          chosen[transition, actual], bellman_target[transition, actual]),
      'q_disagreement_mean': _mean(np.abs(q1[current] - q2[current])),
      'source_factual_return_mae': _mae(
          source_factual, heldout['return_to_go']),
      'source_factual_value_mean': _mean(source_factual),
      'source_nonmatching_value_mean': _mean(source_other),
      'source_both_goals_above_0p5_fraction': _mean(
          (source > .5).all(-1)),
      'value_min': float(value[current].min()),
      'value_max': float(value[current].max()),
  }
  distance = heldout['distance_to_reward']
  distance_masks = {
      'd1': distance == 1,
      'd2_5': (distance >= 2) & (distance <= 5),
      'd6_15': (distance >= 6) & (distance <= 15),
      'd16_plus': distance >= 16,
      'no_future_reward': distance < 0,
  }
  for name, mask in distance_masks.items():
    metrics[f'factual_value_{name}'] = _mean(factual_value[mask])
    metrics[f'factual_return_mae_{name}'] = _mae(
        factual_value[mask], heldout['return_to_go'][mask])
    metrics[f'count_{name}'] = int(mask.sum())
  return metrics


def save_training_checkpoint(
    path: pathlib.Path, update: int, params, target_params, opt_state, key,
    rows: list[dict[str, Any]],
) -> None:
  host = jax.device_get((params, target_params, opt_state, key))
  temporary = path.with_suffix(path.suffix + '.tmp')
  with temporary.open('wb') as handle:
    pickle.dump({
        'format': FORMAT, 'update': update, 'params': host[0],
        'target_params': host[1], 'opt_state': host[2], 'key': host[3],
        'rows': rows,
    }, handle, protocol=pickle.HIGHEST_PROTOCOL)
  temporary.replace(path)


def train_run(
    args, outdir: pathlib.Path, arm: str, seed: int,
    train_data, heldout_data,
) -> None:
  run_name = f'{arm}__seed_{seed:04d}'
  run_dir = outdir / 'runs' / run_name
  complete_path = run_dir / 'complete.json'
  if complete_path.exists():
    print(f'Skipping complete run {run_name}', flush=True)
    return
  run_dir.mkdir(parents=True, exist_ok=True)
  goal_count = len(args.goals)
  deter_size = int(train_data['deter'].shape[1])
  stoch_size = int(np.prod(train_data['stoch'].shape[1:]))
  action_count = int(train_data['actor_probability'].shape[-1])
  input_size = deter_size + stoch_size + goal_count
  optimizer = optax.adam(args.learning_rate)
  checkpoint_path = run_dir / 'training_checkpoint.pkl'
  if checkpoint_path.exists():
    with checkpoint_path.open('rb') as handle:
      saved = pickle.load(handle)
    update = int(saved['update'])
    params = jax.device_put(saved['params'])
    target_params = jax.device_put(saved['target_params'])
    opt_state = jax.device_put(saved['opt_state'])
    key = jax.device_put(saved['key'])
    rows = list(saved['rows'])
    print(f'Resuming {run_name} at update {update}', flush=True)
  else:
    key = jax.random.PRNGKey(seed)
    key, q1_key, q2_key = jax.random.split(key, 3)
    params = (
        init_q_params(q1_key, input_size, args.hidden_size, action_count),
        init_q_params(q2_key, input_size, args.hidden_size, action_count))
    target_params = jax.tree.map(lambda value: value.copy(), params)
    opt_state = optimizer.init(params)
    update = 0
    rows = []
  step = make_train_step(
      optimizer, args.batch_size, goal_count, args.gamma,
      args.target_tau, arm == 'counterfactual')
  started = time.monotonic()

  if not rows or int(rows[-1]['update']) != update:
    metrics = evaluation_metrics(
        params, heldout_data, args.gamma, goal_count,
        args.evaluation_batch_size)
    rows.append({'update': update, 'elapsed_seconds': 0.0, **metrics})
    write_csv(run_dir / 'curve.csv', rows)
  latest_train_metrics = {}
  while update < args.updates:
    params, target_params, opt_state, key, latest_train_metrics = step(
        params, target_params, opt_state, key, train_data)
    update += 1
    due_report = update % args.report_every == 0 or update == args.updates
    due_checkpoint = (
        update % args.checkpoint_every == 0 or update == args.updates)
    if due_report:
      elapsed = time.monotonic() - started
      eval_metrics = evaluation_metrics(
          params, heldout_data, args.gamma, goal_count,
          args.evaluation_batch_size)
      train_metrics = {
          f'train_{key}': float(np.asarray(jax.device_get(value)))
          for key, value in latest_train_metrics.items()}
      rows.append({
          'update': update, 'elapsed_seconds': elapsed,
          **train_metrics, **eval_metrics})
      write_csv(run_dir / 'curve.csv', rows)
      rate = (update - int(rows[0]['update'])) / max(elapsed, 1e-6)
      eta = (args.updates - update) / max(rate, 1e-9)
      print(
          f'PROGRESS run={run_name} update={update}/{args.updates} '
          f'loss={train_metrics["train_loss"]:.6f} '
          f'factual_mae={eval_metrics["factual_return_mae"]:.6f} '
          f'nonmatching={eval_metrics["nonmatching_value_mean"]:.6f} '
          f'elapsed={elapsed_text(elapsed)} eta={elapsed_text(eta)}',
          flush=True)
      update_status(
          outdir, state='training', active_run=run_name,
          active_update=update, updates_per_run=args.updates,
          active_run_eta_seconds=eta)
    if due_checkpoint:
      save_training_checkpoint(
          checkpoint_path, update, params, target_params, opt_state, key, rows)
  final = rows[-1]
  json_dump(complete_path, {
      'format': FORMAT, 'arm': arm, 'seed': seed,
      'update': args.updates, 'metrics': final,
  })
  print(f'COMPLETE run={run_name}', flush=True)


def summarize(outdir: pathlib.Path, args) -> None:
  run_rows = []
  for arm in ARMS:
    for seed in args.q_seeds:
      path = outdir / 'runs' / f'{arm}__seed_{seed:04d}' / 'complete.json'
      if not path.exists():
        raise RuntimeError(f'Missing completed run: {path.parent.name}')
      result = json.loads(path.read_text())
      run_rows.append({
          'arm': arm, 'seed': seed, **result['metrics']})
  write_csv(outdir / 'final_runs.csv', run_rows)
  metric_names = [
      'factual_return_mae', 'factual_return_bias',
      'nonmatching_value_mean', 'matching_minus_nonmatching_mean',
      'both_goals_above_0p5_fraction',
      'successful_initial_goal_rank_accuracy', 'bellman_mae_all_queries',
      'q_disagreement_mean']
  aggregate = []
  for arm in ARMS:
    selected = [row for row in run_rows if row['arm'] == arm]
    row = {'arm': arm, 'seeds': len(selected)}
    for metric in metric_names:
      values = np.asarray([float(item[metric]) for item in selected])
      row[f'{metric}_mean'] = float(values.mean())
      row[f'{metric}_std'] = float(values.std())
    aggregate.append(row)
  write_csv(outdir / 'aggregate.csv', aggregate)
  factual = next(row for row in aggregate if row['arm'] == 'factual_only')
  counterfactual = next(row for row in aggregate if row['arm'] == 'counterfactual')
  decision = {
      'format': FORMAT,
      'question': (
          'Does ordinary counterfactual one-step twin-Q training improve '
          'goal selectivity without materially harming factual calibration?'),
      'counterfactual_minus_factual': {
          metric: counterfactual[f'{metric}_mean'] - factual[f'{metric}_mean']
          for metric in metric_names},
      'interpretation_note': (
          'Held-out return targets are behavior-policy returns. They test '
          'factual calibration but are not counterfactual ground truth.'),
  }
  json_dump(outdir / 'decision.json', decision)
  json_dump(outdir / 'experiment_complete.json', {
      'format': FORMAT, 'name': EXPERIMENT_NAME,
      'completed_at_unix': time.time(), 'runs': len(run_rows),
  })
  update_status(outdir, state='complete', active_run=None, active_update=None)
  print(f'EXPERIMENT_COMPLETE output={outdir}', flush=True)


def train(args, outdir: pathlib.Path) -> None:
  load_runtime_dependencies()
  train_host = load_cache(outdir / 'cache' / 'train')
  heldout_host = load_cache(outdir / 'cache' / 'heldout')
  print(
      f'Loaded train states={len(train_host["deter"])} '
      f'transitions={len(train_host["action"])}; '
      f'heldout states={len(heldout_host["deter"])} '
      f'transitions={len(heldout_host["action"])}', flush=True)
  train_data = device_cache(train_host)
  heldout_data = device_cache(heldout_host)
  for arm in ARMS:
    for seed in args.q_seeds:
      train_run(args, outdir, arm, seed, train_data, heldout_data)


def parser() -> argparse.ArgumentParser:
  result = argparse.ArgumentParser(description=__doc__)
  result.add_argument(
      'stage', nargs='?', choices=('all', 'prepare', 'encode', 'train', 'summarize'),
      default='all')
  result.add_argument('--output-root', required=True)
  result.add_argument('--checkpoint', required=True)
  result.add_argument('--replay-root', required=True)
  result.add_argument('--goals', type=parse_ints, default=(0, 1))
  result.add_argument('--q-seeds', type=parse_ints, default=(0, 1, 2))
  result.add_argument('--selection-seed', type=int, default=20260901)
  result.add_argument('--train-successes-per-goal', type=int, default=64)
  result.add_argument('--train-failures-per-goal', type=int, default=64)
  result.add_argument('--heldout-successes-per-goal', type=int, default=32)
  result.add_argument('--heldout-failures-per-goal', type=int, default=32)
  result.add_argument('--chunk-length', type=int, default=32)
  result.add_argument('--hidden-size', type=int, default=256)
  result.add_argument('--batch-size', type=int, default=128)
  result.add_argument('--updates', type=int, default=10_000)
  result.add_argument('--learning-rate', type=float, default=1e-4)
  result.add_argument('--target-tau', type=float, default=0.01)
  result.add_argument('--gamma', type=float, default=0.99)
  result.add_argument('--report-every', type=int, default=500)
  result.add_argument('--checkpoint-every', type=int, default=2_000)
  result.add_argument('--evaluation-batch-size', type=int, default=256)
  result.add_argument('--jax-platform', default='cuda')
  result.add_argument('--dry-run', action='store_true')
  return result


def validate_args(args) -> None:
  if len(args.goals) != 2 or len(set(args.goals)) != 2:
    raise ValueError('--goals must contain exactly two distinct IDs')
  if not args.q_seeds:
    raise ValueError('--q-seeds cannot be empty')
  for name in (
      'chunk_length', 'hidden_size', 'batch_size', 'updates', 'report_every',
      'checkpoint_every', 'evaluation_batch_size'):
    if getattr(args, name) < 1:
      raise ValueError(f'--{name.replace("_", "-")} must be positive')
  if not 0 < args.gamma <= 1:
    raise ValueError('--gamma must be in (0, 1]')
  if not 0 < args.target_tau <= 1:
    raise ValueError('--target-tau must be in (0, 1]')


def main(argv=None) -> int:
  args = parser().parse_args(argv)
  validate_args(args)
  outdir = pathlib.Path(args.output_root).expanduser().resolve()
  checkpoint = resolve_checkpoint_path(args.checkpoint)
  if args.stage in ('all', 'prepare'):
    prepare(args, outdir, checkpoint)
  if args.dry_run:
    update_status(outdir, state='dry_run_complete')
    print('DRY_RUN_COMPLETE', flush=True)
    return 0
  if args.stage in ('all', 'encode'):
    encode(args, outdir, checkpoint)
  if args.stage in ('all', 'train'):
    train(args, outdir)
  if args.stage in ('all', 'summarize'):
    summarize(outdir, args)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
