"""Offline cloned-head repair on a fixed RLScape episode selection."""

from __future__ import annotations

from functools import partial as bind
import json
import pathlib
import pickle
import time

import elements
import numpy as np

from .head_audit import _pad
from .head_audit import _previous_actions
from .head_audit import _sha256
from .head_audit import discounted_returns
from .head_audit import load_selection


FORMAT = 1
CONDITIONS = ('factual_only', 'counterfactual', 'counterfactual_bounded')
REPAIR_OPT_PREFIXES = ('repair_head_opt/', 'repair_value_opt/')


def _write_json(path: pathlib.Path, value) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
  temporary.replace(path)


def _append_jsonl(path: pathlib.Path, value) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('a') as handle:
    handle.write(json.dumps(value, sort_keys=True) + '\n')
    handle.flush()


def _cache_spec_path(cache: pathlib.Path) -> pathlib.Path:
  return cache.with_suffix(cache.suffix + '.json')


def _select_episode_states(
    data: dict[str, np.ndarray], amount: int, rng: np.random.Generator,
) -> np.ndarray:
  length = len(data['reward'])
  mandatory_mask = (
      np.asarray(data['is_first'], bool) |
      np.asarray(data['goal_complete'], bool) |
      np.asarray(data['is_physical_terminal'], bool) |
      (np.asarray(data['reward'], np.float32) > 0))
  mandatory = np.flatnonzero(mandatory_mask)
  remaining = np.flatnonzero(~mandatory_mask)
  needed = max(0, int(amount) - len(mandatory))
  if needed and len(remaining):
    chosen = rng.choice(remaining, min(needed, len(remaining)), replace=False)
    indices = np.concatenate([mandatory, chosen])
  else:
    indices = mandatory
  if not len(indices):
    indices = np.asarray([rng.integers(0, length)], np.int64)
  return np.unique(indices).astype(np.int64)


def _encode_episode(agent, data: dict[str, np.ndarray], chunk_length: int):
  length = len(data['reward'])
  previous = _previous_actions(data, agent.act_space)
  carry = agent.init_observe(1)
  pieces = {'deter': [], 'stoch': []}
  for start in range(0, length, chunk_length):
    stop = min(length, start + chunk_length)
    valid = stop - start
    pad = chunk_length - valid
    obs = {
        key: _pad(
            np.asarray(data[key][start:stop]), pad,
            first=(key == 'is_first'))[None]
        for key in agent.obs_space}
    prevact = {
        key: _pad(np.asarray(value[start:stop]), pad)[None]
        for key, value in previous.items()}
    carry, feat = agent.posterior_chunk(carry, obs, prevact)
    for key in pieces:
      pieces[key].append(np.asarray(feat[key])[0, :valid])
  return {key: np.concatenate(value, 0) for key, value in pieces.items()}


def build_latent_cache(
    agent, selection: dict, cache: pathlib.Path, *, checkpoint: pathlib.Path,
    chunk_length: int, states_per_episode: int, seed: int,
    representation_digest: str,
) -> dict:
  """Encode a deterministic state subset once for all matched repair arms."""
  cache = cache.expanduser().resolve()
  spec_path = _cache_spec_path(cache)
  expected = {
      'format': FORMAT,
      'selection_sha256': _sha256(pathlib.Path(selection['_path'])),
      'checkpoint': str(checkpoint),
      'representation_sha256': representation_digest,
      'chunk_length': int(chunk_length),
      'states_per_episode': int(states_per_episode),
      'seed': int(seed),
  }
  if cache.is_file() or spec_path.is_file():
    if not (cache.is_file() and spec_path.is_file()):
      raise RuntimeError(f'Incomplete latent cache: {cache}')
    existing = json.loads(spec_path.read_text())
    mismatch = {
        key: (existing.get(key), value) for key, value in expected.items()
        if existing.get(key) != value}
    if mismatch:
      raise RuntimeError(f'Latent cache specification mismatch: {mismatch}')
    if _sha256(cache) != existing.get('cache_sha256'):
      raise RuntimeError(f'Latent cache digest mismatch: {cache}')
    return existing

  cache.parent.mkdir(parents=True, exist_ok=True)
  rng = np.random.default_rng(int(seed))
  arrays: dict[str, list[np.ndarray]] = {
      key: [] for key in (
          'deter', 'stoch', 'actual_goal', 'reward', 'is_terminal',
          'physical_terminal', 'goal_complete', 'value_target',
          'episode_success', 'episode_id', 'timestep')}
  discount = 1 - 1 / float(agent.model.config.horizon)
  episodes = selection['episodes']
  for number, record in enumerate(episodes, 1):
    with np.load(record['path']) as archive:
      data = {key: np.array(archive[key], copy=True) for key in archive.files}
    feat = _encode_episode(agent, data, int(chunk_length))
    indices = _select_episode_states(data, int(states_per_episode), rng)
    inclusive, _ = discounted_returns(
        data['reward'], data['is_terminal'], discount)
    count = len(indices)
    arrays['deter'].append(np.asarray(feat['deter'][indices], np.float32))
    arrays['stoch'].append(np.asarray(feat['stoch'][indices], np.float32))
    arrays['actual_goal'].append(
        np.asarray(data['goal_id'][indices], np.int32))
    arrays['reward'].append(np.asarray(data['reward'][indices], np.float32))
    arrays['is_terminal'].append(
        np.asarray(data['is_terminal'][indices], bool))
    arrays['physical_terminal'].append(
        np.asarray(data['is_physical_terminal'][indices], bool))
    arrays['goal_complete'].append(
        np.asarray(data['goal_complete'][indices], bool))
    arrays['value_target'].append(np.asarray(inclusive[indices], np.float32))
    arrays['episode_success'].append(
        np.full(count, bool(record['success']), bool))
    arrays['episode_id'].append(
        np.full(count, int(record['episode_id']), np.int64))
    arrays['timestep'].append(np.asarray(indices, np.int32))
    print(
        f'Latent cache episode {number}/{len(episodes)}: '
        f'goal={record["goal_id"]} success={record["success"]} '
        f'states={count}', flush=True)

  combined = {key: np.concatenate(value, 0) for key, value in arrays.items()}
  temporary = cache.with_suffix(cache.suffix + '.tmp.npz')
  np.savez_compressed(temporary, **combined)
  temporary.replace(cache)
  counts = {
      str(goal): {
          str(int(success)): int((
              (combined['actual_goal'] == goal) &
              (combined['episode_success'] == success)).sum())
          for success in (False, True)}
      for goal in sorted(set(combined['actual_goal'].tolist()))}
  spec = {
      **expected,
      'states': int(len(combined['actual_goal'])),
      'episodes': int(len(np.unique(combined['episode_id']))),
      'completion_states': int(combined['goal_complete'].sum()),
      'counts_by_goal_and_success': counts,
      'cache_sha256': _sha256(cache),
      'cache_bytes': cache.stat().st_size,
  }
  _write_json(spec_path, spec)
  return spec


def load_cache(path: pathlib.Path) -> dict[str, np.ndarray]:
  with np.load(path) as archive:
    return {key: np.array(archive[key], copy=True) for key in archive.files}


def sample_head_indices(
    data: dict[str, np.ndarray], batch_size: int, event_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
  event = np.flatnonzero(data['goal_complete'] | (data['reward'] > 0))
  ordinary = np.flatnonzero(~(data['goal_complete'] | (data['reward'] > 0)))
  if not len(event) or not len(ordinary):
    raise RuntimeError('Head repair requires both completion and ordinary states')
  event_count = min(batch_size - 1, max(1, round(batch_size * event_fraction)))
  ordinary_count = batch_size - event_count
  indices = np.concatenate([
      rng.choice(event, event_count, replace=True),
      rng.choice(ordinary, ordinary_count, replace=True)])
  rng.shuffle(indices)
  return indices


def sample_value_indices(
    data: dict[str, np.ndarray], batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
  strata = []
  for goal in sorted(set(data['actual_goal'].tolist())):
    for success in (False, True):
      indices = np.flatnonzero(
          (data['actual_goal'] == goal) &
          (data['episode_success'] == success))
      if len(indices):
        strata.append(indices)
  if not strata:
    raise RuntimeError('Bounded-value repair has no populated strata')
  chosen = [rng.choice(strata[index % len(strata)])
            for index in range(batch_size)]
  rng.shuffle(chosen)
  return np.asarray(chosen, np.int64)


def _batch(data: dict[str, np.ndarray], indices: np.ndarray) -> dict:
  keys = (
      'deter', 'stoch', 'actual_goal', 'reward', 'is_terminal',
      'physical_terminal', 'goal_complete', 'value_target')
  return {key: np.asarray(data[key][indices]) for key in keys}


def _save_checkpoint(
    agent, destination: pathlib.Path, *, keep_bounded_value: bool,
) -> dict:
  payload = agent.save()
  removed = []
  for key in list(payload['params']):
    if key.startswith(REPAIR_OPT_PREFIXES) or (
        key.startswith('bval/') and not keep_bounded_value
    ):
      removed.append(key)
      payload['params'].pop(key)
  destination.mkdir(parents=True, exist_ok=False)
  with (destination / 'agent.pkl').open('wb') as handle:
    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
  (destination / 'done').write_bytes(b'')
  return {
      'parameter_arrays': len(payload['params']),
      'removed_arrays': sorted(removed),
      'agent_sha256': _sha256(destination / 'agent.pkl'),
      'agent_bytes': (destination / 'agent.pkl').stat().st_size,
  }


def _mean_metrics(rows: list[dict]) -> dict:
  keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
  return {
      key: float(np.mean([float(row[key]) for row in rows if key in row]))
      for key in keys}


def head_repair(make_agent, args):
  """Train one isolated Stage 4A cloned-head repair condition."""
  cfg = args.head_repair
  if not bool(cfg.enabled):
    raise ValueError('head_repair.enabled must be true')
  condition = str(cfg.condition)
  if condition not in CONDITIONS:
    raise ValueError(f'Unknown head repair condition: {condition!r}')
  if not args.from_checkpoint:
    raise ValueError('Head repair requires run.from_checkpoint')
  selection_path = pathlib.Path(str(cfg.selection)).expanduser().resolve()
  cache_path = pathlib.Path(str(cfg.cache)).expanduser().resolve()
  output_checkpoint = pathlib.Path(
      str(cfg.output_checkpoint)).expanduser().resolve()
  logdir = pathlib.Path(str(args.logdir)).expanduser().resolve()
  if not str(cfg.cache) or not str(cfg.output_checkpoint):
    raise ValueError('head_repair.cache and output_checkpoint are required')
  if output_checkpoint.exists():
    raise FileExistsError(output_checkpoint)
  if int(cfg.chunk_length) < 1 or int(cfg.states_per_episode) < 2:
    raise ValueError('Invalid head-repair encoding settings')
  if int(cfg.head_updates) < 1 or int(cfg.batch_size) < 2:
    raise ValueError('Invalid head-repair update settings')
  if not 0 < float(cfg.event_fraction) < 1:
    raise ValueError('head_repair.event_fraction must be in (0, 1)')
  train_value = condition == 'counterfactual_bounded'
  if train_value and int(cfg.value_updates) < 1:
    raise ValueError('Bounded repair requires value_updates > 0')

  selection = load_selection(selection_path)
  selection['_path'] = str(selection_path)
  agent = make_agent()
  source = pathlib.Path(str(args.from_checkpoint)).expanduser().resolve()
  # Source checkpoints predate the Stage 4A-only bounded head and optimizers.
  elements.checkpoint.load(str(source), {'agent': bind(
      agent.load, regex=(
          '^(?!(bval|repair_head_opt|repair_value_opt)/)'))})
  integrity_before = agent.parameter_digests()
  cache_spec = build_latent_cache(
      agent, selection, cache_path, checkpoint=source,
      chunk_length=int(cfg.chunk_length),
      states_per_episode=int(cfg.states_per_episode), seed=int(cfg.seed),
      representation_digest=integrity_before['representation']['sha256'])
  data = load_cache(cache_path)
  rng = np.random.default_rng(int(cfg.seed))
  metrics_path = logdir / 'repair_metrics.jsonl'
  head_window = []
  for update in range(1, int(cfg.head_updates) + 1):
    indices = sample_head_indices(
        data, int(cfg.batch_size), float(cfg.event_fraction), rng)
    metrics = agent.repair_reward_continuation(
        _batch(data, indices), counterfactual=(condition != 'factual_only'))
    head_window.append(metrics)
    if update == 1 or update % 100 == 0 or update == int(cfg.head_updates):
      row = {
          'kind': 'reward_continuation', 'update': update,
          'condition': condition, **_mean_metrics(head_window)}
      _append_jsonl(metrics_path, row)
      print(
          f'Head repair {condition}: {update}/{int(cfg.head_updates)} '
          f'loss={row.get("repair_head_opt/loss", float("nan")):.5f}',
          flush=True)
      head_window = []

  value_window = []
  if train_value:
    for update in range(1, int(cfg.value_updates) + 1):
      indices = sample_value_indices(data, int(cfg.batch_size), rng)
      metrics = agent.repair_bounded_value(_batch(data, indices))
      value_window.append(metrics)
      if update == 1 or update % 100 == 0 or update == int(cfg.value_updates):
        row = {
            'kind': 'bounded_value', 'update': update,
            'condition': condition, **_mean_metrics(value_window)}
        _append_jsonl(metrics_path, row)
        print(
            f'Bounded-value repair: {update}/{int(cfg.value_updates)} '
            f'loss={row.get("repair_value_opt/loss", float("nan")):.5f}',
            flush=True)
        value_window = []

  integrity_after = agent.parameter_digests()
  unchanged = ('actor', 'critic', 'representation', 'normalizers')
  violations = {
      key: (integrity_before[key], integrity_after[key])
      for key in unchanged if integrity_before[key] != integrity_after[key]}
  if violations:
    raise RuntimeError(f'Head repair changed frozen components: {violations}')
  if (
      integrity_before['reward_continuation'] ==
      integrity_after['reward_continuation']
  ):
    raise RuntimeError('Reward/continuation repair produced no parameter change')
  if train_value and (
      integrity_before['bounded_value'] == integrity_after['bounded_value']
  ):
    raise RuntimeError('Bounded-value repair produced no parameter change')

  checkpoint_summary = _save_checkpoint(
      agent, output_checkpoint, keep_bounded_value=train_value)
  manifest = {
      'format': FORMAT,
      'canonical_stage': 'stage4a',
      'condition': condition,
      'source_checkpoint': str(source),
      'selection': str(selection_path),
      'selection_sha256': _sha256(selection_path),
      'cache': str(cache_path),
      'cache_spec': cache_spec,
      'head_updates': int(cfg.head_updates),
      'value_updates': int(cfg.value_updates) if train_value else 0,
      'batch_size': int(cfg.batch_size),
      'event_fraction': float(cfg.event_fraction),
      'counterfactual_reward_continuation': condition != 'factual_only',
      'counterfactual_value_targets': False,
      'bounded_value': train_value,
      'integrity_before': integrity_before,
      'integrity_after': integrity_after,
      'frozen_component_violations': violations,
      'output_checkpoint': str(output_checkpoint),
      'checkpoint': checkpoint_summary,
      'completed_unix': time.time(),
  }
  _write_json(logdir / 'repair_manifest.json', manifest)
  _write_json(output_checkpoint / 'repair_manifest.json', manifest)
  (logdir / 'repair_complete').write_bytes(b'')
  return manifest
