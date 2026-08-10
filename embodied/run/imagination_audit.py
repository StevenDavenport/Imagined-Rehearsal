"""Restart-safe Stage 1b audit of Dreamer's actor-IR imagination pathway."""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib

import elements
import numpy as np

from .head_audit import _pad
from .head_audit import _previous_actions
from .head_audit import _sha256
from .head_audit import load_selection


FORMAT = 1


def _write_json(path: pathlib.Path, value) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
  temporary.replace(path)


def _digest(value) -> str:
  payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
  return hashlib.sha256(payload).hexdigest()


def select_anchors(data: dict, count: int) -> list[dict]:
  """Choose deterministic decision states spanning one complete episode."""
  count = int(count)
  if count < 1:
    raise ValueError(count)
  is_last = np.asarray(data['is_last'], bool)
  is_terminal = np.asarray(data['is_terminal'], bool)
  valid = np.flatnonzero(~(is_last | is_terminal))
  if not len(valid):
    raise ValueError('Episode has no nonterminal decision states')
  positions = valid[np.rint(np.linspace(0, len(valid) - 1, count)).astype(int)]
  complete = np.flatnonzero(np.asarray(data['goal_complete'], bool))
  if len(complete):
    before = valid[valid < complete[0]]
    if len(before):
      positions[-1] = before[-1]
  positions = list(dict.fromkeys(int(x) for x in positions))
  anchors = []
  for ordinal, timestep in enumerate(positions):
    if ordinal == 0:
      kind = 'start'
    elif ordinal == len(positions) - 1:
      kind = 'pre_completion' if len(complete) else 'late_failure'
    else:
      kind = f'intermediate_{ordinal}'
    anchors.append({
        'ordinal': ordinal,
        'timestep': timestep,
        'kind': kind,
        'fraction': timestep / max(1, len(data['reward']) - 1),
    })
  return anchors


def _atomic_npz(path: pathlib.Path, arrays: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix('.npz.tmp')
  with temporary.open('wb') as handle:
    np.savez_compressed(handle, **arrays)
  temporary.replace(path)


def _host_array(value) -> np.ndarray:
  value = np.asarray(value)
  if value.dtype.kind in ('f', 'c') or str(value.dtype) == 'bfloat16':
    value = value.astype(np.float32)
  return value


def _valid_shard(path: pathlib.Path, spec_digest: str, episode_id: int) -> bool:
  if not path.is_file():
    return False
  try:
    with np.load(path) as data:
      return (
          str(data['spec_digest'].item()) == spec_digest and
          int(data['episode_id'].item()) == int(episode_id))
  except Exception:
    return False


def _encode_anchors(agent, data, anchors, chunk_length, episode_id):
  length = len(data['reward'])
  wanted = {item['timestep']: item for item in anchors}
  previous = _previous_actions(data, agent.act_space)
  carry = agent.init_observe(1)
  found = {}
  for chunk_index, start in enumerate(range(0, length, chunk_length)):
    stop = min(length, start + chunk_length)
    valid = stop - start
    pad = chunk_length - valid
    obs = {}
    for key in agent.obs_space:
      value = np.asarray(data[key][start:stop])
      obs[key] = _pad(value, pad, first=(key == 'is_first'))[None]
    prevact = {
        key: _pad(np.asarray(value[start:stop]), pad)[None]
        for key, value in previous.items()}
    seed_index = int(episode_id) * 10_000 + chunk_index
    carry, feat = agent.posterior_chunk(
        carry, obs, prevact, seed_index=seed_index)
    for timestep in wanted:
      if start <= timestep < stop:
        local = timestep - start
        found[timestep] = {
            key: np.asarray(feat[key])[0, local].astype(np.float32)
            for key in ('deter', 'stoch', 'logit')}
  if set(found) != set(wanted):
    raise RuntimeError(
        f'Failed to encode anchors: wanted={sorted(wanted)} '
        f'found={sorted(found)}')
  return [found[item['timestep']] for item in anchors]


def _run_episode(
    agent, record, data, config, goal_names, spec_digest,
) -> dict:
  anchors = select_anchors(data, int(config.anchors_per_episode))
  starts = _encode_anchors(
      agent, data, anchors, int(config.chunk_length), record['episode_id'])
  outputs = {}
  action_outputs = {}
  for anchor, start in zip(anchors, starts):
    per_goal = {}
    per_goal_actions = {}
    for goal in range(len(goal_names)):
      seed_index = (
          int(config.seed_base) + int(record['episode_id']) * 10_000 +
          int(anchor['ordinal']) * 100)
      result = agent.imagination_audit(
          {key: value[None] for key, value in start.items()},
          np.asarray([goal], np.int32),
          horizon=int(config.horizon),
          start_batch=int(config.particles),
          seed_index=seed_index)
      actions = result.pop('action')
      for key, value in result.items():
        per_goal.setdefault(key, []).append(_host_array(value)[0])
      for key, value in actions.items():
        per_goal_actions.setdefault(key, []).append(_host_array(value)[0])
    for key, values in per_goal.items():
      outputs.setdefault(key, []).append(np.stack(values, 0))
    for key, values in per_goal_actions.items():
      action_outputs.setdefault(key, []).append(np.stack(values, 0))

  arrays = {
      key: np.stack(values, 0) for key, values in outputs.items()}
  arrays.update({
      f'action__{key}': np.stack(values, 0)
      for key, values in action_outputs.items()})
  timesteps = np.asarray([item['timestep'] for item in anchors], np.int32)
  horizon = int(config.horizon)
  arrays.update({
      'format': np.asarray(FORMAT, np.int32),
      'spec_digest': np.asarray(spec_digest),
      'episode_id': np.asarray(record['episode_id'], np.int64),
      'actual_goal': np.asarray(record['goal_id'], np.int32),
      'episode_success': np.asarray(record['success'], bool),
      'anchor_ordinal': np.asarray(
          [item['ordinal'] for item in anchors], np.int32),
      'anchor_timestep': timesteps,
      'anchor_kind': np.asarray([item['kind'] for item in anchors]),
      'anchor_fraction': np.asarray(
          [item['fraction'] for item in anchors], np.float32),
      'recorded_reward_horizon': np.asarray([
          np.asarray(data['reward'])[t + 1:t + horizon + 1].sum()
          for t in timesteps], np.float32),
      'recorded_completion_horizon': np.asarray([
          np.asarray(data['goal_complete'])[t + 1:t + horizon + 1].any()
          for t in timesteps], bool),
  })
  return arrays


def _quantiles(values) -> dict:
  values = np.asarray(values, np.float64)
  return {
      'mean': float(values.mean()),
      'std': float(values.std()),
      'p50': float(np.quantile(values, .50)),
      'p95': float(np.quantile(values, .95)),
      'p99': float(np.quantile(values, .99)),
      'max': float(values.max()),
  }


def summarize_shards(
    shard_paths: list[pathlib.Path], goal_names: list[str], horizon: int,
) -> tuple[list[dict], dict]:
  rows = []
  for path in shard_paths:
    with np.load(path) as shard:
      data = {key: np.asarray(shard[key]) for key in shard.files}
    for anchor in range(len(data['anchor_timestep'])):
      for goal, goal_name in enumerate(goal_names):
        reward = data['reward'][anchor, goal]
        continuation = data['continuation'][anchor, goal]
        future_reward = reward[:, 1:]
        future_stop = 1 - continuation[:, 1:]
        returns = data['return'][anchor, goal, :, 0]
        reward_only = data['reward_only_lambda'][anchor, goal, :, 0]
        bootstrap = data['bootstrap_contribution'][anchor, goal, :, 0]
        reinforce = data['reinforce_loss'][anchor, goal]
        entropy = data['entropy_loss'][anchor, goal]
        row = {
            'episode_id': int(data['episode_id']),
            'actual_goal': int(data['actual_goal']),
            'actual_goal_name': goal_names[int(data['actual_goal'])],
            'probed_goal': goal,
            'probed_goal_name': goal_name,
            'matching_goal': goal == int(data['actual_goal']),
            'episode_success': bool(data['episode_success']),
            'anchor_ordinal': int(data['anchor_ordinal'][anchor]),
            'anchor_timestep': int(data['anchor_timestep'][anchor]),
            'anchor_kind': str(data['anchor_kind'][anchor]),
            'anchor_fraction': float(data['anchor_fraction'][anchor]),
            'recorded_reward_horizon': float(
                data['recorded_reward_horizon'][anchor]),
            'recorded_completion_horizon': bool(
                data['recorded_completion_horizon'][anchor]),
            'reward_step_mean': float(future_reward.mean()),
            'reward_sum_mean': float(future_reward.sum(-1).mean()),
            'reward_event_particle_rate': float(
                (future_reward >= .5).any(-1).mean()),
            'reward_event_step_rate': float((future_reward >= .5).mean()),
            'stop_particle_rate': float((future_stop >= .5).any(-1).mean()),
            'stop_step_mean': float(future_stop.mean()),
            'continuation_mean': float(continuation[:, 1:].mean()),
            'value_start_mean': float(
                data['value'][anchor, goal, :, 0].mean()),
            'slowvalue_start_mean': float(
                data['slowvalue'][anchor, goal, :, 0].mean()),
            'return_mean': float(returns.mean()),
            'return_std': float(returns.std()),
            'return_p50': float(np.quantile(returns, .50)),
            'return_p95': float(np.quantile(returns, .95)),
            'return_p99': float(np.quantile(returns, .99)),
            'return_max': float(returns.max()),
            'reward_only_mean': float(reward_only.mean()),
            'bootstrap_mean': float(bootstrap.mean()),
            'bootstrap_abs_mean': float(np.abs(bootstrap).mean()),
            'bootstrap_fraction': float(
                np.abs(bootstrap).mean() /
                (np.abs(returns).mean() + 1e-8)),
            'advantage_mean': float(
                data['advantage'][anchor, goal].mean()),
            'advantage_normalized_mean': float(
                data['advantage_normalized'][anchor, goal].mean()),
            'positive_advantage_rate': float(
                (data['advantage_normalized'][anchor, goal] > 0).mean()),
            'weight_mean': float(data['weight'][anchor, goal].mean()),
            'policy_entropy_mean': float(
                data['policy_entropy'][anchor, goal].mean()),
            'reinforce_loss_mean': float(reinforce.mean()),
            'reinforce_loss_abs': float(np.abs(reinforce).mean()),
            'entropy_loss_mean': float(entropy.mean()),
            'entropy_loss_abs': float(np.abs(entropy).mean()),
            'entropy_to_reinforce_abs': float(
                np.abs(entropy).mean() /
                (np.abs(reinforce).mean() + 1e-8)),
            'prior_entropy_mean': float(
                data['prior_entropy'][anchor, goal].mean()),
            'deter_rms_mean': float(data['deter_rms'][anchor, goal].mean()),
            'stoch_rms_mean': float(data['stoch_rms'][anchor, goal].mean()),
        }
        rows.append(row)

  numeric = [
      key for key, value in rows[0].items()
      if isinstance(value, (int, float)) and not isinstance(value, bool) and
      key not in ('episode_id', 'actual_goal', 'probed_goal',
                  'anchor_ordinal', 'anchor_timestep')]
  groups = []
  for actual in sorted(set(row['actual_goal'] for row in rows)):
    for probed in range(len(goal_names)):
      selected = [
          row for row in rows
          if row['actual_goal'] == actual and row['probed_goal'] == probed]
      group = {
          'actual_goal': actual,
          'actual_goal_name': goal_names[actual],
          'probed_goal': probed,
          'probed_goal_name': goal_names[probed],
          'anchors': len(selected),
      }
      for key in numeric:
        group[key] = float(np.mean([row[key] for row in selected]))
      groups.append(group)
  factual = [row for row in rows if row['matching_goal']]
  counterfactual = [row for row in rows if not row['matching_goal']]
  def aggregate(selected):
    return {
        key: (
            float(np.mean([row[key] for row in selected]))
            if selected else None)
        for key in numeric}
  summary = {
      'format': FORMAT,
      'episodes': len(shard_paths),
      'anchors': len(rows) // len(goal_names),
      'rollouts': len(rows),
      'horizon': int(horizon),
      'goal_names': goal_names,
      'goal_sweep': groups,
      'factual': aggregate(factual),
      'counterfactual': aggregate(counterfactual),
      'factual_by_anchor_kind': {
          kind: aggregate([row for row in factual if row['anchor_kind'] == kind])
          for kind in sorted(set(row['anchor_kind'] for row in factual))},
      'factual_by_episode_outcome': {
          str(outcome).lower(): aggregate([
              row for row in factual if row['episode_success'] == outcome])
          for outcome in (False, True)},
      'factual_by_recorded_completion': {
          str(outcome).lower(): aggregate([
              row for row in factual
              if row['recorded_completion_horizon'] == outcome])
          for outcome in (False, True)},
  }
  return rows, summary


def summarize_time_profiles(
    shard_paths: list[pathlib.Path], goal_names: list[str], horizon: int,
) -> list[dict]:
  totals = {}
  fields = ('reward', 'reward_event', 'stop', 'value', 'slowvalue')
  for path in shard_paths:
    with np.load(path) as shard:
      actual = int(shard['actual_goal'])
      reward = np.asarray(shard['reward'])
      continuation = np.asarray(shard['continuation'])
      value = np.asarray(shard['value'])
      slowvalue = np.asarray(shard['slowvalue'])
    for goal in range(len(goal_names)):
      for step in range(horizon + 1):
        key = (actual, goal, step)
        bucket = totals.setdefault(
            key, {'count': 0, **{field: 0.0 for field in fields}})
        rew = reward[:, goal, :, step].reshape(-1)
        con = continuation[:, goal, :, step].reshape(-1)
        val = value[:, goal, :, step].reshape(-1)
        slow = slowvalue[:, goal, :, step].reshape(-1)
        bucket['count'] += len(rew)
        bucket['reward'] += float(rew.sum())
        bucket['reward_event'] += float((rew >= .5).sum())
        bucket['stop'] += float((1 - con).sum())
        bucket['value'] += float(val.sum())
        bucket['slowvalue'] += float(slow.sum())
  rows = []
  for (actual, goal, step), bucket in sorted(totals.items()):
    count = bucket.pop('count')
    rows.append({
        'actual_goal': actual,
        'actual_goal_name': goal_names[actual],
        'probed_goal': goal,
        'probed_goal_name': goal_names[goal],
        'step': step,
        'particles': count,
        **{field: total / count for field, total in bucket.items()},
    })
  return rows


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
  with path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def imagination_audit(make_agent, args):
  config = args.imagination_audit
  logdir = pathlib.Path(str(args.logdir)).expanduser().resolve()
  logdir.mkdir(parents=True, exist_ok=True)
  selection_path = pathlib.Path(str(config.selection)).expanduser().resolve()
  selection = load_selection(selection_path)
  goal_names = [str(x) for x in config.goal_names]
  spec = {
      'format': FORMAT,
      'checkpoint': str(args.from_checkpoint),
      'selection': str(selection_path),
      'selection_sha256': _sha256(selection_path),
      'goal_names': goal_names,
      'chunk_length': int(config.chunk_length),
      'horizon': int(config.horizon),
      'particles': int(config.particles),
      'anchors_per_episode': int(config.anchors_per_episode),
      'seed_base': int(config.seed_base),
      'actor_ir_path': True,
  }
  spec_digest = _digest(spec)
  spec['spec_digest'] = spec_digest
  spec_path = logdir / 'audit_spec.json'
  if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
    raise RuntimeError(f'Existing imagination audit has another spec: {logdir}')
  _write_json(spec_path, spec)

  agent = make_agent()
  elements.checkpoint.load(str(args.from_checkpoint), {'agent': agent.load})
  if len(goal_names) != agent.model.goal_count:
    raise ValueError((len(goal_names), agent.model.goal_count))
  integrity_before = agent.parameter_digests()
  shard_dir = logdir / 'shards'
  shard_dir.mkdir(exist_ok=True)
  shard_paths = []
  for number, record in enumerate(selection['episodes'], 1):
    shard = shard_dir / f'episode_{int(record["episode_id"]):016d}.npz'
    shard_paths.append(shard)
    if _valid_shard(shard, spec_digest, record['episode_id']):
      print(
          f'Imagination audit episode {number}/{len(selection["episodes"])}: '
          f'using complete shard', flush=True)
      continue
    with np.load(record['path']) as archive:
      data = {key: np.array(archive[key], copy=True) for key in archive.files}
    arrays = _run_episode(
        agent, record, data, config, goal_names, spec_digest)
    _atomic_npz(shard, arrays)
    print(
        f'Imagination audit episode {number}/{len(selection["episodes"])}: '
        f'goal={record["goal_id"]} success={record["success"]} '
        f'anchors={len(arrays["anchor_timestep"])}', flush=True)

  rows, summary = summarize_shards(
      shard_paths, goal_names, int(config.horizon))
  time_profiles = summarize_time_profiles(
      shard_paths, goal_names, int(config.horizon))
  integrity_after = agent.parameter_digests()
  summary.update({
      'spec_digest': spec_digest,
      'integrity_before': integrity_before,
      'integrity_after': integrity_after,
      'integrity_equal': integrity_before == integrity_after,
  })
  _write_csv(logdir / 'rollout_summary.csv', rows)
  _write_csv(logdir / 'goal_sweep.csv', summary['goal_sweep'])
  _write_csv(logdir / 'time_profile.csv', time_profiles)
  _write_json(logdir / 'summary.json', summary)
  if not summary['integrity_equal']:
    raise RuntimeError('Imagination audit mutated checkpoint parameters')
  (logdir / 'audit_complete').write_bytes(b'')
  return summary
