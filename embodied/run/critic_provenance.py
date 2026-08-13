"""Read-only Stage 3A audit of critic target provenance and calibration."""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib

import elements
import numpy as np

from .head_audit import _sha256
from .head_audit import load_selection
from .imagination_audit import _encode_anchors
from .imagination_audit import select_anchors


FORMAT = 1


def _write_json(path: pathlib.Path, value) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
  temporary.replace(path)


def _digest(value) -> str:
  payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
  return hashlib.sha256(payload).hexdigest()


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
  if not rows:
    raise ValueError(f'Cannot write empty table: {path}')
  with path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _valid_shard(path: pathlib.Path, spec_digest: str, episode_id: int) -> bool:
  if not path.is_file():
    return False
  try:
    payload = json.loads(path.read_text())
    return (
        payload.get('spec_digest') == spec_digest and
        int(payload.get('episode_id')) == int(episode_id) and
        isinstance(payload.get('rows'), list))
  except Exception:
    return False


def _action_diversity(actions: dict) -> tuple[float, float]:
  leaves = [np.asarray(value) for value in actions.values()]
  if not leaves:
    return float('nan'), float('nan')
  flat = np.concatenate([
      value.reshape((value.shape[0], value.shape[1], -1))
      for value in leaves], -1)
  first = flat[:, 0]
  all_actions = flat.reshape((-1, flat.shape[-1]))
  return (
      len(np.unique(first, axis=0)) / max(1, len(first)),
      len(np.unique(all_actions, axis=0)) / max(1, len(all_actions)))


def _finite_mean(values) -> float:
  values = np.asarray(tuple(values), np.float64)
  finite = values[np.isfinite(values)]
  return float(finite.mean()) if len(finite) else float('nan')


def lambda_return_numpy(
    reward: np.ndarray, continuation: np.ndarray,
    bootstrap: np.ndarray, discount: float, lam: float,
) -> np.ndarray:
  """Dreamer lambda return for one [posterior samples, H+1] prefix."""
  reward = np.asarray(reward, np.float64)
  continuation = np.asarray(continuation, np.float64)
  bootstrap = np.asarray(bootstrap, np.float64)
  if not (reward.shape == continuation.shape == bootstrap.shape):
    raise ValueError((reward.shape, continuation.shape, bootstrap.shape))
  if reward.ndim != 2 or reward.shape[1] < 2:
    raise ValueError(reward.shape)
  live = continuation[:, 1:] * float(discount)
  cont = np.full_like(live, float(lam))
  intermediate = reward[:, 1:] + (1 - cont) * live * bootstrap[:, 1:]
  future = bootstrap[:, -1]
  returns = []
  for step in reversed(range(live.shape[1])):
    future = intermediate[:, step] + live[:, step] * cont[:, step] * future
    returns.append(future)
  return np.stack(list(reversed(returns)), 1)


def summarize_result(
    result: dict, *, proposal: str, horizon: int, episode_id: int,
    actual_goal: int, probed_goal: int, episode_success: bool,
    anchor: dict, realized_rewards: np.ndarray | None,
    realized_completion: bool | None, recorded_control_available: bool,
) -> dict:
  arrays = {
      key: np.asarray(value)[0] if key != 'action' else value
      for key, value in result.items()}
  reward = arrays['reward']
  continuation = arrays['continuation']
  reward = reward[:, :horizon + 1]
  continuation = continuation[:, :horizon + 1]
  target_value = arrays['target_value'][:, :horizon + 1]
  discount = float(arrays['discount'][0, 0])
  lam = float(arrays['lambda'][0, 0])
  returns_all = lambda_return_numpy(
      reward, continuation, target_value, discount, lam)
  reward_only_all = lambda_return_numpy(
      reward, continuation, np.zeros_like(target_value), discount, lam)
  returns = returns_all[:, 0]
  reward_only = reward_only_all[:, 0]
  bootstrap = returns - reward_only
  first_unique, all_unique = _action_diversity({
      key: np.asarray(value)[0][:, :horizon]
      for key, value in arrays['action'].items()})
  realized_rewards = (
      None if realized_rewards is None else
      np.asarray(realized_rewards, np.float64))
  realized_sum = (
      float(realized_rewards.sum()) if realized_rewards is not None
      else float('nan'))
  return {
      'episode_id': int(episode_id),
      'actual_goal': int(actual_goal),
      'probed_goal': int(probed_goal),
      'matching_goal': bool(actual_goal == probed_goal),
      'episode_success': bool(episode_success),
      'anchor_ordinal': int(anchor['ordinal']),
      'anchor_timestep': int(anchor['timestep']),
      'anchor_kind': str(anchor['kind']),
      'anchor_fraction': float(anchor['fraction']),
      'proposal': proposal,
      'horizon': int(horizon),
      'posterior_samples': int(len(returns)),
      'recorded_control_available': bool(recorded_control_available),
      'realized_reward_sum': realized_sum,
      'realized_completion': (
          realized_completion if realized_completion is not None else ''),
      'online_value_start_mean': float(arrays['value'][:, 0].mean()),
      'slow_value_start_mean': float(arrays['slowvalue'][:, 0].mean()),
      'online_slow_start_gap': float(
          (arrays['value'][:, 0] - arrays['slowvalue'][:, 0]).mean()),
      'online_value_final_mean': float(
          arrays['value'][:, horizon].mean()),
      'slow_value_final_mean': float(
          arrays['slowvalue'][:, horizon].mean()),
      'return_mean': float(returns.mean()),
      'return_std': float(returns.std()),
      'return_p50': float(np.quantile(returns, .50)),
      'return_p95': float(np.quantile(returns, .95)),
      'return_p99': float(np.quantile(returns, .99)),
      'return_max': float(returns.max()),
      'return_bias_vs_realized': (
          float(returns.mean() - realized_sum)
          if realized_rewards is not None else float('nan')),
      'reward_only_mean': float(reward_only.mean()),
      'bootstrap_mean': float(bootstrap.mean()),
      'bootstrap_abs_mean': float(np.abs(bootstrap).mean()),
      'bootstrap_fraction': float(
          np.abs(bootstrap).mean() / (np.abs(returns).mean() + 1e-8)),
      'imagined_reward_sum_mean': float(reward[:, 1:].sum(-1).mean()),
      'imagined_reward_event_posterior_sample_rate': float(
          (reward[:, 1:] >= .5).any(-1).mean()),
      'imagined_stop_posterior_sample_rate': float(
          ((1 - continuation[:, 1:]) >= .5).any(-1).mean()),
      'prior_entropy_mean': float(
          arrays['prior_entropy'][:, :horizon].mean()),
      'deter_rms_mean': float(arrays['deter_rms'][:, :horizon].mean()),
      'stoch_rms_mean': float(arrays['stoch_rms'][:, :horizon].mean()),
      'first_action_unique_fraction': float(first_unique),
      'all_action_unique_fraction': float(all_unique),
  }


def run_episode(agent, record, data, config, spec_digest: str) -> dict:
  anchors = select_anchors(data, int(config.anchors_per_episode))
  starts = _encode_anchors(
      agent, data, anchors, int(config.chunk_length), record['episode_id'])
  horizons = tuple(int(value) for value in config.horizons)
  max_horizon = max(horizons)
  rows = []
  for anchor, start in zip(anchors, starts):
    start_batch = {key: value[None] for key, value in start.items()}
    timestep = int(anchor['timestep'])
    for goal in range(len(config.goal_names)):
      # The seed is independent of goal, checkpoint, proposal, and horizon.
      # It therefore pairs the posterior draw across these comparisons. Future
      # prior draws cannot be exactly paired once action proposals diverge.
      seed_index = (
          int(config.seed_base) + int(record['episode_id']) * 10_000 +
          int(anchor['ordinal']) * 100)
      actor = agent.imagination_audit(
          start_batch, np.asarray([goal], np.int32), horizon=max_horizon,
          start_batch=int(config.posterior_samples), seed_index=seed_index)
      available = len(data['reward']) - timestep - 1
      recorded_horizons = [value for value in horizons if value <= available]
      recorded = None
      if recorded_horizons:
        recorded_horizon = max(recorded_horizons)
        actions = {
            key: np.asarray(data[key])[
                timestep:timestep + recorded_horizon][None]
            for key in agent.act_space}
        recorded = agent.recorded_imagination_audit(
            start_batch, np.asarray([goal], np.int32), actions,
            horizon=recorded_horizon,
            start_batch=int(config.posterior_samples),
            seed_index=seed_index)
      for horizon in horizons:
        factual = goal == int(record['goal_id'])
        realized = (
            np.asarray(data['reward'])[timestep + 1:timestep + horizon + 1]
            if factual else None)
        completion = (
            bool(np.asarray(data['goal_complete'])[
                timestep + 1:timestep + horizon + 1].any())
            if factual else None)
        rows.append(summarize_result(
            actor, proposal='actor', horizon=horizon,
            episode_id=record['episode_id'], actual_goal=record['goal_id'],
            probed_goal=goal, episode_success=record['success'], anchor=anchor,
            # Actor-proposed actions differ from the recorded behavior, so the
            # observed reward sequence is not a valid calibration target here.
            realized_rewards=None, realized_completion=None,
            recorded_control_available=horizon in recorded_horizons))

        if recorded is not None and horizon in recorded_horizons:
          rows.append(summarize_result(
              recorded, proposal='recorded', horizon=horizon,
              episode_id=record['episode_id'], actual_goal=record['goal_id'],
              probed_goal=goal, episode_success=record['success'],
              anchor=anchor, realized_rewards=realized,
              realized_completion=completion,
              recorded_control_available=True))
  return {
      'format': FORMAT,
      'spec_digest': spec_digest,
      'episode_id': int(record['episode_id']),
      'actual_goal': int(record['goal_id']),
      'episode_success': bool(record['success']),
      'anchors': len(anchors),
      'rows': rows,
  }


def aggregate_rows(rows: list[dict], goal_names: list[str]) -> list[dict]:
  output = []
  scopes = (
      ('actor', 'all'),
      ('actor', 'paired_recorded'),
      ('recorded', 'paired_recorded'),
  )
  for proposal, control_scope in scopes:
    for horizon in sorted({int(row['horizon']) for row in rows}):
      for relationship in ('factual', 'counterfactual'):
        for actual_goal in range(len(goal_names)):
          selected = [
              row for row in rows
              if row['proposal'] == proposal and
              int(row['horizon']) == horizon and
              int(row['actual_goal']) == actual_goal and
              bool(row['matching_goal']) == (relationship == 'factual') and
              (control_scope == 'all' or
               bool(row['recorded_control_available']))]
          if not selected:
            continue
          fields = (
              'online_value_start_mean', 'slow_value_start_mean',
              'online_slow_start_gap', 'online_value_final_mean',
              'slow_value_final_mean', 'return_mean', 'return_p95',
              'return_max', 'return_bias_vs_realized', 'reward_only_mean',
              'bootstrap_mean', 'bootstrap_abs_mean', 'bootstrap_fraction',
              'imagined_reward_sum_mean',
              'imagined_reward_event_posterior_sample_rate',
              'imagined_stop_posterior_sample_rate', 'prior_entropy_mean',
              'first_action_unique_fraction', 'all_action_unique_fraction')
          output.append({
              'proposal': proposal,
              'control_scope': control_scope,
              'horizon': horizon,
              'relationship': relationship,
              'actual_goal': actual_goal,
              'actual_goal_name': goal_names[actual_goal],
              'rows': len(selected),
              **{field: _finite_mean(
                  float(row[field]) for row in selected) for field in fields},
          })
  return output


def critic_provenance(make_agent, args):
  config = args.critic_provenance
  logdir = pathlib.Path(str(args.logdir)).expanduser().resolve()
  logdir.mkdir(parents=True, exist_ok=True)
  selection_path = pathlib.Path(str(config.selection)).expanduser().resolve()
  selection = load_selection(selection_path)
  goal_names = [str(value) for value in config.goal_names]
  horizons = tuple(int(value) for value in config.horizons)
  if not horizons or any(value < 1 for value in horizons):
    raise ValueError(f'Invalid critic-provenance horizons: {horizons}')
  if len(horizons) != len(set(horizons)):
    raise ValueError(f'Duplicate critic-provenance horizons: {horizons}')
  spec = {
      'format': FORMAT,
      'canonical_stage': 'stage3a',
      'checkpoint': str(args.from_checkpoint),
      'selection': str(selection_path),
      'selection_sha256': _sha256(selection_path),
      'goal_names': goal_names,
      'chunk_length': int(config.chunk_length),
      'horizons': list(horizons),
      'mcpb_posterior_samples': int(config.posterior_samples),
      'anchors_per_episode': int(config.anchors_per_episode),
      'seed_base': int(config.seed_base),
      'proposals': ['actor', 'recorded_where_future_actions_exist'],
      'read_only': True,
  }
  digest = _digest(spec)
  spec['spec_digest'] = digest
  spec_path = logdir / 'audit_spec.json'
  if spec_path.is_file() and json.loads(spec_path.read_text()) != spec:
    raise RuntimeError(
        f'Existing critic-provenance audit has another spec: {logdir}')
  _write_json(spec_path, spec)

  agent = make_agent()
  elements.checkpoint.load(str(args.from_checkpoint), {'agent': agent.load})
  if len(goal_names) != agent.model.goal_count:
    raise ValueError((len(goal_names), agent.model.goal_count))
  integrity_before = agent.parameter_digests()
  shard_root = logdir / 'shards'
  shard_root.mkdir(exist_ok=True)
  paths = []
  for number, record in enumerate(selection['episodes'], 1):
    path = shard_root / f'episode_{int(record["episode_id"]):016d}.json'
    paths.append(path)
    if _valid_shard(path, digest, record['episode_id']):
      print(
          f'Critic provenance episode {number}/{len(selection["episodes"])}: '
          'using complete shard', flush=True)
      continue
    with np.load(record['path']) as archive:
      data = {key: np.array(archive[key], copy=True) for key in archive.files}
    payload = run_episode(agent, record, data, config, digest)
    _write_json(path, payload)
    proposals = {row['proposal'] for row in payload['rows']}
    print(
        f'Critic provenance episode {number}/{len(selection["episodes"])}: '
        f'goal={record["goal_id"]} success={record["success"]} '
        f'anchors={payload["anchors"]} proposals={sorted(proposals)}',
        flush=True)

  rows = []
  for path in paths:
    rows.extend(json.loads(path.read_text())['rows'])
  aggregates = aggregate_rows(rows, goal_names)
  integrity_after = agent.parameter_digests()
  summary = {
      'format': FORMAT,
      'canonical_stage': 'stage3a',
      'spec_digest': digest,
      'episodes': len(paths),
      'rows': len(rows),
      'actor_rows': sum(row['proposal'] == 'actor' for row in rows),
      'recorded_rows': sum(row['proposal'] == 'recorded' for row in rows),
      'recorded_control_missing': (
          sum(row['proposal'] == 'actor' for row in rows) -
          sum(row['proposal'] == 'recorded' for row in rows)),
      'horizons': list(horizons),
      'goal_names': goal_names,
      'aggregates': aggregates,
      'integrity_before': integrity_before,
      'integrity_after': integrity_after,
      'integrity_equal': integrity_before == integrity_after,
  }
  _write_csv(logdir / 'rollout_summary.csv', rows)
  _write_csv(logdir / 'aggregate_summary.csv', aggregates)
  _write_json(logdir / 'summary.json', summary)
  if not summary['integrity_equal']:
    raise RuntimeError('Critic-provenance audit mutated checkpoint parameters')
  (logdir / 'audit_complete').touch()
  return summary
