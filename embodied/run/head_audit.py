"""Read-only goal-conditioned head audit for immutable Dreamer checkpoints."""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib

import elements
import numpy as np


FORMAT = 1


def _write_json(path: pathlib.Path, value) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
  temporary.replace(path)


def _sha256(path: pathlib.Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def replay_records(replay_dir: str | pathlib.Path) -> list[dict]:
  """Return the exact retained complete episodes from a replay manifest."""
  replay_dir = pathlib.Path(replay_dir).expanduser().resolve()
  manifest_path = replay_dir / 'manifest.json'
  if not manifest_path.is_file():
    raise FileNotFoundError(f'Missing episode replay manifest: {manifest_path}')
  manifest = json.loads(manifest_path.read_text())
  if manifest.get('kind') != 'episode_replay':
    raise ValueError(
        f'Head audit requires episode replay, got {manifest.get("kind")!r}')
  records = []
  for group, slots in sorted(
      manifest['slots'].items(), key=lambda item: int(item[0])):
    for slot in slots:
      path = replay_dir / slot['filename']
      if not path.is_file():
        raise FileNotFoundError(path)
      records.append({
          'episode_id': int(slot['episode_id']),
          'goal_id': int(slot['goal_id']),
          'length': int(slot['length']),
          'filename': slot['filename'],
          'path': str(path),
          'group': int(group),
      })
  return records


def _episode_outcome(path: pathlib.Path) -> dict:
  # NumPy's NPZ reader decompresses members lazily. Inspecting these small
  # scalar streams does not decode the large image member.
  with np.load(path) as episode:
    reward = np.asarray(episode['reward'], np.float32)
    complete = np.asarray(
        episode['goal_complete'] if 'goal_complete' in episode else reward > 0)
    physical = np.asarray(
        episode['is_physical_terminal']
        if 'is_physical_terminal' in episode else np.zeros_like(complete))
    terminal = np.asarray(episode['is_terminal'])
  return {
      'success': bool(complete.any() or (reward > 0).any()),
      'physical_terminal': bool(physical.any()),
      'terminal': bool(terminal.any()),
      'return': float(reward.sum()),
  }


def create_selection(
    replay_dir: str | pathlib.Path,
    output: str | pathlib.Path,
    *,
    goals: tuple[int, ...] = (0, 1, 2),
    successes_per_goal: int = 64,
    failures_per_goal: int = 64,
    seed: int = 0,
) -> dict:
  """Create a deterministic success/failure-balanced episode bank."""
  replay_dir = pathlib.Path(replay_dir).expanduser().resolve()
  output = pathlib.Path(output).expanduser().resolve()
  rng = np.random.default_rng(int(seed))
  indexed = replay_records(replay_dir)
  enriched = []
  for record in indexed:
    if record['goal_id'] not in goals:
      continue
    enriched.append({
        **record,
        **_episode_outcome(pathlib.Path(record['path'])),
    })

  selected = []
  counts = {}
  for goal in goals:
    success = [x for x in enriched if x['goal_id'] == goal and x['success']]
    failure = [x for x in enriched if x['goal_id'] == goal and not x['success']]
    rng.shuffle(success)
    rng.shuffle(failure)
    chosen_success = success[:max(0, int(successes_per_goal))]
    chosen_failure = failure[:max(0, int(failures_per_goal))]
    selected.extend(chosen_success)
    selected.extend(chosen_failure)
    counts[str(goal)] = {
        'available_success': len(success),
        'available_failure': len(failure),
        'selected_success': len(chosen_success),
        'selected_failure': len(chosen_failure),
    }
  selected.sort(key=lambda x: (x['goal_id'], not x['success'], x['episode_id']))
  manifest_path = replay_dir / 'manifest.json'
  result = {
      'format': FORMAT,
      'kind': 'rlscape_head_audit_selection',
      'replay_dir': str(replay_dir),
      'replay_manifest_sha256': _sha256(manifest_path),
      'goals': list(goals),
      'successes_per_goal': int(successes_per_goal),
      'failures_per_goal': int(failures_per_goal),
      'seed': int(seed),
      'counts': counts,
      'episodes': selected,
  }
  _write_json(output, result)
  return result


def create_split_selections(
    replay_dir: str | pathlib.Path,
    train_output: str | pathlib.Path,
    heldout_output: str | pathlib.Path,
    *,
    goals: tuple[int, ...] = (0, 1, 2),
    train_successes_per_goal: int = 96,
    train_failures_per_goal: int = 96,
    heldout_successes_per_goal: int = 64,
    heldout_failures_per_goal: int = 64,
    seed: int = 0,
) -> tuple[dict, dict]:
  """Create deterministic, exactly sized, episode-disjoint banks.

  Stage 4A deliberately uses one draw per goal/outcome stratum and then
  partitions that draw. This is stronger than independently constructing two
  selections, where overlap would otherwise be possible.
  """
  replay_dir = pathlib.Path(replay_dir).expanduser().resolve()
  train_output = pathlib.Path(train_output).expanduser().resolve()
  heldout_output = pathlib.Path(heldout_output).expanduser().resolve()
  if train_output == heldout_output:
    raise ValueError('Train and held-out selection paths must differ')
  requested = {
      'train_success': int(train_successes_per_goal),
      'train_failure': int(train_failures_per_goal),
      'heldout_success': int(heldout_successes_per_goal),
      'heldout_failure': int(heldout_failures_per_goal),
  }
  if any(amount < 0 for amount in requested.values()):
    raise ValueError(f'Selection counts must be non-negative: {requested}')

  rng = np.random.default_rng(int(seed))
  enriched = []
  for record in replay_records(replay_dir):
    if record['goal_id'] in goals:
      enriched.append({
          **record, **_episode_outcome(pathlib.Path(record['path']))})

  train, heldout, counts = [], [], {}
  for goal in goals:
    counts[str(goal)] = {}
    for outcome, train_amount, heldout_amount in (
        (True, requested['train_success'], requested['heldout_success']),
        (False, requested['train_failure'], requested['heldout_failure'])):
      available = [
          record for record in enriched
          if record['goal_id'] == goal and record['success'] == outcome]
      rng.shuffle(available)
      total = train_amount + heldout_amount
      label = 'success' if outcome else 'failure'
      if len(available) < total:
        raise RuntimeError(
            f'Insufficient goal {goal} {label} episodes: need {total}, '
            f'found {len(available)}')
      train.extend(available[:train_amount])
      heldout.extend(available[train_amount:total])
      counts[str(goal)][f'available_{label}'] = len(available)
      counts[str(goal)][f'train_{label}'] = train_amount
      counts[str(goal)][f'heldout_{label}'] = heldout_amount

  train.sort(key=lambda x: (x['goal_id'], not x['success'], x['episode_id']))
  heldout.sort(
      key=lambda x: (x['goal_id'], not x['success'], x['episode_id']))
  train_ids = {record['episode_id'] for record in train}
  heldout_ids = {record['episode_id'] for record in heldout}
  overlap = train_ids & heldout_ids
  if overlap:
    raise RuntimeError(f'Split selections overlap: {sorted(overlap)[:10]}')

  manifest_path = replay_dir / 'manifest.json'
  shared = {
      'format': FORMAT,
      'kind': 'rlscape_head_audit_selection',
      'replay_dir': str(replay_dir),
      'replay_manifest_sha256': _sha256(manifest_path),
      'goals': list(goals),
      'seed': int(seed),
      'split_counts': requested,
      'counts': counts,
  }
  train_result = {
      **shared, 'split': 'repair_train', 'episodes': train,
      'successes_per_goal': requested['train_success'],
      'failures_per_goal': requested['train_failure'],
  }
  heldout_result = {
      **shared, 'split': 'heldout_audit', 'episodes': heldout,
      'successes_per_goal': requested['heldout_success'],
      'failures_per_goal': requested['heldout_failure'],
  }
  _write_json(train_output, train_result)
  _write_json(heldout_output, heldout_result)
  return train_result, heldout_result


def load_selection(path: str | pathlib.Path) -> dict:
  path = pathlib.Path(path).expanduser().resolve()
  selection = json.loads(path.read_text())
  if (
      selection.get('format') != FORMAT or
      selection.get('kind') != 'rlscape_head_audit_selection'
  ):
    raise ValueError(f'Invalid head-audit selection: {path}')
  replay_dir = pathlib.Path(selection['replay_dir'])
  manifest = replay_dir / 'manifest.json'
  if _sha256(manifest) != selection['replay_manifest_sha256']:
    raise RuntimeError(f'Replay manifest changed after selection: {manifest}')
  for episode in selection['episodes']:
    if not pathlib.Path(episode['path']).is_file():
      raise FileNotFoundError(episode['path'])
  return selection


def discounted_returns(
    reward: np.ndarray,
    terminal: np.ndarray,
    discount: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Return inclusive and exclusive Monte-Carlo returns for one episode."""
  reward = np.asarray(reward, np.float64)
  terminal = np.asarray(terminal, bool)
  inclusive = np.zeros_like(reward, np.float64)
  future = 0.0
  for index in range(len(reward) - 1, -1, -1):
    future = reward[index] + discount * (not terminal[index]) * future
    inclusive[index] = future
  exclusive = np.zeros_like(inclusive)
  if len(inclusive) > 1:
    exclusive[:-1] = (
        discount * (~terminal[:-1]).astype(np.float64) * inclusive[1:])
  return inclusive.astype(np.float32), exclusive.astype(np.float32)


def _pad(array: np.ndarray, amount: int, *, first=False) -> np.ndarray:
  if amount <= 0:
    return array
  padding = np.zeros((amount, *array.shape[1:]), array.dtype)
  if first:
    padding[...] = True
  return np.concatenate([array, padding], 0)


def _previous_actions(data: dict, act_space: dict) -> dict:
  result = {}
  for key in act_space:
    action = np.asarray(data[key])
    previous = np.zeros_like(action)
    if len(action) > 1:
      previous[1:] = action[:-1]
    result[key] = previous
  return result


def _safe_corr(lhs, rhs) -> float | None:
  lhs, rhs = np.asarray(lhs), np.asarray(rhs)
  if len(lhs) < 2 or np.std(lhs) == 0 or np.std(rhs) == 0:
    return None
  return float(np.corrcoef(lhs, rhs)[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
  values = np.asarray(values)
  order = np.argsort(values, kind='mergesort')
  ranks = np.empty(len(values), np.float64)
  ranks[order] = np.arange(len(values), dtype=np.float64)
  unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
  del unique
  sums = np.bincount(inverse, weights=ranks)
  return sums[inverse] / counts[inverse]


def _auc(target, score) -> float | None:
  target = np.asarray(target, bool)
  score = np.asarray(score, np.float64)
  positives, negatives = int(target.sum()), int((~target).sum())
  if not positives or not negatives:
    return None
  ranks = _ranks(score) + 1
  return float(
      (ranks[target].sum() - positives * (positives + 1) / 2) /
      (positives * negatives))


def _average_precision(target, score) -> float | None:
  target = np.asarray(target, bool)
  if not target.any():
    return None
  order = np.argsort(-np.asarray(score), kind='mergesort')
  ordered = target[order]
  precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
  return float(precision[ordered].mean())


def binary_metrics(target, score) -> dict:
  target = np.asarray(target, np.float64)
  score = np.asarray(score, np.float64)
  label = target >= 0.5
  clipped = np.clip(score, 0, 1)
  result = {
      'count': int(len(target)),
      'positives': int(label.sum()),
      'negatives': int((~label).sum()),
      'mean_target': float(target.mean()) if len(target) else None,
      'mean_prediction': float(score.mean()) if len(score) else None,
      'mae': float(np.abs(score - target).mean()) if len(score) else None,
      'brier_clipped': float(np.square(clipped - target).mean()) if len(score) else None,
      'auroc': _auc(label, score),
      'average_precision': _average_precision(label, score),
  }
  for name, mask in [('positive', label), ('negative', ~label)]:
    result[f'{name}_mean'] = float(score[mask].mean()) if mask.any() else None
    result[f'{name}_p95'] = (
        float(np.quantile(score[mask], .95)) if mask.any() else None)
  for threshold in (0.01, 0.1, 0.5):
    predicted = score >= threshold
    suffix = str(threshold).replace('.', 'p')
    result[f'tpr_at_{suffix}'] = (
        float(predicted[label].mean()) if label.any() else None)
    result[f'fpr_at_{suffix}'] = (
        float(predicted[~label].mean()) if (~label).any() else None)
  return result


def regression_metrics(target, prediction) -> dict:
  target = np.asarray(target, np.float64)
  prediction = np.asarray(prediction, np.float64)
  error = prediction - target
  return {
      'count': int(len(target)),
      'target_mean': float(target.mean()),
      'prediction_mean': float(prediction.mean()),
      'bias': float(error.mean()),
      'mae': float(np.abs(error).mean()),
      'rmse': float(np.sqrt(np.square(error).mean())),
      'pearson': _safe_corr(target, prediction),
      'spearman': _safe_corr(_ranks(target), _ranks(prediction)),
  }


def _calibration(target, prediction, bins=10) -> list[dict]:
  target = np.asarray(target)
  prediction = np.asarray(prediction)
  if not len(target):
    return []
  edges = np.quantile(prediction, np.linspace(0, 1, bins + 1))
  edges[0], edges[-1] = -np.inf, np.inf
  rows = []
  for index in range(bins):
    mask = (prediction >= edges[index]) & (prediction <= edges[index + 1])
    if index:
      mask &= prediction > edges[index]
    if not mask.any():
      continue
    rows.append({
        'bin': index,
        'count': int(mask.sum()),
        'prediction_mean': float(prediction[mask].mean()),
        'target_mean': float(target[mask].mean()),
        'prediction_min': float(prediction[mask].min()),
        'prediction_max': float(prediction[mask].max()),
    })
  return rows


def summarize_predictions(arrays: dict, goal_names: list[str]) -> dict:
  actual = arrays['actual_goal'].astype(int)
  reward = arrays['reward_target']
  terminal = arrays['is_terminal'].astype(bool)
  physical = arrays['is_physical_terminal'].astype(bool)
  complete = arrays['goal_complete'].astype(bool)
  discount = float(arrays['discount'])
  reward_pred = arrays['reward_prediction']
  continuation = arrays['continuation_prediction']
  value = arrays['value_prediction']
  slowvalue = arrays['slowvalue_prediction']
  bounded_value = arrays.get('bounded_value_prediction')
  goals = list(range(len(goal_names)))

  head_rows = []
  for observed_goal in sorted(set(actual.tolist())):
    observed = actual == observed_goal
    for probed_goal in goals:
      matching = probed_goal == observed_goal
      reward_target = np.where(matching, reward[observed], 0.0)
      continuation_target = np.where(
          matching,
          discount * (~terminal[observed]),
          discount * (~physical[observed]))
      head_rows.append({
          'observed_goal': int(observed_goal),
          'observed_goal_name': goal_names[observed_goal],
          'probed_goal': int(probed_goal),
          'probed_goal_name': goal_names[probed_goal],
          'reward': binary_metrics(
              reward_target, reward_pred[observed, probed_goal]),
          'continuation': binary_metrics(
              continuation_target, continuation[observed, probed_goal]),
      })

  value_rows, calibration, initial_value_rows = [], [], []
  episode_first = arrays['is_first'].astype(bool)
  for goal in sorted(set(actual.tolist())):
    mask = actual == goal
    prediction = value[mask, goal]
    slow = slowvalue[mask, goal]
    inclusive = arrays['return_inclusive'][mask]
    exclusive = arrays['return_exclusive'][mask]
    first = episode_first[mask]
    episode_success = arrays['episode_success'][mask].astype(bool)
    row = {
        'goal': int(goal),
        'goal_name': goal_names[goal],
        'value_inclusive': regression_metrics(inclusive, prediction),
        'value_exclusive': regression_metrics(exclusive, prediction),
        'slowvalue_inclusive': regression_metrics(inclusive, slow),
        'slowvalue_exclusive': regression_metrics(exclusive, slow),
        'value_slow_gap_mae': float(np.abs(prediction - slow).mean()),
        'value_slow_gap_max': float(np.abs(prediction - slow).max()),
        'initial_value_success_auroc': _auc(
            episode_success[first], prediction[first]),
        'initial_value_success_ap': _average_precision(
            episode_success[first], prediction[first]),
        'initial_success_mean': (
            float(prediction[first & episode_success].mean())
            if (first & episode_success).any() else None),
        'initial_failure_mean': (
            float(prediction[first & ~episode_success].mean())
            if (first & ~episode_success).any() else None),
    }
    if bounded_value is not None:
      bounded_prediction = bounded_value[mask, goal]
      row.update({
          'bounded_value_inclusive': regression_metrics(
              inclusive, bounded_prediction),
          'bounded_value_exclusive': regression_metrics(
              exclusive, bounded_prediction),
          'bounded_value_brier': float(np.square(
              bounded_prediction - inclusive).mean()),
          'bounded_value_min': float(bounded_prediction.min()),
          'bounded_value_max': float(bounded_prediction.max()),
          'bounded_initial_success_auroc': _auc(
              episode_success[first], bounded_prediction[first]),
          'bounded_initial_success_ap': _average_precision(
              episode_success[first], bounded_prediction[first]),
      })
    value_rows.append(row)
    for target_name, target in [('inclusive', inclusive), ('exclusive', exclusive)]:
      for item in _calibration(target, prediction):
        calibration.append({
            'goal': int(goal), 'goal_name': goal_names[goal],
            'target': target_name, **item})

    for probed_goal in goals:
      initial_mask = mask & episode_first
      initial_prediction = value[initial_mask, probed_goal]
      initial_success = arrays['episode_success'][initial_mask].astype(bool)
      initial_value_rows.append({
          'observed_goal': int(goal),
          'observed_goal_name': goal_names[goal],
          'probed_goal': int(probed_goal),
          'probed_goal_name': goal_names[probed_goal],
          'count': int(initial_mask.sum()),
          'successes': int(initial_success.sum()),
          'prediction_mean': float(initial_prediction.mean()),
          'success_mean': (
              float(initial_prediction[initial_success].mean())
              if initial_success.any() else None),
          'failure_mean': (
              float(initial_prediction[~initial_success].mean())
              if (~initial_success).any() else None),
          'success_auroc': _auc(initial_success, initial_prediction),
          'success_average_precision': _average_precision(
              initial_success, initial_prediction),
      })

  completion_rows = []
  for event_goal in sorted(set(actual[complete].tolist())):
    mask = complete & (actual == event_goal)
    for probed_goal in goals:
      completion_rows.append({
          'event_goal': int(event_goal),
          'event_goal_name': goal_names[event_goal],
          'probed_goal': int(probed_goal),
          'probed_goal_name': goal_names[probed_goal],
          'count': int(mask.sum()),
          'reward_mean': float(reward_pred[mask, probed_goal].mean()),
          'continuation_mean': float(continuation[mask, probed_goal].mean()),
          'value_mean': float(value[mask, probed_goal].mean()),
          'slowvalue_mean': float(slowvalue[mask, probed_goal].mean()),
          **({
              'bounded_value_mean': float(
                  bounded_value[mask, probed_goal].mean())}
             if bounded_value is not None else {}),
      })

  correct_reward = reward_pred[np.arange(len(actual)), actual]
  correct_continuation = continuation[np.arange(len(actual)), actual]
  predicted_stop = np.clip(1 - correct_continuation / discount, 0, 1)
  state_kinds = {
      'ordinary': ~(complete | physical),
      'goal_completion': complete & ~physical,
      'physical_terminal': physical,
  }
  result = {
      'format': FORMAT,
      'states': int(len(actual)),
      'episodes': int(len(np.unique(arrays['episode_id']))),
      'goal_names': goal_names,
      'reward_factual': binary_metrics(reward, correct_reward),
      'reward_factual_regression': regression_metrics(reward, correct_reward),
      'continuation_factual': binary_metrics(
          discount * (~terminal), correct_continuation),
      'continuation_stop_by_state': {
          name: {
              'count': int(mask.sum()),
              'predicted_stop_mean': (
                  float(predicted_stop[mask].mean()) if mask.any() else None),
              'predicted_stop_p95': (
                  float(np.quantile(predicted_stop[mask], .95))
                  if mask.any() else None),
          }
          for name, mask in state_kinds.items()
      },
      'head_cross_goal': head_rows,
      'value': value_rows,
      'initial_value_cross_goal': initial_value_rows,
      'calibration': calibration,
      'completion_confusion': completion_rows,
      'representation': {
          'posterior_entropy_mean': float(arrays['posterior_entropy'].mean()),
          'posterior_entropy_std': float(arrays['posterior_entropy'].std()),
          'deter_rms_mean': float(arrays['deter_rms'].mean()),
          'stoch_rms_mean': float(arrays['stoch_rms'].mean()),
      },
      'head_uncertainty': {
          key: {
              'mean': float(arrays[key].mean()),
              'p95': float(np.quantile(arrays[key], .95)),
          }
          for key in (
              'reward_entropy', 'continuation_entropy',
              'value_entropy', 'slowvalue_entropy')
      },
      'bounded_value_enabled': bounded_value is not None,
  }
  if complete.any():
    diagonal = reward_pred[complete, actual[complete]]
    offdiag = np.concatenate([
        np.delete(reward_pred[index], actual[index])
        for index in np.flatnonzero(complete)])
    result['completion_selectivity'] = {
        'count': int(complete.sum()),
        'reward_matching_mean': float(diagonal.mean()),
        'reward_nonmatching_mean': float(offdiag.mean()),
        'reward_margin': float(diagonal.mean() - offdiag.mean()),
        'reward_top1_accuracy': float(
            (reward_pred[complete].argmax(-1) == actual[complete]).mean()),
        'continuation_matching_mean': float(
            continuation[complete, actual[complete]].mean()),
        'continuation_nonmatching_mean': float(np.concatenate([
            np.delete(continuation[index], actual[index])
            for index in np.flatnonzero(complete)]).mean()),
    }
  else:
    result['completion_selectivity'] = None
  return result


def _write_tables(logdir: pathlib.Path, summary: dict) -> None:
  with (logdir / 'head_cross_goal.csv').open('w', newline='') as handle:
    fieldnames = [
        'observed_goal', 'observed_goal_name', 'probed_goal',
        'probed_goal_name', 'reward_mae', 'reward_auroc', 'reward_ap',
        'reward_positive_mean', 'reward_negative_mean',
        'continuation_mae', 'continuation_auroc', 'continuation_ap',
        'continuation_positive_mean', 'continuation_negative_mean']
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in summary['head_cross_goal']:
      writer.writerow({
          'observed_goal': row['observed_goal'],
          'observed_goal_name': row['observed_goal_name'],
          'probed_goal': row['probed_goal'],
          'probed_goal_name': row['probed_goal_name'],
          'reward_mae': row['reward']['mae'],
          'reward_auroc': row['reward']['auroc'],
          'reward_ap': row['reward']['average_precision'],
          'reward_positive_mean': row['reward']['positive_mean'],
          'reward_negative_mean': row['reward']['negative_mean'],
          'continuation_mae': row['continuation']['mae'],
          'continuation_auroc': row['continuation']['auroc'],
          'continuation_ap': row['continuation']['average_precision'],
          'continuation_positive_mean': row['continuation']['positive_mean'],
          'continuation_negative_mean': row['continuation']['negative_mean'],
      })
  for filename, rows in [
      ('completion_confusion.csv', summary['completion_confusion']),
      ('value_calibration.csv', summary['calibration']),
      ('initial_value_cross_goal.csv', summary['initial_value_cross_goal']),
  ]:
    if not rows:
      continue
    with (logdir / filename).open('w', newline='') as handle:
      writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
      writer.writeheader()
      writer.writerows(rows)


def head_audit(make_agent, args):
  """Audit one checkpoint on a fixed, persisted replay selection."""
  config = args.head_audit
  logdir = pathlib.Path(str(args.logdir)).expanduser().resolve()
  logdir.mkdir(parents=True, exist_ok=True)
  selection = load_selection(str(config.selection))
  goal_names = [str(x) for x in config.goal_names]
  chunk_length = int(config.chunk_length)
  if chunk_length < 1:
    raise ValueError('head_audit.chunk_length must be positive')

  agent = make_agent()
  elements.checkpoint.load(str(args.from_checkpoint), {'agent': agent.load})
  if len(goal_names) != agent.model.goal_count:
    raise ValueError(
        f'Configured {len(goal_names)} goal names but checkpoint model has '
        f'{agent.model.goal_count} goal slots')
  integrity_before = agent.parameter_digests()
  discount = 1 - 1 / float(config.horizon)

  state = {
      'episode_id': [], 'timestep': [], 'actual_goal': [],
      'reward_target': [], 'is_first': [], 'is_last': [],
      'is_terminal': [], 'is_physical_terminal': [], 'goal_complete': [],
      'episode_success': [], 'return_inclusive': [], 'return_exclusive': [],
  }
  predicted = {
      key: [] for key in (
          'reward_prediction', 'reward_entropy',
          'continuation_prediction', 'continuation_entropy',
          'value_prediction', 'slowvalue_prediction',
          'value_entropy', 'slowvalue_entropy',
          'posterior_entropy', 'deter_rms', 'stoch_rms')
  }
  prediction_map = {
      'reward': 'reward_prediction',
      'continuation': 'continuation_prediction',
      'value': 'value_prediction',
      'slowvalue': 'slowvalue_prediction',
  }
  if getattr(agent.model, 'bounded_value_enabled', False):
    predicted['bounded_value_prediction'] = []
    prediction_map['bounded_value'] = 'bounded_value_prediction'

  for number, record in enumerate(selection['episodes'], 1):
    path = pathlib.Path(record['path'])
    with np.load(path) as archive:
      data = {key: np.array(archive[key], copy=True) for key in archive.files}
    length = len(data['reward'])
    actual_goal = np.asarray(data['goal_id'], np.int32).reshape(length)
    if not np.all(actual_goal == int(record['goal_id'])):
      raise RuntimeError(f'Goal changed within selected episode: {path}')
    inclusive, exclusive = discounted_returns(
        data['reward'], data['is_terminal'], discount)
    previous = _previous_actions(data, agent.act_space)
    carry = agent.init_observe(1)
    episode_outputs = {key: [] for key in predicted}

    for start in range(0, length, chunk_length):
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
      carry, outputs = agent.head_audit_chunk(carry, obs, prevact)
      for source, destination in prediction_map.items():
        episode_outputs[destination].append(np.asarray(outputs[source])[0, :valid])
      for key in (
          'reward_entropy', 'continuation_entropy',
          'value_entropy', 'slowvalue_entropy',
          'posterior_entropy', 'deter_rms', 'stoch_rms'):
        episode_outputs[key].append(np.asarray(outputs[key])[0, :valid])

    for key, pieces in episode_outputs.items():
      predicted[key].append(np.concatenate(pieces, 0))
    state['episode_id'].append(np.full(length, int(record['episode_id']), np.int64))
    state['timestep'].append(np.arange(length, dtype=np.int32))
    state['actual_goal'].append(actual_goal)
    state['reward_target'].append(np.asarray(data['reward'], np.float32))
    for key in (
        'is_first', 'is_last', 'is_terminal',
        'is_physical_terminal', 'goal_complete'):
      state[key].append(np.asarray(data[key], bool))
    state['episode_success'].append(np.full(length, bool(record['success'])))
    state['return_inclusive'].append(inclusive)
    state['return_exclusive'].append(exclusive)
    print(
        f'Head audit episode {number}/{len(selection["episodes"])}: '
        f'goal={record["goal_id"]} success={record["success"]} '
        f'length={length}', flush=True)

  arrays = {
      key: np.concatenate(values, 0) for key, values in {**state, **predicted}.items()}
  arrays['discount'] = np.asarray(discount, np.float32)
  arrays['goal_names'] = np.asarray(goal_names)
  np.savez_compressed(logdir / 'predictions.npz', **arrays)
  summary = summarize_predictions(arrays, goal_names)
  integrity_after = agent.parameter_digests()
  summary.update({
      'checkpoint': str(args.from_checkpoint),
      'selection': str(config.selection),
      'selection_sha256': _sha256(pathlib.Path(str(config.selection))),
      'integrity_before': integrity_before,
      'integrity_after': integrity_after,
      'integrity_equal': integrity_before == integrity_after,
      'chunk_length': chunk_length,
      'discount': discount,
  })
  _write_json(logdir / 'summary.json', summary)
  _write_tables(logdir, summary)
  if not summary['integrity_equal']:
    raise RuntimeError('Head audit mutated checkpoint parameters')
  (logdir / 'audit_complete').write_bytes(b'')
  return summary
