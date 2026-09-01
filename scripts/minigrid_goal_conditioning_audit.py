#!/usr/bin/env python3
"""Analyze cross-goal leakage in completed MiniGrid diagnostics.

This audit never loads a trainable checkpoint and never calls an optimizer. It
reuses the immutable same-state predictions and imagined-rollout provenance
already produced by ``minigrid_sequential_baseline.py``. The analysis asks
whether reward or continuation credit leaks to nonmatching goals, whether the
critic is optimistic for many goals on the same posterior state, and whether
cross-goal reward differences covary with cross-goal value differences.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import sys
from collections import defaultdict
from typing import Iterable

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.minigrid_common import spec_digest  # noqa: E402
from scripts.minigrid_common import write_csv  # noqa: E402
from scripts.minigrid_common import write_json  # noqa: E402


FORMAT = 1
DEFAULT_REWARD_THRESHOLD = 0.5
DEFAULT_VALUE_THRESHOLD = 1.0


def _sha256(path: pathlib.Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def _finite(values: Iterable[float]) -> np.ndarray:
  array = np.asarray(list(values), np.float64)
  return array[np.isfinite(array)]


def _mean(values: Iterable[float]) -> float | None:
  array = _finite(values)
  return float(array.mean()) if len(array) else None


def _correlation(left, right) -> float | None:
  left = np.asarray(left, np.float64).reshape(-1)
  right = np.asarray(right, np.float64).reshape(-1)
  finite = np.isfinite(left) & np.isfinite(right)
  left, right = left[finite], right[finite]
  if len(left) < 2 or left.std() <= 1e-12 or right.std() <= 1e-12:
    return None
  return float(np.corrcoef(left, right)[0, 1])


def _slope(predictor, outcome) -> float | None:
  predictor = np.asarray(predictor, np.float64).reshape(-1)
  outcome = np.asarray(outcome, np.float64).reshape(-1)
  finite = np.isfinite(predictor) & np.isfinite(outcome)
  predictor, outcome = predictor[finite], outcome[finite]
  if len(predictor) < 2:
    return None
  variance = float(np.var(predictor))
  if variance <= 1e-12:
    return None
  covariance = float(np.mean(
      (predictor - predictor.mean()) * (outcome - outcome.mean())))
  return covariance / variance


def _center_goals(values: np.ndarray) -> np.ndarray:
  values = np.asarray(values, np.float64)
  return values - values.mean(-1, keepdims=True)


def _off_diagonal(
    values: np.ndarray, actual: np.ndarray, goal_ids: tuple[int, ...]
) -> np.ndarray:
  values = np.asarray(values)
  actual = np.asarray(actual, np.int32)
  selected = values[:, goal_ids]
  matching = actual[:, None] == np.asarray(goal_ids)[None]
  if not matching.any(1).all():
    missing = np.unique(actual[~matching.any(1)]).tolist()
    raise ValueError(f'Actual goals absent from query scope: {missing}')
  return selected[~matching].reshape((len(actual), len(goal_ids) - 1))


def _pairwise_actor_js(probability: np.ndarray) -> tuple[float, float]:
  probability = np.asarray(probability, np.float64)
  if probability.ndim != 3 or probability.shape[1] < 2:
    return float('nan'), float('nan')
  epsilon = 1e-12
  divergences = []
  for left in range(probability.shape[1]):
    for right in range(left + 1, probability.shape[1]):
      first, second = probability[:, left], probability[:, right]
      middle = 0.5 * (first + second)
      js = 0.5 * (
          np.sum(first * np.log((first + epsilon) / (middle + epsilon)), -1) +
          np.sum(second * np.log(
              (second + epsilon) / (middle + epsilon)), -1))
      divergences.append(js / math.log(2))
  values = np.stack(divergences, 1).mean(1)
  return float(values.mean()), float(np.quantile(values, 0.95))


def _actor_modal_diversity(probability: np.ndarray) -> float | None:
  probability = np.asarray(probability)
  if probability.ndim != 3 or not len(probability):
    return None
  modes = probability.argmax(-1)
  fractions = [
      len(np.unique(row)) / modes.shape[1] for row in modes]
  return float(np.mean(fractions))


def _scope_goal_ids(spec: dict, goal_count: int) -> dict[str, tuple[int, ...]]:
  trained = tuple(
      int(spec['task_goal_ids'][task]) for task in spec['tasks'])
  if len(trained) != len(set(trained)):
    raise ValueError(f'Duplicate trained goal IDs: {trained}')
  if not trained or min(trained) < 0 or max(trained) >= goal_count:
    raise ValueError((trained, goal_count))
  return {'trained': trained, 'all_slots': tuple(range(goal_count))}


def _state_masks(arrays: dict) -> dict[str, np.ndarray]:
  complete = np.asarray(arrays['goal_complete'], bool)
  physical = np.asarray(arrays['is_physical_terminal'], bool)
  first = np.asarray(arrays['is_first'], bool)
  return {
      'all': np.ones(len(complete), bool),
      'ordinary': ~(complete | physical),
      'goal_completion': complete & ~physical,
      'physical_terminal': physical,
      'episode_initial': first,
  }


def summarize_head_scope(
    arrays: dict, goal_ids: tuple[int, ...], mask: np.ndarray,
    *, reward_threshold: float, value_threshold: float,
) -> dict:
  actual_all = np.asarray(arrays['actual_goal'], np.int32)
  eligible = np.isin(actual_all, goal_ids) & np.asarray(mask, bool)
  if not eligible.any():
    return {'states': 0, 'goal_queries': 0}
  actual = actual_all[eligible]
  reward = np.asarray(arrays['reward_prediction'])[eligible]
  continuation = np.asarray(arrays['continuation_prediction'])[eligible]
  value = np.asarray(arrays['value_prediction'])[eligible]
  slowvalue = np.asarray(arrays['slowvalue_prediction'])[eligible]
  reward = reward[:, goal_ids]
  continuation = continuation[:, goal_ids]
  value = value[:, goal_ids]
  slowvalue = slowvalue[:, goal_ids]
  positions = {
      goal: index for index, goal in enumerate(goal_ids)}
  factual_index = np.asarray([positions[int(goal)] for goal in actual])
  row_index = np.arange(len(actual))
  reward_factual = reward[row_index, factual_index]
  continuation_factual = continuation[row_index, factual_index]
  value_factual = value[row_index, factual_index]
  slowvalue_factual = slowvalue[row_index, factual_index]
  factual_return = np.asarray(arrays['return_inclusive'])[eligible]
  local_goals = tuple(range(len(goal_ids)))
  reward_other = _off_diagonal(reward, factual_index, local_goals)
  continuation_other = _off_diagonal(
      continuation, factual_index, local_goals)
  value_other = _off_diagonal(value, factual_index, local_goals)
  slowvalue_other = _off_diagonal(
      slowvalue, factual_index, local_goals)
  discount = float(np.asarray(arrays['discount']))
  stop = np.clip(1 - continuation / discount, 0, 1)
  stop_factual = stop[row_index, factual_index]
  stop_other = _off_diagonal(stop, factual_index, local_goals)
  centered_reward = _center_goals(reward)
  centered_value = _center_goals(value)
  centered_slowvalue = _center_goals(slowvalue)
  actor = arrays.get('actor_probability')
  actor_js_mean = actor_js_p95 = actor_diversity = None
  if actor is not None:
    actor = np.asarray(actor)[eligible][:, goal_ids]
    actor_js_mean, actor_js_p95 = _pairwise_actor_js(actor)
    actor_diversity = _actor_modal_diversity(actor)
  top_reward = np.asarray(goal_ids)[reward.argmax(-1)]
  top_value = np.asarray(goal_ids)[value.argmax(-1)]
  top_stop = np.asarray(goal_ids)[stop.argmax(-1)]
  return {
      'states': int(len(actual)),
      'goal_queries': int(len(actual) * len(goal_ids)),
      'reward_all_mean': float(reward.mean()),
      'reward_factual_mean': float(reward_factual.mean()),
      'reward_nonmatching_mean': float(reward_other.mean()),
      'reward_factual_minus_nonmatching': float(
          reward_factual.mean() - reward_other.mean()),
      'reward_nonmatching_event_rate': float(
          (reward_other >= reward_threshold).mean()),
      'reward_goal_std_mean': float(reward.std(-1).mean()),
      'reward_goal_range_mean': float(np.ptp(reward, axis=-1).mean()),
      'reward_goal_top1_accuracy': float((top_reward == actual).mean()),
      'continuation_all_mean': float(continuation.mean()),
      'continuation_factual_mean': float(continuation_factual.mean()),
      'continuation_nonmatching_mean': float(continuation_other.mean()),
      'stop_factual_mean': float(stop_factual.mean()),
      'stop_nonmatching_mean': float(stop_other.mean()),
      'stop_factual_minus_nonmatching': float(
          stop_factual.mean() - stop_other.mean()),
      'stop_goal_top1_accuracy': float((top_stop == actual).mean()),
      'value_all_mean': float(value.mean()),
      'value_factual_mean': float(value_factual.mean()),
      'value_nonmatching_mean': float(value_other.mean()),
      'value_factual_minus_nonmatching': float(
          value_factual.mean() - value_other.mean()),
      'value_factual_bias_vs_realized': float(
          (value_factual - factual_return).mean()),
      'value_factual_mae_vs_realized': float(
          np.abs(value_factual - factual_return).mean()),
      'value_goal_std_mean': float(value.std(-1).mean()),
      'value_goal_range_mean': float(np.ptp(value, axis=-1).mean()),
      'value_goal_relative_range_mean': float(np.mean(
          np.ptp(value, axis=-1) /
          (np.abs(value).mean(-1) + 1e-8))),
      'value_goal_top1_accuracy': float((top_value == actual).mean()),
      'value_query_over_threshold_rate': float(
          (value > value_threshold).mean()),
      'value_all_goals_over_threshold_state_rate': float(
          (value > value_threshold).all(-1).mean()),
      'value_query_above_factual_return_rate': float(
          (value > factual_return[:, None]).mean()),
      'value_all_goals_above_factual_return_state_rate': float(
          (value > factual_return[:, None]).all(-1).mean()),
      'value_min_across_goals_mean': float(value.min(-1).mean()),
      'value_max_across_goals_mean': float(value.max(-1).mean()),
      'slowvalue_all_mean': float(slowvalue.mean()),
      'slowvalue_factual_mean': float(slowvalue_factual.mean()),
      'slowvalue_nonmatching_mean': float(slowvalue_other.mean()),
      'slowvalue_query_over_threshold_rate': float(
          (slowvalue > value_threshold).mean()),
      'reward_value_within_state_correlation': _correlation(
          centered_reward, centered_value),
      'reward_value_within_state_slope': _slope(
          centered_reward, centered_value),
      'reward_slowvalue_within_state_correlation': _correlation(
          centered_reward, centered_slowvalue),
      'value_slowvalue_within_state_correlation': _correlation(
          centered_value, centered_slowvalue),
      'actor_pairwise_goal_js_mean': actor_js_mean,
      'actor_pairwise_goal_js_p95': actor_js_p95,
      'actor_modal_goal_diversity': actor_diversity,
  }


def _read_csv(path: pathlib.Path) -> list[dict]:
  with path.open(newline='') as handle:
    return list(csv.DictReader(handle))


def _number(row: dict, key: str) -> float:
  try:
    return float(row[key])
  except (KeyError, TypeError, ValueError):
    return float('nan')


def _truth(value) -> bool:
  return str(value).strip().lower() in ('1', 'true', 'yes')


def summarize_imagination_scope(
    raw_rows: list[dict], goal_ids: tuple[int, ...], *, proposal: str,
    comparison_scope: str, horizon: int, value_threshold: float,
) -> dict:
  selected = []
  for row in raw_rows:
    if row.get('proposal') != proposal:
      continue
    if int(row['horizon']) != int(horizon):
      continue
    if int(row['actual_goal']) not in goal_ids:
      continue
    if int(row['probed_goal']) not in goal_ids:
      continue
    if comparison_scope == 'paired_recorded' and not _truth(
        row.get('recorded_control_available')):
      continue
    selected.append(row)
  fields = (
      'online_value_start_mean', 'slow_value_start_mean', 'return_mean',
      'reward_only_mean', 'bootstrap_mean', 'imagined_reward_sum_mean',
      'imagined_reward_event_posterior_sample_rate',
      'imagined_stop_posterior_sample_rate')
  grouped = defaultdict(list)
  for row in selected:
    key = (
        int(row['episode_id']), int(row['anchor_ordinal']),
        int(row['actual_goal']))
    grouped[key].append(row)
  anchor_rows = []
  goal_matrices = {field: [] for field in fields}
  incomplete = 0
  for (_, _, actual), rows in grouped.items():
    by_goal = {int(row['probed_goal']): row for row in rows}
    if any(goal not in by_goal for goal in goal_ids):
      incomplete += 1
      continue
    ordered = [by_goal[goal] for goal in goal_ids]
    actual_index = goal_ids.index(actual)
    values = {
        field: np.asarray([_number(row, field) for row in ordered])
        for field in fields}
    for field, array in values.items():
      goal_matrices[field].append(array)
    anchor = {'actual_goal': actual}
    for field, array in values.items():
      factual = array[actual_index]
      nonmatching = np.delete(array, actual_index)
      anchor[f'{field}_factual'] = factual
      anchor[f'{field}_nonmatching'] = float(nonmatching.mean())
      anchor[f'{field}_goal_std'] = float(array.std())
      anchor[f'{field}_goal_range'] = float(np.ptp(array))
    anchor['all_goal_value_over_threshold'] = bool(
        (values['online_value_start_mean'] > value_threshold).all())
    anchor['minimum_goal_value'] = float(
        values['online_value_start_mean'].min())
    anchor_rows.append(anchor)
  if not anchor_rows:
    return {
        'anchors': 0, 'raw_rows': len(selected),
        'incomplete_goal_sweeps': incomplete}
  output = {
      'anchors': len(anchor_rows),
      'raw_rows': len(selected),
      'incomplete_goal_sweeps': incomplete,
  }
  for field in fields:
    for suffix in ('factual', 'nonmatching', 'goal_std', 'goal_range'):
      key = f'{field}_{suffix}'
      output[f'{key}_mean'] = _mean(row[key] for row in anchor_rows)
  output.update({
      'all_goal_value_over_threshold_anchor_rate': _mean(
          row['all_goal_value_over_threshold'] for row in anchor_rows),
      'minimum_goal_value_mean': _mean(
          row['minimum_goal_value'] for row in anchor_rows),
  })
  matrices = {
      field: np.stack(values, 0) for field, values in goal_matrices.items()}
  centered = {
      field: _center_goals(values) for field, values in matrices.items()}
  output.update({
      'reward_only_value_within_anchor_correlation': _correlation(
          centered['reward_only_mean'],
          centered['online_value_start_mean']),
      'reward_only_return_within_anchor_correlation': _correlation(
          centered['reward_only_mean'], centered['return_mean']),
      'reward_sum_return_within_anchor_correlation': _correlation(
          centered['imagined_reward_sum_mean'], centered['return_mean']),
      'reward_only_bootstrap_within_anchor_correlation': _correlation(
          centered['reward_only_mean'], centered['bootstrap_mean']),
  })
  return output


def _aggregate_rows(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
  grouped = defaultdict(list)
  for row in rows:
    grouped[tuple(row[key] for key in group_keys)].append(row)
  output = []
  for key, selected in sorted(grouped.items()):
    result = dict(zip(group_keys, key))
    result['jobs'] = len(selected)
    metrics = sorted(set().union(*(row.keys() for row in selected)) - set(
        group_keys) - {'arm', 'seed', 'step', 'goal_ids', 'goal_names'})
    for metric in metrics:
      values = [
          row.get(metric) for row in selected
          if isinstance(row.get(metric), (int, float, np.number)) and
          not isinstance(row.get(metric), (bool, np.bool_))]
      mean = _mean(values)
      if mean is not None:
        result[metric] = mean
    output.append(result)
  return output


def _checkpoint_rows(
    head_rows: list[dict], imagination_rows: list[dict], final_step: int,
) -> list[dict]:
  keys = sorted({(row['arm'], row['seed'], row['step']) for row in head_rows})
  output = []
  for arm, seed, step in keys:
    row = {'arm': arm, 'seed': seed, 'step': step,
           'is_final_step': step == final_step}
    for state_kind in ('all', 'ordinary', 'goal_completion'):
      match = next((item for item in head_rows if
                    item['arm'] == arm and item['seed'] == seed and
                    item['step'] == step and item['scope'] == 'trained' and
                    item['state_kind'] == state_kind), None)
      if match:
        for field in (
            'states', 'reward_factual_mean', 'reward_nonmatching_mean',
            'reward_factual_minus_nonmatching',
            'reward_nonmatching_event_rate', 'reward_goal_top1_accuracy',
            'stop_factual_mean', 'stop_nonmatching_mean',
            'stop_factual_minus_nonmatching', 'stop_goal_top1_accuracy',
            'value_all_mean', 'value_factual_mean',
            'value_nonmatching_mean', 'value_factual_minus_nonmatching',
            'value_factual_bias_vs_realized',
            'value_factual_mae_vs_realized',
            'value_goal_range_mean', 'value_goal_relative_range_mean',
            'value_query_over_threshold_rate',
            'value_all_goals_over_threshold_state_rate',
            'value_query_above_factual_return_rate',
            'value_all_goals_above_factual_return_state_rate',
            'value_min_across_goals_mean',
            'reward_value_within_state_correlation',
            'actor_pairwise_goal_js_mean'):
          row[f'{state_kind}_{field}'] = match.get(field)
    for proposal, comparison in (
        ('actor', 'all'), ('recorded', 'paired_recorded')):
      match = next((item for item in imagination_rows if
                    item['arm'] == arm and item['seed'] == seed and
                    item['step'] == step and item['scope'] == 'trained' and
                    item['proposal'] == proposal and
                    item['comparison_scope'] == comparison and
                    item['horizon'] == 15), None)
      if match:
        prefix = 'imag_actor_h15' if proposal == 'actor' else 'imag_recorded_h15'
        for field in (
            'online_value_start_mean_factual_mean',
            'online_value_start_mean_nonmatching_mean',
            'return_mean_factual_mean', 'return_mean_nonmatching_mean',
            'reward_only_mean_factual_mean',
            'reward_only_mean_nonmatching_mean',
            'imagined_reward_sum_mean_factual_mean',
            'imagined_reward_sum_mean_nonmatching_mean',
            'all_goal_value_over_threshold_anchor_rate',
            'reward_only_value_within_anchor_correlation'):
          row[f'{prefix}_{field}'] = match.get(field)
    output.append(row)
  return output


def _format(value, digits=3) -> str:
  if value is None:
    return 'n/a'
  try:
    value = float(value)
  except (TypeError, ValueError):
    return str(value)
  return 'n/a' if not math.isfinite(value) else f'{value:.{digits}f}'


def _write_findings(
    path: pathlib.Path, checkpoint_rows: list[dict], final_step: int,
    first_task_step: int, input_count: int, input_bytes: int,
) -> None:
  final = [row for row in checkpoint_rows if row['step'] == final_step]
  first = [row for row in checkpoint_rows if row['step'] == first_task_step]
  grouped = defaultdict(list)
  for row in final:
    grouped[row['arm']].append(row)
  first_grouped = defaultdict(list)
  for row in first:
    first_grouped[row['arm']].append(row)
  lines = [
      '# MiniGrid cross-goal leakage audit', '',
      'This is a read-only analysis of the immutable same-state and '
      'imagination diagnostics produced by the completed sequential run. It '
      'does not modify or retrain any checkpoint.', '',
      f'- Inputs verified: {input_count} files ({input_bytes / 2**20:.1f} MiB)',
      f'- Final checkpoint: {final_step:,} environment steps',
      '- Primary scope: the four trained goals; all six configured slots are '
      'also retained in the CSV tables.', '',
  ]
  if first:
    lines.extend([
        f'## First task boundary ({first_task_step:,} steps)', '',
        'At this boundary only the first task has been trained. Similar '
        'predictions for the three future goal IDs are therefore especially '
        'direct evidence that a head is relying on shared state features '
        'rather than the requested goal.', '',
        '| Arm | Completion reward factual | Reward nonmatch | Minimum goal '
        'value | Value goal range | All goals above factual return | Actor goal JS |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ])
    for arm, rows in sorted(first_grouped.items()):
      mean = lambda key: _mean(row.get(key) for row in rows)
      lines.append(
          f'| {arm} | '
          f'{_format(mean("goal_completion_reward_factual_mean"))} | '
          f'{_format(mean("goal_completion_reward_nonmatching_mean"))} | '
          f'{_format(mean("all_value_min_across_goals_mean"))} | '
          f'{_format(mean("all_value_goal_range_mean"))} | '
          f'{_format(mean("all_value_all_goals_above_factual_return_state_rate"))} | '
          f'{_format(mean("all_actor_pairwise_goal_js_mean"))} |')
    lines.append('')
  lines.extend([
      '## Final-checkpoint same-state results', '',
      '| Arm | Reward factual | Reward nonmatch | Reward top-1 | Stop factual | '
      'Stop nonmatch | Stop top-1 |',
      '|---|---:|---:|---:|---:|---:|---:|',
  ])
  for arm, rows in sorted(grouped.items()):
    mean = lambda key: _mean(row.get(key) for row in rows)
    lines.append(
        f'| {arm} | '
        f'{_format(mean("goal_completion_reward_factual_mean"))} | '
        f'{_format(mean("goal_completion_reward_nonmatching_mean"))} | '
        f'{_format(mean("goal_completion_reward_goal_top1_accuracy"))} | '
        f'{_format(mean("goal_completion_stop_factual_mean"))} | '
        f'{_format(mean("goal_completion_stop_nonmatching_mean"))} | '
        f'{_format(mean("goal_completion_stop_goal_top1_accuracy"))} |')
  lines.extend([
      '', '## Final-checkpoint critic and actor goal dependence', '',
      '| Arm | Factual value bias | Minimum goal value | All goals above '
      'factual return | Value goal range | Reward-value corr. | Actor goal JS |',
      '|---|---:|---:|---:|---:|---:|---:|',
  ])
  for arm, rows in sorted(grouped.items()):
    mean = lambda key: _mean(row.get(key) for row in rows)
    lines.append(
        f'| {arm} | '
        f'{_format(mean("all_value_factual_bias_vs_realized"))} | '
        f'{_format(mean("all_value_min_across_goals_mean"))} | '
        f'{_format(mean("all_value_all_goals_above_factual_return_state_rate"))} | '
        f'{_format(mean("all_value_goal_range_mean"))} | '
        f'{_format(mean("all_reward_value_within_state_correlation"))} | '
        f'{_format(mean("all_actor_pairwise_goal_js_mean"))} |')
  lines.extend([
      '', '## Final-checkpoint imagined-return decomposition', '',
      '| Arm | Actor H15 full factual | Actor full nonmatch | Actor '
      'reward-only factual | Actor reward-only nonmatch | Recorded H15 full '
      'factual | Recorded reward-only factual |',
      '|---|---:|---:|---:|---:|---:|---:|',
  ])
  for arm, rows in sorted(grouped.items()):
    mean = lambda key: _mean(row.get(key) for row in rows)
    lines.append(
        f'| {arm} | '
        f'{_format(mean("imag_actor_h15_return_mean_factual_mean"))} | '
        f'{_format(mean("imag_actor_h15_return_mean_nonmatching_mean"))} | '
        f'{_format(mean("imag_actor_h15_reward_only_mean_factual_mean"))} | '
        f'{_format(mean("imag_actor_h15_reward_only_mean_nonmatching_mean"))} | '
        f'{_format(mean("imag_recorded_h15_return_mean_factual_mean"))} | '
        f'{_format(mean("imag_recorded_h15_reward_only_mean_factual_mean"))} |')
  lines.extend([
      '', '## Interpretation', '',
      'The reward-leakage theory is supported most directly when completion '
      'reward remains high for nonmatching goals, critic optimism is broad '
      'across those same goal swaps, and centered reward-value correlation is '
      'positive on identical states. Flat, high reward and value across every '
      'goal can also support broad leakage even when centered correlation is '
      'small because there is little remaining cross-goal variance. The '
      'imagination table adds a temporal '
      'check: counterfactual reward-only return or reward sum that grows with '
      'horizon is a mechanism by which leakage can be amplified. When the '
      'full imagined return remains large but its reward-only component is '
      'small, the current numerical error is primarily carried by the critic '
      'bootstrap even if weak goal semantics helped create it upstream.', '',
      'This audit is correlational. A weak same-checkpoint correlation does '
      'not rule out historical reward leakage because bootstrapped critic '
      'targets can retain earlier errors. The clean causal test remains a '
      'from-scratch counterfactual-head intervention with the ordinary critic '
      'objective left unchanged.', '',
  ])
  path.write_text('\n'.join(lines))


def generate_audit(
    experiment_root: pathlib.Path, *, output: pathlib.Path | None = None,
    arms: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
    steps: tuple[int, ...] | None = None,
    reward_threshold: float = DEFAULT_REWARD_THRESHOLD,
    value_threshold: float = DEFAULT_VALUE_THRESHOLD,
    skip_imagination: bool = False,
) -> dict:
  root = pathlib.Path(experiment_root).expanduser().resolve()
  spec_path = root / 'experiment_spec.json'
  if not spec_path.is_file():
    raise FileNotFoundError(spec_path)
  spec = json.loads(spec_path.read_text())
  selected_arms = tuple(arms or tuple(spec['arms']))
  selected_seeds = tuple(int(seed) for seed in (seeds or spec['seeds']))
  selected_steps = tuple(int(step) for step in (
      steps or spec['phase_boundaries']))
  unknown_arms = set(selected_arms) - set(spec['arms'])
  unknown_steps = set(selected_steps) - set(spec['phase_boundaries'])
  if unknown_arms or unknown_steps:
    raise ValueError({
        'unknown_arms': sorted(unknown_arms),
        'unknown_steps': sorted(unknown_steps)})
  if len(selected_seeds) != len(set(selected_seeds)):
    raise ValueError(f'Duplicate seeds: {selected_seeds}')
  output = pathlib.Path(output or (
      root / 'analysis' / 'goal_conditioning_audit')).expanduser().resolve()
  diagnostics = (root / 'diagnostics').resolve()
  runs = (root / 'runs').resolve()
  if output == diagnostics or diagnostics in output.parents:
    raise ValueError(f'Output must not modify diagnostics: {output}')
  if output == runs or runs in output.parents:
    raise ValueError(f'Output must not modify training runs: {output}')

  inventory = []
  head_jobs = []
  imagination_jobs = []
  for arm in selected_arms:
    for seed in selected_seeds:
      for step in selected_steps:
        head_dir = (
            diagnostics / 'head' / arm / f'seed_{seed:04d}' /
            f'step_{step:012d}')
        head_path = head_dir / 'predictions.npz'
        head_summary_path = head_dir / 'summary.json'
        required = [head_path, head_summary_path, head_dir / 'audit_complete']
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
          raise FileNotFoundError(f'Missing head-audit inputs: {missing}')
        head_summary = json.loads(head_summary_path.read_text())
        if not head_summary.get('integrity_equal'):
          raise RuntimeError(f'Unverified head audit: {head_dir}')
        head_jobs.append((arm, seed, step, head_path, head_summary))
        for kind, path in (
            ('head_predictions', head_path),
            ('head_summary', head_summary_path)):
          inventory.append({
              'kind': kind, 'arm': arm, 'seed': seed, 'step': step,
              'path': str(path), 'size_bytes': path.stat().st_size,
              'sha256_before': _sha256(path)})
        if skip_imagination:
          continue
        critic_dir = (
            diagnostics / 'critic_provenance' / arm /
            f'seed_{seed:04d}' / f'step_{step:012d}')
        rollout_path = critic_dir / 'rollout_summary.csv'
        critic_summary_path = critic_dir / 'summary.json'
        required = [
            rollout_path, critic_summary_path, critic_dir / 'audit_complete']
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
          raise FileNotFoundError(
              f'Missing critic-provenance inputs: {missing}')
        critic_summary = json.loads(critic_summary_path.read_text())
        if not critic_summary.get('integrity_equal'):
          raise RuntimeError(f'Unverified critic audit: {critic_dir}')
        imagination_jobs.append(
            (arm, seed, step, rollout_path, critic_summary))
        for kind, path in (
            ('critic_rollouts', rollout_path),
            ('critic_summary', critic_summary_path)):
          inventory.append({
              'kind': kind, 'arm': arm, 'seed': seed, 'step': step,
              'path': str(path), 'size_bytes': path.stat().st_size,
              'sha256_before': _sha256(path)})

  head_rows = []
  goal_names_reference = None
  for arm, seed, step, path, summary in head_jobs:
    with np.load(path) as archive:
      arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    goal_names = [str(value) for value in arrays['goal_names'].tolist()]
    if goal_names_reference is None:
      goal_names_reference = goal_names
    elif goal_names != goal_names_reference:
      raise RuntimeError(f'Goal-name mismatch: {path}')
    if goal_names != [str(value) for value in summary['goal_names']]:
      raise RuntimeError(f'Prediction/summary goal mismatch: {path}')
    scopes = _scope_goal_ids(spec, len(goal_names))
    for scope, goal_ids in scopes.items():
      for state_kind, mask in _state_masks(arrays).items():
        head_rows.append({
            'arm': arm, 'seed': seed, 'step': step,
            'scope': scope, 'state_kind': state_kind,
            'goal_ids': ','.join(str(goal) for goal in goal_ids),
            'goal_names': ','.join(goal_names[goal] for goal in goal_ids),
            **summarize_head_scope(
                arrays, goal_ids, mask,
                reward_threshold=reward_threshold,
                value_threshold=value_threshold),
        })

  imagination_rows = []
  for arm, seed, step, path, summary in imagination_jobs:
    raw_rows = _read_csv(path)
    goal_names = [str(value) for value in summary['goal_names']]
    if goal_names_reference is not None and goal_names != goal_names_reference:
      raise RuntimeError(f'Goal-name mismatch: {path}')
    scopes = _scope_goal_ids(spec, len(goal_names))
    horizons = sorted({int(row['horizon']) for row in raw_rows})
    for scope, goal_ids in scopes.items():
      for proposal, comparison_scope in (
          ('actor', 'all'), ('actor', 'paired_recorded'),
          ('recorded', 'paired_recorded')):
        for horizon in horizons:
          imagination_rows.append({
              'arm': arm, 'seed': seed, 'step': step,
              'scope': scope, 'proposal': proposal,
              'comparison_scope': comparison_scope,
              'horizon': horizon,
              'goal_ids': ','.join(str(goal) for goal in goal_ids),
              'goal_names': ','.join(goal_names[goal] for goal in goal_ids),
              **summarize_imagination_scope(
                  raw_rows, goal_ids, proposal=proposal,
                  comparison_scope=comparison_scope, horizon=horizon,
                  value_threshold=value_threshold),
          })

  for record in inventory:
    path = pathlib.Path(record['path'])
    record['sha256_after'] = _sha256(path)
    record['input_unchanged'] = (
        record['sha256_before'] == record['sha256_after'])
  if not all(record['input_unchanged'] for record in inventory):
    raise RuntimeError('A diagnostic input changed during analysis')

  final_step = max(selected_steps)
  checkpoint_rows = _checkpoint_rows(
      head_rows, imagination_rows, final_step)
  head_aggregate = _aggregate_rows(
      head_rows, ('arm', 'step', 'scope', 'state_kind'))
  imagination_aggregate = _aggregate_rows(
      imagination_rows,
      ('arm', 'step', 'scope', 'proposal', 'comparison_scope', 'horizon'))
  analysis_spec = {
      'format': FORMAT,
      'name': 'minigrid_goal_conditioning_leakage_audit',
      'experiment_root': str(root),
      'experiment_spec_sha256': _sha256(spec_path),
      'arms': list(selected_arms), 'seeds': list(selected_seeds),
      'steps': list(selected_steps),
      'reward_threshold': float(reward_threshold),
      'value_threshold': float(value_threshold),
      'skip_imagination': bool(skip_imagination),
      'goal_names': goal_names_reference,
      'trained_goal_ids': [
          int(spec['task_goal_ids'][task]) for task in spec['tasks']],
      'input_hashes': {
          record['path']: record['sha256_before'] for record in inventory},
      'read_only_inputs': True,
  }
  analysis_spec['spec_digest'] = spec_digest(analysis_spec)
  output.mkdir(parents=True, exist_ok=True)
  write_json(output / 'audit_spec.json', analysis_spec)
  write_csv(output / 'input_inventory.csv', inventory)
  write_csv(output / 'head_goal_conditioning.csv', head_rows)
  write_csv(output / 'head_goal_conditioning_aggregate.csv', head_aggregate)
  write_csv(output / 'imagination_goal_conditioning.csv', imagination_rows)
  write_csv(
      output / 'imagination_goal_conditioning_aggregate.csv',
      imagination_aggregate)
  write_csv(output / 'checkpoint_summary.csv', checkpoint_rows)
  _write_findings(
      output / 'FINDINGS.md', checkpoint_rows, final_step,
      int(spec['phase_steps']), len(inventory),
      sum(record['size_bytes'] for record in inventory))
  result = {
      'format': FORMAT,
      'spec_digest': analysis_spec['spec_digest'],
      'experiment_root': str(root),
      'output': str(output),
      'read_only_inputs': True,
      'input_files': len(inventory),
      'input_bytes': sum(record['size_bytes'] for record in inventory),
      'all_inputs_unchanged': all(
          record['input_unchanged'] for record in inventory),
      'head_jobs': len(head_jobs),
      'critic_jobs': len(imagination_jobs),
      'head_rows': len(head_rows),
      'imagination_rows': len(imagination_rows),
      'checkpoint_rows': len(checkpoint_rows),
      'final_step': final_step,
      'tables': {
          'checkpoint_summary': str(output / 'checkpoint_summary.csv'),
          'head_goal_conditioning': str(
              output / 'head_goal_conditioning.csv'),
          'head_goal_conditioning_aggregate': str(
              output / 'head_goal_conditioning_aggregate.csv'),
          'imagination_goal_conditioning': str(
              output / 'imagination_goal_conditioning.csv'),
          'imagination_goal_conditioning_aggregate': str(
              output / 'imagination_goal_conditioning_aggregate.csv'),
          'input_inventory': str(output / 'input_inventory.csv'),
          'findings': str(output / 'FINDINGS.md'),
      },
      'interpretation_guardrails': [
          'All comparisons hold the posterior state fixed and change only '
          'the queried goal.',
          'The trained-goal scope is primary; all six configured slots are '
          'reported separately.',
          'Counterfactual critic values diagnose goal dependence but do not '
          'have factual Monte Carlo targets under another policy.',
          'Same-checkpoint association is correlational; a counterfactual-head '
          'training intervention is required for a causal claim.',
      ],
  }
  write_json(output / 'summary.json', result)
  (output / 'audit_complete').touch()
  return result


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Analyze same-state cross-goal leakage without training.')
  parser.add_argument('--experiment-root', type=pathlib.Path, required=True)
  parser.add_argument('--output', type=pathlib.Path)
  parser.add_argument('--arms', nargs='+')
  parser.add_argument('--seeds', nargs='+', type=int)
  parser.add_argument('--steps', nargs='+', type=int)
  parser.add_argument(
      '--reward-threshold', type=float, default=DEFAULT_REWARD_THRESHOLD)
  parser.add_argument(
      '--value-threshold', type=float, default=DEFAULT_VALUE_THRESHOLD)
  parser.add_argument('--skip-imagination', action='store_true')
  return parser.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)
  if not math.isfinite(args.reward_threshold):
    raise ValueError('--reward-threshold must be finite')
  if not math.isfinite(args.value_threshold):
    raise ValueError('--value-threshold must be finite')
  result = generate_audit(
      args.experiment_root, output=args.output,
      arms=tuple(args.arms) if args.arms else None,
      seeds=tuple(args.seeds) if args.seeds else None,
      steps=tuple(args.steps) if args.steps else None,
      reward_threshold=args.reward_threshold,
      value_threshold=args.value_threshold,
      skip_imagination=args.skip_imagination)
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
