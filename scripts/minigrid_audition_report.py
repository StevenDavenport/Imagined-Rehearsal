#!/usr/bin/env python3
"""Aggregate MiniGrid audition learning curves and paired evaluations."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embodied.envs.minigrid_registry import ALTERNATING_CHAIN  # noqa: E402
from embodied.envs.minigrid_registry import PROGRESSIVE_CHAIN  # noqa: E402
from embodied.envs.minigrid_registry import TASK_BY_NAME  # noqa: E402
from scripts.minigrid_common import read_jsonl  # noqa: E402
from scripts.minigrid_common import spec_digest  # noqa: E402
from scripts.minigrid_common import write_csv  # noqa: E402
from scripts.minigrid_common import write_json  # noqa: E402


def _mean(values):
  values = [float(value) for value in values if value is not None]
  return statistics.fmean(values) if values else None


def _std(values):
  values = [float(value) for value in values if value is not None]
  return statistics.pstdev(values) if len(values) > 1 else 0.0


def _curve_metrics(rows, *, budget: int, threshold: float, window: int):
  points = []
  returns = []
  successes = []
  for row in rows:
    if 'episode/score' not in row:
      continue
    score = float(row['episode/score'])
    returns.append(score)
    successes.append(float(score > 0))
    rolling_success = statistics.fmean(successes[-window:])
    rolling_return = statistics.fmean(returns[-window:])
    points.append((
        min(int(row.get('step', 0)), budget), rolling_success,
        rolling_return, len(successes) >= window))
  if not points:
    return {
        'training_episodes': 0,
        'training_auc': 0.0,
        'training_final_rolling_success': 0.0,
        'training_final_rolling_return': 0.0,
        'training_steps_to_threshold': None,
        'training_saturation_step': None,
    }
  compact = [(0, 0.0)]
  for step, success, _return, _full_window in points:
    value = success
    if step == compact[-1][0]:
      compact[-1] = (step, value)
    else:
      compact.append((step, value))
  if compact[-1][0] < budget:
    compact.append((budget, compact[-1][1]))
  area = sum(
      (right[0] - left[0]) * (left[1] + right[1]) / 2
      for left, right in zip(compact, compact[1:]))
  threshold_steps = [
      step for step, success, _return, full_window in points
      if full_window and success >= threshold]
  saturation_steps = [
      step for step, success, _return, full_window in points
      if full_window and success >= 0.9]
  return {
      'training_episodes': len(returns),
      'training_auc': area / max(1, budget),
      'training_final_rolling_success': points[-1][1],
      'training_final_rolling_return': points[-1][2],
      'training_steps_to_threshold': (
          min(threshold_steps) if threshold_steps else None),
      'training_saturation_step': (
          min(saturation_steps) if saturation_steps else None),
  }


def _matched(left, right, tolerance):
  if left is None or right is None:
    return False
  left, right = float(left), float(right)
  if min(abs(left), abs(right)) < 1e-9:
    return abs(left - right) <= tolerance
  return max(abs(left), abs(right)) / min(abs(left), abs(right)) <= 1 + tolerance


def generate_report(experiment_root: pathlib.Path) -> dict[str, Any]:
  experiment_root = pathlib.Path(experiment_root).resolve()
  spec = json.loads((experiment_root / 'experiment_spec.json').read_text())
  tasks = tuple(spec['tasks'])
  seeds = tuple(int(seed) for seed in spec['seeds'])
  budget = int(spec['steps'])
  threshold = float(spec['difficulty_threshold'])
  window = int(spec['difficulty_window'])
  checkpoint_steps = tuple(int(step) for step in spec['checkpoint_steps'])
  source_root = spec.get('source_experiment_root')
  source_root = (
      pathlib.Path(source_root).expanduser().resolve()
      if source_root else None)
  source_spec = (
      json.loads((source_root / 'experiment_spec.json').read_text())
      if source_root else None)
  if source_spec and spec_digest(source_spec) != spec.get('source_spec_digest'):
    raise RuntimeError(
        f'Source experiment spec changed after extension: {source_root}')
  inherited_steps = tuple(
      int(step) for step in
      (source_spec.get('checkpoint_steps', ()) if source_spec else ()))

  run_rows = []
  eval_rows = []
  for task in tasks:
    for seed in seeds:
      run_root = experiment_root / 'runs' / task / f'seed_{seed:04d}'
      training_rows = []
      if source_root:
        training_rows.extend(read_jsonl(
            source_root / 'runs' / task / f'seed_{seed:04d}' /
            'training' / 'episodes.jsonl'))
      training_rows.extend(read_jsonl(
          run_root / 'training' / 'episodes.jsonl'))
      curve = _curve_metrics(
          training_rows,
          budget=budget, threshold=threshold, window=window)
      row = {
          'task': task,
          'family': TASK_BY_NAME[task]['family'],
          'seed': seed,
          **curve,
      }
      evaluation_sources = [(run_root, checkpoint_steps)]
      if source_root:
        evaluation_sources.insert(0, (
            source_root / 'runs' / task / f'seed_{seed:04d}',
            inherited_steps))
      for evaluation_root, selected_steps in evaluation_sources:
        for checkpoint in selected_steps:
          for mode in spec['policy_modes']:
            condition = (
                evaluation_root / 'evaluations' /
                f'step_{checkpoint:012d}' / mode)
            status_path = condition / 'condition_status.json'
            if not status_path.is_file():
              continue
            status = json.loads(status_path.read_text())
            summary = status.get('summary')
            if summary:
              eval_rows.append(summary)
      final = [
          item for item in eval_rows
          if item['task'] == task and item['training_seed'] == seed and
          int(item['checkpoint']) == budget]
      for mode in spec['policy_modes']:
        matches = [item for item in final if item['policy_mode'] == mode]
        if matches:
          row[f'final_{mode}_return'] = matches[-1]['mean_return']
          row[f'final_{mode}_success'] = matches[-1]['success_rate']
      run_rows.append(row)

  task_rows = []
  for task in tasks:
    selected = [row for row in run_rows if row['task'] == task]
    threshold_steps = [
        row['training_steps_to_threshold'] for row in selected
        if row['training_steps_to_threshold'] is not None]
    row = {
        'task': task,
        'family': TASK_BY_NAME[task]['family'],
        'seeds': len(selected),
        'solved_seeds': len(threshold_steps),
        'mean_training_auc': _mean(
            row['training_auc'] for row in selected),
        'std_training_auc': _std(
            row['training_auc'] for row in selected),
        'mean_training_final_rolling_success': _mean(
            row['training_final_rolling_success'] for row in selected),
        'std_training_final_rolling_success': _std(
            row['training_final_rolling_success'] for row in selected),
        'mean_training_final_rolling_return': _mean(
            row['training_final_rolling_return'] for row in selected),
        'std_training_final_rolling_return': _std(
            row['training_final_rolling_return'] for row in selected),
        'median_training_steps_to_threshold': (
            statistics.median(threshold_steps) if threshold_steps else None),
    }
    saturation_steps = [
        item['training_saturation_step'] for item in selected
        if item['training_saturation_step'] is not None]
    row.update({
        'saturated_seeds': len(saturation_steps),
        'median_training_saturation_step': (
            statistics.median(saturation_steps)
            if saturation_steps else None),
        'early_saturation_seeds': sum(
            step <= budget * 0.25 for step in saturation_steps),
    })
    for mode in spec['policy_modes']:
      row[f'mean_final_{mode}_return'] = _mean(
          item.get(f'final_{mode}_return') for item in selected)
      row[f'std_final_{mode}_return'] = _std(
          item.get(f'final_{mode}_return') for item in selected)
      row[f'mean_final_{mode}_success'] = _mean(
          item.get(f'final_{mode}_success') for item in selected)
      row[f'std_final_{mode}_success'] = _std(
          item.get(f'final_{mode}_success') for item in selected)
    task_rows.append(row)

  tolerance = float(spec['difficulty_tolerance'])
  pairwise = []
  for index, left in enumerate(task_rows):
    for right in task_rows[index + 1:]:
      auc_match = _matched(
          left['mean_training_auc'], right['mean_training_auc'], tolerance)
      step_match = _matched(
          left['median_training_steps_to_threshold'],
          right['median_training_steps_to_threshold'], tolerance)
      pairwise.append({
          'left': left['task'],
          'right': right['task'],
          'auc_match': auc_match,
          'threshold_step_match': step_match,
          'matched': bool(auc_match and step_match),
      })

  def chain_status(chain):
    chain = tuple(task for task in chain if task in tasks)
    pairs = {
        frozenset((row['left'], row['right'])): row['matched']
        for row in pairwise}
    qualified = bool(len(chain) >= 2 and all(
        pairs.get(frozenset((left, right)), False)
        for index, left in enumerate(chain)
        for right in chain[index + 1:]))
    return {'tasks': list(chain), 'all_pairwise_matched': qualified}

  report = {
      'format': 1,
      'experiment_root': str(experiment_root),
      'source_experiment_root': (
          str(source_root) if source_root else None),
      'difficulty_threshold': threshold,
      'difficulty_window': window,
      'difficulty_tolerance': tolerance,
      'task_summaries': task_rows,
      'pairwise_matches': pairwise,
      'candidate_chains': {
          'alternating': chain_status(ALTERNATING_CHAIN),
          'progressive': chain_status(PROGRESSIVE_CHAIN),
      },
      'interpretation': (
          'Training AUC and threshold crossings use rolling binary success, '
          'with a full window required before a crossing. Pairwise matching '
          'requires both normalized success AUC and median '
          'steps-to-threshold to agree within the predeclared tolerance. '
          'Early saturation means reaching 90% rolling success in the first '
          'quarter of the budget. Sampled and deterministic evaluation '
          'results remain separate.'),
  }
  outdir = experiment_root / 'analysis'
  write_csv(outdir / 'run_summary.csv', run_rows)
  write_csv(outdir / 'evaluation_summary.csv', eval_rows)
  write_csv(outdir / 'task_summary.csv', task_rows)
  write_csv(outdir / 'pairwise_difficulty.csv', pairwise)
  write_json(outdir / 'summary.json', report)
  return report


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Aggregate a MiniGrid audition experiment.')
  parser.add_argument('--experiment-root', type=pathlib.Path, required=True)
  return parser.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)
  report = generate_report(args.experiment_root)
  print(json.dumps(report['candidate_chains'], indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
