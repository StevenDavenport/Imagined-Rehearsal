#!/usr/bin/env python3
"""Aggregate the sequential MiniGrid baseline into analysis tables."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.minigrid_common import read_jsonl
from scripts.minigrid_common import write_csv
from scripts.minigrid_common import write_json


def _mean(values):
  values = [float(value) for value in values if value is not None]
  return statistics.fmean(values) if values else None


def evaluation_rows(root: pathlib.Path, spec: dict, status: dict) -> list[dict]:
  rows = []
  for arm in spec['arms']:
    for seed in spec['seeds']:
      state = status.get('states', {}).get(f'{arm}__seed_{seed:04d}', {})
      for condition in state.get('evaluations', {}).values():
        summary = condition.get('summary')
        if not summary:
          continue
        rows.append({
            'arm': arm,
            'seed': seed,
            'phase': condition.get('phase'),
            **{key: summary.get(key) for key in (
                'step', 'checkpoint', 'task', 'policy_mode', 'episodes',
                'mean_return', 'median_return', 'return_std', 'successes',
                'success_rate', 'mean_action_length', 'integrity_verified')},
        })
  for row in rows:
    if row.get('step') is None:
      row['step'] = row.pop('checkpoint')
    else:
      row.pop('checkpoint', None)
  return sorted(
      rows, key=lambda row: (
          row['arm'], row['seed'], row['step'], row['task'],
          row['policy_mode']))


def transfer_rows(rows: list[dict], spec: dict) -> list[dict]:
  lookup = {
      (row['arm'], row['seed'], row['task'], row['policy_mode'], row['step']):
      row['success_rate'] for row in rows}
  result = []
  final_step = int(spec['total_steps_per_run'])
  phase_steps = int(spec['phase_steps'])
  for arm in spec['arms']:
    for seed in spec['seeds']:
      for task_index, task in enumerate(spec['tasks']):
        start = task_index * phase_steps
        learned = (task_index + 1) * phase_steps
        for mode in spec['policy_modes']:
          initial = lookup.get((arm, seed, task, mode, 0))
          pretrain = lookup.get((arm, seed, task, mode, start))
          learned_value = lookup.get((arm, seed, task, mode, learned))
          final = lookup.get((arm, seed, task, mode, final_step))
          post_training = [
              row['success_rate'] for row in rows
              if row['arm'] == arm and row['seed'] == seed and
              row['task'] == task and row['policy_mode'] == mode and
              learned <= row['step'] <= final_step]
          peak = max(post_training) if post_training else None
          result.append({
              'arm': arm, 'seed': seed, 'task': task,
              'task_index': task_index, 'policy_mode': mode,
              'initial_success': initial,
              'pretraining_success': pretrain,
              'forward_transfer_vs_initial': (
                  pretrain - initial
                  if pretrain is not None and initial is not None else None),
              'post_acquisition_success': learned_value,
              'peak_post_acquisition_success': peak,
              'final_success': final,
              'forgetting_from_peak': (
                  peak - final if peak is not None and final is not None
                  else None),
              'backward_transfer_from_acquisition': (
                  final - learned_value
                  if final is not None and learned_value is not None else None),
          })
  return result


def arm_rows(transfers: list[dict], spec: dict) -> list[dict]:
  result = []
  for arm in spec['arms']:
    for mode in spec['policy_modes']:
      selected = [
          row for row in transfers
          if row['arm'] == arm and row['policy_mode'] == mode]
      result.append({
          'arm': arm,
          'policy_mode': mode,
          'task_seed_pairs': len(selected),
          'mean_initial_success': _mean(
              row['initial_success'] for row in selected),
          'mean_forward_transfer': _mean(
              row['forward_transfer_vs_initial'] for row in selected),
          'mean_post_acquisition_success': _mean(
              row['post_acquisition_success'] for row in selected),
          'mean_final_success': _mean(
              row['final_success'] for row in selected),
          'mean_forgetting_from_peak': _mean(
              row['forgetting_from_peak'] for row in selected),
          'mean_backward_transfer': _mean(
              row['backward_transfer_from_acquisition'] for row in selected),
      })
  return result


def replay_rows(root: pathlib.Path, spec: dict) -> list[dict]:
  """Preserve every logged replay allocation snapshot as a flat table."""
  output = []
  for arm in spec['arms']:
    for seed in spec['seeds']:
      path = root / 'runs' / arm / f'seed_{seed:04d}' / 'training' / 'metrics.jsonl'
      for record in read_jsonl(path):
        values = {
            key: value for key, value in record.items()
            if key.startswith('replay/')}
        if values:
          output.append({
              'arm': arm, 'seed': seed, 'step': record.get('step'), **values})
  return output


def component_loss_rows(root: pathlib.Path, spec: dict) -> list[dict]:
  """Preserve per-goal component losses without averaging failures away."""
  output = []
  for arm in spec['arms']:
    for seed in spec['seeds']:
      path = root / 'runs' / arm / f'seed_{seed:04d}' / 'training' / 'metrics.jsonl'
      for record in read_jsonl(path):
        values = {
            key: value for key, value in record.items()
            if key.startswith(('train/loss_by_goal/', 'train/batch_goal/'))}
        if values:
          output.append({
              'arm': arm, 'seed': seed, 'step': record.get('step'), **values})
  return output


def diagnostic_rows(root: pathlib.Path, spec: dict) -> list[dict]:
  rows = []
  boundaries = spec['phase_boundaries']
  for kind in ('head', 'critic_provenance'):
    for arm in spec['arms']:
      for seed in spec['seeds']:
        for step in boundaries:
          folder = (
              root / 'diagnostics' / kind / arm / f'seed_{seed:04d}' /
              f'step_{step:012d}')
          summary_path = folder / 'summary.json'
          summary = (
              json.loads(summary_path.read_text())
              if summary_path.is_file() else {})
          row = {
              'kind': kind, 'arm': arm, 'seed': seed, 'step': step,
              'complete': (folder / 'audit_complete').is_file(),
              'summary_path': str(summary_path),
              'episodes': summary.get('episodes'),
              'integrity_equal': summary.get('integrity_equal'),
          }
          if kind == 'head':
            row.update({
                'states': summary.get('states'),
                'reward_factual_auroc': (
                    summary.get('reward_factual', {}).get('auroc')),
                'reward_factual_average_precision': (
                    summary.get('reward_factual', {}).get(
                        'average_precision')),
                'completion_reward_margin': (
                    (summary.get('completion_selectivity') or {}).get(
                        'reward_margin')),
                'actor_distribution_enabled': summary.get(
                    'actor_distribution_enabled'),
            })
          else:
            row.update({
                'rows': summary.get('rows'),
                'actor_rows': summary.get('actor_rows'),
                'recorded_rows': summary.get('recorded_rows'),
                'horizons': json.dumps(summary.get('horizons')),
            })
          rows.append(row)
  return rows


def actor_drift_rows(root: pathlib.Path, spec: dict) -> list[dict]:
  """Compare the actor on identical bank states across phase boundaries."""
  rows = []
  boundaries = tuple(spec['phase_boundaries'])
  for arm in spec['arms']:
    for seed in spec['seeds']:
      for before, after in zip(boundaries[:-1], boundaries[1:]):
        folders = [
            root / 'diagnostics' / 'head' / arm / f'seed_{seed:04d}' /
            f'step_{step:012d}' / 'predictions.npz'
            for step in (before, after)]
        if not all(path.is_file() for path in folders):
          continue
        with np.load(folders[0]) as archive:
          left = {key: np.array(archive[key], copy=True) for key in (
              'episode_id', 'timestep', 'actual_goal', 'actor_probability')}
        with np.load(folders[1]) as archive:
          right = {key: np.array(archive[key], copy=True) for key in (
              'episode_id', 'timestep', 'actual_goal', 'actor_probability')}
        for key in ('episode_id', 'timestep', 'actual_goal'):
          if not np.array_equal(left[key], right[key]):
            raise RuntimeError(
                f'Actor drift inputs are not identical: {folders}, {key}')
        actual = left['actual_goal'].astype(int)
        for task in spec['tasks']:
          goal = int(spec['task_goal_ids'][task])
          mask = actual == goal
          if not mask.any():
            continue
          first = left['actor_probability'][mask, goal]
          second = right['actor_probability'][mask, goal]
          middle = 0.5 * (first + second)
          epsilon = 1e-8
          js = 0.5 * (
              np.sum(first * np.log((first + epsilon) / (middle + epsilon)), -1) +
              np.sum(second * np.log((second + epsilon) / (middle + epsilon)), -1))
          rows.append({
              'arm': arm, 'seed': seed, 'task': task, 'goal_id': goal,
              'before_step': before, 'after_step': after,
              'states': int(mask.sum()),
              'js_divergence_mean': float(js.mean()),
              'js_divergence_p95': float(np.quantile(js, 0.95)),
              'modal_action_agreement': float(
                  (first.argmax(-1) == second.argmax(-1)).mean()),
              'entropy_before': float((-first * np.log(
                  first + epsilon)).sum(-1).mean()),
              'entropy_after': float((-second * np.log(
                  second + epsilon)).sum(-1).mean()),
          })
  return rows


def generate_report(experiment_root: pathlib.Path) -> dict:
  root = pathlib.Path(experiment_root).expanduser().resolve()
  spec = json.loads((root / 'experiment_spec.json').read_text())
  status = json.loads((root / 'supervisor_status.json').read_text())
  analysis = root / 'analysis'
  analysis.mkdir(parents=True, exist_ok=True)
  evaluations = evaluation_rows(root, spec, status)
  transfers = transfer_rows(evaluations, spec)
  arms = arm_rows(transfers, spec)
  replay = replay_rows(root, spec)
  losses = component_loss_rows(root, spec)
  diagnostics = (
      diagnostic_rows(root, spec)
      if spec.get('diagnostics_enabled', True) else [])
  actor_drift = (
      actor_drift_rows(root, spec)
      if spec.get('diagnostics_enabled', True) else [])
  write_csv(analysis / 'evaluation_curves.csv', evaluations)
  write_csv(analysis / 'transfer_and_forgetting.csv', transfers)
  write_csv(analysis / 'arm_summary.csv', arms)
  write_csv(analysis / 'replay_allocation.csv', replay)
  write_csv(analysis / 'component_losses_by_goal.csv', losses)
  write_csv(analysis / 'diagnostic_inventory.csv', diagnostics)
  write_csv(analysis / 'actor_same_state_drift.csv', actor_drift)
  summary = {
      'format': 1,
      'experiment': spec['name'],
      'evaluation_rows': len(evaluations),
      'expected_evaluation_rows': (
          spec['evaluation_units_per_run'] * len(spec['arms']) *
          len(spec['seeds'])
          if spec.get('evaluation_enabled', True) else 0),
      'evaluation_complete': bool(
          len(evaluations) == (
              spec['evaluation_units_per_run'] * len(spec['arms']) *
              len(spec['seeds'])
              if spec.get('evaluation_enabled', True) else 0) and
          all(bool(row['integrity_verified']) for row in evaluations)),
      'arm_summaries': arms,
      'transfer_rows': len(transfers),
      'replay_snapshots': len(replay),
      'component_loss_snapshots': len(losses),
      'diagnostic_jobs': len(diagnostics),
      'diagnostic_jobs_complete': sum(
          bool(row['complete']) for row in diagnostics),
      'actor_drift_comparisons': len(actor_drift),
      'interpretation_guardrails': [
          'Uniform reservoir is the primary CIR baseline, not an allocation '
          'mistake.',
          'The 50:50 arm tests whether ordinary current-biased replay already '
          'solves any apparent CIR benefit.',
          'Counterfactual critic values are diagnostics, not supervised '
          'targets in this baseline.',
          'Sampled and deterministic policy results remain separate.',
      ],
      'tables': {
          'evaluation_curves': str(analysis / 'evaluation_curves.csv'),
          'transfer_and_forgetting': str(
              analysis / 'transfer_and_forgetting.csv'),
          'arm_summary': str(analysis / 'arm_summary.csv'),
          'replay_allocation': str(analysis / 'replay_allocation.csv'),
          'component_losses_by_goal': str(
              analysis / 'component_losses_by_goal.csv'),
          'diagnostic_inventory': str(analysis / 'diagnostic_inventory.csv'),
          'actor_same_state_drift': str(
              analysis / 'actor_same_state_drift.csv'),
      },
  }
  write_json(analysis / 'summary.json', summary)
  return summary


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Aggregate a sequential MiniGrid baseline.')
  parser.add_argument('--experiment-root', type=pathlib.Path, required=True)
  return parser.parse_args(argv)


def main(argv=None):
  summary = generate_report(parse_args(argv).experiment_root)
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
