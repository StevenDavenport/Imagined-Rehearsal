#!/usr/bin/env python3
"""Run RLScape Stage 3A critic provenance and calibration audits."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import pathlib
import platform
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embodied.run import milestones  # noqa: E402
from embodied.run.head_audit import _sha256  # noqa: E402
from embodied.run.head_audit import load_selection  # noqa: E402
from scripts.rlscape_m1_pilot import spec_digest  # noqa: E402
from scripts.rlscape_m1_pilot import write_csv  # noqa: E402
from scripts.rlscape_m1_pilot import write_json  # noqa: E402
from scripts.rlscape_m3_process import run_managed  # noqa: E402
from scripts.rlscape_m3_sequential import git_revision  # noqa: E402


GOALS = ('kill_goblin', 'bury_bones', 'chop_logs')
MODEL_GOALS = (*GOALS, 'light_fire', 'catch_fish')
DEFAULT_HORIZONS = (1, 3, 6, 15, 20)
DEFAULT_POSTERIOR_SEED_BASE = 20260813


def parse_ints(text: str) -> tuple[int, ...]:
  values = tuple(int(item.strip()) for item in text.split(',') if item.strip())
  if not values or any(value < 1 for value in values):
    raise ValueError(f'Expected positive comma-separated integers: {text!r}')
  if len(values) != len(set(values)):
    raise ValueError(f'Duplicate values are not allowed: {text!r}')
  return values


def resolve_milestone(experiment_root: pathlib.Path, step: int) -> pathlib.Path:
  archive = milestones.archive_path(experiment_root / 'milestones', step)
  milestones.validate(archive, expected_step=step, full=False)
  return archive


def build_command(
    *, python: pathlib.Path, repo_root: pathlib.Path,
    logdir: pathlib.Path, checkpoint: pathlib.Path,
    selection: pathlib.Path, seed: int, horizons: tuple[int, ...],
    posterior_samples: int, anchors_per_episode: int, chunk_length: int,
    posterior_seed_base: int = DEFAULT_POSTERIOR_SEED_BASE,
) -> list[str]:
  return [
      str(python), str(repo_root / 'dreamerv3' / 'main.py'),
      '--configs', 'rlscape', 'rlscape_click_grid', 'rlscape_chop_logs',
      'rlscape_stage3a_critic_provenance',
      '--seed', str(seed),
      '--logdir', str(logdir),
      '--run.from_checkpoint', str(checkpoint),
      '--critic_provenance.selection', str(selection),
      '--critic_provenance.horizons', ','.join(map(str, horizons)),
      '--critic_provenance.posterior_samples', str(int(posterior_samples)),
      '--critic_provenance.anchors_per_episode', str(int(anchors_per_episode)),
      '--critic_provenance.chunk_length', str(int(chunk_length)),
      '--critic_provenance.seed_base', str(int(posterior_seed_base)),
  ]


def _read_csv(path: pathlib.Path) -> list[dict]:
  with path.open(newline='') as handle:
    rows = list(csv.DictReader(handle))
  for row in rows:
    for key, value in tuple(row.items()):
      try:
        row[key] = float(value)
      except (TypeError, ValueError):
        pass
  return rows


def combine(args, labels: list[str]) -> list[dict]:
  rows = []
  for label in labels:
    for row in _read_csv(args.output_root / label / 'aggregate_summary.csv'):
      rows.append({'checkpoint': label, **row})
  return rows


def load_real_state_summary(args, label: str) -> dict:
  path = args.head_audit_root / label / 'summary.json'
  if not path.is_file():
    raise FileNotFoundError(
        f'Stage 3A requires the completed Stage 1A summary: {path}')
  return json.loads(path.read_text())


def real_state_rows(args, labels: list[str]) -> list[dict]:
  rows = []
  for label in labels:
    summary = load_real_state_summary(args, label)
    for item in summary['value']:
      goal = int(item['goal'])
      online = item['value_inclusive']
      slow = item['slowvalue_inclusive']
      rows.append({
          'checkpoint': label,
          'goal': goal,
          'goal_name': item['goal_name'],
          'states': int(online['count']),
          'target_mean': float(online['target_mean']),
          'online_prediction_mean': float(online['prediction_mean']),
          'online_bias': float(online['bias']),
          'online_mae': float(online['mae']),
          'online_pearson': float(online['pearson']),
          'online_spearman': float(online['spearman']),
          'slow_prediction_mean': float(slow['prediction_mean']),
          'slow_bias': float(slow['bias']),
          'slow_mae': float(slow['mae']),
          'slow_pearson': float(slow['pearson']),
          'slow_spearman': float(slow['spearman']),
          'online_slow_gap_mae': float(item['value_slow_gap_mae']),
          'online_slow_gap_max': float(item['value_slow_gap_max']),
      })
  return rows


def actor_recorded_contrast(rows: list[dict]) -> list[dict]:
  lookup = {
      (row['checkpoint'], int(row['horizon']), row['relationship'],
       int(row['actual_goal']), row['proposal']): row for row in rows
      if row.get('control_scope') == 'paired_recorded'}
  output = []
  keys = sorted({
      (row['checkpoint'], int(row['horizon']), row['relationship'],
       int(row['actual_goal'])) for row in rows})
  for checkpoint, horizon, relationship, goal in keys:
    actor = lookup.get((checkpoint, horizon, relationship, goal, 'actor'))
    recorded = lookup.get((
        checkpoint, horizon, relationship, goal, 'recorded'))
    if not actor or not recorded:
      continue
    output.append({
        'checkpoint': checkpoint,
        'horizon': horizon,
        'relationship': relationship,
        'actual_goal': goal,
        'actual_goal_name': actor['actual_goal_name'],
        'actor_rows': int(actor['rows']),
        'recorded_rows': int(recorded['rows']),
        'actor_return_mean': float(actor['return_mean']),
        'recorded_return_mean': float(recorded['return_mean']),
        'actor_minus_recorded_return': (
            float(actor['return_mean']) - float(recorded['return_mean'])),
        'actor_bootstrap_mean': float(actor['bootstrap_mean']),
        'recorded_bootstrap_mean': float(recorded['bootstrap_mean']),
        'actor_minus_recorded_bootstrap': (
            float(actor['bootstrap_mean']) -
            float(recorded['bootstrap_mean'])),
        'actor_reward_only_mean': float(actor['reward_only_mean']),
        'recorded_reward_only_mean': float(recorded['reward_only_mean']),
        'actor_minus_recorded_reward_only': (
            float(actor['reward_only_mean']) -
            float(recorded['reward_only_mean'])),
        'actor_prior_entropy_mean': float(actor['prior_entropy_mean']),
        'recorded_prior_entropy_mean': float(recorded['prior_entropy_mean']),
        'actor_minus_recorded_prior_entropy': (
            float(actor['prior_entropy_mean']) -
            float(recorded['prior_entropy_mean'])),
  })
  return output


def make_figures(
    output_root: pathlib.Path, rollout_rows: list[dict],
    real_rows: list[dict], contrasts: list[dict],
) -> None:
  import matplotlib.pyplot as plt

  figure_root = output_root / 'figures'
  figure_root.mkdir(exist_ok=True)
  plt.rcParams.update({
      'font.size': 9, 'axes.spines.top': False,
      'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .2})
  labels = list(dict.fromkeys(row['checkpoint'] for row in rollout_rows))
  colors = {'strong': '#4c78a8', 'weak': '#e45756'}

  fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
  x = np.arange(len(GOALS))
  width = .35
  for index, label in enumerate(labels):
    selected = {int(row['goal']): row for row in real_rows
                if row['checkpoint'] == label}
    online = [float(selected[g]['online_bias']) for g in range(len(GOALS))]
    slow = [float(selected[g]['slow_bias']) for g in range(len(GOALS))]
    gap = [float(selected[g]['online_slow_gap_mae'])
           for g in range(len(GOALS))]
    offset = (index - (len(labels) - 1) / 2) * width
    axes[0].bar(x + offset, online, width, label=label,
                color=colors.get(label))
    axes[1].bar(x + offset, slow, width, label=label,
                color=colors.get(label))
    axes[2].bar(x + offset, gap, width, label=label,
                color=colors.get(label))
  for axis, title, ylabel in zip(
      axes,
      ('Online critic', 'Slow critic', 'Online--slow disagreement'),
      ('Bias vs realized return', 'Bias vs realized return', 'Mean absolute gap')):
    axis.set_xticks(x, [goal.replace('_', ' ') for goal in GOALS],
                    rotation=25, ha='right')
    axis.set_title(title)
    axis.set_ylabel(ylabel)
  axes[-1].legend(frameon=False)
  fig.suptitle('Stage 3A: critic calibration on real posterior states')
  for suffix in ('png', 'pdf'):
    fig.savefig(figure_root / f'real_state_calibration.{suffix}', dpi=200)
  plt.close(fig)

  fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
  for label in labels:
    selected = [row for row in rollout_rows
                if row['checkpoint'] == label and
                row['proposal'] == 'actor' and
                row['control_scope'] == 'all' and
                row['relationship'] == 'factual']
    horizons = sorted({int(row['horizon']) for row in selected})
    for axis, field in zip(
        axes, ('return_mean', 'reward_only_mean', 'bootstrap_mean')):
      values = [np.mean([
          float(row[field]) for row in selected
          if int(row['horizon']) == horizon]) for horizon in horizons]
      axis.plot(horizons, values, 'o-', label=label,
                color=colors.get(label))
  for axis, title in zip(
      axes, ('Full IR target', 'Reward-only component', 'Bootstrap component')):
    axis.set_title(title)
    axis.set_xlabel('Imagination horizon H')
    axis.set_ylabel('Mean target contribution')
  axes[-1].legend(frameon=False)
  fig.suptitle('Horizon-dependent provenance of the actor-selected IR target')
  for suffix in ('png', 'pdf'):
    fig.savefig(figure_root / f'horizon_target_decomposition.{suffix}', dpi=200)
  plt.close(fig)

  fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), constrained_layout=True)
  for label in labels:
    selected = [row for row in contrasts
                if row['checkpoint'] == label and
                row['relationship'] == 'factual']
    horizons = sorted({int(row['horizon']) for row in selected})
    for axis, field in zip(
        axes, ('actor_minus_recorded_return',
               'actor_minus_recorded_bootstrap')):
      values = [np.mean([
          float(row[field]) for row in selected
          if int(row['horizon']) == horizon]) for horizon in horizons]
      axis.plot(horizons, values, 'o-', label=label,
                color=colors.get(label))
  axes[0].set_title('Full return difference')
  axes[1].set_title('Bootstrap difference')
  for axis in axes:
    axis.axhline(0, color='black', lw=1)
    axis.set_xlabel('Imagination horizon H')
    axis.set_ylabel('Actor minus recorded proposal')
  axes[-1].legend(frameon=False)
  fig.suptitle('Does the actor select more optimistic imagined trajectories?')
  for suffix in ('png', 'pdf'):
    fig.savefig(figure_root / f'actor_vs_recorded_proposal.{suffix}', dpi=200)
  plt.close(fig)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Run Stage 3A RLScape critic provenance audit.')
  parser.add_argument('--experiment-root', type=pathlib.Path, required=True)
  parser.add_argument('--output-root', type=pathlib.Path)
  parser.add_argument('--head-audit-root', type=pathlib.Path)
  parser.add_argument('--selection', type=pathlib.Path)
  parser.add_argument('--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument('--python', type=pathlib.Path,
                      default=pathlib.Path(sys.executable))
  parser.add_argument('--strong-step', type=int, default=1_250_000)
  parser.add_argument('--weak-step', type=int, default=1_500_000)
  parser.add_argument('--horizons', default='1,3,6,15,20')
  parser.add_argument('--posterior-samples', type=int, default=128)
  parser.add_argument('--anchors-per-episode', type=int, default=4)
  parser.add_argument('--chunk-length', type=int, default=32)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument(
      '--posterior-seed-base', type=int,
      default=DEFAULT_POSTERIOR_SEED_BASE)
  parser.add_argument('--poll-seconds', type=float, default=30)
  parser.add_argument('--startup-timeout', type=float, default=3600)
  parser.add_argument('--stall-timeout', type=float, default=14400)
  parser.add_argument('--stream-output', action='store_true')
  parser.add_argument('--dry-run', action='store_true')
  return parser.parse_args(argv)


def main(argv=None) -> int:
  args = parse_args(argv)
  args.repo_root = args.repo_root.resolve()
  args.experiment_root = args.experiment_root.resolve()
  args.python = args.python.resolve()
  args.output_root = (
      args.output_root.resolve() if args.output_root else
      args.experiment_root / 'stage3a_critic_provenance_calibration')
  args.head_audit_root = (
      args.head_audit_root.resolve() if args.head_audit_root else
      args.experiment_root / 'head_audit')
  args.selection = (
      args.selection.resolve() if args.selection else
      args.head_audit_root / 'selection.json')
  args.horizons = parse_ints(args.horizons)
  if min(
      args.posterior_samples, args.anchors_per_episode,
      args.chunk_length) < 1:
    raise ValueError(
        'posterior samples, anchors, and chunk length must be positive')
  if args.posterior_seed_base < 0:
    raise ValueError('posterior seed base must be nonnegative')
  if not args.python.is_file():
    raise FileNotFoundError(args.python)
  selection = load_selection(args.selection)
  args.output_root.mkdir(parents=True, exist_ok=True)

  jobs = [('strong', args.strong_step), ('weak', args.weak_step)]
  spec = {
      'format': 1,
      'canonical_stage': 'stage3a',
      'name': 'critic_provenance_calibration',
      'git_commit': git_revision(args.repo_root),
      'python_version': platform.python_version(),
      'rlscape_version': importlib.metadata.version('rl-scape'),
      'experiment_root': str(args.experiment_root),
      'selection': str(args.selection),
      'selection_sha256': _sha256(args.selection),
      'selection_episodes': len(selection['episodes']),
      'checkpoints': {label: step for label, step in jobs},
      'horizons': list(args.horizons),
      'mcpb_posterior_samples': args.posterior_samples,
      'anchors_per_episode': args.anchors_per_episode,
      'checkpoint_seed': args.seed,
      'posterior_seed_base': args.posterior_seed_base,
      'proposals': ['actor', 'recorded_where_future_actions_exist'],
      'counterfactual_goal_sweep': list(MODEL_GOALS),
      'real_state_source': str(args.head_audit_root),
      'read_only': True,
      'claim_boundary': (
          'diagnostic attribution; no actor, critic, or world-model update'),
  }
  digest = spec_digest(spec)
  spec_path = args.output_root / 'experiment_spec.json'
  if spec_path.is_file():
    if spec_digest(json.loads(spec_path.read_text())) != digest:
      raise RuntimeError(f'Existing Stage 3A specification differs: {spec_path}')
  else:
    write_json(spec_path, spec)
  write_json(
      args.output_root / 'experiment_spec_digest.json', {'sha256': digest})

  commands = []
  for label, step in jobs:
    archive = resolve_milestone(args.experiment_root, step)
    commands.append((label, build_command(
        python=args.python, repo_root=args.repo_root,
        logdir=args.output_root / label,
        checkpoint=archive / 'checkpoint', selection=args.selection,
        seed=args.seed, horizons=args.horizons,
        posterior_samples=args.posterior_samples,
        anchors_per_episode=args.anchors_per_episode,
        chunk_length=args.chunk_length,
        posterior_seed_base=args.posterior_seed_base)))
  write_json(
      args.output_root / 'commands.json',
      {label: command for label, command in commands})
  if args.dry_run:
    for label, command in commands:
      print(f'\n=== Stage 3A {label} ===')
      print(' '.join(command))
    print(
        f'Plan: {len(selection["episodes"])} episodes x '
        f'{args.anchors_per_episode} anchors x {len(MODEL_GOALS)} goals x '
        f'{len(args.horizons)} reported horizon prefixes x 2 checkpoints. '
        'Each proposal uses one maximum-horizon rollout; recorded controls '
        'are omitted where the episode lacks enough future actions.')
    print(f'Specification: {spec_path}')
    return 0

  active_path = args.output_root / 'active_child.json'
  completed = []
  for label, command in commands:
    logdir = args.output_root / label
    print(f'\n=== Stage 3A {label} ===', flush=True)
    print(' '.join(command), flush=True)
    if (logdir / 'audit_complete').is_file():
      summary = json.loads((logdir / 'summary.json').read_text())
      if summary.get('integrity_equal'):
        completed.append(label)
        print(f'Using complete existing Stage 3A audit: {logdir}', flush=True)
        continue
      raise RuntimeError(f'Existing Stage 3A audit failed integrity: {logdir}')
    logdir.mkdir(parents=True, exist_ok=True)
    child = run_managed(
        command, cwd=args.repo_root, active_path=active_path,
        console_path=logdir / 'console.log',
        activity_paths=[logdir / 'console.log', logdir / 'shards'],
        stream_output=args.stream_output, poll_seconds=args.poll_seconds,
        startup_timeout=args.startup_timeout, stall_timeout=args.stall_timeout)
    if child['returncode'] not in (0, None) or not (
        logdir / 'audit_complete').is_file():
      raise RuntimeError(
          f'Stage 3A failed for {label}: returncode={child["returncode"]}')
    completed.append(label)

  rollout_rows = combine(args, completed)
  real_rows = real_state_rows(args, completed)
  contrasts = actor_recorded_contrast(rollout_rows)
  write_csv(args.output_root / 'combined_rollout_summary.csv', rollout_rows)
  write_csv(args.output_root / 'real_state_calibration.csv', real_rows)
  write_csv(args.output_root / 'actor_vs_recorded.csv', contrasts)
  make_figures(args.output_root, rollout_rows, real_rows, contrasts)
  summaries = {
      label: json.loads((args.output_root / label / 'summary.json').read_text())
      for label in completed}
  write_json(args.output_root / 'comparison_summary.json', {
      'format': 1,
      'canonical_stage': 'stage3a',
      'spec_digest': digest,
      'labels': completed,
      'summaries': summaries,
      'real_state_rows': len(real_rows),
      'rollout_aggregate_rows': len(rollout_rows),
      'actor_recorded_contrasts': len(contrasts),
  })
  (args.output_root / 'stage3a_complete').touch()
  print(f'Stage 3A complete: {args.output_root}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
