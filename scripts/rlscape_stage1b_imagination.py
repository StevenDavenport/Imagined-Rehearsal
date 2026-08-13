#!/usr/bin/env python3
"""Run the paired Stage 1b actor-IR imagination-path audit."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embodied.run import milestones  # noqa: E402
from embodied.run.head_audit import _sha256  # noqa: E402
from embodied.run.head_audit import load_selection  # noqa: E402
from scripts.rlscape_m1_pilot import write_json  # noqa: E402
from scripts.rlscape_m3_process import run_managed  # noqa: E402


TRAINED_GOALS = ('kill_goblin', 'bury_bones', 'chop_logs')
MODEL_GOALS = (*TRAINED_GOALS, 'light_fire', 'catch_fish')


def resolve_milestone(experiment_root: pathlib.Path, step: int) -> pathlib.Path:
  archive = milestones.archive_path(experiment_root / 'milestones', step)
  milestones.validate(archive, expected_step=step, full=False)
  return archive


def build_command(
    *, python, repo_root, logdir, checkpoint, selection, seed,
    horizon, particles, anchors_per_episode,
) -> list[str]:
  return [
      str(python), str(repo_root / 'dreamerv3' / 'main.py'),
      '--configs', 'rlscape', 'rlscape_click_grid', 'rlscape_chop_logs',
      'rlscape_m3_imagination_audit',
      '--seed', str(seed),
      '--logdir', str(logdir),
      '--run.from_checkpoint', str(checkpoint),
      '--imagination_audit.selection', str(selection),
      '--imagination_audit.horizon', str(horizon),
      '--imagination_audit.particles', str(particles),
      '--imagination_audit.anchors_per_episode', str(anchors_per_episode),
  ]


def _read_csv(path: pathlib.Path) -> list[dict]:
  with path.open() as handle:
    rows = list(csv.DictReader(handle))
  for row in rows:
    for key, value in tuple(row.items()):
      try:
        row[key] = float(value)
      except (TypeError, ValueError):
        pass
  return rows


def _matrix(rows, field):
  result = np.full((len(TRAINED_GOALS), len(MODEL_GOALS)), np.nan)
  for row in rows:
    actual, probed = int(row['actual_goal']), int(row['probed_goal'])
    if actual < len(TRAINED_GOALS) and probed < len(MODEL_GOALS):
      result[actual, probed] = float(row[field])
  return result


def _heat(ax, matrix, title, vmin=None, vmax=None, cmap='viridis'):
  image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
  ax.set_xticks(
      range(len(MODEL_GOALS)),
      [name.replace('_', ' ') for name in MODEL_GOALS],
      rotation=25, ha='right')
  ax.set_yticks(
      range(len(TRAINED_GOALS)),
      [name.replace('_', ' ') for name in TRAINED_GOALS])
  ax.set_xlabel('Probed goal')
  ax.set_ylabel('Goal of replay anchor')
  ax.set_title(title)
  for row in range(matrix.shape[0]):
    for col in range(matrix.shape[1]):
      if np.isfinite(matrix[row, col]):
        ax.text(col, row, f'{matrix[row, col]:.2f}', ha='center',
                va='center', fontsize=8)
  return image


def make_figures(output_root: pathlib.Path, labels=('strong', 'weak')):
  import matplotlib.pyplot as plt

  figure_dir = output_root / 'figures'
  figure_dir.mkdir(exist_ok=True)
  rows = {
      label: _read_csv(output_root / label / 'goal_sweep.csv')
      for label in labels}
  plt.rcParams.update({
      'font.size': 9, 'axes.spines.top': False,
      'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .2})

  for field, filename, title, cmap in (
      ('return_mean', 'imagined_return', 'IR lambda return at the anchor',
       'magma'),
      ('return_p95', 'imagined_return_p95', '95th percentile IR return',
       'magma'),
      ('reward_event_particle_rate', 'imagined_reward_events',
       'MCPB posterior samples with an imagined reward event', 'viridis'),
      ('stop_particle_rate', 'imagined_stop_events',
       'MCPB posterior samples with an imagined stop event', 'viridis'),
  ):
    matrices = {label: _matrix(rows[label], field) for label in labels}
    values = np.concatenate([matrix[np.isfinite(matrix)]
                             for matrix in matrices.values()])
    vmin = 0 if field.endswith(('rate', 'events')) else float(values.min())
    vmax = 1 if field.endswith(('rate', 'events')) else float(values.max())
    fig, axes = plt.subplots(
        1, len(labels), figsize=(4.8 * len(labels), 3.7),
        constrained_layout=True, squeeze=False)
    for col, label in enumerate(labels):
      image = _heat(
          axes[0, col], matrices[label], label, vmin, vmax, cmap)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.8, label=title)
    fig.suptitle(title)
    fig.savefig(figure_dir / f'{filename}.png', dpi=220, bbox_inches='tight')
    fig.savefig(figure_dir / f'{filename}.pdf', bbox_inches='tight')
    plt.close(fig)

  x = np.arange(len(TRAINED_GOALS))
  width = .35
  fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
  for offset, label, color in zip(
      (-.5, .5), labels, ('#4c78a8', '#e45756')):
    factual = {
        int(row['actual_goal']): row for row in rows[label]
        if int(row['actual_goal']) == int(row['probed_goal'])}
    reward_only = np.asarray([
        float(factual[goal]['reward_only_mean'])
        for goal in range(len(TRAINED_GOALS))])
    bootstrap = np.asarray([
        float(factual[goal]['bootstrap_mean'])
        for goal in range(len(TRAINED_GOALS))])
    axes[0].bar(
        x + offset * width, reward_only, width, color=color,
        label=f'{label}: reward only')
    axes[0].bar(
        x + offset * width, bootstrap, width, bottom=reward_only,
        color=color, alpha=.45, hatch='//', label=f'{label}: bootstrap')
    reinforce = np.asarray([
        float(factual[goal]['reinforce_loss_abs'])
        for goal in range(len(TRAINED_GOALS))])
    entropy = np.asarray([
        float(factual[goal]['entropy_loss_abs'])
        for goal in range(len(TRAINED_GOALS))])
    axes[1].bar(
        x + offset * width, reinforce, width, color=color,
        label=f'{label}: policy-gradient term')
    axes[1].bar(
        x + offset * width, entropy, width, bottom=reinforce,
        color=color, alpha=.45, hatch='//', label=f'{label}: entropy term')
  axes[0].set_title('Lambda-return decomposition')
  axes[0].set_ylabel('Mean return contribution')
  axes[1].set_title('Actor-loss magnitude decomposition')
  axes[1].set_ylabel('Mean absolute weighted loss')
  for axis in axes:
    axis.set_xticks(
        x, [name.replace('_', ' ') for name in TRAINED_GOALS],
        rotation=20, ha='right')
    axis.legend(frameon=False, fontsize=7)
  fig.suptitle('What drives the six-step actor-IR objective')
  fig.savefig(
      figure_dir / 'ir_objective_decomposition.png', dpi=220,
      bbox_inches='tight')
  fig.savefig(
      figure_dir / 'ir_objective_decomposition.pdf', bbox_inches='tight')
  plt.close(fig)

  profiles = {
      label: _read_csv(output_root / label / 'time_profile.csv')
      for label in labels}
  fig, axes = plt.subplots(
      2, len(TRAINED_GOALS), figsize=(10.5, 5.8),
      constrained_layout=True, sharex=True, sharey='row')
  for goal, goal_name in enumerate(TRAINED_GOALS):
    for label, color in zip(labels, ('#4c78a8', '#e45756')):
      selected = sorted([
          row for row in profiles[label]
          if int(row['actual_goal']) == goal and
          int(row['probed_goal']) == goal], key=lambda row: row['step'])
      steps = [int(row['step']) for row in selected]
      axes[0, goal].plot(
          steps, [row['reward_event'] for row in selected], 'o-',
          color=color, label=label)
      axes[1, goal].plot(
          steps, [row['stop'] for row in selected], 'o-',
          color=color, label=label)
    axes[0, goal].set_title(goal_name.replace('_', ' '))
    axes[1, goal].set_xlabel('Imagined state (0 = posterior anchor)')
  axes[0, 0].set_ylabel('Reward-event probability')
  axes[1, 0].set_ylabel('Mean stop probability')
  axes[0, -1].legend(frameon=False)
  fig.suptitle('When reward and termination enter six-step imagination')
  fig.savefig(
      figure_dir / 'imagination_time_profile.png', dpi=220,
      bbox_inches='tight')
  fig.savefig(
      figure_dir / 'imagination_time_profile.pdf', bbox_inches='tight')
  plt.close(fig)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Audit the six-step actor-IR imagination pathway.')
  parser.add_argument('--experiment-root', type=pathlib.Path, required=True)
  parser.add_argument('--output-root', type=pathlib.Path)
  parser.add_argument('--selection', type=pathlib.Path)
  parser.add_argument('--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument('--python', type=pathlib.Path,
                      default=pathlib.Path(sys.executable))
  parser.add_argument('--strong-step', type=int, default=1_250_000)
  parser.add_argument('--weak-step', type=int, default=1_500_000)
  parser.add_argument('--horizon', type=int, default=6)
  parser.add_argument(
      '--particles', type=int, default=128,
      help=(
          'Number of MCPB posterior samples. The legacy flag name is kept '
          'for compatibility with completed Stage 1B manifests.'))
  parser.add_argument('--anchors-per-episode', type=int, default=4)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--poll-seconds', type=float, default=30)
  parser.add_argument('--startup-timeout', type=float, default=3600)
  parser.add_argument('--stall-timeout', type=float, default=7200)
  parser.add_argument('--stream-output', action='store_true')
  parser.add_argument('--dry-run', action='store_true')
  return parser.parse_args(argv)


def main(argv=None) -> int:
  args = parse_args(argv)
  args.repo_root = args.repo_root.expanduser().resolve()
  args.experiment_root = args.experiment_root.expanduser().resolve()
  args.python = args.python.expanduser().resolve()
  args.output_root = (
      args.output_root.expanduser().resolve() if args.output_root else
      args.experiment_root / 'imagination_audit')
  args.selection = (
      args.selection.expanduser().resolve() if args.selection else
      args.experiment_root / 'head_audit' / 'selection.json')
  if min(args.horizon, args.particles, args.anchors_per_episode) < 1:
    raise ValueError(
        'horizon, MCPB posterior samples, and anchors must be positive')
  if not args.python.is_file():
    raise FileNotFoundError(args.python)
  selection = load_selection(args.selection)
  args.output_root.mkdir(parents=True, exist_ok=True)
  workflow_spec = {
      'format': 1,
      'experiment_root': str(args.experiment_root),
      'selection': str(args.selection),
      'selection_sha256': _sha256(args.selection),
      'selection_episodes': len(selection['episodes']),
      'strong_step': args.strong_step,
      'weak_step': args.weak_step,
      'horizon': args.horizon,
      'particles': args.particles,
      'anchors_per_episode': args.anchors_per_episode,
      'goal_sweep': list(MODEL_GOALS),
      'paired_randomness': True,
      'actor_ir_path': True,
  }
  spec_path = args.output_root / 'workflow_spec.json'
  if spec_path.exists() and json.loads(spec_path.read_text()) != workflow_spec:
    raise RuntimeError(f'Existing Stage 1b workflow differs: {spec_path}')
  write_json(spec_path, workflow_spec)

  active_path = args.output_root / 'active_child.json'
  completed = []
  for label, step in (('strong', args.strong_step), ('weak', args.weak_step)):
    archive = resolve_milestone(args.experiment_root, step)
    logdir = args.output_root / label
    command = build_command(
        python=args.python, repo_root=args.repo_root, logdir=logdir,
        checkpoint=archive / 'checkpoint', selection=args.selection,
        seed=args.seed, horizon=args.horizon, particles=args.particles,
        anchors_per_episode=args.anchors_per_episode)
    print(f'\n=== Stage 1b {label}: step={step} ===', flush=True)
    print(' '.join(command), flush=True)
    if args.dry_run:
      continue
    if (logdir / 'audit_complete').is_file():
      completed.append(label)
      print(f'Using complete existing audit: {logdir}', flush=True)
      continue
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
          f'Stage 1b failed for {label}: returncode={child["returncode"]}')
    completed.append(label)
  if args.dry_run:
    return 0
  make_figures(args.output_root, completed)
  summaries = {
      label: json.loads((args.output_root / label / 'summary.json').read_text())
      for label in completed}
  write_json(args.output_root / 'comparison_summary.json', {
      'format': 1,
      'workflow_spec': workflow_spec,
      'labels': completed,
      'summaries': summaries,
  })
  (args.output_root / 'comparison_complete').write_bytes(b'')
  print(f'Stage 1b complete: {args.output_root}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
