#!/usr/bin/env python3
"""Run RLScape Stage 1A same-state audits on strong and weak checkpoints."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embodied.run.head_audit import _sha256  # noqa: E402
from embodied.run.head_audit import create_selection  # noqa: E402
from embodied.run.head_audit import load_selection  # noqa: E402
from embodied.run import milestones  # noqa: E402
from scripts.rlscape_m1_pilot import write_json  # noqa: E402
from scripts.rlscape_m3_process import run_managed  # noqa: E402


GOALS = ('kill_goblin', 'bury_bones', 'chop_logs')
MODEL_GOALS = (*GOALS, 'light_fire', 'catch_fish')


def build_command(
    *,
    python: pathlib.Path,
    repo_root: pathlib.Path,
    logdir: pathlib.Path,
    checkpoint: pathlib.Path,
    selection: pathlib.Path,
    seed: int,
    chunk_length: int,
) -> list[str]:
  return [
      str(python), str(repo_root / 'dreamerv3' / 'main.py'),
      '--configs', 'rlscape', 'rlscape_click_grid', 'rlscape_chop_logs',
      'rlscape_m3_head_audit',
      '--seed', str(seed),
      '--logdir', str(logdir),
      '--run.from_checkpoint', str(checkpoint),
      '--head_audit.selection', str(selection),
      '--head_audit.chunk_length', str(chunk_length),
  ]


def resolve_milestone(experiment_root: pathlib.Path, step: int) -> pathlib.Path:
  archive = milestones.archive_path(experiment_root / 'milestones', step)
  milestones.validate(archive, expected_step=step, full=False)
  return archive


def prepare_selection(args) -> dict:
  data_archive = resolve_milestone(args.experiment_root, args.data_step)
  selection_path = args.output_root / 'selection.json'
  if selection_path.exists():
    selection = load_selection(selection_path)
    expected_replay = str((data_archive / 'replay').resolve())
    if selection['replay_dir'] != expected_replay:
      raise RuntimeError(
          f'Existing selection uses {selection["replay_dir"]}, expected '
          f'{expected_replay}')
    return selection
  return create_selection(
      data_archive / 'replay', selection_path,
      goals=tuple(range(len(GOALS))),
      successes_per_goal=args.successes_per_goal,
      failures_per_goal=args.failures_per_goal,
      seed=args.selection_seed)


def _load_predictions(path: pathlib.Path) -> dict:
  with np.load(path) as archive:
    return {key: np.array(archive[key], copy=True) for key in archive.files}


def _completion_matrix(
    data: dict, field: str, observed_goals: int, probed_goals: int,
) -> np.ndarray:
  matrix = np.full((observed_goals, probed_goals), np.nan, np.float64)
  actual = data['actual_goal'].astype(int)
  complete = data['goal_complete'].astype(bool)
  values = data[field]
  for event_goal in range(observed_goals):
    mask = complete & (actual == event_goal)
    if mask.any():
      matrix[event_goal] = values[mask, :probed_goals].mean(0)
  return matrix


def _initial_value_matrix(data: dict, success: bool) -> np.ndarray:
  matrix = np.full((len(GOALS), len(MODEL_GOALS)), np.nan, np.float64)
  actual = data['actual_goal'].astype(int)
  first = data['is_first'].astype(bool)
  outcome = data['episode_success'].astype(bool)
  for observed_goal in range(len(GOALS)):
    mask = first & (actual == observed_goal) & (outcome == success)
    if mask.any():
      matrix[observed_goal] = data['value_prediction'][
          mask, :len(MODEL_GOALS)].mean(0)
  return matrix


def _heat(
    ax, matrix, title, xnames, ynames=None,
    vmin=0, vmax=1, cmap='viridis', value_format='.2f',
):
  ynames = xnames if ynames is None else ynames
  image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
  ax.set_xticks(range(len(xnames)), [x.replace('_', ' ') for x in xnames],
                rotation=25, ha='right')
  ax.set_yticks(range(len(ynames)), [x.replace('_', ' ') for x in ynames])
  ax.set_xlabel('Probed goal')
  ax.set_ylabel('Observed completion event')
  ax.set_title(title)
  for row in range(matrix.shape[0]):
    for col in range(matrix.shape[1]):
      value = matrix[row, col]
      if np.isfinite(value):
        ax.text(col, row, format(value, value_format), ha='center', va='center',
                color='white' if value > .55 else 'black', fontsize=8)
  return image


def make_figures(output_root: pathlib.Path, labels: list[str]) -> None:
  import matplotlib.pyplot as plt

  plt.rcParams.update({
      'font.size': 9,
      'axes.spines.top': False,
      'axes.spines.right': False,
      'axes.grid': True,
      'grid.alpha': .2,
      'savefig.bbox': 'tight',
      'savefig.dpi': 220,
  })
  datasets = {
      label: _load_predictions(output_root / label / 'predictions.npz')
      for label in labels}
  figure_dir = output_root / 'figures'
  figure_dir.mkdir(exist_ok=True)

  fig, axes = plt.subplots(2, len(labels), figsize=(4.1 * len(labels), 6.3),
                           constrained_layout=True, squeeze=False)
  for col, label in enumerate(labels):
    data = datasets[label]
    reward = _completion_matrix(
        data, 'reward_prediction', len(GOALS), len(MODEL_GOALS))
    continuation = _completion_matrix(
        data, 'continuation_prediction', len(GOALS), len(MODEL_GOALS))
    discount = float(data['discount'])
    stop = np.clip(1 - continuation / discount, 0, 1)
    im1 = _heat(
        axes[0, col], reward, f'{label}: reward', MODEL_GOALS, GOALS)
    im2 = _heat(
        axes[1, col], stop, f'{label}: predicted stop', MODEL_GOALS, GOALS,
        cmap='magma')
  fig.colorbar(im1, ax=axes[0].tolist(), shrink=.8, label='Predicted reward')
  fig.colorbar(im2, ax=axes[1].tolist(), shrink=.8,
               label='Predicted termination probability')
  fig.suptitle('Goal-conditioning confusion on genuine completion states')
  fig.savefig(figure_dir / 'completion_confusion.pdf')
  fig.savefig(figure_dir / 'completion_confusion.png')
  plt.close(fig)

  all_values = np.concatenate([
      dataset['value_prediction'][:, :len(MODEL_GOALS)].reshape(-1)
      for dataset in datasets.values()])
  value_min, value_max = np.quantile(all_values, [.01, .99])
  fig, axes = plt.subplots(
      2, len(labels), figsize=(4.8 * len(labels), 6.4),
      constrained_layout=True, squeeze=False)
  for col, label in enumerate(labels):
    data = datasets[label]
    successful = _initial_value_matrix(data, True)
    failed = _initial_value_matrix(data, False)
    im = _heat(
        axes[0, col], successful, f'{label}: successful episodes',
        MODEL_GOALS, GOALS, vmin=value_min, vmax=value_max, cmap='coolwarm')
    _heat(
        axes[1, col], failed, f'{label}: failed episodes',
        MODEL_GOALS, GOALS, vmin=value_min, vmax=value_max, cmap='coolwarm')
  fig.colorbar(
      im, ax=axes.ravel().tolist(), shrink=.8,
      label='Predicted initial value')
  fig.suptitle('Critic goal sweep at the identical episode-start states')
  fig.savefig(figure_dir / 'initial_value_goal_confusion.pdf')
  fig.savefig(figure_dir / 'initial_value_goal_confusion.png')
  plt.close(fig)

  fig, axes = plt.subplots(1, len(GOALS), figsize=(10.5, 3.2),
                           constrained_layout=True, sharex=True, sharey=True)
  colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(labels))))
  for goal, ax in enumerate(axes):
    for color, label in zip(colors, labels):
      data = datasets[label]
      mask = data['actual_goal'] == goal
      target = data['return_inclusive'][mask]
      prediction = data['value_prediction'][mask, goal]
      edges = np.quantile(prediction, np.linspace(0, 1, 9))
      xs, ys = [], []
      for low, high in zip(edges[:-1], edges[1:]):
        inside = (prediction >= low) & (prediction <= high)
        if inside.any():
          xs.append(prediction[inside].mean())
          ys.append(target[inside].mean())
      ax.plot(xs, ys, 'o-', color=color, label=label)
    ax.plot([0, 1], [0, 1], '--', color='0.45', lw=1)
    ax.set_title(GOALS[goal].replace('_', ' '))
    ax.set_xlabel('Predicted value')
  axes[0].set_ylabel('Empirical discounted return')
  axes[-1].legend(frameon=True)
  fig.suptitle('Critic calibration on the identical replay state bank')
  fig.savefig(figure_dir / 'value_calibration.pdf')
  fig.savefig(figure_dir / 'value_calibration.png')
  plt.close(fig)

  fig, axes = plt.subplots(2, len(GOALS), figsize=(10.5, 5.8),
                           constrained_layout=True, sharey='row')
  for goal in range(len(GOALS)):
    reward_groups, stop_groups, ticklabels = [], [], []
    for label in labels:
      data = datasets[label]
      actual = data['actual_goal'].astype(int)
      matching = actual == goal
      positive = matching & data['goal_complete'].astype(bool)
      negative = matching & ~data['goal_complete'].astype(bool)
      reward_groups.extend([
          data['reward_prediction'][positive, goal],
          data['reward_prediction'][negative, goal]])
      discount = float(data['discount'])
      stop = np.clip(1 - data['continuation_prediction'][:, goal] / discount, 0, 1)
      stop_groups.extend([stop[positive], stop[negative]])
      ticklabels.extend([f'{label}\ncomplete', f'{label}\nother'])
    axes[0, goal].boxplot(reward_groups, showfliers=False)
    axes[1, goal].boxplot(stop_groups, showfliers=False)
    axes[0, goal].set_title(GOALS[goal].replace('_', ' '))
    axes[1, goal].set_xticks(range(1, len(ticklabels) + 1), ticklabels,
                            rotation=25, ha='right', fontsize=7)
  axes[0, 0].set_ylabel('Predicted reward')
  axes[1, 0].set_ylabel('Predicted stop probability')
  fig.suptitle('Separation of completion states from ordinary states')
  fig.savefig(figure_dir / 'head_separation.pdf')
  fig.savefig(figure_dir / 'head_separation.png')
  plt.close(fig)

  if len(labels) == 2:
    strong, weak = [datasets[x] for x in labels]
    if not np.array_equal(strong['episode_id'], weak['episode_id']):
      raise RuntimeError('Checkpoint audits do not use identical state order')
    fig, axes = plt.subplots(1, len(GOALS), figsize=(10.5, 3.2),
                             constrained_layout=True, sharex=True, sharey=True)
    for goal, ax in enumerate(axes):
      mask = strong['actual_goal'] == goal
      ax.hexbin(
          strong['value_prediction'][mask, goal],
          weak['value_prediction'][mask, goal],
          gridsize=35, mincnt=1, cmap='Blues')
      lo = min(
          strong['value_prediction'][mask, goal].min(),
          weak['value_prediction'][mask, goal].min())
      hi = max(
          strong['value_prediction'][mask, goal].max(),
          weak['value_prediction'][mask, goal].max())
      ax.plot([lo, hi], [lo, hi], '--', color='0.4', lw=1)
      ax.set_title(GOALS[goal].replace('_', ' '))
      ax.set_xlabel(f'{labels[0]} value')
    axes[0].set_ylabel(f'{labels[1]} value')
    fig.suptitle('Per-state critic drift between checkpoints')
    fig.savefig(figure_dir / 'critic_checkpoint_drift.pdf')
    fig.savefig(figure_dir / 'critic_checkpoint_drift.png')
    plt.close(fig)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Audit reward, continuation, value, and slow-value heads.')
  parser.add_argument('--experiment-root', type=pathlib.Path, required=True)
  parser.add_argument('--output-root', type=pathlib.Path)
  parser.add_argument('--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument('--python', type=pathlib.Path,
                      default=pathlib.Path(sys.executable))
  parser.add_argument('--strong-step', type=int, default=1_250_000)
  parser.add_argument('--weak-step', type=int, default=1_500_000)
  parser.add_argument('--data-step', type=int, default=1_250_000)
  parser.add_argument('--successes-per-goal', type=int, default=64)
  parser.add_argument('--failures-per-goal', type=int, default=64)
  parser.add_argument('--selection-seed', type=int, default=20260810)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--chunk-length', type=int, default=32)
  parser.add_argument('--poll-seconds', type=float, default=30)
  parser.add_argument('--startup-timeout', type=float, default=1800)
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
      args.experiment_root / 'head_audit')
  if not args.python.is_file():
    raise FileNotFoundError(args.python)
  if min(args.successes_per_goal, args.failures_per_goal) < 1:
    raise ValueError('success/failure episode counts must be positive')
  if args.chunk_length < 1:
    raise ValueError('--chunk-length must be positive')
  args.output_root.mkdir(parents=True, exist_ok=True)
  selection = prepare_selection(args)
  selection_path = args.output_root / 'selection.json'
  write_json(args.output_root / 'audit_spec.json', {
      'format': 1,
      'experiment_root': str(args.experiment_root),
      'strong_step': args.strong_step,
      'weak_step': args.weak_step,
      'data_step': args.data_step,
      'selection': str(selection_path),
      'selection_counts': selection['counts'],
      'same_state_bank': True,
      'heads': ['reward', 'continuation', 'value', 'slowvalue'],
      'counterfactual_goal_sweep': list(MODEL_GOALS),
  })

  jobs = [
      ('strong', args.strong_step),
      ('weak', args.weak_step),
  ]
  active_path = args.output_root / 'active_child.json'
  completed = []
  for label, step in jobs:
    archive = resolve_milestone(args.experiment_root, step)
    logdir = args.output_root / label
    command = build_command(
        python=args.python, repo_root=args.repo_root, logdir=logdir,
        checkpoint=archive / 'checkpoint', selection=selection_path,
        seed=args.seed, chunk_length=args.chunk_length)
    print(f'\n=== Head audit {label}: step={step} ===', flush=True)
    print(' '.join(command), flush=True)
    if args.dry_run:
      continue
    if (logdir / 'audit_complete').is_file():
      summary = json.loads((logdir / 'summary.json').read_text())
      if summary.get('selection_sha256') == _sha256(selection_path):
        completed.append(label)
        print(f'Using complete existing audit: {logdir}', flush=True)
        continue
      raise RuntimeError(f'Existing audit uses a different selection: {logdir}')
    logdir.mkdir(parents=True, exist_ok=True)
    child = run_managed(
        command, cwd=args.repo_root, active_path=active_path,
        console_path=logdir / 'console.log',
        activity_paths=[logdir / 'console.log', logdir / 'predictions.npz'],
        stream_output=args.stream_output, poll_seconds=args.poll_seconds,
        startup_timeout=args.startup_timeout, stall_timeout=args.stall_timeout)
    if child['returncode'] not in (0, None) or not (logdir / 'audit_complete').is_file():
      raise RuntimeError(
          f'Head audit failed for {label}: returncode={child["returncode"]}')
    completed.append(label)

  if args.dry_run:
    return 0
  make_figures(args.output_root, completed)
  summaries = {
      label: json.loads((args.output_root / label / 'summary.json').read_text())
      for label in completed}
  write_json(args.output_root / 'comparison_summary.json', {
      'format': 1,
      'labels': completed,
      'selection_sha256': _sha256(selection_path),
      'same_state_bank': True,
      'summaries': summaries,
  })
  (args.output_root / 'comparison_complete').write_bytes(b'')
  print(f'Head audit complete: {args.output_root}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
