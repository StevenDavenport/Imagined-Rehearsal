#!/usr/bin/env python3
"""Run Stage 4A counterfactual goal-head repair and matched IR evaluation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import pathlib
import platform
import shutil
import sys
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embodied.run.head_audit import _sha256  # noqa: E402
from embodied.run.head_audit import create_split_selections  # noqa: E402
from embodied.run.head_audit import load_selection  # noqa: E402
from scripts import rlscape_stage1a_head_audit as stage1a  # noqa: E402
from scripts import rlscape_stage2_actor_recovery as stage2  # noqa: E402
from scripts.rlscape_m1_pilot import spec_digest  # noqa: E402
from scripts.rlscape_m1_pilot import write_csv  # noqa: E402
from scripts.rlscape_m1_pilot import write_json  # noqa: E402
from scripts.rlscape_m3_process import active_matches  # noqa: E402
from scripts.rlscape_m3_process import run_managed  # noqa: E402
from scripts.rlscape_m3_sequential import git_revision  # noqa: E402


GOALS = stage2.GOALS
POLICY_MODES = stage2.POLICY_MODES
REPAIR_CONDITIONS = (
    'factual_only', 'counterfactual', 'counterfactual_bounded')


def installed_version(package: str) -> str:
  try:
    return importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:
    return 'not-installed'


def build_eval_conditions() -> list[dict[str, Any]]:
  conditions = [
      dict(
          name='original_frozen', family='original', checkpoint='original',
          enabled=False, objective='standard', horizon=0, actent=0.0,
          bounded=False),
      dict(
          name='original_standard_h6', family='original_standard',
          checkpoint='original', enabled=True, objective='standard',
          horizon=6, actent=3e-4, bounded=False),
      dict(
          name='original_reward_only_h15', family='original_reward_only',
          checkpoint='original', enabled=True, objective='reward_only',
          horizon=15, actent=0.0, bounded=False),
      dict(
          name='factual_reward_only_h15', family='factual_reward_only',
          checkpoint='factual_only', enabled=True, objective='reward_only',
          horizon=15, actent=0.0, bounded=False),
      dict(
          name='counterfactual_reward_only_h15',
          family='counterfactual_reward_only', checkpoint='counterfactual',
          enabled=True, objective='reward_only', horizon=15, actent=0.0,
          bounded=False),
      dict(
          name='counterfactual_bounded_reward_only_h15',
          family='counterfactual_bounded_reward_only',
          checkpoint='counterfactual_bounded', enabled=True,
          objective='reward_only', horizon=15, actent=0.0, bounded=True),
      dict(
          name='counterfactual_bounded_h6', family='bounded_ir',
          checkpoint='counterfactual_bounded', enabled=True,
          objective='bounded', horizon=6, actent=0.0, bounded=True),
      dict(
          name='counterfactual_bounded_h15', family='bounded_ir',
          checkpoint='counterfactual_bounded', enabled=True,
          objective='bounded', horizon=15, actent=0.0, bounded=True),
  ]
  names = [condition['name'] for condition in conditions]
  if len(names) != len(set(names)):
    raise AssertionError(names)
  return conditions


def build_audit_command(
    args, *, logdir: pathlib.Path, checkpoint: pathlib.Path,
    bounded: bool,
) -> list[str]:
  configs = [
      'rlscape', 'rlscape_click_grid', 'rlscape_chop_logs',
      'rlscape_m3_head_audit']
  if bounded:
    configs.append('rlscape_stage4a_bounded_value')
  return [
      str(args.python), str(args.repo_root / 'dreamerv3' / 'main.py'),
      '--configs', *configs,
      '--seed', str(args.seed),
      '--logdir', str(logdir),
      '--run.from_checkpoint', str(checkpoint),
      '--head_audit.selection', str(args.heldout_selection),
      '--head_audit.chunk_length', str(args.chunk_length),
  ]


def build_repair_command(
    args, *, condition: str, logdir: pathlib.Path,
    output_checkpoint: pathlib.Path,
) -> list[str]:
  return [
      str(args.python), str(args.repo_root / 'dreamerv3' / 'main.py'),
      '--configs', 'rlscape', 'rlscape_click_grid', 'rlscape_chop_logs',
      'rlscape_stage4a_head_repair',
      '--seed', str(args.seed),
      '--logdir', str(logdir),
      '--run.from_checkpoint', str(args.reservoir_checkpoint),
      '--head_repair.selection', str(args.train_selection),
      '--head_repair.cache', str(args.latent_cache),
      '--head_repair.output_checkpoint', str(output_checkpoint),
      '--head_repair.condition', condition,
      '--head_repair.chunk_length', str(args.chunk_length),
      '--head_repair.states_per_episode', str(args.states_per_episode),
      '--head_repair.head_updates', str(args.head_updates),
      '--head_repair.value_updates', str(args.value_updates),
      '--head_repair.batch_size', str(args.repair_batch_size),
      '--head_repair.event_fraction', str(args.event_fraction),
      '--head_repair.learning_rate', str(args.repair_lr),
      '--head_repair.seed', str(args.repair_seed),
  ]


def _heartbeat(args, snapshot):
  args.state['active_child'] = snapshot
  args.state['updated_unix'] = time.time()
  write_json(args.state_path, args.state)


def run_child(
    args, *, unit_id: str, command_factory, complete, activity_paths,
) -> dict:
  """Run or adopt an idempotent non-evaluation unit with bounded retries."""
  state = args.state['units'].setdefault(
      unit_id, {'status': 'pending', 'attempts': []})
  latest_logdir = (
      pathlib.Path(state['attempts'][-1]['logdir'])
      if state['attempts'] else None)
  if latest_logdir is not None and complete(latest_logdir):
    state['status'] = 'complete'
    write_json(args.state_path, args.state)
    return state
  if state.get('status') == 'complete':
    state['status'] = 'pending'

  attempt = state['attempts'][-1] if state['attempts'] else None
  if attempt and attempt.get('status') == 'running':
    command = command_factory(pathlib.Path(attempt['logdir']))
    if complete(pathlib.Path(attempt['logdir'])):
      attempt.update(status='complete', finished_unix=time.time())
      state['status'] = 'complete'
      return state
    if not active_matches(args.active_path, command, args.repo_root):
      attempt.update(status='interrupted', finished_unix=time.time())
      attempt = None

  if attempt is None or attempt.get('status') != 'running':
    attempt_number = len(state['attempts']) + 1
    if attempt_number > args.max_attempts:
      raise RuntimeError(f'Exhausted attempts for {unit_id}')
    logdir = args.output_root / 'attempts' / unit_id / f'{attempt_number:02d}'
    logdir.mkdir(parents=True, exist_ok=False)
    command = command_factory(logdir)
    attempt = {
        'attempt': attempt_number, 'status': 'running',
        'started_unix': time.time(), 'logdir': str(logdir),
        'command': command}
    state['attempts'].append(attempt)
  else:
    attempt_number = int(attempt['attempt'])
    logdir = pathlib.Path(attempt['logdir'])
    command = command_factory(logdir)

  state['status'] = 'running'
  write_json(args.state_path, args.state)
  print(f'\n=== Stage 4A {unit_id}, attempt {attempt_number} ===', flush=True)
  print(' '.join(command), flush=True)
  child = run_managed(
      command, cwd=args.repo_root, active_path=args.active_path,
      console_path=logdir / 'console.log',
      activity_paths=activity_paths(logdir),
      stream_output=args.stream_output, poll_seconds=args.poll_seconds,
      startup_timeout=args.startup_timeout, stall_timeout=args.stall_timeout,
      heartbeat=lambda value: _heartbeat(args, value))
  args.state.pop('active_child', None)
  succeeded = child['returncode'] in (0, None) and complete(logdir)
  attempt.update(
      status='complete' if succeeded else 'failed',
      finished_unix=time.time(), child=child)
  state['status'] = attempt['status']
  write_json(args.state_path, args.state)
  if not succeeded:
    if attempt_number < args.max_attempts:
      return run_child(
          args, unit_id=unit_id, command_factory=command_factory,
          complete=complete, activity_paths=activity_paths)
    raise RuntimeError(
        f'Incomplete Stage 4A unit {unit_id}: {logdir / "console.log"}')
  return state


def execute_eval(
    args, *, condition: dict, goal: str, policy_mode: str, episodes: int,
) -> dict:
  unit_id = f'eval/{condition["name"]}/{goal}/{policy_mode}'
  state = args.state['units'].setdefault(
      unit_id, {'status': 'pending', 'attempts': []})
  checkpoint = args.checkpoints[condition['checkpoint']]

  def summarize(logdir, returncode):
    result = stage2.summarize_condition(
        logdir, condition=condition, goal=goal, policy_mode=policy_mode,
        episodes=episodes, seed=args.eval_seed, returncode=returncode)
    result.update(
        canonical_stage='stage4a', source_checkpoint=str(checkpoint),
        repaired_checkpoint=condition['checkpoint'],
        bounded_value=bool(condition['bounded']))
    return result

  if state.get('status') == 'complete':
    saved = state['attempts'][-1]
    recovered = summarize(pathlib.Path(saved['logdir']), 0)
    if recovered['complete']:
      saved['summary'] = recovered
      return recovered
    state['status'] = 'pending'

  attempt = state['attempts'][-1] if state['attempts'] else None
  if attempt and attempt.get('status') == 'running':
    attempt_number = int(attempt['attempt'])
    logdir = pathlib.Path(attempt['logdir'])
  else:
    attempt_number = len(state['attempts']) + 1
    if attempt_number > args.max_attempts:
      raise RuntimeError(f'Exhausted attempts for {unit_id}')
    logdir = args.output_root / unit_id / f'attempt_{attempt_number:02d}'
    logdir.mkdir(parents=True, exist_ok=False)
    attempt = {
        'attempt': attempt_number, 'status': 'running',
        'started_unix': time.time(), 'logdir': str(logdir)}
    state['attempts'].append(attempt)

  extra_configs = (
      ('rlscape_stage4a_bounded_value',) if condition['bounded'] else ())
  command = stage2.build_command(
      python=args.python, repo_root=args.repo_root, logdir=logdir,
      checkpoint=checkpoint, reference_checkpoint=checkpoint,
      condition=condition, goal=goal, policy_mode=policy_mode,
      episodes=episodes, seed=args.eval_seed,
      episode_length=args.episode_length, grid_columns=args.grid_columns,
      grid_rows=args.grid_rows, adapt_steps=args.adapt_steps,
      adapt_lr=args.adapt_lr, adapt_every_k=args.adapt_every_k,
      start_batch=args.start_batch, mvn_path=args.mvn_path,
      java_home=args.java_home, extra_configs=extra_configs)
  attempt['command'] = command
  if attempt.get('status') == 'running':
    recovered = summarize(logdir, 0)
    if recovered['complete']:
      attempt.update(
          status='complete', finished_unix=time.time(), summary=recovered)
      state['status'] = 'complete'
      return recovered
    if len(state['attempts']) > 1 or active_matches(
        args.active_path, command, args.repo_root):
      pass
    elif attempt.get('started_child'):
      attempt.update(status='interrupted', finished_unix=time.time())
      return execute_eval(
          args, condition=condition, goal=goal, policy_mode=policy_mode,
          episodes=episodes)

  attempt['started_child'] = True
  state['status'] = 'running'
  write_json(args.state_path, args.state)
  print(f'\n=== Stage 4A {unit_id}, attempt {attempt_number} ===', flush=True)
  print(' '.join(command), flush=True)
  child = run_managed(
      command, cwd=args.repo_root, active_path=args.active_path,
      console_path=logdir / 'console.log',
      activity_paths=(
          logdir / 'metrics.jsonl', logdir / 'episodes.jsonl',
          logdir / 'audit_env0.jsonl'),
      stream_output=args.stream_output, poll_seconds=args.poll_seconds,
      startup_timeout=args.startup_timeout, stall_timeout=args.stall_timeout,
      heartbeat=lambda value: _heartbeat(args, value))
  args.state.pop('active_child', None)
  summary = summarize(logdir, child['returncode'])
  attempt.update(
      status='complete' if summary['complete'] else 'failed',
      finished_unix=time.time(), child=child, summary=summary)
  state['status'] = attempt['status']
  write_json(args.state_path, args.state)
  if not summary['complete']:
    if attempt_number < args.max_attempts:
      return execute_eval(
          args, condition=condition, goal=goal, policy_mode=policy_mode,
          episodes=episodes)
    raise RuntimeError(f'Incomplete Stage 4A evaluation: {unit_id}')
  return summary


def paired_effect(baseline: list[int], treatment: list[int], seed: int) -> dict:
  baseline = np.asarray(baseline, np.float64)
  treatment = np.asarray(treatment, np.float64)
  if baseline.shape != treatment.shape or not len(baseline):
    raise ValueError((baseline.shape, treatment.shape))
  difference = treatment - baseline
  rng = np.random.default_rng(seed)
  indices = rng.integers(0, len(difference), size=(10_000, len(difference)))
  boot = difference[indices].mean(1)
  return {
      'paired_delta_vs_original_frozen': float(difference.mean()),
      'paired_delta_ci_low': float(np.quantile(boot, .025)),
      'paired_delta_ci_high': float(np.quantile(boot, .975)),
      'discordant_improved': int(((treatment == 1) & (baseline == 0)).sum()),
      'discordant_harmed': int(((treatment == 0) & (baseline == 1)).sum()),
  }


def add_paired_effects(rows: list[dict]) -> None:
  def successes(logdir: pathlib.Path) -> list[int]:
    episode_rows = stage2.read_jsonl(logdir / 'episodes.jsonl')
    return [int(
        float(row.get('episode/score', 0.0)) > 0 or
        float(row.get('episode/log/task_success/max', 0.0)) > 0)
            for row in episode_rows]

  for mode in POLICY_MODES:
    for goal in GOALS:
      group = [row for row in rows
               if row['policy_mode'] == mode and row['goal'] == goal]
      baseline = next(
          row for row in group if row['condition'] == 'original_frozen')
      baseline_success = successes(pathlib.Path(baseline['logdir']))
      baseline_resets = stage2.reset_identities(pathlib.Path(baseline['logdir']))
      for row in group:
        resets = stage2.reset_identities(pathlib.Path(row['logdir']))
        if resets != baseline_resets:
          raise RuntimeError(
              f'Unpaired resets for {goal}/{mode}/{row["condition"]}')
        row.update(paired_effect(
            baseline_success, successes(pathlib.Path(row['logdir'])),
            args_seed(row['condition'], goal, mode)))


def args_seed(*parts: str) -> int:
  return int(spec_digest({'parts': parts})[:8], 16)


def make_figures(output_root: pathlib.Path, rows: list[dict]) -> None:
  import matplotlib.pyplot as plt

  figure_root = output_root / 'figures'
  figure_root.mkdir(exist_ok=True)
  plt.rcParams.update({
      'font.size': 9, 'axes.spines.top': False,
      'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .2})
  conditions = build_eval_conditions()
  names = [condition['name'] for condition in conditions]
  labels = {
      'original_frozen': 'Original\nfrozen',
      'original_standard_h6': 'Original\nstandard H6',
      'original_reward_only_h15': 'Original\nreward H15',
      'factual_reward_only_h15': 'Factual\nreward H15',
      'counterfactual_reward_only_h15': 'CF\nreward H15',
      'counterfactual_bounded_reward_only_h15': 'CF+BVal\nreward H15',
      'counterfactual_bounded_h6': 'CF+BVal\nbounded H6',
      'counterfactual_bounded_h15': 'CF+BVal\nbounded H15',
  }
  for mode in POLICY_MODES:
    fig, axes = plt.subplots(
        1, len(GOALS), figsize=(14, 4.2), sharey=True,
        constrained_layout=True)
    for axis, goal in zip(axes, GOALS):
      selected = {
          row['condition']: row for row in rows
          if row['policy_mode'] == mode and row['goal'] == goal}
      values = [float(selected[name]['success_rate']) for name in names]
      colors = [
          '#777777', '#4c78a8', '#72b7b2', '#f2cf5b',
          '#e45756', '#b279a2', '#59a14f', '#2f7d32']
      axis.bar(range(len(names)), values, color=colors)
      axis.set_ylim(0, 1)
      axis.set_title(goal.replace('_', ' '))
      axis.set_xticks(
          range(len(names)), [labels[name] for name in names],
          rotation=32, ha='right', fontsize=7)
    axes[0].set_ylabel('Success rate')
    fig.suptitle(f'Stage 4A IR outcomes — {mode} policy')
    for suffix in ('png', 'pdf'):
      fig.savefig(figure_root / f'ir_success_{mode}.{suffix}', dpi=220)
    plt.close(fig)

  fig, axes = plt.subplots(
      1, len(GOALS), figsize=(13, 3.8), sharey=True,
      constrained_layout=True)
  adapted = names[1:]
  for axis, goal in zip(axes, GOALS):
    selected = {
        row['condition']: row for row in rows
        if row['policy_mode'] == 'sampled' and row['goal'] == goal}
    values = [
        float(selected[name]['paired_delta_vs_original_frozen'])
        for name in adapted]
    lows = [float(selected[name]['paired_delta_ci_low']) for name in adapted]
    highs = [float(selected[name]['paired_delta_ci_high']) for name in adapted]
    axis.bar(range(len(adapted)), values, color='#4c78a8')
    axis.errorbar(
        range(len(adapted)), values,
        yerr=[np.asarray(values) - lows, np.asarray(highs) - values],
        fmt='none', ecolor='black', capsize=2, lw=.8)
    axis.axhline(0, color='black', lw=1)
    axis.set_title(goal.replace('_', ' '))
    axis.set_xticks(
        range(len(adapted)), [labels[name] for name in adapted],
        rotation=32, ha='right', fontsize=7)
  axes[0].set_ylabel('Paired success change vs original frozen')
  fig.suptitle('Stage 4A sampled-policy paired treatment effects')
  for suffix in ('png', 'pdf'):
    fig.savefig(figure_root / f'ir_paired_effects_sampled.{suffix}', dpi=220)
  plt.close(fig)


def make_head_repair_figures(
    audit_root: pathlib.Path, labels: list[str],
) -> None:
  """Plot the Stage 4A-specific selectivity and bounded-value endpoints."""
  import matplotlib.pyplot as plt

  datasets = {}
  summaries = {}
  for label in labels:
    with np.load(audit_root / label / 'predictions.npz') as archive:
      datasets[label] = {
          key: np.array(archive[key], copy=True) for key in archive.files}
    summaries[label] = json.loads(
        (audit_root / label / 'summary.json').read_text())
  figure_root = audit_root / 'figures'
  figure_root.mkdir(exist_ok=True)
  display = {
      'fifo_original': 'FIFO', 'reservoir_original': 'Reservoir',
      'factual_only': 'Factual', 'counterfactual': 'Counterfactual',
      'counterfactual_bounded': 'Counterfactual + bounded value'}

  top1 = [
      float(summaries[label]['completion_selectivity'][
          'reward_top1_accuracy']) for label in labels]
  margin = [
      float(summaries[label]['completion_selectivity']['reward_margin'])
      for label in labels]
  fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8),
                           constrained_layout=True)
  colors = ['#9d755d', '#777777', '#f2cf5b', '#e45756', '#59a14f']
  axes[0].bar(range(len(labels)), top1, color=colors)
  axes[0].axhline(1 / 5, color='black', ls='--', lw=1, label='chance')
  axes[0].set_ylim(0, 1)
  axes[0].set_ylabel('Completion-state top-1 goal accuracy')
  axes[0].legend(frameon=False)
  axes[1].bar(range(len(labels)), margin, color=colors)
  axes[1].axhline(0, color='black', lw=1)
  axes[1].set_ylabel('Matching minus nonmatching reward')
  for axis in axes:
    axis.set_xticks(
        range(len(labels)), [display[label] for label in labels],
        rotation=25, ha='right')
  fig.suptitle('Held-out counterfactual goal selectivity')
  for suffix in ('png', 'pdf'):
    fig.savefig(figure_root / f'counterfactual_selectivity.{suffix}', dpi=220)
  plt.close(fig)

  bounded_labels = [
      label for label in labels
      if 'bounded_value_prediction' in datasets[label]]
  if bounded_labels:
    fig, axes = plt.subplots(
        1, len(GOALS), figsize=(10.5, 3.4), sharex=True, sharey=True,
        squeeze=False, constrained_layout=True)
    data = datasets[bounded_labels[-1]]
    for goal, axis in enumerate(axes[0]):
      mask = data['actual_goal'].astype(int) == goal
      prediction = data['bounded_value_prediction'][mask, goal]
      target = np.clip(data['return_inclusive'][mask], 0, 1)
      edges = np.linspace(0, 1, 9)
      xs, ys, counts = [], [], []
      for low, high in zip(edges[:-1], edges[1:]):
        inside = (prediction >= low) & (prediction <= high)
        if low:
          inside &= prediction > low
        if inside.any():
          xs.append(float(prediction[inside].mean()))
          ys.append(float(target[inside].mean()))
          counts.append(int(inside.sum()))
      axis.plot(xs, ys, 'o-', color='#59a14f')
      axis.plot([0, 1], [0, 1], '--', color='0.45', lw=1)
      for x, y, count in zip(xs, ys, counts):
        axis.annotate(str(count), (x, y), xytext=(3, 3),
                      textcoords='offset points', fontsize=7)
      axis.set_title(goal.replace('_', ' '))
      axis.set_xlabel('Predicted bounded value')
    axes[0, 0].set_ylabel('Empirical discounted completion return')
    fig.suptitle('Held-out bounded success-value calibration')
    for suffix in ('png', 'pdf'):
      fig.savefig(figure_root / f'bounded_value_calibration.{suffix}', dpi=220)
    plt.close(fig)


def prepare_selections(args) -> None:
  args.train_selection = args.output_root / 'banks' / 'repair_train.json'
  args.heldout_selection = args.output_root / 'banks' / 'heldout_audit.json'
  if args.train_selection.exists() or args.heldout_selection.exists():
    if not (args.train_selection.exists() and args.heldout_selection.exists()):
      raise RuntimeError('Incomplete existing Stage 4A bank split')
    train = load_selection(args.train_selection)
    heldout = load_selection(args.heldout_selection)
    expected_replay = str(args.reservoir_replay.resolve())
    if train['replay_dir'] != expected_replay or heldout['replay_dir'] != expected_replay:
      raise RuntimeError('Existing Stage 4A bank uses a different replay')
  else:
    train, heldout = create_split_selections(
        args.reservoir_replay, args.train_selection, args.heldout_selection,
        goals=tuple(range(len(GOALS))),
        train_successes_per_goal=args.train_successes_per_goal,
        train_failures_per_goal=args.train_failures_per_goal,
        heldout_successes_per_goal=args.heldout_successes_per_goal,
        heldout_failures_per_goal=args.heldout_failures_per_goal,
        seed=args.selection_seed)
  train_ids = {row['episode_id'] for row in train['episodes']}
  heldout_ids = {row['episode_id'] for row in heldout['episodes']}
  if train_ids & heldout_ids:
    raise RuntimeError('Existing Stage 4A train and held-out banks overlap')


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Run Stage 4A counterfactual goal-head repair.')
  parser.add_argument('--fifo-root', type=pathlib.Path, required=True)
  parser.add_argument('--reservoir-root', type=pathlib.Path, required=True)
  parser.add_argument('--output-root', type=pathlib.Path)
  parser.add_argument('--checkpoint-step', type=int, default=1_250_000)
  parser.add_argument('--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument('--python', type=pathlib.Path,
                      default=pathlib.Path(sys.executable))
  parser.add_argument('--train-successes-per-goal', type=int, default=96)
  parser.add_argument('--train-failures-per-goal', type=int, default=96)
  parser.add_argument('--heldout-successes-per-goal', type=int, default=64)
  parser.add_argument('--heldout-failures-per-goal', type=int, default=64)
  parser.add_argument('--selection-seed', type=int, default=20260816)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--chunk-length', type=int, default=32)
  parser.add_argument('--states-per-episode', type=int, default=24)
  parser.add_argument('--head-updates', type=int, default=2000)
  parser.add_argument('--value-updates', type=int, default=2000)
  parser.add_argument('--repair-batch-size', type=int, default=256)
  parser.add_argument('--event-fraction', type=float, default=.5)
  parser.add_argument('--repair-lr', type=float, default=4e-5)
  parser.add_argument('--repair-seed', type=int, default=20260816)
  parser.add_argument('--sampled-episodes', type=int, default=100)
  parser.add_argument('--deterministic-episodes', type=int, default=50)
  parser.add_argument('--eval-seed', type=int, default=3000)
  parser.add_argument('--episode-length', type=int, default=200)
  parser.add_argument('--grid-columns', type=int, default=28)
  parser.add_argument('--grid-rows', type=int, default=18)
  parser.add_argument('--adapt-steps', type=int, default=1)
  parser.add_argument('--adapt-lr', type=float, default=4e-5)
  parser.add_argument('--adapt-every-k', type=int, default=1)
  parser.add_argument('--start-batch', type=int, default=128)
  parser.add_argument('--max-attempts', type=int, default=3)
  parser.add_argument('--poll-seconds', type=float, default=30)
  parser.add_argument('--startup-timeout', type=float, default=3600)
  parser.add_argument('--stall-timeout', type=float, default=14400)
  parser.add_argument('--mvn-path', default=os.environ.get(
      'RL_SCAPE_MVN', shutil.which('mvn') or ''))
  parser.add_argument('--java-home', default=os.environ.get(
      'RL_SCAPE_JAVA_HOME', os.environ.get('JAVA_HOME', '')))
  parser.add_argument('--stream-output', action='store_true')
  parser.add_argument('--dry-run', action='store_true')
  return parser.parse_args(argv)


def validate_args(args) -> None:
  args.repo_root = args.repo_root.expanduser().resolve()
  args.fifo_root = args.fifo_root.expanduser().resolve()
  args.reservoir_root = args.reservoir_root.expanduser().resolve()
  args.python = args.python.expanduser().resolve()
  args.output_root = (
      args.output_root.expanduser().resolve() if args.output_root else
      args.reservoir_root / 'stage4a_counterfactual_head_repair')
  if not args.python.is_file():
    raise FileNotFoundError(args.python)
  positive = (
      args.checkpoint_step, args.train_successes_per_goal,
      args.train_failures_per_goal, args.heldout_successes_per_goal,
      args.heldout_failures_per_goal, args.chunk_length,
      args.states_per_episode, args.head_updates, args.value_updates,
      args.repair_batch_size, args.sampled_episodes,
      args.deterministic_episodes, args.episode_length, args.grid_columns,
      args.grid_rows, args.adapt_steps, args.start_batch, args.max_attempts)
  if any(value < 1 for value in positive):
    raise ValueError('Stage 4A count arguments must be positive')
  if not 0 < args.event_fraction < 1:
    raise ValueError('--event-fraction must lie strictly between 0 and 1')
  if args.repair_lr <= 0 or args.adapt_lr <= 0 or args.adapt_every_k < 0:
    raise ValueError('Invalid repair or adaptation schedule')


def main(argv=None) -> int:
  args = parse_args(argv)
  validate_args(args)
  fifo_archive = stage2.resolve_milestone(args.fifo_root, args.checkpoint_step)
  reservoir_archive = stage2.resolve_milestone(
      args.reservoir_root, args.checkpoint_step)
  args.fifo_checkpoint = fifo_archive / 'checkpoint'
  args.reservoir_checkpoint = reservoir_archive / 'checkpoint'
  args.reservoir_replay = reservoir_archive / 'replay'
  args.output_root.mkdir(parents=True, exist_ok=True)
  args.latent_cache = args.output_root / 'banks' / 'repair_latents.npz'
  prepare_selections(args)

  args.checkpoints = {'original': args.reservoir_checkpoint}
  for condition in REPAIR_CONDITIONS:
    args.checkpoints[condition] = (
        args.output_root / 'checkpoints' / condition)
  conditions = build_eval_conditions()
  spec = {
      'format': 1, 'canonical_stage': 'stage4a',
      'git_commit': git_revision(args.repo_root),
      'python_version': platform.python_version(),
      'rlscape_version': installed_version('rl-scape'),
      'fifo_root': str(args.fifo_root),
      'reservoir_root': str(args.reservoir_root),
      'checkpoint_step': args.checkpoint_step,
      'fifo_checkpoint': str(args.fifo_checkpoint),
      'reservoir_checkpoint': str(args.reservoir_checkpoint),
      'train_selection': str(args.train_selection),
      'train_selection_sha256': _sha256(args.train_selection),
      'heldout_selection': str(args.heldout_selection),
      'heldout_selection_sha256': _sha256(args.heldout_selection),
      'repair': {
          'states_per_episode': args.states_per_episode,
          'head_updates': args.head_updates, 'value_updates': args.value_updates,
          'batch_size': args.repair_batch_size,
          'event_fraction': args.event_fraction, 'lr': args.repair_lr,
          'seed': args.repair_seed},
      'evaluation': {
          'sampled_episodes': args.sampled_episodes,
          'deterministic_episodes': args.deterministic_episodes,
          'seed': args.eval_seed, 'conditions': conditions},
  }
  spec['digest'] = spec_digest(spec)
  spec_path = args.output_root / 'experiment_spec.json'
  if spec_path.is_file():
    existing = json.loads(spec_path.read_text())
    if existing.get('digest') != spec['digest']:
      raise RuntimeError('Existing Stage 4A root has a different specification')
  else:
    write_json(spec_path, spec)
  write_json(
      args.output_root / 'experiment_spec_digest.json',
      {'sha256': spec['digest']})
  if args.dry_run:
    print(json.dumps(spec, indent=2))
    return 0

  args.state_path = args.output_root / 'queue.json'
  args.active_path = args.output_root / 'active_child.json'
  args.state = (
      json.loads(args.state_path.read_text()) if args.state_path.is_file() else
      {'format': 1, 'canonical_stage': 'stage4a',
       'spec_digest': spec['digest'], 'status': 'running', 'units': {}})
  if args.state.get('spec_digest') != spec['digest']:
    raise RuntimeError('Stage 4A status belongs to a different specification')
  args.state.update(status='running', updated_unix=time.time())
  write_json(args.state_path, args.state)

  audits = [
      ('fifo_original', args.fifo_checkpoint, False),
      ('reservoir_original', args.reservoir_checkpoint, False),
  ]
  for label, checkpoint, bounded in audits:
    run_child(
        args, unit_id=f'audit/{label}',
        command_factory=lambda logdir, c=checkpoint, b=bounded: (
            build_audit_command(args, logdir=logdir, checkpoint=c, bounded=b)),
        complete=lambda logdir: (logdir / 'audit_complete').is_file(),
        activity_paths=lambda logdir: (
            logdir / 'console.log', logdir / 'predictions.npz'))

  for condition in REPAIR_CONDITIONS:
    target = args.checkpoints[condition]
    run_child(
        args, unit_id=f'repair/{condition}',
        command_factory=lambda logdir, c=condition, t=target: (
            build_repair_command(
                args, condition=c, logdir=logdir, output_checkpoint=t)),
        complete=lambda logdir, target=target: (
            (target / 'done').is_file() and
            (target / 'repair_manifest.json').is_file() and
            (logdir / 'repair_complete').is_file()),
        activity_paths=lambda logdir: (
            logdir / 'repair_metrics.jsonl', logdir / 'repair_complete'))

  repair_manifests = {
      condition: json.loads(
          (args.checkpoints[condition] / 'repair_manifest.json').read_text())
      for condition in REPAIR_CONDITIONS}
  counterfactual_equal = (
      repair_manifests['counterfactual']['integrity_after'][
          'reward_continuation'] ==
      repair_manifests['counterfactual_bounded']['integrity_after'][
          'reward_continuation'])
  if not counterfactual_equal:
    raise RuntimeError(
        'Matched counterfactual arms produced different repaired heads')
  source_actor_equal = len({
      value['integrity_before']['actor']['sha256']
      for value in repair_manifests.values()}) == 1
  exported_actor_equal = len({
      value['integrity_after']['actor']['sha256']
      for value in repair_manifests.values()}) == 1
  if not (source_actor_equal and exported_actor_equal):
    raise RuntimeError('Stage 4A repair arms do not share an identical actor')
  write_json(args.output_root / 'repair_comparison.json', {
      'format': 1,
      'source_actor_equal': source_actor_equal,
      'exported_actor_equal': exported_actor_equal,
      'counterfactual_reward_continuation_equal': counterfactual_equal,
      'manifests': {
          condition: str(
              args.checkpoints[condition] / 'repair_manifest.json')
          for condition in REPAIR_CONDITIONS},
  })

  for condition in REPAIR_CONDITIONS:
    label = condition
    checkpoint = args.checkpoints[condition]
    bounded = condition == 'counterfactual_bounded'
    run_child(
        args, unit_id=f'audit/{label}',
        command_factory=lambda logdir, c=checkpoint, b=bounded: (
            build_audit_command(args, logdir=logdir, checkpoint=c, bounded=b)),
        complete=lambda logdir: (logdir / 'audit_complete').is_file(),
        activity_paths=lambda logdir: (
            logdir / 'console.log', logdir / 'predictions.npz'))

  rows = []
  for condition in conditions:
    for goal in GOALS:
      for mode in POLICY_MODES:
        episodes = (
            args.sampled_episodes if mode == 'sampled'
            else args.deterministic_episodes)
        rows.append(execute_eval(
            args, condition=condition, goal=goal, policy_mode=mode,
            episodes=episodes))
  add_paired_effects(rows)
  write_json(args.output_root / 'pairing.json', {
      'format': 1, 'equal': True,
      'keys': ['goal', 'policy_mode', 'reset_seed'],
      'conditions': [condition['name'] for condition in conditions],
      'note': 'All reset identity lists matched original_frozen.'})
  write_json(args.output_root / 'main_results.json', {
      'format': 1, 'canonical_stage': 'stage4a',
      'spec_digest': spec['digest'], 'rows': rows})
  write_csv(args.output_root / 'main_results.csv', rows)

  # Copy canonical audit outputs out of attempt directories for simple figure
  # generation and later archival.
  audit_labels = [label for label, _, _ in audits] + list(REPAIR_CONDITIONS)
  canonical_audits = args.output_root / 'head_audits'
  canonical_audits.mkdir(parents=True, exist_ok=True)
  for label in audit_labels:
    unit = args.state['units'][f'audit/{label}']
    source = pathlib.Path(unit['attempts'][-1]['logdir'])
    destination = canonical_audits / label
    if not destination.exists():
      destination.symlink_to(source, target_is_directory=True)
  stage1a.make_figures(canonical_audits, audit_labels)
  make_head_repair_figures(canonical_audits, audit_labels)
  make_figures(args.output_root, rows)
  write_json(args.output_root / 'comparison_summary.json', {
      'format': 1, 'canonical_stage': 'stage4a',
      'spec_digest': spec['digest'],
      'audits': {
          label: json.loads(
              (canonical_audits / label / 'summary.json').read_text())
          for label in audit_labels},
      'evaluation_rows': rows,
  })
  args.state.update(status='complete', updated_unix=time.time())
  write_json(args.state_path, args.state)
  (args.output_root / 'stage4a_complete').write_bytes(b'')
  print(f'Stage 4A complete: {args.output_root}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
