#!/usr/bin/env python3
"""Run RLScape Stage 3B natural-forgetting recovery evaluations."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
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

from scripts.rlscape_m1_pilot import spec_digest  # noqa: E402
from scripts.rlscape_m1_pilot import write_csv  # noqa: E402
from scripts.rlscape_m1_pilot import write_json  # noqa: E402
from scripts.rlscape_m3_process import active_matches  # noqa: E402
from scripts.rlscape_m3_process import run_managed  # noqa: E402
from scripts.rlscape_m3_sequential import git_revision  # noqa: E402
from scripts import rlscape_stage2_actor_recovery as stage2  # noqa: E402


GOALS = stage2.GOALS
POLICY_MODES = stage2.POLICY_MODES
DEFAULT_CHECKPOINTS = (1_250_000, 1_500_000)
DEFAULT_HORIZONS = (6, 15, 20)
HISTORICAL_SAMPLED_SUCCESS = {
    1_250_000: {
        'kill_goblin': .56, 'bury_bones': .49, 'chop_logs': .88},
    1_500_000: {
        'kill_goblin': .16, 'bury_bones': .08, 'chop_logs': .05},
}


def parse_ints(text: str) -> tuple[int, ...]:
  values = tuple(int(item.strip()) for item in text.split(',') if item.strip())
  if not values or any(value < 1 for value in values):
    raise ValueError(f'Expected positive comma-separated integers: {text!r}')
  if len(values) != len(set(values)):
    raise ValueError(f'Duplicate values are not allowed: {text!r}')
  return values


def checkpoint_label(step: int) -> str:
  return f'step_{int(step):010d}'


def build_conditions(horizons=DEFAULT_HORIZONS) -> list[dict[str, Any]]:
  conditions = [dict(
      name='frozen', family='frozen', checkpoint='natural', enabled=False,
      objective='standard', horizon=0, actent=0.0)]
  conditions.append(dict(
      name='standard_ir_h6', family='standard_ir', checkpoint='natural',
      enabled=True, objective='standard', horizon=6, actent=3e-4))
  for horizon in horizons:
    conditions.append(dict(
        name=f'reward_only_ir_h{int(horizon)}', family='reward_only_ir',
        checkpoint='natural', enabled=True, objective='reward_only',
        horizon=int(horizon), actent=0.0))
  names = [item['name'] for item in conditions]
  if len(names) != len(set(names)):
    raise AssertionError(names)
  return conditions


def historical_classification(
    step: int, goal: str, threshold: float,
) -> tuple[float, str]:
  if step not in HISTORICAL_SAMPLED_SUCCESS:
    raise ValueError(
        f'No prespecified Stage 0 classification for checkpoint {step}')
  score = float(HISTORICAL_SAMPLED_SUCCESS[step][goal])
  return score, ('weak' if score < threshold else 'competent')


def paired_effect(
    baseline: np.ndarray, treatment: np.ndarray, *, seed: int,
    samples: int = 10_000,
) -> dict[str, Any]:
  baseline = np.asarray(baseline, np.float64)
  treatment = np.asarray(treatment, np.float64)
  if not len(baseline) or baseline.shape != treatment.shape:
    raise ValueError((baseline.shape, treatment.shape))
  difference = treatment - baseline
  rng = np.random.default_rng(seed)
  indices = rng.integers(0, len(difference), size=(samples, len(difference)))
  boot = difference[indices].mean(1)
  return {
      'paired_delta_vs_frozen': float(difference.mean()),
      'paired_delta_ci_low': float(np.quantile(boot, .025)),
      'paired_delta_ci_high': float(np.quantile(boot, .975)),
      'paired_discordant_improved': int(
          ((treatment == 1) & (baseline == 0)).sum()),
      'paired_discordant_harmed': int(
          ((treatment == 0) & (baseline == 1)).sum()),
  }


def validate_pairing(rows: list[dict[str, Any]]) -> dict[str, Any]:
  groups: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
  for row in rows:
    key = (int(row['checkpoint_step']), row['goal'], row['policy_mode'])
    groups.setdefault(key, []).append(row)
  mismatches = []
  for key, group in groups.items():
    reference = stage2.reset_identities(pathlib.Path(group[0]['logdir']))
    for row in group[1:]:
      current = stage2.reset_identities(pathlib.Path(row['logdir']))
      if current != reference:
        mismatches.append({
            'checkpoint_step': key[0], 'goal': key[1],
            'policy_mode': key[2], 'condition': row['condition'],
            'reference_condition': group[0]['condition'],
            'reference_count': len(reference), 'current_count': len(current),
        })
  return {'equal': not mismatches, 'mismatches': mismatches}


def comparison_rows(
    rows: list[dict[str, Any]], *, classification_threshold: float,
) -> list[dict[str, Any]]:
  lookup = {
      (int(row['checkpoint_step']), row['goal'], row['policy_mode'],
       row['condition']): row for row in rows}
  output = []
  for row in rows:
    step = int(row['checkpoint_step'])
    baseline = lookup[(step, row['goal'], row['policy_mode'], 'frozen')]
    baseline_outcomes = stage2.episode_outcomes(
        pathlib.Path(baseline['logdir']))
    outcomes = stage2.episode_outcomes(pathlib.Path(row['logdir']))
    seed = int(spec_digest({
        'stage': '3b', 'checkpoint': step, 'goal': row['goal'],
        'mode': row['policy_mode'], 'condition': row['condition']})[:8], 16)
    historical, status = historical_classification(
        step, row['goal'], classification_threshold)
    output.append({
        **row,
        'historical_sampled_success': historical,
        'prespecified_status': status,
        'classification_threshold': classification_threshold,
        'frozen_success_rate': float(baseline['success_rate']),
        'delta_vs_frozen': (
            float(row['success_rate']) - float(baseline['success_rate'])),
        **paired_effect(baseline_outcomes, outcomes, seed=seed),
    })
  return output


def summarize_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  output = []
  for step in sorted({int(row['checkpoint_step']) for row in rows}):
    for mode in POLICY_MODES:
      for condition in sorted({row['condition'] for row in rows}):
        selected = [
            row for row in rows
            if int(row['checkpoint_step']) == step and
            row['policy_mode'] == mode and row['condition'] == condition]
        if not selected:
          continue
        for status in ('all', 'weak', 'competent'):
          subset = selected if status == 'all' else [
              row for row in selected if row['prespecified_status'] == status]
          if not subset:
            continue
          output.append({
              'checkpoint_step': step,
              'checkpoint_label': checkpoint_label(step),
              'policy_mode': mode,
              'condition': condition,
              'family': subset[0]['family'],
              'horizon': int(subset[0]['horizon']),
              'prespecified_status': status,
              'goals': len(subset),
              'mean_success_rate': float(np.mean([
                  float(row['success_rate']) for row in subset])),
              'mean_delta_vs_frozen': float(np.mean([
                  float(row['delta_vs_frozen']) for row in subset])),
          })
  return output


def make_figures(output_root: pathlib.Path, rows: list[dict[str, Any]]) -> None:
  import matplotlib.pyplot as plt

  figure_root = output_root / 'figures'
  figure_root.mkdir(exist_ok=True)
  plt.rcParams.update({
      'font.size': 9, 'axes.spines.top': False,
      'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .2})
  steps = sorted({int(row['checkpoint_step']) for row in rows})
  conditions = [item['name'] for item in build_conditions(tuple(sorted({
      int(row['horizon']) for row in rows
      if row['family'] == 'reward_only_ir'})))]
  labels = {
      'frozen': 'Frozen', 'standard_ir_h6': 'Standard IR H=6',
      **{name: name.replace('reward_only_ir_h', 'Reward-only H=')
         for name in conditions if name.startswith('reward_only')},
  }

  for mode in POLICY_MODES:
    fig, axes = plt.subplots(
        len(steps), len(GOALS), figsize=(11, 5.8), sharey=True,
        squeeze=False, constrained_layout=True)
    for row_index, step in enumerate(steps):
      for col_index, goal in enumerate(GOALS):
        ax = axes[row_index, col_index]
        selected = {
            row['condition']: float(row['success_rate']) for row in rows
            if int(row['checkpoint_step']) == step and row['goal'] == goal and
            row['policy_mode'] == mode}
        values = [selected.get(name, math.nan) for name in conditions]
        colors = ['#777777', '#4c78a8'] + ['#54a24b'] * (len(conditions) - 2)
        ax.bar(range(len(conditions)), values, color=colors)
        ax.set_ylim(0, 1)
        ax.set_title(
            f'{step / 1e6:.2f}M — {goal.replace("_", " ")}')
        ax.set_xticks(
            range(len(conditions)), [labels[name] for name in conditions],
            rotation=32, ha='right', fontsize=7)
        if col_index == 0:
          ax.set_ylabel('Success rate')
    fig.suptitle(f'Stage 3B natural-forgetting recovery — {mode} policy')
    for suffix in ('png', 'pdf'):
      fig.savefig(figure_root / f'success_matrix_{mode}.{suffix}', dpi=200)
    plt.close(fig)

  fig, axes = plt.subplots(1, len(steps), figsize=(10.5, 3.8), sharey=True,
                           constrained_layout=True)
  axes = np.atleast_1d(axes)
  adapted = [name for name in conditions if name != 'frozen']
  for axis, step in zip(axes, steps):
    positions = np.arange(len(GOALS))
    width = .8 / len(adapted)
    for index, condition in enumerate(adapted):
      selected = [
          row for row in rows
          if int(row['checkpoint_step']) == step and
          row['policy_mode'] == 'sampled' and row['condition'] == condition]
      values = [next(
          float(row['paired_delta_vs_frozen']) for row in selected
          if row['goal'] == goal) for goal in GOALS]
      lows = [next(
          float(row['paired_delta_ci_low']) for row in selected
          if row['goal'] == goal) for goal in GOALS]
      highs = [next(
          float(row['paired_delta_ci_high']) for row in selected
          if row['goal'] == goal) for goal in GOALS]
      xpos = positions - .4 + width / 2 + index * width
      axis.bar(xpos, values, width=width, label=labels[condition])
      axis.errorbar(
          xpos, values,
          yerr=[np.asarray(values) - lows, np.asarray(highs) - values],
          fmt='none', ecolor='black', capsize=2, lw=.8)
    axis.axhline(0, color='black', lw=1)
    axis.set_title(f'{step / 1e6:.2f}M checkpoint')
    axis.set_xticks(positions, [goal.replace('_', ' ') for goal in GOALS],
                    rotation=25, ha='right')
  axes[0].set_ylabel('Paired success change vs frozen')
  handles, legend_labels = axes[-1].get_legend_handles_labels()
  fig.legend(handles, legend_labels, loc='upper center', ncol=4, frameon=False)
  fig.suptitle('Sampled-policy paired treatment effects')
  for suffix in ('png', 'pdf'):
    fig.savefig(figure_root / f'paired_effects_sampled.{suffix}', dpi=200)
  plt.close(fig)


def _heartbeat(args, snapshot):
  args.queue['active_child'] = snapshot
  args.queue['updated_unix'] = time.time()
  write_json(args.queue_path, args.queue)


def execute_unit(
    args, *, states: dict[str, Any], unit_id: str, root: pathlib.Path,
    checkpoint: pathlib.Path, condition: dict[str, Any], goal: str,
    policy_mode: str, episodes: int, checkpoint_step: int,
) -> dict[str, Any]:
  state = states.setdefault(unit_id, {'status': 'pending', 'attempts': []})

  def summarize(logdir, returncode):
    result = stage2.summarize_condition(
        logdir, condition=condition, goal=goal, policy_mode=policy_mode,
        episodes=episodes, seed=args.eval_seed, returncode=returncode)
    result.update(
        checkpoint_step=int(checkpoint_step),
        checkpoint_label=checkpoint_label(checkpoint_step),
        source_checkpoint=str(checkpoint))
    return result

  if state.get('status') == 'complete':
    saved = state['attempts'][-1]
    recovered = summarize(pathlib.Path(saved['logdir']), 0)
    if recovered['complete']:
      saved['summary'] = recovered
      return recovered
    saved.update(status='invalidated', finished_unix=time.time())
    state['status'] = 'pending'

  active_path = args.output_root / 'active_child.json'
  attempt = state['attempts'][-1] if state['attempts'] else None
  if attempt and attempt.get('status') == 'running':
    attempt_number = int(attempt['attempt'])
    logdir = pathlib.Path(attempt['logdir'])
    command = stage2.build_command(
        python=args.python, repo_root=args.repo_root, logdir=logdir,
        checkpoint=checkpoint, reference_checkpoint=checkpoint,
        condition=condition, goal=goal, policy_mode=policy_mode,
        episodes=episodes, seed=args.eval_seed,
        episode_length=args.episode_length, grid_columns=args.grid_columns,
        grid_rows=args.grid_rows, adapt_steps=args.adapt_steps,
        adapt_lr=args.adapt_lr, adapt_every_k=args.adapt_every_k,
        start_batch=args.start_batch, mvn_path=args.mvn_path,
        java_home=args.java_home)
    recovered = summarize(logdir, 0)
    if recovered['complete']:
      attempt.update(
          status='complete', finished_unix=time.time(), summary=recovered)
      state['status'] = 'complete'
      return recovered
    if not active_matches(active_path, command, args.repo_root):
      attempt.update(status='interrupted', finished_unix=time.time())
      attempt = None

  if attempt is None or attempt.get('status') != 'running':
    attempt_number = len(state['attempts']) + 1
    if attempt_number > args.max_attempts:
      raise RuntimeError(f'Exhausted attempts for {unit_id}')
    logdir = root / f'attempt_{attempt_number:02d}'
    logdir.mkdir(parents=True, exist_ok=False)
    command = stage2.build_command(
        python=args.python, repo_root=args.repo_root, logdir=logdir,
        checkpoint=checkpoint, reference_checkpoint=checkpoint,
        condition=condition, goal=goal, policy_mode=policy_mode,
        episodes=episodes, seed=args.eval_seed,
        episode_length=args.episode_length, grid_columns=args.grid_columns,
        grid_rows=args.grid_rows, adapt_steps=args.adapt_steps,
        adapt_lr=args.adapt_lr, adapt_every_k=args.adapt_every_k,
        start_batch=args.start_batch, mvn_path=args.mvn_path,
        java_home=args.java_home)
    attempt = {
        'attempt': attempt_number, 'status': 'running',
        'started_unix': time.time(), 'logdir': str(logdir),
        'command': command}
    state['attempts'].append(attempt)
  state['status'] = 'running'
  args.queue['updated_unix'] = time.time()
  write_json(args.queue_path, args.queue)

  print(f'\n=== Stage 3B {unit_id}, attempt {attempt_number} ===', flush=True)
  print(' '.join(command), flush=True)
  child = run_managed(
      command, cwd=args.repo_root, active_path=active_path,
      console_path=logdir / 'console.log',
      activity_paths=(
          logdir / 'metrics.jsonl', logdir / 'episodes.jsonl',
          logdir / 'audit_env0.jsonl'),
      stream_output=args.stream_output, poll_seconds=args.poll_seconds,
      startup_timeout=args.startup_timeout, stall_timeout=args.stall_timeout,
      heartbeat=lambda value: _heartbeat(args, value))
  args.queue.pop('active_child', None)
  summary = summarize(logdir, child['returncode'])
  attempt.update(
      status='complete' if summary['complete'] else 'failed',
      finished_unix=time.time(), child=child, summary=summary)
  state['status'] = attempt['status']
  args.queue['updated_unix'] = time.time()
  write_json(args.queue_path, args.queue)
  if not summary['complete']:
    if attempt_number < args.max_attempts:
      return execute_unit(
          args, states=states, unit_id=unit_id, root=root,
          checkpoint=checkpoint, condition=condition, goal=goal,
          policy_mode=policy_mode, episodes=episodes,
          checkpoint_step=checkpoint_step)
    raise RuntimeError(
        f'Incomplete Stage 3B unit {unit_id}: {logdir / "console.log"}')
  return summary


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Run Stage 3B RLScape natural-forgetting recovery.')
  parser.add_argument('--experiment-root', type=pathlib.Path, required=True)
  parser.add_argument('--output-root', type=pathlib.Path)
  parser.add_argument('--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument('--python', type=pathlib.Path,
                      default=pathlib.Path(sys.executable))
  parser.add_argument('--checkpoints', default='1250000,1500000')
  parser.add_argument('--reward-only-horizons', default='6,15,20')
  parser.add_argument('--classification-threshold', type=float, default=.50)
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


def validate_args(args):
  args.repo_root = args.repo_root.resolve()
  args.experiment_root = args.experiment_root.resolve()
  args.python = args.python.resolve()
  args.output_root = (
      args.output_root.resolve() if args.output_root else
      args.experiment_root / 'stage3b_natural_forgetting_recovery')
  args.checkpoints = parse_ints(args.checkpoints)
  args.reward_only_horizons = parse_ints(args.reward_only_horizons)
  if any(step not in HISTORICAL_SAMPLED_SUCCESS for step in args.checkpoints):
    raise ValueError(
        'Stage 3B classifications are prespecified only for checkpoints '
        f'{sorted(HISTORICAL_SAMPLED_SUCCESS)}')
  if not 0 < args.classification_threshold < 1:
    raise ValueError('--classification-threshold must be between zero and one')
  if not args.python.is_file():
    raise FileNotFoundError(args.python)
  if any(value <= 0 for value in (
      args.sampled_episodes, args.deterministic_episodes,
      args.episode_length, args.grid_columns, args.grid_rows,
      args.adapt_steps, args.start_batch, args.max_attempts)):
    raise ValueError('Episode, grid, adaptation, and attempt counts must be positive')
  if args.adapt_lr <= 0 or args.adapt_every_k < 0:
    raise ValueError('Invalid adaptation schedule')


def main(argv=None) -> int:
  args = parse_args(argv)
  validate_args(args)
  checkpoints = {
      step: stage2.resolve_milestone(args.experiment_root, step) / 'checkpoint'
      for step in args.checkpoints}
  conditions = build_conditions(args.reward_only_horizons)
  classifications = {
      checkpoint_label(step): {
          goal: dict(zip(
              ('historical_sampled_success', 'status'),
              historical_classification(
                  step, goal, args.classification_threshold)))
          for goal in GOALS}
      for step in args.checkpoints}
  spec = {
      'format': 1,
      'canonical_stage': 'stage3b',
      'name': 'natural_forgetting_recovery',
      'git_commit': git_revision(args.repo_root),
      'python_version': platform.python_version(),
      'rlscape_version': importlib.metadata.version('rl-scape'),
      'experiment_root': str(args.experiment_root),
      'checkpoints': {
          checkpoint_label(step): str(path)
          for step, path in checkpoints.items()},
      'goals': list(GOALS),
      'policy_modes': list(POLICY_MODES),
      'conditions': conditions,
      'episodes': {
          'sampled': args.sampled_episodes,
          'deterministic': args.deterministic_episodes},
      'eval_seed': args.eval_seed,
      'episode_length': args.episode_length,
      'grid': [args.grid_columns, args.grid_rows],
      'adaptation': {
          'steps_per_trigger': args.adapt_steps,
          'lr': args.adapt_lr,
          'every_k': args.adapt_every_k,
          'posterior_samples': args.start_batch,
          'critic_frozen': True,
          'world_model_frozen': True,
          'episode_local': True},
      'classification': {
          'source': 'Stage 0 frozen sampled-policy evaluation',
          'threshold': args.classification_threshold,
          'rule': 'weak iff historical sampled success < threshold',
          'prespecified': True,
          'cells': classifications},
      'pairing': (
          'same checkpoint, goal, policy mode, and reset seed across conditions'),
      'claim_boundary': (
          'checkpoint-local recovery without new training experience; '
          'not persistent continual retention'),
  }
  digest = spec_digest(spec)
  args.output_root.mkdir(parents=True, exist_ok=True)
  spec_path = args.output_root / 'experiment_spec.json'
  if spec_path.is_file():
    if spec_digest(json.loads(spec_path.read_text())) != digest:
      raise RuntimeError(f'Existing Stage 3B specification differs: {spec_path}')
  else:
    write_json(spec_path, spec)
  write_json(
      args.output_root / 'experiment_spec_digest.json', {'sha256': digest})

  units = (
      len(checkpoints) * len(conditions) * len(GOALS) * len(POLICY_MODES))
  episodes_total = len(checkpoints) * len(conditions) * len(GOALS) * (
      args.sampled_episodes + args.deterministic_episodes)
  if args.dry_run:
    print(f'Stage 3B plan: {units} paired units, {episodes_total} episodes.')
    print(f'Checkpoints: {args.checkpoints}')
    print(f'Conditions: {[item["name"] for item in conditions]}')
    print(f'Specification: {spec_path}')
    return 0

  queue_path = args.output_root / 'queue.json'
  existing = json.loads(queue_path.read_text()) if queue_path.is_file() else {}
  if existing.get('spec_digest') not in (None, digest):
    raise RuntimeError(f'Existing Stage 3B queue differs: {queue_path}')
  queue = {
      'format': 1, 'canonical_stage': 'stage3b', 'spec_digest': digest,
      'created_unix': existing.get('created_unix', time.time()),
      'updated_unix': time.time(), 'status': 'running',
      'states': existing.get('states', {})}
  args.queue = queue
  args.queue_path = queue_path
  write_json(queue_path, queue)

  episode_counts = {
      'sampled': args.sampled_episodes,
      'deterministic': args.deterministic_episodes}
  rows = []
  for step, checkpoint in checkpoints.items():
    label = checkpoint_label(step)
    for condition in conditions:
      for goal in GOALS:
        for mode in POLICY_MODES:
          unit_id = f'{label}__{condition["name"]}__{goal}__{mode}'
          summary = execute_unit(
              args, states=queue['states'], unit_id=unit_id,
              root=(args.output_root / 'evaluations' / label /
                    condition['name'] / goal / mode),
              checkpoint=checkpoint, condition=condition, goal=goal,
              policy_mode=mode, episodes=episode_counts[mode],
              checkpoint_step=step)
          rows.append(summary)
          write_csv(args.output_root / 'main_results.csv', rows)

  pairing = validate_pairing(rows)
  write_json(args.output_root / 'pairing.json', pairing)
  if not pairing['equal']:
    raise RuntimeError('Stage 3B reset identities are not paired')
  compared = comparison_rows(
      rows, classification_threshold=args.classification_threshold)
  write_csv(args.output_root / 'comparison_vs_frozen.csv', compared)
  write_csv(args.output_root / 'aggregate_results.csv', summarize_groups(compared))
  make_figures(args.output_root, compared)
  queue.update(status='complete', updated_unix=time.time())
  write_json(queue_path, queue)
  (args.output_root / 'stage3b_complete').touch()
  print(f'Stage 3B complete: {args.output_root}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
