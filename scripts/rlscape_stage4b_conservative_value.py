#!/usr/bin/env python3
"""Run Stage 4B conservative bounded-value dose and stability assays."""

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
from scripts import rlscape_stage2_actor_recovery as stage2  # noqa: E402
from scripts.rlscape_m1_pilot import read_jsonl  # noqa: E402
from scripts.rlscape_m1_pilot import spec_digest  # noqa: E402
from scripts.rlscape_m1_pilot import write_csv  # noqa: E402
from scripts.rlscape_m1_pilot import write_json  # noqa: E402
from scripts.rlscape_m3_process import active_matches  # noqa: E402
from scripts.rlscape_m3_process import run_managed  # noqa: E402
from scripts.rlscape_m3_sequential import git_revision  # noqa: E402


GOALS = stage2.GOALS
POLICY_MODES = stage2.POLICY_MODES


def installed_version(package: str) -> str:
  try:
    return importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:
    return 'not-installed'


def build_conditions() -> list[dict[str, Any]]:
  """Return the prespecified matrix; order is part of the protocol."""
  conditions = [
      dict(
          name='frozen', family='frozen', enabled=False,
          objective='standard', horizon=0, mix_beta=0.0,
          value_correction_clip=0.0, update_rms_clip=0.0, actent=0.0),
      dict(
          name='reward_only_h15', family='dose_h15', enabled=True,
          objective='mixed_bounded', horizon=15, mix_beta=0.0,
          value_correction_clip=0.0, update_rms_clip=0.0, actent=0.0),
      dict(
          name='mixed_b005_h15', family='dose_h15', enabled=True,
          objective='mixed_bounded', horizon=15, mix_beta=.05,
          value_correction_clip=0.0, update_rms_clip=0.0, actent=0.0),
      dict(
          name='mixed_b010_h15', family='dose_h15', enabled=True,
          objective='mixed_bounded', horizon=15, mix_beta=.10,
          value_correction_clip=0.0, update_rms_clip=0.0, actent=0.0),
      dict(
          name='mixed_b025_h15', family='dose_h15', enabled=True,
          objective='mixed_bounded', horizon=15, mix_beta=.25,
          value_correction_clip=0.0, update_rms_clip=0.0, actent=0.0),
      dict(
          name='mixed_b025_clip050_h15', family='correction_clip',
          enabled=True, objective='mixed_bounded', horizon=15,
          mix_beta=.25, value_correction_clip=.50, update_rms_clip=0.0,
          actent=0.0),
      dict(
          name='mixed_b025_ucap15e6_h15', family='update_cap',
          enabled=True, objective='mixed_bounded', horizon=15,
          mix_beta=.25, value_correction_clip=0.0,
          update_rms_clip=1.5e-5, actent=0.0),
      dict(
          name='mixed_b025_h6', family='horizon_control', enabled=True,
          objective='mixed_bounded', horizon=6, mix_beta=.25,
          value_correction_clip=0.0, update_rms_clip=0.0, actent=0.0),
      dict(
          name='mixed_b100_h15', family='bounded_anchor', enabled=True,
          objective='mixed_bounded', horizon=15, mix_beta=1.0,
          value_correction_clip=0.0, update_rms_clip=0.0, actent=0.0),
  ]
  for condition in conditions:
    condition['checkpoint'] = 'counterfactual_bounded'
  names = [condition['name'] for condition in conditions]
  if len(names) != len(set(names)):
    raise AssertionError(names)
  return conditions


def build_command(
    args, *, logdir: pathlib.Path, condition: dict[str, Any], goal: str,
    policy_mode: str, episodes: int, replicate: int,
) -> list[str]:
  environment_seed = args.eval_seed_base + replicate
  agent_seed = args.agent_seed_base + replicate
  command = stage2.build_command(
      python=args.python, repo_root=args.repo_root, logdir=logdir,
      checkpoint=args.checkpoint, reference_checkpoint=args.checkpoint,
      condition=condition, goal=goal, policy_mode=policy_mode,
      episodes=episodes, seed=environment_seed,
      episode_length=args.episode_length, grid_columns=args.grid_columns,
      grid_rows=args.grid_rows, adapt_steps=args.adapt_steps,
      adapt_lr=args.adapt_lr, adapt_every_k=args.adapt_every_k,
      start_batch=args.start_batch, mvn_path=args.mvn_path,
      java_home=args.java_home,
      extra_configs=('rlscape_stage4a_bounded_value',))
  command += [
      '--eval_adapt.mix_beta', str(float(condition['mix_beta'])),
      '--eval_adapt.value_correction_clip',
      str(float(condition['value_correction_clip'])),
      '--eval_adapt.update_rms_clip',
      str(float(condition['update_rms_clip'])),
      '--eval_adapt.paired_rng', 'True',
      '--eval_adapt.paired_rng_seed', str(agent_seed),
      '--eval_adapt.paired_rng_stride', str(args.paired_rng_stride),
  ]
  return command


def _metric_mean(logdir: pathlib.Path, suffix: str) -> float:
  key = f'epstats/log/eval_adapt/{suffix}/avg'
  values = [float(row[key]) for row in read_jsonl(
      logdir / 'metrics.jsonl') if key in row]
  return float(np.mean(values)) if values else float('nan')


def summarize(
    args, *, logdir: pathlib.Path, condition: dict[str, Any], goal: str,
    policy_mode: str, episodes: int, replicate: int,
    returncode: int | None,
) -> dict[str, Any]:
  environment_seed = args.eval_seed_base + replicate
  agent_seed = args.agent_seed_base + replicate
  result = stage2.summarize_condition(
      logdir, condition=condition, goal=goal, policy_mode=policy_mode,
      episodes=episodes, seed=environment_seed, returncode=returncode)
  integrity_path = logdir / 'eval_integrity.json'
  integrity = (
      json.loads(integrity_path.read_text()) if integrity_path.is_file()
      else {})
  rng_equal = bool(
      integrity.get('paired_rng') is True and
      integrity.get('paired_rng_seed') == agent_seed and
      integrity.get('paired_rng_stride') == args.paired_rng_stride)
  result.update({
      'canonical_stage': 'stage4b',
      'source_checkpoint': str(args.checkpoint),
      'replicate': replicate,
      'environment_seed': environment_seed,
      'agent_seed': agent_seed,
      'mix_beta': float(condition['mix_beta']),
      'value_correction_clip': float(condition['value_correction_clip']),
      'update_rms_clip': float(condition['update_rms_clip']),
      'paired_rng_verified': rng_equal,
      'adapt_value_correction_abs_mean': _metric_mean(
          logdir, 'adapt_actor/value_correction_abs'),
      'adapt_value_correction_used_abs_mean': _metric_mean(
          logdir, 'adapt_actor/value_correction_used_abs'),
      'adapt_value_correction_clip_rate_mean': _metric_mean(
          logdir, 'adapt_actor/value_correction_clip_rate'),
  })
  result['complete'] = bool(result['complete'] and rng_equal)
  return result


def _heartbeat(args, snapshot):
  args.state['active_child'] = snapshot
  args.state['updated_unix'] = time.time()
  write_json(args.state_path, args.state)


def execute_eval(
    args, *, condition: dict[str, Any], goal: str, policy_mode: str,
    episodes: int, replicate: int,
) -> dict[str, Any]:
  unit_id = (
      f'replicate_{replicate:02d}/{condition["name"]}/'
      f'{goal}/{policy_mode}')
  state = args.state['units'].setdefault(
      unit_id, {'status': 'pending', 'attempts': []})

  def recover(logdir, returncode):
    return summarize(
        args, logdir=logdir, condition=condition, goal=goal,
        policy_mode=policy_mode, episodes=episodes, replicate=replicate,
        returncode=returncode)

  if state.get('status') == 'complete':
    saved = state['attempts'][-1]
    result = recover(pathlib.Path(saved['logdir']), 0)
    if result['complete']:
      saved['summary'] = result
      return result
    state['status'] = 'pending'

  attempt = state['attempts'][-1] if state['attempts'] else None
  if attempt and attempt.get('status') == 'running':
    attempt_number = int(attempt['attempt'])
    logdir = pathlib.Path(attempt['logdir'])
    command = build_command(
        args, logdir=logdir, condition=condition, goal=goal,
        policy_mode=policy_mode, episodes=episodes, replicate=replicate)
    result = recover(logdir, 0)
    if result['complete']:
      attempt.update(
          status='complete', finished_unix=time.time(), summary=result)
      state['status'] = 'complete'
      return result
    if not active_matches(args.active_path, command, args.repo_root):
      attempt.update(status='interrupted', finished_unix=time.time())
      attempt = None

  if attempt is None or attempt.get('status') != 'running':
    attempt_number = len(state['attempts']) + 1
    if attempt_number > args.max_attempts:
      raise RuntimeError(f'Exhausted attempts for {unit_id}')
    logdir = (
        args.output_root / 'evaluations' / unit_id /
        f'attempt_{attempt_number:02d}')
    logdir.mkdir(parents=True, exist_ok=False)
    command = build_command(
        args, logdir=logdir, condition=condition, goal=goal,
        policy_mode=policy_mode, episodes=episodes, replicate=replicate)
    attempt = {
        'attempt': attempt_number, 'status': 'running',
        'started_unix': time.time(), 'logdir': str(logdir),
        'command': command}
    state['attempts'].append(attempt)
  else:
    command = build_command(
        args, logdir=logdir, condition=condition, goal=goal,
        policy_mode=policy_mode, episodes=episodes, replicate=replicate)

  command_path = args.output_root / 'commands' / f'{unit_id}.json'
  command_path.parent.mkdir(parents=True, exist_ok=True)
  write_json(command_path, {'unit_id': unit_id, 'command': command})
  state['status'] = 'running'
  write_json(args.state_path, args.state)
  print(f'\n=== Stage 4B {unit_id}, attempt {attempt_number} ===', flush=True)
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
  result = recover(logdir, child['returncode'])
  attempt.update(
      status='complete' if result['complete'] else 'failed',
      finished_unix=time.time(), child=child, summary=result)
  state['status'] = attempt['status']
  write_json(args.state_path, args.state)
  if not result['complete']:
    if attempt_number < args.max_attempts:
      return execute_eval(
          args, condition=condition, goal=goal, policy_mode=policy_mode,
          episodes=episodes, replicate=replicate)
    raise RuntimeError(
        f'Incomplete Stage 4B evaluation: {unit_id}; '
        f'see {logdir / "console.log"}')
  return result


def episode_outcomes(logdir: pathlib.Path) -> np.ndarray:
  return np.asarray([
      int(
          float(row.get('episode/score', 0.0)) > 0 or
          float(row.get('episode/log/task_success/max', 0.0)) > 0)
      for row in read_jsonl(logdir / 'episodes.jsonl')], np.float64)


def paired_effect(
    baseline: np.ndarray, treatment: np.ndarray, *, seed: int,
    prefix: str,
) -> dict[str, Any]:
  if baseline.shape != treatment.shape or not len(baseline):
    raise ValueError((baseline.shape, treatment.shape))
  difference = treatment - baseline
  rng = np.random.default_rng(seed)
  indices = rng.integers(0, len(difference), size=(10_000, len(difference)))
  bootstrap = difference[indices].mean(1)
  return {
      f'paired_delta_vs_{prefix}': float(difference.mean()),
      f'paired_delta_vs_{prefix}_ci_low': float(
          np.quantile(bootstrap, .025)),
      f'paired_delta_vs_{prefix}_ci_high': float(
          np.quantile(bootstrap, .975)),
      f'paired_improved_vs_{prefix}': int(
          ((treatment == 1) & (baseline == 0)).sum()),
      f'paired_harmed_vs_{prefix}': int(
          ((treatment == 0) & (baseline == 1)).sum()),
  }


def add_paired_effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
  mismatches = []
  for replicate in sorted({int(row['replicate']) for row in rows}):
    for mode in POLICY_MODES:
      for goal in GOALS:
        group = [
            row for row in rows
            if int(row['replicate']) == replicate and
            row['policy_mode'] == mode and row['goal'] == goal]
        by_name = {row['condition']: row for row in group}
        reward = by_name['reward_only_h15']
        frozen = by_name['frozen']
        reference_resets = stage2.reset_identities(
            pathlib.Path(reward['logdir']))
        reward_outcomes = episode_outcomes(pathlib.Path(reward['logdir']))
        frozen_outcomes = episode_outcomes(pathlib.Path(frozen['logdir']))
        for row in group:
          current_resets = stage2.reset_identities(pathlib.Path(row['logdir']))
          if current_resets != reference_resets:
            mismatches.append({
                'replicate': replicate, 'goal': goal,
                'policy_mode': mode, 'condition': row['condition']})
            continue
          outcomes = episode_outcomes(pathlib.Path(row['logdir']))
          seed = int(spec_digest({
              'stage': '4b', 'replicate': replicate, 'goal': goal,
              'mode': mode, 'condition': row['condition']})[:8], 16)
          row.update(paired_effect(
              reward_outcomes, outcomes, seed=seed, prefix='reward_only'))
          row.update(paired_effect(
              frozen_outcomes, outcomes, seed=seed + 1, prefix='frozen'))
  if mismatches:
    raise RuntimeError(f'Stage 4B reset pairing failed: {mismatches[:3]}')
  return {
      'format': 1, 'equal': True,
      'environment_keys': ['replicate', 'goal', 'policy_mode', 'reset_seed'],
      'agent_rng_keys': [
          'replicate', 'episode_index', 'episode_step', 'agent_seed'],
      'note': (
          'Every condition shares RLScape resets and explicit episode-step '
          'posterior/action random keys within each replicate.'),
  }


def aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
  goal_rows = []
  conditions = [condition['name'] for condition in build_conditions()]
  for mode in POLICY_MODES:
    for condition in conditions:
      for goal in GOALS:
        group = [
            row for row in rows if row['policy_mode'] == mode and
            row['condition'] == condition and row['goal'] == goal]
        if not group:
          raise RuntimeError((mode, condition, goal))
        episodes = sum(int(row['episodes']) for row in group)
        successes = sum(int(row['successes']) for row in group)
        rates = [float(row['success_rate']) for row in group]
        combined_effect = {}
        if all('logdir' in row for row in group):
          baseline_group = [
              row for row in rows if row['policy_mode'] == mode and
              row['condition'] == 'reward_only_h15' and row['goal'] == goal]
          baseline_by_replicate = {
              int(row['replicate']): row for row in baseline_group}
          baseline_outcomes = np.concatenate([
              episode_outcomes(pathlib.Path(
                  baseline_by_replicate[int(row['replicate'])]['logdir']))
              for row in group])
          treatment_outcomes = np.concatenate([
              episode_outcomes(pathlib.Path(row['logdir'])) for row in group])
          combined_effect = paired_effect(
              baseline_outcomes, treatment_outcomes,
              seed=int(spec_digest({
                  'stage': '4b_aggregate', 'mode': mode,
                  'condition': condition, 'goal': goal})[:8], 16),
              prefix='reward_only')
        goal_rows.append({
            'policy_mode': mode, 'condition': condition, 'goal': goal,
            'replicates': len(group), 'episodes': episodes,
            'successes': successes, 'success_rate': successes / episodes,
            'replicate_success_std': float(np.std(rates)),
            'replicate_success_min': min(rates),
            'replicate_success_max': max(rates),
            'mix_beta': float(group[0]['mix_beta']),
            'value_correction_clip': float(
                group[0]['value_correction_clip']),
            'update_rms_clip': float(group[0]['update_rms_clip']),
            'horizon': int(group[0]['horizon']),
            'paired_delta_vs_reward_only': float(np.mean([
                row['paired_delta_vs_reward_only'] for row in group])),
            'paired_delta_vs_reward_only_ci_low': combined_effect.get(
                'paired_delta_vs_reward_only_ci_low', float('nan')),
            'paired_delta_vs_reward_only_ci_high': combined_effect.get(
                'paired_delta_vs_reward_only_ci_high', float('nan')),
            'paired_delta_vs_frozen': float(np.mean([
                row['paired_delta_vs_frozen'] for row in group])),
            'paired_improved_vs_reward_only': sum(
                int(row['paired_improved_vs_reward_only']) for row in group),
            'paired_harmed_vs_reward_only': sum(
                int(row['paired_harmed_vs_reward_only']) for row in group),
        })
  condition_rows = []
  for mode in POLICY_MODES:
    for condition in conditions:
      goals = [row for row in goal_rows
               if row['policy_mode'] == mode and
               row['condition'] == condition]
      raw = [row for row in rows
             if row['policy_mode'] == mode and
             row['condition'] == condition]
      replicate_macros = []
      replicate_deltas = []
      for replicate in sorted({int(row['replicate']) for row in raw}):
        macro = float(np.mean([
            row['success_rate'] for row in raw
            if int(row['replicate']) == replicate]))
        baseline_macro = float(np.mean([
            row['success_rate'] for row in rows
            if row['policy_mode'] == mode and
            row['condition'] == 'reward_only_h15' and
            int(row['replicate']) == replicate]))
        replicate_macros.append(macro)
        replicate_deltas.append(macro - baseline_macro)
      condition_rows.append({
          'policy_mode': mode, 'condition': condition,
          'mix_beta': goals[0]['mix_beta'],
          'value_correction_clip': goals[0]['value_correction_clip'],
          'update_rms_clip': goals[0]['update_rms_clip'],
          'horizon': goals[0]['horizon'],
          'mean_goal_success': float(np.mean([
              row['success_rate'] for row in goals])),
          'worst_goal_success': float(min(
              row['success_rate'] for row in goals)),
          'best_goal_success': float(max(
              row['success_rate'] for row in goals)),
          'mean_goal_delta_vs_reward_only': float(np.mean([
              row['paired_delta_vs_reward_only'] for row in goals])),
          'worst_goal_delta_vs_reward_only': float(min(
              row['paired_delta_vs_reward_only'] for row in goals)),
          'mean_goal_delta_vs_frozen': float(np.mean([
              row['paired_delta_vs_frozen'] for row in goals])),
          'replicate_macro_mean': float(np.mean(replicate_macros)),
          'replicate_macro_std': float(np.std(replicate_macros)),
          'replicate_macro_min': min(replicate_macros),
          'replicate_macro_max': max(replicate_macros),
          'replicate_delta_mean': float(np.mean(replicate_deltas)),
          'replicate_delta_std': float(np.std(replicate_deltas)),
          'replicate_delta_min': min(replicate_deltas),
          'replicate_delta_max': max(replicate_deltas),
          'replicate_positive_delta_count': sum(
              value > 0 for value in replicate_deltas),
          'episodes': sum(int(row['episodes']) for row in goals),
      })
  return goal_rows, condition_rows


def make_figures(
    output_root: pathlib.Path, rows: list[dict], goal_rows: list[dict],
    condition_rows: list[dict],
) -> None:
  import matplotlib.pyplot as plt

  figure_root = output_root / 'figures'
  figure_root.mkdir(exist_ok=True)
  plt.rcParams.update({
      'font.size': 9, 'axes.spines.top': False,
      'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .2})
  dose_names = [
      'reward_only_h15', 'mixed_b005_h15', 'mixed_b010_h15',
      'mixed_b025_h15', 'mixed_b100_h15']
  for mode in POLICY_MODES:
    selected = [row for row in goal_rows if row['policy_mode'] == mode]
    fig, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for goal, color in zip(GOALS, ('#4c78a8', '#f58518', '#54a24b')):
      lookup = {(row['condition'], row['goal']): row for row in selected}
      xs = [lookup[(name, goal)]['mix_beta'] for name in dose_names]
      ys = [lookup[(name, goal)]['success_rate'] for name in dose_names]
      axis.plot(xs, ys, marker='o', color=color,
                label=goal.replace('_', ' '))
    axis.set_xscale('symlog', linthresh=.05)
    axis.set_xticks([0, .05, .1, .25, 1], ['0', '.05', '.10', '.25', '1'])
    axis.set_ylim(0, 1)
    axis.set_xlabel(r'Bounded-value dose $\beta$ (H=15, unclipped)')
    axis.set_ylabel('Success rate')
    axis.set_title(f'Stage 4B value-dose response — {mode}')
    axis.legend(frameon=False)
    for suffix in ('png', 'pdf'):
      fig.savefig(figure_root / f'value_dose_{mode}.{suffix}', dpi=220)
    plt.close(fig)

  labels = {
      'frozen': 'Frozen', 'reward_only_h15': 'Reward\nonly',
      'mixed_b005_h15': r'$\beta=.05$',
      'mixed_b010_h15': r'$\beta=.10$',
      'mixed_b025_h15': r'$\beta=.25$',
      'mixed_b025_clip050_h15': r'$\beta=.25$' + '\nclip .50',
      'mixed_b025_ucap15e6_h15': r'$\beta=.25$' + '\nupdate cap',
      'mixed_b025_h6': r'$\beta=.25$' + '\nH=6',
      'mixed_b100_h15': r'$\beta=1$',
  }
  names = [condition['name'] for condition in build_conditions()]
  fig, axes = plt.subplots(
      1, 2, figsize=(12.2, 4.3), sharey=True, constrained_layout=True)
  for axis, mode in zip(axes, POLICY_MODES):
    lookup = {(row['condition']): row for row in condition_rows
              if row['policy_mode'] == mode}
    means = [lookup[name]['mean_goal_success'] for name in names]
    worst = [lookup[name]['worst_goal_success'] for name in names]
    x = np.arange(len(names))
    axis.bar(x - .18, means, width=.36, label='Mean goal', color='#4c78a8')
    axis.bar(x + .18, worst, width=.36, label='Worst goal', color='#e45756')
    axis.set_xticks(x, [labels[name] for name in names],
                    rotation=30, ha='right')
    axis.set_ylim(0, 1)
    axis.set_title(mode)
  axes[0].set_ylabel('Success rate')
  axes[0].legend(frameon=False)
  fig.suptitle('Stage 4B mean and worst-goal safety')
  for suffix in ('png', 'pdf'):
    fig.savefig(figure_root / f'mean_worst_goal.{suffix}', dpi=220)
  plt.close(fig)

  fig, axes = plt.subplots(
      1, 2, figsize=(12.2, 4.3), sharey=True, constrained_layout=True)
  for axis, mode in zip(axes, POLICY_MODES):
    lookup = {(row['condition'], row['goal']): row for row in goal_rows
              if row['policy_mode'] == mode}
    matrix = np.asarray([
        [lookup[(name, goal)]['paired_delta_vs_reward_only']
         for name in names] for goal in GOALS])
    image = axis.imshow(matrix, vmin=-.5, vmax=.5, cmap='RdBu', aspect='auto')
    axis.set_xticks(range(len(names)), [labels[name] for name in names],
                    rotation=30, ha='right')
    axis.set_yticks(range(len(GOALS)), [goal.replace('_', ' ') for goal in GOALS])
    axis.set_title(mode)
    for y in range(len(GOALS)):
      for x in range(len(names)):
        axis.text(x, y, f'{matrix[y, x]:+.2f}', ha='center', va='center',
                  fontsize=7)
  fig.colorbar(image, ax=axes.tolist(), shrink=.8,
               label='Paired success change vs reward-only')
  fig.suptitle('Stage 4B benefit and harm relative to reward-only IR')
  for suffix in ('png', 'pdf'):
    fig.savefig(figure_root / f'paired_delta_heatmap.{suffix}', dpi=220)
  plt.close(fig)

  fig, axes = plt.subplots(
      1, 2, figsize=(12.2, 4.3), sharey=True, constrained_layout=True)
  for axis, mode in zip(axes, POLICY_MODES):
    for replicate in sorted({int(row['replicate']) for row in rows}):
      values = []
      for name in names:
        selected = [
            row['success_rate'] for row in rows
            if row['policy_mode'] == mode and row['condition'] == name and
            int(row['replicate']) == replicate]
        values.append(float(np.mean(selected)))
      axis.plot(range(len(names)), values, marker='o', alpha=.75,
                label=f'replicate {replicate}')
    axis.set_xticks(range(len(names)), [labels[name] for name in names],
                    rotation=30, ha='right')
    axis.set_ylim(0, 1)
    axis.set_title(mode)
  axes[0].set_ylabel('Macro success across goals')
  axes[0].legend(frameon=False)
  fig.suptitle('Stage 4B adaptation-seed stability')
  for suffix in ('png', 'pdf'):
    fig.savefig(figure_root / f'replicate_stability.{suffix}', dpi=220)
  plt.close(fig)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Run Stage 4B conservative value-dose and stability assay.')
  parser.add_argument('--stage4a-root', type=pathlib.Path, required=True)
  parser.add_argument('--output-root', type=pathlib.Path)
  parser.add_argument('--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument('--python', type=pathlib.Path,
                      default=pathlib.Path(sys.executable))
  parser.add_argument('--replicates', type=int, default=3)
  parser.add_argument('--sampled-episodes', type=int, default=50)
  parser.add_argument('--deterministic-episodes', type=int, default=25)
  parser.add_argument('--eval-seed-base', type=int, default=4000)
  parser.add_argument('--agent-seed-base', type=int, default=20260817)
  parser.add_argument('--paired-rng-stride', type=int, default=1000)
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
  args.stage4a_root = args.stage4a_root.expanduser().resolve()
  args.python = args.python.expanduser().resolve()
  args.output_root = (
      args.output_root.expanduser().resolve() if args.output_root else
      args.stage4a_root.parent / 'stage4b_conservative_value')
  args.checkpoint = args.stage4a_root / 'checkpoints' / 'counterfactual_bounded'
  if not args.python.is_file():
    raise FileNotFoundError(args.python)
  for path in (
      args.checkpoint / 'done', args.checkpoint / 'agent.pkl',
      args.checkpoint / 'repair_manifest.json'):
    if not path.is_file():
      raise FileNotFoundError(path)
  repair = json.loads(
      (args.checkpoint / 'repair_manifest.json').read_text())
  if not (
      repair.get('condition') == 'counterfactual_bounded' and
      repair.get('counterfactual_reward_continuation') is True and
      repair.get('bounded_value') is True and
      int(repair.get('value_updates', 0)) > 0 and
      not repair.get('frozen_component_violations')):
    raise RuntimeError(
        'Stage 4B requires the complete, integrity-clean Stage 4A '
        'counterfactual_bounded checkpoint')
  positive = (
      args.replicates, args.sampled_episodes, args.deterministic_episodes,
      args.episode_length, args.grid_columns, args.grid_rows,
      args.adapt_steps, args.start_batch, args.max_attempts)
  if any(value < 1 for value in positive):
    raise ValueError('Stage 4B count arguments must be positive')
  if args.adapt_lr <= 0 or args.adapt_every_k < 0:
    raise ValueError('Invalid Stage 4B adaptation schedule')
  if args.paired_rng_stride <= args.episode_length:
    raise ValueError(
        '--paired-rng-stride must exceed --episode-length to avoid collisions')


def main(argv=None) -> int:
  args = parse_args(argv)
  validate_args(args)
  args.output_root.mkdir(parents=True, exist_ok=True)
  conditions = build_conditions()
  manifest = args.checkpoint / 'repair_manifest.json'
  spec = {
      'format': 1, 'canonical_stage': 'stage4b',
      'slug': 'conservative_value_dose_stability',
      'hypothesis': (
          'A small bounded-success-value correction can improve reward-only '
          'IR without reducing worst-goal performance.'),
      'git_commit': git_revision(args.repo_root),
      'python_version': platform.python_version(),
      'rlscape_version': installed_version('rl-scape'),
      'stage4a_root': str(args.stage4a_root),
      'checkpoint': str(args.checkpoint),
      'repair_manifest_sha256': _sha256(manifest),
      'goals': list(GOALS), 'policy_modes': list(POLICY_MODES),
      'conditions': conditions,
      'replicates': args.replicates,
      'evaluation': {
          'sampled_episodes_per_replicate': args.sampled_episodes,
          'deterministic_episodes_per_replicate':
              args.deterministic_episodes,
          'environment_seeds': [
              args.eval_seed_base + index for index in range(args.replicates)],
          'agent_seeds': [
              args.agent_seed_base + index for index in range(args.replicates)],
          'paired_rng_stride': args.paired_rng_stride,
          'episode_length': args.episode_length,
          'adapt_steps': args.adapt_steps, 'adapt_lr': args.adapt_lr,
          'adapt_every_k': args.adapt_every_k,
          'posterior_samples': args.start_batch,
      },
      'primary_endpoints': [
          'mean_goal_success', 'worst_goal_success',
          'mean_goal_delta_vs_reward_only',
          'worst_goal_delta_vs_reward_only'],
  }
  spec['digest'] = spec_digest(spec)
  spec_path = args.output_root / 'experiment_spec.json'
  if spec_path.is_file():
    existing = json.loads(spec_path.read_text())
    if existing.get('digest') != spec['digest']:
      raise RuntimeError('Existing Stage 4B root has a different specification')
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
      {'format': 1, 'canonical_stage': 'stage4b',
       'spec_digest': spec['digest'], 'status': 'running', 'units': {}})
  if args.state.get('spec_digest') != spec['digest']:
    raise RuntimeError('Stage 4B queue belongs to a different specification')
  args.state.update(status='running', updated_unix=time.time())
  write_json(args.state_path, args.state)

  rows = []
  for replicate in range(args.replicates):
    for condition in conditions:
      for goal in GOALS:
        for mode in POLICY_MODES:
          episodes = (
              args.sampled_episodes if mode == 'sampled'
              else args.deterministic_episodes)
          rows.append(execute_eval(
              args, condition=condition, goal=goal, policy_mode=mode,
              episodes=episodes, replicate=replicate))

  pairing = add_paired_effects(rows)
  write_json(args.output_root / 'pairing.json', pairing)
  goal_rows, condition_rows = aggregate(rows)
  write_json(args.output_root / 'main_results.json', {
      'format': 1, 'canonical_stage': 'stage4b',
      'spec_digest': spec['digest'], 'rows': rows})
  write_csv(args.output_root / 'main_results.csv', rows)
  write_csv(args.output_root / 'goal_summary.csv', goal_rows)
  write_csv(args.output_root / 'condition_summary.csv', condition_rows)
  write_json(args.output_root / 'aggregate_summary.json', {
      'format': 1, 'canonical_stage': 'stage4b',
      'spec_digest': spec['digest'], 'goal_rows': goal_rows,
      'condition_rows': condition_rows})
  make_figures(args.output_root, rows, goal_rows, condition_rows)
  args.state.update(status='complete', updated_unix=time.time())
  write_json(args.state_path, args.state)
  (args.output_root / 'stage4b_complete').write_bytes(b'')
  print(f'Stage 4B complete: {args.output_root}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
