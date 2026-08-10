#!/usr/bin/env python3
"""Run the RLScape Stage 2 controlled actor corruption/recovery assay."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import shutil
import sys
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'scripts'))

from embodied.run import milestones  # noqa: E402
from scripts import corrupt_actor_checkpoint as corrupt  # noqa: E402
from scripts.rlscape_m1_pilot import read_jsonl  # noqa: E402
from scripts.rlscape_m1_pilot import spec_digest  # noqa: E402
from scripts.rlscape_m1_pilot import write_csv  # noqa: E402
from scripts.rlscape_m1_pilot import write_json  # noqa: E402
from scripts.rlscape_m2_eval import summarize_run  # noqa: E402
from scripts.rlscape_m3_process import active_matches  # noqa: E402
from scripts.rlscape_m3_process import run_managed  # noqa: E402


GOALS = ('kill_goblin', 'bury_bones', 'chop_logs')
DEFAULT_HORIZONS = (3, 6, 15, 20)
POLICY_MODES = ('sampled', 'deterministic')


def parse_ints(text: str) -> tuple[int, ...]:
  values = tuple(int(item.strip()) for item in text.split(',') if item.strip())
  if not values or any(value < 1 for value in values):
    raise ValueError(f'Expected positive comma-separated integers: {text!r}')
  if len(set(values)) != len(values):
    raise ValueError(f'Duplicate values are not allowed: {text!r}')
  return values


def parse_floats(text: str) -> tuple[float, ...]:
  values = tuple(float(item.strip()) for item in text.split(',') if item.strip())
  if not values or any(not 0 < value < 1 for value in values):
    raise ValueError(f'Expected values strictly between zero and one: {text!r}')
  if len(set(values)) != len(values):
    raise ValueError(f'Duplicate values are not allowed: {text!r}')
  return values


def resolve_milestone(experiment_root: pathlib.Path, step: int) -> pathlib.Path:
  archive = milestones.archive_path(experiment_root / 'milestones', step)
  milestones.validate(archive, expected_step=step, full=False)
  return archive


def build_conditions(horizons=DEFAULT_HORIZONS) -> list[dict[str, Any]]:
  conditions = [
      dict(
          name='clean_no_adapt', family='clean', checkpoint='clean',
          enabled=False, objective='standard', horizon=0, actent=0.0),
      dict(
          name='corrupt_no_adapt', family='corrupt', checkpoint='corrupt',
          enabled=False, objective='standard', horizon=0, actent=0.0),
  ]
  for family, objective, actent in (
      ('standard_ir', 'standard', 3e-4),
      ('entropy_off_ir', 'standard', 0.0),
      ('reward_only_ir', 'reward_only', 0.0),
  ):
    for horizon in horizons:
      conditions.append(dict(
          name=f'{family}_h{horizon}', family=family,
          checkpoint='corrupt', enabled=True, objective=objective,
          horizon=int(horizon), actent=float(actent)))
  conditions.append(dict(
      name='reference_distill', family='reference_distill',
      checkpoint='corrupt', enabled=True, objective='distill',
      horizon=0, actent=0.0))
  names = [item['name'] for item in conditions]
  if len(names) != len(set(names)):
    raise AssertionError(names)
  return conditions


def build_command(
    *, python: pathlib.Path, repo_root: pathlib.Path,
    logdir: pathlib.Path, checkpoint: pathlib.Path,
    reference_checkpoint: pathlib.Path, condition: dict[str, Any],
    goal: str, policy_mode: str, episodes: int, seed: int,
    episode_length: int, grid_columns: int, grid_rows: int,
    adapt_steps: int, adapt_lr: float, adapt_every_k: int,
    start_batch: int, mvn_path: str = '', java_home: str = '',
) -> list[str]:
  if goal not in GOALS:
    raise ValueError(goal)
  if policy_mode not in POLICY_MODES:
    raise ValueError(policy_mode)
  command = [
      str(python), str(repo_root / 'dreamerv3' / 'main.py'),
      '--configs', 'rlscape', 'rlscape_click_grid', f'rlscape_{goal}',
      'rlscape_m3_eval',
      '--seed', str(seed),
      '--logdir', str(logdir),
      '--run.from_checkpoint', str(checkpoint),
      '--run.eval_episodes', str(int(episodes)),
      '--run.eval_policy_mode', policy_mode,
      '--env.rlscape.episode_length', str(int(episode_length)),
      '--env.rlscape.grid_columns', str(int(grid_columns)),
      '--env.rlscape.grid_rows', str(int(grid_rows)),
      '--eval_adapt.enabled', str(bool(condition['enabled'])),
      '--eval_adapt.train_critic', 'False',
      '--eval_adapt.objective', str(condition['objective']),
      '--eval_adapt.steps', str(int(adapt_steps)),
      '--eval_adapt.imag_length', str(max(1, int(condition['horizon']))),
      '--eval_adapt.lr', str(float(adapt_lr)),
      '--eval_adapt.every_k', str(int(adapt_every_k)),
      '--eval_adapt.actent', str(float(condition['actent'])),
      '--eval_adapt.start_batch', str(int(start_batch)),
  ]
  if condition['objective'] == 'distill':
    command += [
        '--eval_adapt.reference_checkpoint', str(reference_checkpoint)]
  if mvn_path:
    command += ['--env.rlscape.mvn_path', mvn_path]
  if java_home:
    command += ['--env.rlscape.java_home', java_home]
  return command


def reset_identities(logdir: pathlib.Path) -> list[dict[str, Any]]:
  return [{
      'goal': row.get('goal'),
      'reset_seed': row.get('reset_seed'),
  } for row in read_jsonl(logdir / 'audit_env0.jsonl')
          if row.get('kind') == 'reset']


def summarize_condition(
    logdir: pathlib.Path, *, condition: dict[str, Any], goal: str,
    policy_mode: str, episodes: int, seed: int,
    returncode: int | None,
) -> dict[str, Any]:
  summary = summarize_run(
      logdir, goal=goal, seed=seed, requested_episodes=episodes,
      returncode=returncode)
  integrity_path = logdir / 'eval_integrity.json'
  integrity = (
      json.loads(integrity_path.read_text())
      if integrity_path.is_file() else {})
  resets = reset_identities(logdir)
  metrics_rows = read_jsonl(logdir / 'metrics.jsonl')
  diagnostic_keys = {
      'adapt_triggers': 'epstats/log/eval_adapt/trigger/sum',
      'actor_steps': 'epstats/log/eval_adapt/actor_steps/sum',
      'actor_loss': 'epstats/log/eval_adapt/adapt_actor_opt/loss/avg',
      'actor_grad_norm': (
          'epstats/log/eval_adapt/adapt_actor_opt/grad_norm/avg'),
      'actor_update_rms': (
          'epstats/log/eval_adapt/adapt_actor_opt/update_rms/avg'),
  }
  diagnostics = {}
  for name, key in diagnostic_keys.items():
    values = [float(row[key]) for row in metrics_rows if key in row]
    diagnostics[f'adapt_{name}_mean'] = (
        sum(values) / len(values) if values else float('nan'))
  summary.update({
      'condition': condition['name'],
      'family': condition['family'],
      'checkpoint_kind': condition['checkpoint'],
      'objective': condition['objective'],
      'horizon': int(condition['horizon']),
      'actent': float(condition['actent']),
      'policy_mode': policy_mode,
      'integrity_equal': bool(integrity.get('equal', False)),
      'reset_count': len(resets),
      'logdir': str(logdir),
      **diagnostics,
  })
  summary['complete'] = bool(
      returncode in (None, 0) and summary['episode_count_exact'] and
      summary['integrity_equal'] and len(resets) == episodes)
  return summary


def _heartbeat(args, snapshot):
  args.queue['active_child'] = snapshot
  args.queue['updated_unix'] = time.time()
  write_json(args.queue_path, args.queue)


def execute_unit(
    args, *, states: dict[str, Any], unit_id: str,
    root: pathlib.Path, checkpoint: pathlib.Path,
    reference_checkpoint: pathlib.Path, condition: dict[str, Any],
    goal: str, policy_mode: str, episodes: int,
) -> dict[str, Any]:
  state = states.setdefault(unit_id, {'status': 'pending', 'attempts': []})
  if state.get('status') == 'complete':
    saved = state['attempts'][-1]
    recovered = summarize_condition(
        pathlib.Path(saved['logdir']), condition=condition, goal=goal,
        policy_mode=policy_mode, episodes=episodes, seed=args.eval_seed,
        returncode=0)
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
    command = build_command(
        python=args.python, repo_root=args.repo_root, logdir=logdir,
        checkpoint=checkpoint, reference_checkpoint=reference_checkpoint,
        condition=condition, goal=goal, policy_mode=policy_mode,
        episodes=episodes, seed=args.eval_seed,
        episode_length=args.episode_length, grid_columns=args.grid_columns,
        grid_rows=args.grid_rows, adapt_steps=args.adapt_steps,
        adapt_lr=args.adapt_lr, adapt_every_k=args.adapt_every_k,
        start_batch=args.start_batch, mvn_path=args.mvn_path,
        java_home=args.java_home)
    recovered = summarize_condition(
        logdir, condition=condition, goal=goal, policy_mode=policy_mode,
        episodes=episodes, seed=args.eval_seed, returncode=0)
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
    command = build_command(
        python=args.python, repo_root=args.repo_root, logdir=logdir,
        checkpoint=checkpoint, reference_checkpoint=reference_checkpoint,
        condition=condition, goal=goal, policy_mode=policy_mode,
        episodes=episodes, seed=args.eval_seed,
        episode_length=args.episode_length, grid_columns=args.grid_columns,
        grid_rows=args.grid_rows, adapt_steps=args.adapt_steps,
        adapt_lr=args.adapt_lr, adapt_every_k=args.adapt_every_k,
        start_batch=args.start_batch, mvn_path=args.mvn_path,
        java_home=args.java_home)
    attempt = {
        'attempt': attempt_number,
        'status': 'running',
        'started_unix': time.time(),
        'logdir': str(logdir),
        'command': command,
    }
    state['attempts'].append(attempt)
  state['status'] = 'running'
  args.queue['updated_unix'] = time.time()
  write_json(args.queue_path, args.queue)

  print(f'\n=== Stage 2 {unit_id}, attempt {attempt_number} ===', flush=True)
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
  returncode = child['returncode']
  summary = summarize_condition(
      logdir, condition=condition, goal=goal, policy_mode=policy_mode,
      episodes=episodes, seed=args.eval_seed, returncode=returncode)
  attempt.update(
      status='complete' if summary['complete'] else 'failed',
      finished_unix=time.time(), child=child, summary=summary)
  state['status'] = attempt['status']
  args.queue['updated_unix'] = time.time()
  write_json(args.queue_path, args.queue)
  if not summary['complete']:
    if attempt_number < args.max_attempts:
      print(
          f'Retrying incomplete Stage 2 unit {unit_id} '
          f'({attempt_number}/{args.max_attempts})', flush=True)
      return execute_unit(
          args, states=states, unit_id=unit_id, root=root,
          checkpoint=checkpoint, reference_checkpoint=reference_checkpoint,
          condition=condition, goal=goal, policy_mode=policy_mode,
          episodes=episodes)
    raise RuntimeError(
        f'Incomplete Stage 2 unit {unit_id}: {logdir / "console.log"}')
  return summary


def ensure_corruptions(
    source: pathlib.Path, output_root: pathlib.Path,
    strengths: tuple[float, ...], seed: int,
) -> list[tuple[str, pathlib.Path, dict[str, Any]]]:
  root = output_root / 'corruptions'
  root.mkdir(parents=True, exist_ok=True)
  base = corrupt.load_agent_checkpoint(source)
  variants = []
  for strength in strengths:
    name = f'zero_mask_all_s{corrupt.fmt_strength(strength)}_seed{seed}'
    target = root / name
    expected = {
        'source_checkpoint': str(source), 'target': 'actor',
        'method': 'zero_mask', 'scope': 'all',
        'strength': float(strength), 'seed': int(seed),
    }
    manifest_path = target / 'manifest.json'
    if manifest_path.is_file() and (target / 'agent.pkl').is_file():
      manifest = json.loads(manifest_path.read_text())
      mismatch = {
          key: (manifest.get(key), value) for key, value in expected.items()
          if manifest.get(key) != value}
      if mismatch:
        raise RuntimeError(
            f'Existing corruption has a different specification: '
            f'{target}: {mismatch}')
    elif target.exists():
      raise RuntimeError(f'Incomplete corruption directory: {target}')
    else:
      spec = dict(
          name=name, method='zero_mask', scope='all', strength=strength)
      manifest = corrupt.save_variant(
          source, target, corrupt.clone_agent_checkpoint(base),
          spec, seed, 'actor')
    variants.append((name, target, manifest))
  return variants


def select_corruption(
    rows: list[dict[str, Any]], candidate_names: list[str],
    *, target_retention: float, retention_min: float,
    retention_max: float, max_goal_retention: float = .85,
) -> dict[str, Any]:
  clean_rows = [row for row in rows if row['condition'] == 'clean']
  if {row['goal'] for row in clean_rows} != set(GOALS):
    raise ValueError('Calibration is missing clean results for one or more goals')
  clean_by_goal = {row['goal']: float(row['success_rate']) for row in clean_rows}
  clean_mean = sum(clean_by_goal.values()) / len(clean_by_goal)
  if clean_mean <= 0:
    raise RuntimeError('Clean calibration performance is zero')

  candidates = []
  for name in candidate_names:
    selected = [row for row in rows if row['condition'] == name]
    if {row['goal'] for row in selected} != set(GOALS):
      raise ValueError(f'Calibration is incomplete for {name}')
    by_goal = {row['goal']: float(row['success_rate']) for row in selected}
    mean = sum(by_goal.values()) / len(by_goal)
    retention = mean / clean_mean
    goal_retentions = [
        by_goal[goal] / max(clean_by_goal[goal], 1e-8) for goal in GOALS]
    improved = sum(
        max(0.0, by_goal[goal] - clean_by_goal[goal]) for goal in GOALS)
    spread = (
        sum((value - sum(goal_retentions) / len(goal_retentions)) ** 2
            for value in goal_retentions) / len(goal_retentions)) ** .5
    qualified = bool(
        retention_min <= retention <= retention_max and improved <= .10 and
        max(goal_retentions) <= max_goal_retention)
    rank_score = (
        abs(retention - target_retention) + .20 * spread + 2.0 * improved +
        (0.0 if qualified else 10.0))
    candidates.append({
        'name': name, 'success_rate': mean, 'retention': retention,
        'goal_success': by_goal, 'goal_retention': dict(zip(
            GOALS, goal_retentions)), 'improvement_penalty': improved,
        'retention_spread': spread, 'qualified': qualified,
        'rank_score': rank_score,
    })
  candidates.sort(key=lambda item: item['rank_score'])
  return {
      'clean_success_rate': clean_mean,
      'clean_goal_success': clean_by_goal,
      'target_retention': target_retention,
      'retention_range': [retention_min, retention_max],
      'max_goal_retention': max_goal_retention,
      'selected': candidates[0],
      'candidates': candidates,
  }


def validate_pairing(rows: list[dict[str, Any]]) -> dict[str, Any]:
  groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
  for row in rows:
    groups.setdefault((row['goal'], row['policy_mode']), []).append(row)
  mismatches = []
  for key, group in groups.items():
    reference = reset_identities(pathlib.Path(group[0]['logdir']))
    for row in group[1:]:
      current = reset_identities(pathlib.Path(row['logdir']))
      if current != reference:
        mismatches.append({
            'goal': key[0], 'policy_mode': key[1],
            'condition': row['condition'],
            'reference_condition': group[0]['condition'],
            'reference_count': len(reference), 'current_count': len(current),
        })
  return {'equal': not mismatches, 'mismatches': mismatches}


def episode_outcomes(logdir: pathlib.Path) -> np.ndarray:
  rows = read_jsonl(logdir / 'episodes.jsonl')
  return np.asarray([
      bool(
          float(row.get('episode/score', 0.0)) > 0 or
          float(row.get('episode/log/task_success/max', 0.0)) > 0)
      for row in rows], np.float64)


def _paired_intervals(
    clean: np.ndarray, corrupt_values: np.ndarray, values: np.ndarray,
    *, seed: int, samples: int = 10_000,
) -> dict[str, float]:
  if not (len(clean) == len(corrupt_values) == len(values)) or not len(values):
    raise ValueError((len(clean), len(corrupt_values), len(values)))
  difference = values - corrupt_values
  denominator = clean.mean() - corrupt_values.mean()
  rng = np.random.default_rng(seed)
  indices = rng.integers(0, len(values), size=(samples, len(values)))
  delta_samples = difference[indices].mean(1)
  clean_samples = clean[indices].mean(1)
  corrupt_samples = corrupt_values[indices].mean(1)
  value_samples = values[indices].mean(1)
  boot_denominator = clean_samples - corrupt_samples
  valid = np.abs(boot_denominator) >= .05
  recovery_samples = (
      (value_samples[valid] - corrupt_samples[valid]) /
      boot_denominator[valid])
  result = {
      'paired_delta_vs_corrupt': float(difference.mean()),
      'paired_delta_vs_corrupt_ci_low': float(np.quantile(
          delta_samples, .025)),
      'paired_delta_vs_corrupt_ci_high': float(np.quantile(
          delta_samples, .975)),
      'paired_discordant_improved': int(
          ((values == 1) & (corrupt_values == 0)).sum()),
      'paired_discordant_harmed': int(
          ((values == 0) & (corrupt_values == 1)).sum()),
  }
  if abs(denominator) >= .05 and len(recovery_samples):
    result.update({
        'recovery_fraction_ci_low': float(np.quantile(
            recovery_samples, .025)),
        'recovery_fraction_ci_high': float(np.quantile(
            recovery_samples, .975)),
        'recovery_bootstrap_valid_rate': float(valid.mean()),
    })
  else:
    result.update({
        'recovery_fraction_ci_low': float('nan'),
        'recovery_fraction_ci_high': float('nan'),
        'recovery_bootstrap_valid_rate': float(valid.mean()),
    })
  return result


def comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  lookup = {
      (row['goal'], row['policy_mode'], row['condition']): row
      for row in rows}
  output = []
  for row in rows:
    clean = lookup[(row['goal'], row['policy_mode'], 'clean_no_adapt')]
    damaged = lookup[(row['goal'], row['policy_mode'], 'corrupt_no_adapt')]
    clean_score = float(clean['success_rate'])
    corrupt_score = float(damaged['success_rate'])
    score = float(row['success_rate'])
    denominator = clean_score - corrupt_score
    recovery = (
        (score - corrupt_score) / denominator
        if abs(denominator) >= .05 else float('nan'))
    seed = int(spec_digest({
        'goal': row['goal'], 'mode': row['policy_mode'],
        'condition': row['condition']})[:8], 16)
    paired = _paired_intervals(
        episode_outcomes(pathlib.Path(clean['logdir'])),
        episode_outcomes(pathlib.Path(damaged['logdir'])),
        episode_outcomes(pathlib.Path(row['logdir'])), seed=seed)
    output.append({
        **row,
        'clean_success_rate': clean_score,
        'corrupt_success_rate': corrupt_score,
        'delta_vs_corrupt': score - corrupt_score,
        'delta_vs_clean': score - clean_score,
        'recovery_fraction': recovery,
        **paired,
    })
  return output


def make_figures(output_root: pathlib.Path, rows: list[dict[str, Any]]) -> None:
  import matplotlib.pyplot as plt

  figure_root = output_root / 'figures'
  figure_root.mkdir(exist_ok=True)
  plt.rcParams.update({
      'font.size': 9, 'axes.spines.top': False,
      'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .2})
  families = (
      ('standard_ir', 'Standard IR', '#4c78a8'),
      ('entropy_off_ir', 'Entropy-off IR', '#f58518'),
      ('reward_only_ir', 'Reward-only IR', '#54a24b'),
  )
  horizons = sorted({
      int(row['horizon']) for row in rows if int(row['horizon']) > 0})

  for field, filename, ylabel in (
      ('success_rate', 'success_by_horizon', 'Success rate'),
      ('recovery_fraction', 'normalized_recovery_by_horizon',
       'Fraction of clean-to-corrupt gap recovered'),
  ):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for row_index, mode in enumerate(POLICY_MODES):
      for col_index, goal in enumerate(GOALS):
        ax = axes[row_index, col_index]
        subset = [
            row for row in rows
            if row['policy_mode'] == mode and row['goal'] == goal]
        clean = next(row for row in subset
                     if row['condition'] == 'clean_no_adapt')
        damaged = next(row for row in subset
                       if row['condition'] == 'corrupt_no_adapt')
        if field == 'success_rate':
          ax.axhline(float(clean[field]), color='black', ls='--', lw=1,
                     label='Clean')
          ax.axhline(float(damaged[field]), color='gray', ls=':', lw=1,
                     label='Corrupt')
        else:
          ax.axhline(1.0, color='black', ls='--', lw=1, label='Clean')
          ax.axhline(0.0, color='gray', ls=':', lw=1, label='Corrupt')
        for family, label, color in families:
          values = {int(row['horizon']): float(row[field]) for row in subset
                    if row['family'] == family}
          ax.plot(horizons, [values.get(h, math.nan) for h in horizons],
                  marker='o', label=label, color=color)
        distill = next(row for row in subset
                       if row['condition'] == 'reference_distill')
        ax.axhline(float(distill[field]), color='#b279a2', ls='-.', lw=1,
                   label='Reference distillation')
        ax.set_title(f'{goal.replace("_", " ")} — {mode}')
        ax.set_xticks(horizons)
        if col_index == 0:
          ax.set_ylabel(ylabel)
        if row_index == 1:
          ax.set_xlabel('Imagination horizon H')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .92))
    for suffix in ('png', 'pdf'):
      fig.savefig(figure_root / f'{filename}.{suffix}', dpi=180)
    plt.close(fig)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Run Stage 2 RLScape actor corruption and recovery.')
  parser.add_argument('--experiment-root', type=pathlib.Path, required=True)
  parser.add_argument('--strong-step', type=int, default=1_250_000)
  parser.add_argument('--output-root', type=pathlib.Path, default=None)
  parser.add_argument('--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument('--python', type=pathlib.Path,
                      default=pathlib.Path(sys.executable))
  parser.add_argument('--horizons', default='3,6,15,20')
  parser.add_argument('--corruption-strengths', default='.1,.2,.3,.4,.5,.6')
  parser.add_argument('--corruption-seed', type=int, default=0)
  parser.add_argument('--corruption-checkpoint', type=pathlib.Path, default=None)
  parser.add_argument('--target-retention', type=float, default=.35)
  parser.add_argument('--retention-min', type=float, default=.10)
  parser.add_argument('--retention-max', type=float, default=.65)
  parser.add_argument('--allow-unqualified-corruption', action='store_true')
  parser.add_argument('--calibration-episodes', type=int, default=20)
  parser.add_argument('--sampled-episodes', type=int, default=100)
  parser.add_argument('--deterministic-episodes', type=int, default=50)
  parser.add_argument('--eval-seed', type=int, default=2000)
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
      args.experiment_root / 'stage2_actor_recovery')
  if args.corruption_checkpoint:
    args.corruption_checkpoint = args.corruption_checkpoint.resolve()
  args.horizons = parse_ints(args.horizons)
  args.corruption_strengths = parse_floats(args.corruption_strengths)
  if args.horizons != DEFAULT_HORIZONS:
    print(f'Non-default Stage 2 horizons requested: {args.horizons}')
  if not args.python.is_file():
    raise FileNotFoundError(args.python)
  if any(value <= 0 for value in (
      args.calibration_episodes, args.sampled_episodes,
      args.deterministic_episodes, args.episode_length,
      args.grid_columns, args.grid_rows, args.adapt_steps,
      args.start_batch, args.max_attempts)):
    raise ValueError('Episode, grid, adaptation, and attempt counts must be positive')
  if args.adapt_every_k < 0 or args.adapt_lr <= 0:
    raise ValueError('Invalid adaptation schedule')
  if not 0 <= args.retention_min <= args.target_retention <= args.retention_max:
    raise ValueError('Target retention must lie inside the accepted range')


def main(argv=None) -> int:
  args = parse_args(argv)
  validate_args(args)
  source = resolve_milestone(args.experiment_root, args.strong_step) / 'checkpoint'
  args.output_root.mkdir(parents=True, exist_ok=True)
  conditions = build_conditions(args.horizons)
  spec = {
      'format': 1,
      'experiment_root': str(args.experiment_root),
      'source_checkpoint': str(source),
      'strong_step': args.strong_step,
      'goals': list(GOALS),
      'horizons': list(args.horizons),
      'conditions': conditions,
      'corruption': {
          'method': 'zero_mask', 'scope': 'all', 'target': 'actor',
          'strengths': list(args.corruption_strengths),
          'seed': args.corruption_seed,
          'provided_checkpoint': (
              str(args.corruption_checkpoint)
              if args.corruption_checkpoint else ''),
          'target_retention': args.target_retention,
          'retention_range': [args.retention_min, args.retention_max],
          'max_goal_retention': .85,
      },
      'episodes': {
          'calibration': args.calibration_episodes,
          'sampled': args.sampled_episodes,
          'deterministic': args.deterministic_episodes,
      },
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
          'episode_local': True,
      },
      'pairing': 'same_goal_mode_reset_seed_across_all_conditions',
      'future_conditioning_repair': 'counterfactual_goal_training',
  }
  digest = spec_digest(spec)
  spec_path = args.output_root / 'stage2_spec.json'
  if spec_path.is_file():
    existing_spec = json.loads(spec_path.read_text())
    if spec_digest(existing_spec) != digest:
      raise RuntimeError(f'Existing Stage 2 specification differs: {spec_path}')
  else:
    write_json(spec_path, spec)

  queue_path = args.output_root / 'queue.json'
  existing = json.loads(queue_path.read_text()) if queue_path.is_file() else {}
  if existing.get('spec_digest') not in (None, digest):
    raise RuntimeError(f'Existing Stage 2 queue differs: {queue_path}')
  queue = {
      'format': 1, 'spec_digest': digest,
      'created_unix': existing.get('created_unix', time.time()),
      'updated_unix': time.time(), 'status': 'planned' if args.dry_run else 'running',
      'calibration_states': existing.get('calibration_states', {}),
      'main_states': existing.get('main_states', {}),
  }
  args.queue = queue
  args.queue_path = queue_path
  write_json(queue_path, queue)

  if args.dry_run:
    print(
        f'Stage 2 plan: {len(conditions)} conditions x {len(GOALS)} goals '
        f'x {len(POLICY_MODES)} modes = '
        f'{len(conditions) * len(GOALS) * len(POLICY_MODES)} main units.')
    print(f'Horizons: {args.horizons}')
    print(f'Specification: {spec_path}')
    return 0

  if args.corruption_checkpoint:
    selected_checkpoint = args.corruption_checkpoint.resolve()
    manifest_path = selected_checkpoint / 'manifest.json'
    if not manifest_path.is_file():
      raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('source_checkpoint') != str(source):
      raise RuntimeError('Provided corruption does not derive from strong checkpoint')
    selection = {
        'provided': True, 'selected': {
            'name': selected_checkpoint.name,
            'checkpoint': str(selected_checkpoint),
            'manifest': manifest}}
  else:
    variants = ensure_corruptions(
        source, args.output_root, args.corruption_strengths,
        args.corruption_seed)
    calibration_conditions = [dict(
        name='clean', family='calibration', checkpoint='clean',
        enabled=False, objective='standard', horizon=0, actent=0.0)]
    calibration_conditions += [dict(
        name=name, family='calibration', checkpoint='corrupt',
        enabled=False, objective='standard', horizon=0, actent=0.0)
        for name, _, _ in variants]
    checkpoint_by_name = {'clean': source, **{
        name: path for name, path, _ in variants}}
    calibration_rows = []
    for condition in calibration_conditions:
      for goal in GOALS:
        unit_id = f'calibration__{condition["name"]}__{goal}'
        summary = execute_unit(
            args, states=queue['calibration_states'], unit_id=unit_id,
            root=args.output_root / 'calibration' / condition['name'] / goal,
            checkpoint=checkpoint_by_name[condition['name']],
            reference_checkpoint=source, condition=condition, goal=goal,
            policy_mode='sampled', episodes=args.calibration_episodes)
        calibration_rows.append(summary)
        write_csv(args.output_root / 'calibration_results.csv', calibration_rows)
    calibration_pairing = validate_pairing(calibration_rows)
    write_json(args.output_root / 'calibration_pairing.json', calibration_pairing)
    if not calibration_pairing['equal']:
      raise RuntimeError('Calibration reset identities are not paired')
    selection = select_corruption(
        calibration_rows, [name for name, _, _ in variants],
        target_retention=args.target_retention,
        retention_min=args.retention_min,
        retention_max=args.retention_max)
    selected_name = selection['selected']['name']
    selected_checkpoint = checkpoint_by_name[selected_name]
    selection['selected']['checkpoint'] = str(selected_checkpoint)
    selection['selected']['manifest'] = next(
        manifest for name, _, manifest in variants if name == selected_name)
    if (
        not selection['selected']['qualified'] and
        not args.allow_unqualified_corruption
    ):
      write_json(args.output_root / 'corruption_selection.json', selection)
      raise RuntimeError(
          'No corruption met the qualification range; inspect '
          f'{args.output_root / "corruption_selection.json"} or pass '
          '--allow-unqualified-corruption explicitly.')
  write_json(args.output_root / 'corruption_selection.json', selection)

  episode_counts = {
      'sampled': args.sampled_episodes,
      'deterministic': args.deterministic_episodes,
  }
  main_rows = []
  for condition in conditions:
    checkpoint = source if condition['checkpoint'] == 'clean' else selected_checkpoint
    for goal in GOALS:
      for mode in POLICY_MODES:
        unit_id = f'main__{condition["name"]}__{goal}__{mode}'
        summary = execute_unit(
            args, states=queue['main_states'], unit_id=unit_id,
            root=(args.output_root / 'main' / condition['name'] / goal / mode),
            checkpoint=checkpoint, reference_checkpoint=source,
            condition=condition, goal=goal, policy_mode=mode,
            episodes=episode_counts[mode])
        main_rows.append(summary)
        write_csv(args.output_root / 'main_results.csv', main_rows)

  pairing = validate_pairing(main_rows)
  write_json(args.output_root / 'main_pairing.json', pairing)
  if not pairing['equal']:
    raise RuntimeError('Main reset identities are not paired across conditions')
  compared = comparison_rows(main_rows)
  write_csv(args.output_root / 'comparison_vs_baselines.csv', compared)
  make_figures(args.output_root, compared)
  queue.update(status='complete', updated_unix=time.time())
  write_json(queue_path, queue)
  (args.output_root / 'stage2_complete').touch()
  print(f'Stage 2 complete: {args.output_root}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
