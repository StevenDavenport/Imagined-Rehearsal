#!/usr/bin/env python3
"""Generate and evaluate corruption candidates for paper baselines."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


TASKS = {
    'dmc_cup_catch': {
        'configs': 'dmc_vision',
        'checkpoint': 'checkpoints/paper_baselines/dmc_cup_catch',
    },
    'dmc_cartpole_swingup': {
        'configs': 'dmc_proprio',
        'checkpoint': 'checkpoints/paper_baselines/dmc_cartpole_swingup',
    },
    'dmc_cheetah_run': {
        'configs': 'dmc_vision',
        'checkpoint': 'checkpoints/paper_baselines/dmc_cheetah_run',
    },
    'dmc_finger_turn_hard': {
        'configs': 'dmc_vision',
        'checkpoint': 'checkpoints/paper_baselines/dmc_finger_turn_hard',
    },
    'dmc_reacher_hard': {
        'configs': 'dmc_vision',
        'checkpoint': 'checkpoints/paper_baselines/dmc_reacher_hard',
    },
    'dmc_walker_run': {
        'configs': 'dmc_vision',
        'checkpoint': 'checkpoints/paper_baselines/dmc_walker_run',
    },
    'dmc_walker_walk': {
        'configs': 'dmc_vision',
        'checkpoint': 'checkpoints/paper_baselines/dmc_walker_walk',
    },
    'dmc_quadruped_run': {
        'configs': 'dmc_vision',
        'checkpoint': 'checkpoints/paper_baselines/dmc_quadruped_run',
    },
    'dmc_hopper_hop': {
        'configs': 'dmc_vision',
        'checkpoint': 'checkpoints/paper_baselines/dmc_hopper_hop',
    },
    'dmc_fish_swim': {
        'configs': 'dmc_vision',
        'checkpoint': 'checkpoints/paper_baselines/dmc_fish_swim',
    },
}

DEFAULT_STRENGTHS = (
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90, 0.95,
)


def parse_csv(text: str) -> list[str]:
  return [x.strip() for x in text.split(',') if x.strip()]


def fmt_strength(value: float) -> str:
  text = f'{value:.3f}'.rstrip('0').rstrip('.')
  return text.replace('.', 'p')


def candidate_seed(strength: float, fallback_index: int, corrupt_seed: int) -> int:
  for index, default_strength in enumerate(DEFAULT_STRENGTHS):
    if abs(strength - default_strength) < 1e-9:
      return corrupt_seed + index
  return corrupt_seed + fallback_index


def candidate_name(family: str, strength: float, seed: int) -> str:
  return f'{family}_all_s{fmt_strength(strength)}_seed{seed}'


def run_command(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
) -> int:
  print(' '.join(shlex.quote(x) for x in cmd))
  if dry_run:
    return 0
  return subprocess.run(cmd, cwd=cwd, env=env, check=False).returncode


def read_summary(path: Path) -> list[dict[str, str]]:
  if not path.exists():
    return []
  with path.open(newline='') as f:
    return list(csv.DictReader(f))


def write_calibration_summary(task_outdir: Path) -> None:
  rows = read_summary(task_outdir / 'summary_by_checkpoint.csv')
  if not rows:
    return

  keep = []
  for row in rows:
    keep.append({
        'checkpoint_name': row.get('checkpoint_name', ''),
        'strength': row.get('corruption_strength', ''),
        'relative_l2': row.get('corruption_relative_l2', ''),
        'runs_ok': row.get('runs_ok', ''),
        'episodes_total': row.get('episodes_total', ''),
        'score_mean': row.get('score_mean_weighted_by_episodes', ''),
        'score_ci95': row.get('score_mean_ci95', ''),
        'fps_policy': row.get('fps_policy_mean_mean', ''),
    })

  keep.sort(key=lambda x: float(x['strength'] or '0'))
  with (task_outdir / 'calibration_summary.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(keep[0]))
    writer.writeheader()
    writer.writerows(keep)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--repo_root', type=Path, default=Path('.'))
  parser.add_argument(
      '--tasks',
      default=','.join([
          'dmc_walker_walk',
          'dmc_walker_run',
          'dmc_cheetah_run',
          'dmc_cartpole_swingup',
          'dmc_reacher_hard',
          'dmc_finger_turn_hard',
          'dmc_cup_catch',
      ]),
      help='Comma-separated task names, or "all".')
  parser.add_argument('--family', default='zero_mask')
  parser.add_argument(
      '--strengths',
      default=','.join(f'{x:.2f}' for x in DEFAULT_STRENGTHS),
      help='Comma-separated corruption strengths to generate/evaluate.')
  parser.add_argument('--seeds', default='0')
  parser.add_argument('--corrupt_seed', type=int, default=0)
  parser.add_argument('--run_steps', type=int, default=5000)
  parser.add_argument('--run_envs', type=int, default=1)
  parser.add_argument('--run_log_every', type=int, default=50)
  parser.add_argument('--jax_platform', default='cuda')
  parser.add_argument('--mujoco_gl', default='egl')
  parser.add_argument('--python_bin', default=sys.executable)
  parser.add_argument('--generate_only', action='store_true')
  parser.add_argument('--eval_only', action='store_true')
  parser.add_argument('--dry_run', action='store_true')
  parser.add_argument('--force_corruptions', action='store_true')
  parser.add_argument('--skip_existing', action='store_true')
  parser.add_argument('--continue_on_error', action='store_true')
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  repo = args.repo_root.resolve()
  task_names = sorted(TASKS) if args.tasks == 'all' else parse_csv(args.tasks)
  strengths = [float(x) for x in parse_csv(args.strengths)]
  seeds = parse_csv(args.seeds)

  unknown = [x for x in task_names if x not in TASKS]
  if unknown:
    raise ValueError(f'Unknown tasks: {unknown}. Known: {sorted(TASKS)}')

  env = os.environ.copy()
  if args.mujoco_gl:
    env['MUJOCO_GL'] = args.mujoco_gl

  plan = {
      'tasks': task_names,
      'family': args.family,
      'strengths': strengths,
      'eval_seeds': seeds,
      'corrupt_seed': args.corrupt_seed,
      'run_steps': args.run_steps,
      'run_envs': args.run_envs,
      'jax_platform': args.jax_platform,
      'mujoco_gl': args.mujoco_gl,
  }
  plan_path = repo / 'eval_suite/paper/corruption_calibration/calibration_plan.json'
  plan_path.parent.mkdir(parents=True, exist_ok=True)
  plan_path.write_text(json.dumps(plan, indent=2))

  for task in task_names:
    meta = TASKS[task]
    corrupt_root = repo / 'checkpoints/paper_corruptions' / task / args.family / 'candidates'
    eval_outdir = repo / 'eval_suite/paper/corruption_calibration' / task / args.family
    corrupt_root.mkdir(parents=True, exist_ok=True)
    eval_outdir.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
      for idx, strength in enumerate(strengths):
        seed = candidate_seed(strength, idx, args.corrupt_seed)
        name = candidate_name(args.family, strength, seed)
        outdir = corrupt_root / name
        if outdir.exists() and not args.force_corruptions:
          print(f'Skip existing corruption: {outdir}')
          continue
        cmd = [
            args.python_bin,
            'scripts/corrupt_actor_checkpoint.py',
            '--source', meta['checkpoint'],
            '--out_root', str(corrupt_root),
            '--method', args.family,
            '--scope', 'all',
            '--strength', str(strength),
            '--seed', str(seed),
            '--name', name,
        ]
        if args.force_corruptions:
          cmd.append('--force')
        code = run_command(cmd, repo, env, args.dry_run)
        if code != 0:
          return code

    if args.generate_only:
      continue

    candidate_paths = []
    missing = []
    for idx, strength in enumerate(strengths):
      seed = candidate_seed(strength, idx, args.corrupt_seed)
      path = corrupt_root / candidate_name(args.family, strength, seed)
      if (path / 'done').exists():
        candidate_paths.append(str(path))
      else:
        missing.append(str(path))
    if missing:
      raise FileNotFoundError(
          'Missing candidate checkpoints. Generate them first or remove '
          f'these strengths from --strengths:\n' + '\n'.join(missing))

    cmd = [
        args.python_bin,
        'scripts/checkpoint_eval_sweep.py',
        '--repo_root', str(repo),
        '--outdir', str(eval_outdir),
        '--checkpoints', ','.join(candidate_paths),
        '--configs', meta['configs'],
        '--task', task,
        '--seeds', ','.join(seeds),
        '--run_steps', str(args.run_steps),
        '--run_envs', str(args.run_envs),
        '--run_log_every', str(args.run_log_every),
        '--jax_platform', args.jax_platform,
        '--logger_filter', 'score|length|fps',
    ]
    if args.skip_existing:
      cmd.append('--skip_existing')
    if args.continue_on_error:
      cmd.append('--continue_on_error')
    code = run_command(cmd, repo, env, args.dry_run)
    if code != 0 and not args.continue_on_error:
      return code
    if not args.dry_run:
      write_calibration_summary(eval_outdir)

  print(f'Wrote plan: {plan_path}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
