#!/usr/bin/env python3
"""Evaluate one RLScape checkpoint for an exact episode count per goal."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.checkpoint_utils import resolve_checkpoint_path  # noqa: E402
from scripts.rlscape_m1_pilot import GOALS  # noqa: E402
from scripts.rlscape_m1_pilot import read_jsonl  # noqa: E402
from scripts.rlscape_m1_pilot import spec_digest  # noqa: E402
from scripts.rlscape_m1_pilot import write_csv  # noqa: E402
from scripts.rlscape_m1_pilot import write_json  # noqa: E402


def build_command(
    *,
    python: pathlib.Path,
    repo_root: pathlib.Path,
    logdir: pathlib.Path,
    checkpoint: pathlib.Path,
    goal: str,
    seed: int,
    episodes: int,
    episode_length: int,
    action_interface: str = 'mixed',
    grid_columns: int = 28,
    grid_rows: int = 18,
    mvn_path: str = '',
    java_home: str = '',
) -> list[str]:
  configs = ['rlscape']
  if action_interface == 'click_grid':
    configs.append('rlscape_click_grid')
  elif action_interface != 'mixed':
    raise ValueError(f'Unknown action interface {action_interface!r}')
  configs.extend((f'rlscape_{goal}', 'rlscape_m2_eval'))
  command = [
      str(python),
      str(repo_root / 'dreamerv3' / 'main.py'),
      '--configs', *configs,
      '--seed', str(seed),
      '--logdir', str(logdir),
      '--run.from_checkpoint', str(checkpoint),
      '--run.eval_episodes', str(episodes),
      '--env.rlscape.episode_length', str(episode_length),
  ]
  if action_interface == 'click_grid':
    command += [
        '--env.rlscape.grid_columns', str(grid_columns),
        '--env.rlscape.grid_rows', str(grid_rows),
    ]
  if mvn_path:
    command += ['--env.rlscape.mvn_path', mvn_path]
  if java_home:
    command += ['--env.rlscape.java_home', java_home]
  return command


def summarize_run(
    logdir: pathlib.Path,
    *,
    goal: str,
    seed: int,
    requested_episodes: int,
    returncode: int | None = None,
) -> dict[str, Any]:
  rows = read_jsonl(logdir / 'episodes.jsonl')
  scores = [float(row.get('episode/score', 0.0)) for row in rows]
  successes = [
      bool(score > 0 or row.get('episode/log/task_success/max', 0) > 0)
      for row, score in zip(rows, scores)
  ]
  lengths = [max(0.0, float(row['episode/length']) - 1) for row in rows]
  deaths = [
      bool(row.get('episode/log/player_death/max', 0) > 0) for row in rows
  ]
  result = {
      'goal': goal,
      'seed': int(seed),
      'returncode': returncode,
      'requested_episodes': int(requested_episodes),
      'episodes': len(rows),
      'episode_count_exact': len(rows) == requested_episodes,
      'successes': int(sum(successes)),
      'success_rate': sum(successes) / len(rows) if rows else 0.0,
      'deaths': int(sum(deaths)),
      'mean_action_length': sum(lengths) / len(lengths) if lengths else 0.0,
      'median_action_length': statistics.median(lengths) if lengths else 0.0,
      'last_step': max(
          (int(row.get('step', 0)) for row in rows), default=0),
  }
  return result


def parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description='Run frozen deterministic RLScape evaluation by goal.')
  parser.add_argument('--checkpoint', type=pathlib.Path, required=True)
  parser.add_argument('--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument(
      '--python', type=pathlib.Path, default=pathlib.Path(sys.executable))
  parser.add_argument(
      '--log-root', type=pathlib.Path,
      default=ROOT / 'logs' / 'rlscape' / 'm2_multitask_eval')
  parser.add_argument('--goals', nargs='+', choices=GOALS, default=list(GOALS))
  parser.add_argument('--seed', type=int, default=1000)
  parser.add_argument('--episodes', type=int, default=20)
  parser.add_argument('--episode-length', type=int, default=200)
  parser.add_argument(
      '--action-interface', choices=('mixed', 'click_grid'), default='mixed')
  parser.add_argument('--grid-columns', type=int, default=28)
  parser.add_argument('--grid-rows', type=int, default=18)
  parser.add_argument(
      '--mvn-path', default=os.environ.get(
          'RL_SCAPE_MVN', shutil.which('mvn') or ''))
  parser.add_argument(
      '--java-home', default=os.environ.get(
          'RL_SCAPE_JAVA_HOME', os.environ.get('JAVA_HOME', '')))
  parser.add_argument('--dry-run', action='store_true')
  parser.add_argument('--keep-going', action='store_true')
  parser.add_argument(
      '--stream-output', action='store_true',
      help='Stream child output instead of writing each run to console.log.')
  return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
  if args.episodes <= 0:
    raise ValueError('--episodes must be positive')
  if args.episode_length <= 0:
    raise ValueError('--episode-length must be positive')
  if args.grid_columns <= 0 or args.grid_rows <= 0:
    raise ValueError('grid dimensions must be positive')
  if not args.python.exists():
    raise FileNotFoundError(args.python)
  if not (args.repo_root / 'dreamerv3' / 'main.py').exists():
    raise FileNotFoundError(args.repo_root / 'dreamerv3' / 'main.py')


def main(argv=None) -> int:
  args = parse_args(argv)
  validate_args(args)
  args.repo_root = args.repo_root.resolve()
  args.log_root = args.log_root.resolve()
  checkpoint = resolve_checkpoint_path(args.checkpoint)
  args.log_root.mkdir(parents=True, exist_ok=True)

  suite = {
      'created_unix': time.time(),
      'checkpoint': str(checkpoint),
      'repo_root': str(args.repo_root),
      'python': str(args.python),
      'goals': list(args.goals),
      'seed': args.seed,
      'episodes_per_goal': args.episodes,
      'episode_length': args.episode_length,
      'action_interface': args.action_interface,
      'grid_columns': args.grid_columns,
      'grid_rows': args.grid_rows,
      'policy_mode': 'eval',
      'actor_ir': False,
      'runs': [],
  }
  write_json(args.log_root / 'suite_manifest.json', suite)
  summaries = []
  overall = 0

  for goal in args.goals:
    logdir = args.log_root / f'{goal}_seed{args.seed}'
    spec = {
        'checkpoint': str(checkpoint),
        'goal': goal,
        'seed': args.seed,
        'episodes': args.episodes,
        'episode_length': args.episode_length,
        'action_interface': args.action_interface,
        'grid_columns': args.grid_columns,
        'grid_rows': args.grid_rows,
        'policy_mode': 'eval',
        'actor_ir': False,
    }
    digest = spec_digest(spec)
    status_path = logdir / 'eval_status.json'
    existing = (
        json.loads(status_path.read_text())
        if status_path.exists() else {})
    if existing.get('spec_digest') not in (None, digest):
      raise RuntimeError(
          f'Existing evaluation has a different specification: {logdir}')
    if existing.get('status') == 'complete':
      summary = summarize_run(
          logdir, goal=goal, seed=args.seed,
          requested_episodes=args.episodes,
          returncode=int(existing.get('returncode', 0)))
      if summary['episode_count_exact']:
        print(f'Skipping completed evaluation: {goal}', flush=True)
        summaries.append(summary)
        suite['runs'].append(existing)
        continue
    if status_path.exists():
      raise RuntimeError(
          f'Incomplete evaluation artifacts already exist at {logdir}; '
          'use a new --log-root to keep episode counts unambiguous.')

    command = build_command(
        python=args.python,
        repo_root=args.repo_root,
        logdir=logdir,
        checkpoint=checkpoint,
        goal=goal,
        seed=args.seed,
        episodes=args.episodes,
        episode_length=args.episode_length,
        action_interface=args.action_interface,
        grid_columns=args.grid_columns,
        grid_rows=args.grid_rows,
        mvn_path=args.mvn_path,
        java_home=args.java_home,
    )
    status = {
        'spec': spec,
        'spec_digest': digest,
        'command': command,
        'status': 'dry_run' if args.dry_run else 'running',
        'started_unix': time.time(),
    }
    logdir.mkdir(parents=True, exist_ok=True)
    write_json(status_path, status)
    print(f'\n=== RLScape M2 eval: {goal} ===', flush=True)
    print(' '.join(command), flush=True)
    if args.dry_run:
      suite['runs'].append(status)
      continue

    console = None
    if not args.stream_output:
      console = (logdir / 'console.log').open('a')
    try:
      process = subprocess.run(
          command,
          cwd=args.repo_root,
          check=False,
          env=os.environ.copy(),
          stdout=console,
          stderr=subprocess.STDOUT if console else None,
      )
    finally:
      if console:
        console.close()

    summary = summarize_run(
        logdir, goal=goal, seed=args.seed,
        requested_episodes=args.episodes,
        returncode=process.returncode)
    complete = process.returncode == 0 and summary['episode_count_exact']
    status.update(
        returncode=process.returncode,
        finished_unix=time.time(),
        status='complete' if complete else 'failed',
    )
    write_json(status_path, status)
    suite['runs'].append(status)
    summaries.append(summary)
    write_json(args.log_root / 'summary.json', summaries)
    write_csv(args.log_root / 'summary.csv', summaries)
    print(
        f"Finished {goal}: rc={process.returncode} "
        f"episodes={summary['episodes']}/{args.episodes} "
        f"successes={summary['successes']}",
        flush=True)
    if not complete:
      overall = process.returncode or 1
      if not args.keep_going:
        break

  write_json(args.log_root / 'suite_manifest.json', suite)
  if summaries:
    write_json(args.log_root / 'summary.json', summaries)
    write_csv(args.log_root / 'summary.csv', summaries)
  return overall


if __name__ == '__main__':
  raise SystemExit(main())
