#!/usr/bin/env python3
"""Run and summarize resumable RLScape milestone-1 single-task pilots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
GOALS = (
    'kill_goblin',
    'bury_bones',
    'chop_logs',
    'light_fire',
    'catch_fish',
)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
  if not path.exists():
    return []
  rows = []
  with path.open() as handle:
    for line in handle:
      if line.strip():
        rows.append(json.loads(line))
  return rows


def write_json(path: pathlib.Path, value: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
  temporary.replace(path)


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
  if not rows:
    return
  keys = sorted({key for row in rows for key in row})
  temporary = path.with_suffix(path.suffix + '.tmp')
  with temporary.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def spec_digest(spec: dict[str, Any]) -> str:
  payload = json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()
  return hashlib.sha256(payload).hexdigest()


def build_command(
    *,
    python: pathlib.Path,
    repo_root: pathlib.Path,
    logdir: pathlib.Path,
    goal: str,
    seed: int,
    steps: int,
    episode_length: int,
    train_ratio: float,
    log_every: int,
    save_every: int,
    mvn_path: str = '',
    java_home: str = '',
) -> list[str]:
  command = [
      str(python),
      str(repo_root / 'dreamerv3' / 'main.py'),
      '--configs', 'rlscape', f'rlscape_{goal}', 'rlscape_m1_pilot',
      '--seed', str(seed),
      '--logdir', str(logdir),
      '--run.steps', str(steps),
      '--run.train_ratio', str(train_ratio),
      '--run.log_every', str(log_every),
      '--run.save_every', str(save_every),
      '--env.rlscape.episode_length', str(episode_length),
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
    returncode: int | None = None,
) -> dict[str, Any]:
  episodes = read_jsonl(logdir / 'episodes.jsonl')
  metrics = read_jsonl(logdir / 'metrics.jsonl')
  scores = [float(row.get('episode/score', 0.0)) for row in episodes]
  successes = [
      bool(score > 0 or row.get('episode/log/task_success/max', 0) > 0)
      for row, score in zip(episodes, scores)
  ]
  progress = [
      float(row.get('episode/log/task_progress/max', 0.0))
      for row in episodes
  ]
  deaths = [
      float(row.get('episode/log/player_death/max', 0.0))
      for row in episodes
  ]
  lengths = [float(row['episode/length']) for row in episodes]
  first_success = next((
      int(row['step']) for row, success in zip(episodes, successes)
      if success > 0), None)
  training = [
      row for row in metrics if any(key.startswith('train/') for key in row)
  ]
  latest = training[-1] if training else {}
  result = {
      'goal': goal,
      'seed': int(seed),
      'returncode': returncode,
      'episodes': len(episodes),
      'successes': int(sum(value > 0 for value in successes)),
      'success_rate': (
          sum(value > 0 for value in successes) / len(episodes)
          if episodes else 0.0),
      'first_success_step': first_success,
      'max_progress': max(progress, default=0.0),
      'deaths': int(sum(value > 0 for value in deaths)),
      'mean_episode_length': (
          sum(lengths) / len(lengths) if lengths else 0.0),
      'last_episode_step': max(
          (int(row.get('step', 0)) for row in episodes), default=0),
      'last_metric_step': max(
          (int(row.get('step', 0)) for row in metrics), default=0),
      'latest_mean_update_index': float(
          latest.get('train/opt/updates', 0.0)),
      'policy_fps': float(latest.get('fps/policy', 0.0)),
      'train_fps': float(latest.get('fps/train', 0.0)),
  }
  for index in range(4):
    total = sum(float(row.get(
        f'episode/log/action_mode_{index}/sum', 0.0)) for row in episodes)
    result[f'action_mode_{index}_count'] = int(total)
  action_steps = sum(
      result[f'action_mode_{index}_count'] for index in range(4))
  position_sum = sum(float(row.get(
      'episode/log/action_position_abs/sum', 0.0)) for row in episodes)
  result['action_steps'] = action_steps
  result['expected_action_steps'] = int(sum(lengths) - len(episodes))
  result['action_log_complete'] = (
      action_steps == result['expected_action_steps'])
  result['action_position_abs_mean'] = (
      position_sum / sum(lengths) if lengths else 0.0)
  for name in ('con', 'dyn', 'image', 'policy', 'rew', 'value'):
    result[f'loss_{name}'] = float(latest.get(f'train/loss/{name}', 0.0))
  return result


def parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          'Run ordinary single-task Dreamer pilots with sparse RLScape reward.'))
  parser.add_argument(
      '--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument(
      '--python', type=pathlib.Path, default=pathlib.Path(sys.executable))
  parser.add_argument(
      '--log-root', type=pathlib.Path,
      default=ROOT / 'logs' / 'rlscape' / 'm1_pilot')
  parser.add_argument('--goals', nargs='+', choices=GOALS, default=list(GOALS))
  parser.add_argument('--seeds', nargs='+', type=int, default=[0])
  parser.add_argument('--steps', type=int, default=1000)
  parser.add_argument('--episode-length', type=int, default=200)
  parser.add_argument('--train-ratio', type=float, default=4.0)
  parser.add_argument('--log-every', type=int, default=30)
  parser.add_argument('--save-every', type=int, default=60)
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
  if args.steps <= 0:
    raise ValueError('--steps must be positive')
  if args.episode_length <= 0:
    raise ValueError('--episode-length must be positive')
  if args.train_ratio < 0:
    raise ValueError('--train-ratio must be nonnegative')
  if args.log_every <= 0 or args.save_every <= 0:
    raise ValueError('logging and checkpoint intervals must be positive')
  if not args.python.exists():
    raise FileNotFoundError(args.python)
  if not (args.repo_root / 'dreamerv3' / 'main.py').exists():
    raise FileNotFoundError(args.repo_root / 'dreamerv3' / 'main.py')


def main(argv=None) -> int:
  args = parse_args(argv)
  validate_args(args)
  args.repo_root = args.repo_root.resolve()
  args.log_root = args.log_root.resolve()
  args.log_root.mkdir(parents=True, exist_ok=True)
  suite = {
      'created_unix': time.time(),
      'repo_root': str(args.repo_root),
      'python': str(args.python),
      'goals': list(args.goals),
      'seeds': list(args.seeds),
      'steps': args.steps,
      'episode_length': args.episode_length,
      'train_ratio': args.train_ratio,
      'log_every': args.log_every,
      'save_every': args.save_every,
      'reward_mode': 'sparse_success',
      'actor_ir': False,
      'replay': 'fifo',
      'runs': [],
  }
  write_json(args.log_root / 'suite_manifest.json', suite)

  summaries = []
  overall = 0
  for goal in args.goals:
    for seed in args.seeds:
      logdir = args.log_root / f'{goal}_seed{seed}'
      spec = {
          key: suite[key] for key in (
              'repo_root', 'python', 'steps', 'episode_length', 'train_ratio',
              'log_every', 'save_every', 'reward_mode', 'actor_ir', 'replay')
      }
      spec.update(goal=goal, seed=seed)
      digest = spec_digest(spec)
      status_path = logdir / 'pilot_status.json'
      existing = (
          json.loads(status_path.read_text()) if status_path.exists() else {})
      if existing.get('spec_digest') not in (None, digest):
        raise RuntimeError(
            f'Existing run has a different specification: {logdir}')
      if existing.get('status') == 'complete':
        print(f'Skipping completed run: {goal} seed={seed}', flush=True)
        summary = summarize_run(
            logdir, goal=goal, seed=seed,
            returncode=int(existing.get('returncode', 0)))
        summaries.append(summary)
        suite['runs'].append(existing)
        continue

      command = build_command(
          python=args.python,
          repo_root=args.repo_root,
          logdir=logdir,
          goal=goal,
          seed=seed,
          steps=args.steps,
          episode_length=args.episode_length,
          train_ratio=args.train_ratio,
          log_every=args.log_every,
          save_every=args.save_every,
          mvn_path=args.mvn_path,
          java_home=args.java_home,
      )
      status = {
          'goal': goal,
          'seed': seed,
          'spec': spec,
          'spec_digest': digest,
          'command': command,
          'status': 'dry_run' if args.dry_run else 'running',
          'started_unix': time.time(),
      }
      logdir.mkdir(parents=True, exist_ok=True)
      write_json(status_path, status)
      print(f'\n=== RLScape M1: {goal} seed={seed} ===', flush=True)
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
      status.update(
          returncode=process.returncode,
          finished_unix=time.time(),
          status='complete' if process.returncode == 0 else 'failed',
      )
      write_json(status_path, status)
      suite['runs'].append(status)
      summary = summarize_run(
          logdir, goal=goal, seed=seed, returncode=process.returncode)
      print(
          f"Finished {goal} seed={seed}: rc={process.returncode} "
          f"episodes={summary['episodes']} "
          f"successes={summary['successes']} "
          f"max_progress={summary['max_progress']}",
          flush=True)
      summaries.append(summary)
      write_json(args.log_root / 'summary.json', summaries)
      write_csv(args.log_root / 'summary.csv', summaries)
      if process.returncode:
        overall = process.returncode
        if not args.keep_going:
          write_json(args.log_root / 'suite_manifest.json', suite)
          return overall

  write_json(args.log_root / 'suite_manifest.json', suite)
  if summaries:
    write_json(args.log_root / 'summary.json', summaries)
    write_csv(args.log_root / 'summary.csv', summaries)
  return overall


if __name__ == '__main__':
  raise SystemExit(main())
