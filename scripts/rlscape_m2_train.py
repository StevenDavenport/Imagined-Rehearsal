#!/usr/bin/env python3
"""Run a fault-tolerant RLScape random-goal multitask training job."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import shutil
import subprocess
import sys
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.checkpoint_utils import resolve_checkpoint_path  # noqa: E402
from scripts.rlscape_m1_pilot import spec_digest  # noqa: E402
from scripts.rlscape_m1_pilot import write_json  # noqa: E402


def build_command(
    *,
    python: pathlib.Path,
    repo_root: pathlib.Path,
    logdir: pathlib.Path,
    seed: int,
    steps: int,
    batch_size: int,
    replay_size: int,
    mvn_path: str = '',
    java_home: str = '',
) -> list[str]:
  command = [
      str(python),
      str(repo_root / 'dreamerv3' / 'main.py'),
      '--configs', 'rlscape', 'rlscape_m2_multitask',
      '--seed', str(seed),
      '--logdir', str(logdir),
      '--run.steps', str(steps),
      '--batch_size', str(batch_size),
      '--replay.size', str(replay_size),
  ]
  if mvn_path:
    command += ['--env.rlscape.mvn_path', mvn_path]
  if java_home:
    command += ['--env.rlscape.java_home', java_home]
  return command


def checkpoint_step(logdir: pathlib.Path) -> int:
  try:
    checkpoint = resolve_checkpoint_path(logdir)
  except FileNotFoundError:
    return 0
  with (checkpoint / 'step.pkl').open('rb') as handle:
    value = pickle.load(handle)
  return int(value)


def parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description='Run the 2M-transition RLScape M2 multitask experiment.')
  parser.add_argument('--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument(
      '--python', type=pathlib.Path, default=pathlib.Path(sys.executable))
  parser.add_argument(
      '--logdir', type=pathlib.Path,
      default=ROOT / 'logs' / 'rlscape' / 'm2_multitask_2m_seed0')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--steps', type=int, default=2000000)
  parser.add_argument('--batch-size', type=int, default=8)
  parser.add_argument('--replay-size', type=int, default=350000)
  parser.add_argument('--max-restarts', type=int, default=20)
  parser.add_argument('--max-stalled-restarts', type=int, default=3)
  parser.add_argument('--restart-delay', type=float, default=10)
  parser.add_argument(
      '--mvn-path', default=os.environ.get(
          'RL_SCAPE_MVN', shutil.which('mvn') or ''))
  parser.add_argument(
      '--java-home', default=os.environ.get(
          'RL_SCAPE_JAVA_HOME', os.environ.get('JAVA_HOME', '')))
  parser.add_argument('--dry-run', action='store_true')
  parser.add_argument(
      '--stream-output', action='store_true',
      help='Stream Dreamer output instead of appending it to console.log.')
  return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
  if args.steps <= 0:
    raise ValueError('--steps must be positive')
  if args.batch_size <= 0:
    raise ValueError('--batch-size must be positive')
  if args.replay_size <= 0:
    raise ValueError('--replay-size must be positive')
  if args.max_restarts < 0 or args.max_stalled_restarts < 0:
    raise ValueError('restart limits must be nonnegative')
  if args.restart_delay < 0:
    raise ValueError('--restart-delay must be nonnegative')
  if not args.python.exists():
    raise FileNotFoundError(args.python)
  if not (args.repo_root / 'dreamerv3' / 'main.py').exists():
    raise FileNotFoundError(args.repo_root / 'dreamerv3' / 'main.py')


def main(argv=None) -> int:
  args = parse_args(argv)
  validate_args(args)
  args.repo_root = args.repo_root.resolve()
  args.logdir = args.logdir.resolve()
  args.logdir.mkdir(parents=True, exist_ok=True)
  status_path = args.logdir / 'm2_train_status.json'
  spec = {
      'repo_root': str(args.repo_root),
      'python': str(args.python),
      'seed': args.seed,
      'steps': args.steps,
      'goals': [
          'kill_goblin', 'bury_bones', 'chop_logs', 'light_fire', 'catch_fish'],
      'goal_schedule': 'random_per_episode',
      'episode_length': 200,
      'model_size': '50m',
      'batch_size': args.batch_size,
      'batch_length': 32,
      'imag_length': 15,
      'train_ratio': 32,
      'replay': {'kind': 'fifo', 'capacity': args.replay_size},
      'actor_ir': False,
  }
  digest = spec_digest(spec)
  existing: dict[str, Any] = (
      json.loads(status_path.read_text()) if status_path.exists() else {})
  if existing.get('spec_digest') not in (None, digest):
    raise RuntimeError(
        f'Existing run has a different specification: {args.logdir}')
  if existing.get('status') == 'complete':
    print(f'Run already complete at step {checkpoint_step(args.logdir)}')
    return 0

  command = build_command(
      python=args.python,
      repo_root=args.repo_root,
      logdir=args.logdir,
      seed=args.seed,
      steps=args.steps,
      batch_size=args.batch_size,
      replay_size=args.replay_size,
      mvn_path=args.mvn_path,
      java_home=args.java_home,
  )
  attempts = list(existing.get('attempts', []))
  status = {
      'spec': spec,
      'spec_digest': digest,
      'command': command,
      'status': 'dry_run' if args.dry_run else 'running',
      'started_unix': existing.get('started_unix', time.time()),
      'attempts': attempts,
  }
  write_json(status_path, status)
  print(' '.join(command), flush=True)
  if args.dry_run:
    return 0

  restarts = 0
  stalled = 0
  while True:
    before = checkpoint_step(args.logdir)
    started = time.time()
    print(
        f'\n=== RLScape M2 train attempt {len(attempts) + 1}: '
        f'step={before}/{args.steps} ===',
        flush=True)
    console = None
    if not args.stream_output:
      console = (args.logdir / 'console.log').open('a')
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
    after = checkpoint_step(args.logdir)
    attempt = {
        'attempt': len(attempts) + 1,
        'started_unix': started,
        'finished_unix': time.time(),
        'returncode': process.returncode,
        'step_before': before,
        'step_after': after,
    }
    attempts.append(attempt)
    status['attempts'] = attempts

    if process.returncode == 0 and after >= args.steps:
      status.update(status='complete', finished_unix=time.time())
      write_json(status_path, status)
      print(f'Completed RLScape M2 training at step {after}.', flush=True)
      return 0

    stalled = stalled + 1 if after <= before else 0
    restarts += 1
    status.update(
        status='recovering',
        last_returncode=process.returncode,
        last_checkpoint_step=after,
        restarts=restarts,
        stalled_restarts=stalled,
    )
    write_json(status_path, status)
    if restarts > args.max_restarts or stalled > args.max_stalled_restarts:
      status.update(status='failed', finished_unix=time.time())
      write_json(status_path, status)
      print(
          f'Stopping after {restarts} restarts; checkpoint step={after}, '
          f'stalled restarts={stalled}.',
          flush=True)
      return process.returncode or 1
    print(
        f'Child exited rc={process.returncode} at step {after}; '
        f'restarting in {args.restart_delay:g}s.',
        flush=True)
    time.sleep(args.restart_delay)


if __name__ == '__main__':
  raise SystemExit(main())
