#!/usr/bin/env python3
"""Run, extend, evaluate, and summarize RLScape single-task learnability checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
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


GOALS = (
    'kill_goblin',
    'bury_bones',
    'chop_logs',
    'light_fire',
    'catch_fish',
)
TRAINING_SPEC_KEYS = (
    'repo_root',
    'python',
    'episode_length',
    'train_ratio',
    'log_every',
    'save_every',
    'batch_size',
    'replay_size',
    'model_size',
    'action_interface',
    'grid_columns',
    'grid_rows',
    'reward_mode',
    'actor_ir',
    'replay',
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


def training_spec(
    suite: dict[str, Any],
    *,
    goal: str,
    seed: int,
) -> dict[str, Any]:
  spec = {key: suite[key] for key in TRAINING_SPEC_KEYS}
  return {**spec, 'goal': goal, 'seed': int(seed)}


def checkpoint_step(logdir: pathlib.Path) -> int:
  try:
    checkpoint = resolve_checkpoint_path(logdir)
  except FileNotFoundError:
    return 0
  with (checkpoint / 'step.pkl').open('rb') as handle:
    return int(pickle.load(handle))


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
    batch_size: int = 8,
    replay_size: int = 200000,
    grid_columns: int = 28,
    grid_rows: int = 18,
    mvn_path: str = '',
    java_home: str = '',
) -> list[str]:
  command = [
      str(python),
      str(repo_root / 'dreamerv3' / 'main.py'),
      '--configs', 'rlscape', 'rlscape_click_grid',
      f'rlscape_{goal}', 'rlscape_m1_learnability',
      '--seed', str(seed),
      '--logdir', str(logdir),
      '--run.steps', str(steps),
      '--run.train_ratio', str(train_ratio),
      '--run.log_every', str(log_every),
      '--run.save_every', str(save_every),
      '--env.rlscape.episode_length', str(episode_length),
      '--env.rlscape.grid_columns', str(grid_columns),
      '--env.rlscape.grid_rows', str(grid_rows),
      '--batch_size', str(batch_size),
      '--replay.size', str(replay_size),
  ]
  if mvn_path:
    command += ['--env.rlscape.mvn_path', mvn_path]
  if java_home:
    command += ['--env.rlscape.java_home', java_home]
  return command


def build_eval_command(
    *,
    python: pathlib.Path,
    repo_root: pathlib.Path,
    log_root: pathlib.Path,
    checkpoint: pathlib.Path,
    goal: str,
    seed: int,
    episodes: int,
    episode_length: int,
    grid_columns: int,
    grid_rows: int,
    mvn_path: str = '',
    java_home: str = '',
    stream_output: bool = False,
) -> list[str]:
  command = [
      str(python),
      str(repo_root / 'scripts' / 'rlscape_m2_eval.py'),
      '--repo-root', str(repo_root),
      '--python', str(python),
      '--checkpoint', str(checkpoint),
      '--log-root', str(log_root),
      '--goals', goal,
      '--seed', str(seed),
      '--episodes', str(episodes),
      '--episode-length', str(episode_length),
      '--action-interface', 'click_grid',
      '--grid-columns', str(grid_columns),
      '--grid-rows', str(grid_rows),
  ]
  if mvn_path:
    command += ['--mvn-path', mvn_path]
  if java_home:
    command += ['--java-home', java_home]
  if stream_output:
    command.append('--stream-output')
  return command


def evaluation_summary(
    log_root: pathlib.Path,
    *,
    goal: str,
    seed: int,
) -> dict[str, Any] | None:
  path = log_root / 'summary.json'
  if not path.exists():
    return None
  rows = json.loads(path.read_text())
  return next((
      row for row in rows
      if row.get('goal') == goal and int(row.get('seed', -1)) == seed
  ), None)


def evaluation_complete(
    summary: dict[str, Any] | None,
    *,
    episodes: int,
) -> bool:
  return bool(
      summary and int(summary.get('returncode', -1)) == 0 and
      summary.get('episode_count_exact') and
      int(summary.get('requested_episodes', -1)) == episodes and
      int(summary.get('episodes', -1)) == episodes)


def evaluation_log_root(
    logdir: pathlib.Path,
    *,
    step: int,
    goal: str,
    seed: int,
    episodes: int,
) -> pathlib.Path:
  base = (
      logdir / 'evaluations' /
      f'step_{step}_seed{seed}_n{episodes}')
  candidate = base
  attempt = 1
  while candidate.exists():
    summary = evaluation_summary(candidate, goal=goal, seed=seed)
    if evaluation_complete(summary, episodes=episodes):
      return candidate
    attempt += 1
    candidate = pathlib.Path(f'{base}_attempt{attempt}')
  return candidate


def upsert_summary(
    summaries: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
  key = (summary['goal'], int(summary['seed']))
  result = list(summaries)
  for index, existing in enumerate(result):
    if (existing.get('goal'), int(existing.get('seed', -1))) == key:
      result[index] = summary
      return result
  result.append(summary)
  return result


def merge_evaluation(
    training: dict[str, Any],
    evaluation: dict[str, Any] | None,
) -> dict[str, Any]:
  result = dict(training)
  if evaluation is not None:
    for key, value in evaluation.items():
      if key not in ('goal', 'seed'):
        result[f'eval_{key}'] = value
  return result


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
      'final_100_success_rate': (
          sum(successes[-100:]) / len(successes[-100:])
          if successes else 0.0),
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
  grid_action_steps = int(sum(float(row.get(
      'episode/log/action_grid_valid/sum', 0.0)) for row in episodes))
  position_sum = sum(float(row.get(
      'episode/log/action_position_abs/sum', 0.0)) for row in episodes)
  result['action_interface'] = (
      'click_grid' if grid_action_steps else 'mixed')
  result['action_steps'] = grid_action_steps or action_steps
  result['expected_action_steps'] = int(sum(lengths) - len(episodes))
  result['action_log_complete'] = (
      result['action_steps'] == result['expected_action_steps'])
  result['action_position_abs_mean'] = (
      position_sum / sum(lengths) if lengths else 0.0)
  grid_index_sum = sum(float(row.get(
      'episode/log/action_grid_index/sum', 0.0)) for row in episodes)
  result['action_grid_index_mean'] = (
      grid_index_sum / grid_action_steps if grid_action_steps else 0.0)
  for name in ('con', 'dyn', 'image', 'policy', 'rew', 'value'):
    result[f'loss_{name}'] = float(latest.get(f'train/loss/{name}', 0.0))
  return result


def parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          'Run one fresh click-grid Dreamer agent per RLScape goal.'))
  parser.add_argument(
      '--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument(
      '--python', type=pathlib.Path, default=pathlib.Path(sys.executable))
  parser.add_argument(
      '--log-root', type=pathlib.Path,
      default=ROOT / 'logs' / 'rlscape' / 'm1_learnability_200k_seed0')
  parser.add_argument('--goals', nargs='+', choices=GOALS, default=list(GOALS))
  parser.add_argument('--seeds', nargs='+', type=int, default=[0])
  parser.add_argument('--steps', type=int, default=200000)
  parser.add_argument('--episode-length', type=int, default=200)
  parser.add_argument('--train-ratio', type=float, default=32.0)
  parser.add_argument('--log-every', type=int, default=300)
  parser.add_argument('--save-every', type=int, default=1800)
  parser.add_argument('--batch-size', type=int, default=8)
  parser.add_argument('--replay-size', type=int, default=200000)
  parser.add_argument('--grid-columns', type=int, default=28)
  parser.add_argument('--grid-rows', type=int, default=18)
  parser.add_argument('--eval-episodes', type=int, default=20)
  parser.add_argument('--eval-seed', type=int, default=1000)
  parser.add_argument('--skip-eval', action='store_true')
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
  if args.batch_size <= 0 or args.replay_size <= 0:
    raise ValueError('batch size and replay size must be positive')
  if args.grid_columns <= 0 or args.grid_rows <= 0:
    raise ValueError('grid dimensions must be positive')
  if args.eval_episodes <= 0:
    raise ValueError('--eval-episodes must be positive')
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
      'target_steps': args.steps,
      'episode_length': args.episode_length,
      'train_ratio': args.train_ratio,
      'log_every': args.log_every,
      'save_every': args.save_every,
      'batch_size': args.batch_size,
      'replay_size': args.replay_size,
      'model_size': '50m',
      'action_interface': 'click_grid',
      'grid_columns': args.grid_columns,
      'grid_rows': args.grid_rows,
      'eval_episodes': args.eval_episodes,
      'eval_seed': args.eval_seed,
      'reward_mode': 'sparse_success',
      'actor_ir': False,
      'replay': 'fifo',
      'runs': [],
  }
  write_json(args.log_root / 'suite_manifest.json', suite)

  summary_path = args.log_root / 'summary.json'
  summaries = (
      json.loads(summary_path.read_text()) if summary_path.exists() else [])
  overall = 0
  for goal in args.goals:
    for seed in args.seeds:
      logdir = args.log_root / f'{goal}_seed{seed}'
      immutable_spec = training_spec(suite, goal=goal, seed=seed)
      digest = spec_digest(immutable_spec)
      status_path = logdir / 'pilot_status.json'
      existing = (
          json.loads(status_path.read_text()) if status_path.exists() else {})
      if existing.get('spec_digest') not in (None, digest):
        raise RuntimeError(
            f'Existing run has a different specification: {logdir}')

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
          batch_size=args.batch_size,
          replay_size=args.replay_size,
          grid_columns=args.grid_columns,
          grid_rows=args.grid_rows,
          mvn_path=args.mvn_path,
          java_home=args.java_home,
      )
      current_step = checkpoint_step(logdir)
      status = {
          **existing,
          'goal': goal,
          'seed': seed,
          'spec': immutable_spec,
          'spec_digest': digest,
          'target_steps': args.steps,
          'command': command,
          'status': 'dry_run' if args.dry_run else 'training',
      }
      logdir.mkdir(parents=True, exist_ok=True)
      print(f'\n=== RLScape M1 learnability: {goal} seed={seed} ===', flush=True)
      print(' '.join(command), flush=True)
      if args.dry_run:
        write_json(status_path, status)
        suite['runs'].append(status)
        continue

      returncode = 0
      if current_step < args.steps:
        attempt = {
            'started_unix': time.time(),
            'from_step': current_step,
            'target_steps': args.steps,
            'command': command,
        }
        status['started_unix'] = attempt['started_unix']
        status['training_attempts'] = [
            *existing.get('training_attempts', []), attempt]
        write_json(status_path, status)
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
        returncode = process.returncode
        current_step = checkpoint_step(logdir)
        status['training_attempts'][-1].update(
            finished_unix=time.time(),
            returncode=returncode,
            final_step=current_step,
        )
      else:
        print(
            f'Training checkpoint already at step {current_step}; '
            f'target is {args.steps}.', flush=True)

      training_complete = returncode == 0 and current_step >= args.steps
      status.update(
          returncode=returncode,
          finished_unix=time.time(),
          final_step=current_step,
          status='trained' if training_complete else 'failed',
      )
      write_json(status_path, status)
      if not training_complete:
        overall = returncode or 1
        suite['runs'].append(status)
        if not args.keep_going:
          write_json(args.log_root / 'suite_manifest.json', suite)
          return overall
        continue

      evaluation = None
      if not args.skip_eval:
        checkpoint = resolve_checkpoint_path(logdir)
        eval_root = evaluation_log_root(
            logdir, step=current_step, goal=goal, seed=args.eval_seed,
            episodes=args.eval_episodes)
        evaluation = evaluation_summary(
            eval_root, goal=goal, seed=args.eval_seed)
        if evaluation is None:
          eval_command = build_eval_command(
              python=args.python,
              repo_root=args.repo_root,
              log_root=eval_root,
              checkpoint=checkpoint,
              goal=goal,
              seed=args.eval_seed,
              episodes=args.eval_episodes,
              episode_length=args.episode_length,
              grid_columns=args.grid_columns,
              grid_rows=args.grid_rows,
              mvn_path=args.mvn_path,
              java_home=args.java_home,
              stream_output=args.stream_output,
          )
          status.update(
              status='evaluating',
              evaluation_log_root=str(eval_root),
              evaluation_command=eval_command,
          )
          write_json(status_path, status)
          print(
              f'\n=== Held-out evaluation: {goal} '
              f'checkpoint_step={current_step} ===', flush=True)
          print(' '.join(eval_command), flush=True)
          process = subprocess.run(
              eval_command,
              cwd=args.repo_root,
              check=False,
              env=os.environ.copy(),
          )
          evaluation = evaluation_summary(
              eval_root, goal=goal, seed=args.eval_seed)
          eval_complete = bool(
              process.returncode == 0 and
              evaluation_complete(evaluation, episodes=args.eval_episodes))
          status.update(
              evaluation_returncode=process.returncode,
              status='complete' if eval_complete else 'failed',
          )
          write_json(status_path, status)
          if not eval_complete:
            overall = process.returncode or 1
            suite['runs'].append(status)
            if not args.keep_going:
              write_json(args.log_root / 'suite_manifest.json', suite)
              return overall
            continue
        else:
          status.update(
              status='complete',
              evaluation_log_root=str(eval_root),
              evaluation_returncode=int(evaluation.get('returncode', 0)),
          )
      else:
        status['status'] = 'complete'

      write_json(status_path, status)
      suite['runs'].append(status)
      summary = merge_evaluation(summarize_run(
          logdir, goal=goal, seed=seed, returncode=returncode), evaluation)
      print(
          f"Finished {goal} seed={seed}: step={current_step} "
          f"train_successes={summary['successes']} "
          f"train_final100={summary['final_100_success_rate']:.3f} "
          f"eval_success={summary.get('eval_success_rate', 'skipped')}",
          flush=True)
      summaries = upsert_summary(summaries, summary)
      write_json(summary_path, summaries)
      write_csv(args.log_root / 'summary.csv', summaries)

  write_json(args.log_root / 'suite_manifest.json', suite)
  if summaries:
    write_json(summary_path, summaries)
    write_csv(args.log_root / 'summary.csv', summaries)
  return overall


if __name__ == '__main__':
  raise SystemExit(main())
