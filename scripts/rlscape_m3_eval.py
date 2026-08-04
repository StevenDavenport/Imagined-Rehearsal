#!/usr/bin/env python3
"""Drain the paired Experiment 1 RLScape milestone evaluation queue."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embodied.run import milestones  # noqa: E402
from scripts.rlscape_m1_pilot import GOALS  # noqa: E402
from scripts.rlscape_m1_pilot import read_jsonl  # noqa: E402
from scripts.rlscape_m1_pilot import spec_digest  # noqa: E402
from scripts.rlscape_m1_pilot import write_csv  # noqa: E402
from scripts.rlscape_m1_pilot import write_json  # noqa: E402
from scripts.rlscape_m2_eval import summarize_run  # noqa: E402
from scripts.rlscape_m3_process import active_matches  # noqa: E402
from scripts.rlscape_m3_process import run_managed  # noqa: E402


PHASE_STEPS = 500_000
MILESTONE_STEPS = 250_000
POLICY_MODES = {
    'sampled': 100,
    'deterministic': 50,
}


def build_units(goals=GOALS) -> list[dict[str, Any]]:
  units = []
  for step in range(MILESTONE_STEPS, PHASE_STEPS * len(goals) + 1,
                    MILESTONE_STEPS):
    phase = (step - 1) // PHASE_STEPS
    for goal in goals[:phase + 1]:
      for policy_mode, episodes in POLICY_MODES.items():
        units.append({
            'id': f'step_{step:012d}__{goal}__{policy_mode}',
            'step': step,
            'phase': phase,
            'trained_goal': goals[phase],
            'goal': goal,
            'policy_mode': policy_mode,
            'episodes': episodes,
        })
  return units


def build_command(
    *,
    python: pathlib.Path,
    repo_root: pathlib.Path,
    logdir: pathlib.Path,
    checkpoint: pathlib.Path,
    goal: str,
    seed: int,
    episodes: int,
    policy_mode: str,
    actor_ir: bool,
    episode_length: int = 200,
    grid_columns: int = 28,
    grid_rows: int = 18,
    mvn_path: str = '',
    java_home: str = '',
) -> list[str]:
  configs = [
      'rlscape', 'rlscape_click_grid', f'rlscape_{goal}',
      'rlscape_m3_eval']
  if actor_ir:
    configs.append('rlscape_m3_actor_ir')
  command = [
      str(python), str(repo_root / 'dreamerv3' / 'main.py'),
      '--configs', *configs,
      '--seed', str(seed),
      '--logdir', str(logdir),
      '--run.from_checkpoint', str(checkpoint),
      '--run.eval_episodes', str(episodes),
      '--run.eval_policy_mode', policy_mode,
      '--env.rlscape.episode_length', str(episode_length),
      '--env.rlscape.grid_columns', str(grid_columns),
      '--env.rlscape.grid_rows', str(grid_rows),
  ]
  if mvn_path:
    command += ['--env.rlscape.mvn_path', mvn_path]
  if java_home:
    command += ['--env.rlscape.java_home', java_home]
  return command


def reset_identities(logdir: pathlib.Path) -> list[dict[str, Any]]:
  rows = read_jsonl(logdir / 'audit_env0.jsonl')
  return [{
      'goal': row.get('goal'),
      'reset_seed': row.get('reset_seed'),
  } for row in rows if row.get('kind') == 'reset']


def reset_frame_hashes(logdir: pathlib.Path) -> list[str | None]:
  rows = read_jsonl(logdir / 'audit_env0.jsonl')
  return [row.get('frame_sha256')
          for row in rows if row.get('kind') == 'reset']


def condition_summary(
    logdir: pathlib.Path,
    *,
    unit: dict[str, Any],
    actor_ir: bool,
    seed: int,
    returncode: int,
) -> dict[str, Any]:
  summary = summarize_run(
      logdir, goal=unit['goal'], seed=seed,
      requested_episodes=unit['episodes'], returncode=returncode)
  integrity_path = logdir / 'eval_integrity.json'
  integrity = (
      json.loads(integrity_path.read_text())
      if integrity_path.exists() else {})
  resets = reset_identities(logdir)
  summary.update({
      'step': unit['step'],
      'phase': unit['phase'],
      'trained_goal': unit['trained_goal'],
      'policy_mode': unit['policy_mode'],
      'actor_ir': bool(actor_ir),
      'integrity_equal': integrity.get('equal', False),
      'reset_count': len(resets),
      'logdir': str(logdir),
  })
  summary['complete'] = bool(
      returncode == 0 and summary['episode_count_exact'] and
      summary['integrity_equal'] and len(resets) == unit['episodes'])
  return summary


def run_condition(
    args,
    *,
    unit: dict[str, Any],
    checkpoint: pathlib.Path,
    logdir: pathlib.Path,
    actor_ir: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
  command = build_command(
      python=args.python, repo_root=args.repo_root, logdir=logdir,
      checkpoint=checkpoint, goal=unit['goal'], seed=args.eval_seed,
      episodes=unit['episodes'], policy_mode=unit['policy_mode'],
      actor_ir=actor_ir, episode_length=args.episode_length,
      grid_columns=args.grid_columns, grid_rows=args.grid_rows,
      mvn_path=args.mvn_path, java_home=args.java_home)
  if logdir.exists():
    recovered = condition_summary(
        logdir, unit=unit, actor_ir=actor_ir, seed=args.eval_seed,
        returncode=0)
    if recovered['complete']:
      return recovered, reset_identities(logdir), command
    if not active_matches(args.active_path, command, args.repo_root):
      recovered['returncode'] = 1
      return recovered, reset_identities(logdir), command
  else:
    logdir.mkdir(parents=True, exist_ok=False)
  print(' '.join(command), flush=True)
  child = run_managed(
      command,
      cwd=args.repo_root,
      active_path=args.active_path,
      console_path=logdir / 'console.log',
      activity_paths=[
          logdir / 'metrics.jsonl',
          logdir / 'episodes.jsonl',
          logdir / 'audit_env0.jsonl',
      ],
      stream_output=args.stream_output,
      poll_seconds=args.poll_seconds,
      startup_timeout=args.startup_timeout,
      stall_timeout=args.stall_timeout,
      heartbeat=args.heartbeat,
  )
  args.queue.pop('active_child', None)
  args.queue['updated_unix'] = time.time()
  write_json(args.queue_path, args.queue)
  returncode = child['returncode']
  if returncode is None:
    tentative = condition_summary(
        logdir, unit=unit, actor_ir=actor_ir, seed=args.eval_seed,
        returncode=0)
    returncode = 0 if tentative['complete'] else 1
  return (
      condition_summary(
          logdir, unit=unit, actor_ir=actor_ir, seed=args.eval_seed,
          returncode=returncode),
      reset_identities(logdir),
      command,
  )


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Run the paired Experiment 1 milestone evaluation queue.')
  parser.add_argument('--experiment-root', type=pathlib.Path, required=True)
  parser.add_argument('--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument(
      '--python', type=pathlib.Path, default=pathlib.Path(sys.executable))
  parser.add_argument('--eval-seed', type=int, default=1000)
  parser.add_argument('--episode-length', type=int, default=200)
  parser.add_argument('--grid-columns', type=int, default=28)
  parser.add_argument('--grid-rows', type=int, default=18)
  parser.add_argument('--max-attempts', type=int, default=3)
  parser.add_argument('--poll-seconds', type=float, default=30)
  parser.add_argument('--startup-timeout', type=float, default=3600)
  parser.add_argument('--stall-timeout', type=float, default=14400)
  parser.add_argument(
      '--mvn-path', default=os.environ.get(
          'RL_SCAPE_MVN', shutil.which('mvn') or ''))
  parser.add_argument(
      '--java-home', default=os.environ.get(
          'RL_SCAPE_JAVA_HOME', os.environ.get('JAVA_HOME', '')))
  parser.add_argument('--stream-output', action='store_true')
  parser.add_argument('--dry-run', action='store_true')
  return parser.parse_args(argv)


def main(argv=None) -> int:
  args = parse_args(argv)
  args.repo_root = args.repo_root.resolve()
  args.experiment_root = args.experiment_root.resolve()
  args.python = args.python.resolve()
  if args.max_attempts < 1:
    raise ValueError('--max-attempts must be positive')
  if min(args.poll_seconds, args.startup_timeout, args.stall_timeout) <= 0:
    raise ValueError('poll and timeout values must be positive')
  if not args.python.is_file():
    raise FileNotFoundError(args.python)

  milestone_root = args.experiment_root / 'milestones'
  eval_root = args.experiment_root / 'evaluations'
  eval_root.mkdir(parents=True, exist_ok=True)
  experiment_path = args.experiment_root / 'experiment_spec.json'
  if not experiment_path.is_file():
    raise FileNotFoundError(experiment_path)
  experiment = json.loads(experiment_path.read_text())
  goals = tuple(experiment.get('goals', ()))
  if not goals:
    raise ValueError(f'Experiment goal list is empty: {experiment_path}')
  if goals != tuple(GOALS[:len(goals)]):
    raise ValueError(
        'Experiment goals must be an ordered prefix of the canonical ladder: '
        f'{goals}')
  units = build_units(goals)
  spec = {
      'format': 1,
      'experiment_root': str(args.experiment_root),
      'goals': list(goals),
      'units': units,
      'eval_seed': args.eval_seed,
      'episode_length': args.episode_length,
      'grid': [args.grid_columns, args.grid_rows],
      'pairing': 'same_goal_and_environment_seed',
      'initial_frame_hash': 'diagnostic_only_across_process_lifetimes',
      'conditions': {
          'frozen': {'actor_ir': False},
          'actor_ir': {
              'steps_per_action': 1,
              'every_k': 1,
              'horizon': 6,
              'posterior_samples': 128,
              'lr': 4e-5,
              'entropy': 3e-4,
              'critic_frozen': True,
              'world_model_frozen': True,
          },
      },
  }
  digest = spec_digest(spec)
  queue_path = eval_root / 'queue.json'
  existing = json.loads(queue_path.read_text()) if queue_path.exists() else {}
  if existing.get('spec_digest') not in (None, digest):
    raise RuntimeError(
        f'Existing evaluation queue has a different spec: {queue_path}')
  states = existing.get('states', {})
  queue = {
      'spec': spec,
      'spec_digest': digest,
      'created_unix': existing.get('created_unix', time.time()),
      'updated_unix': time.time(),
      'status': 'dry_run' if args.dry_run else 'running',
      'states': states,
  }
  write_json(queue_path, queue)
  args.active_path = eval_root / 'active_child.json'
  args.queue = queue
  args.queue_path = queue_path

  def heartbeat(snapshot):
    queue['active_child'] = snapshot
    queue['updated_unix'] = time.time()
    write_json(queue_path, queue)

  args.heartbeat = heartbeat
  if args.dry_run:
    queue.update(status='dry_run', updated_unix=time.time())
    write_json(queue_path, queue)
    print(
        f'Planned {len(units)} paired evaluation units '
        f'({2 * len(units)} conditions).', flush=True)
    return 0

  results = []
  for unit in units:
    state = states.setdefault(unit['id'], {'status': 'pending', 'attempts': []})
    if state.get('status') == 'complete':
      results.extend(state['attempts'][-1]['summaries'])
  write_json(eval_root / 'results.json', results)

  while True:
    pending = [
        unit for unit in units
        if states[unit['id']].get('status') != 'complete' and
        (
            states[unit['id']].get('status') == 'running' or
            len(states[unit['id']].get('attempts', ())) < args.max_attempts
        )]
    if not pending:
      break
    for unit in pending:
      state = states[unit['id']]
      if (
          state.get('status') == 'running' and state.get('attempts') and
          state['attempts'][-1].get('status') == 'running'
      ):
        attempt = state['attempts'][-1]
        attempt_number = int(attempt['attempt'])
      else:
        attempt_number = len(state['attempts']) + 1
        attempt = {
            'attempt': attempt_number,
            'started_unix': time.time(),
            'summaries': [],
            'commands': [],
            'status': 'running',
        }
        state['attempts'].append(attempt)
      attempt_root = (
          eval_root / f'step_{unit["step"]:012d}' / unit['goal'] /
          unit['policy_mode'] / f'attempt_{attempt_number:02d}')
      checkpoint = (
          milestones.archive_path(milestone_root, unit['step']) /
          'checkpoint')
      milestones.validate(
          checkpoint.parent, expected_step=unit['step'], full=False)
      print(
          f'\n=== Eval pair {unit["id"]}, attempt {attempt_number} ===',
          flush=True)
      attempt['checkpoint'] = str(checkpoint)
      state.update(status='running')
      queue['updated_unix'] = time.time()
      write_json(queue_path, queue)
      frozen = next((
          summary for summary in attempt['summaries']
          if not summary.get('actor_ir')), None)
      if frozen and frozen.get('complete'):
        frozen_resets = reset_identities(
            pathlib.Path(frozen['logdir']))
      else:
        frozen, frozen_resets, frozen_command = run_condition(
            args, unit=unit, checkpoint=checkpoint,
            logdir=attempt_root / 'frozen', actor_ir=False)
        attempt['summaries'] = [
            summary for summary in attempt['summaries']
            if summary.get('actor_ir')]
        attempt['summaries'].insert(0, frozen)
        attempt['commands'].append(frozen_command)
        queue['updated_unix'] = time.time()
        write_json(queue_path, queue)
      actor_ir = None
      actor_resets = []
      if frozen.get('complete'):
        actor_ir = next((
            summary for summary in attempt['summaries']
            if summary.get('actor_ir')), None)
        if actor_ir and actor_ir.get('complete'):
          actor_resets = reset_identities(
              pathlib.Path(actor_ir['logdir']))
        else:
          actor_ir, actor_resets, actor_command = run_condition(
              args, unit=unit, checkpoint=checkpoint,
              logdir=attempt_root / 'actor_ir', actor_ir=True)
          attempt['summaries'] = [
              summary for summary in attempt['summaries']
              if not summary.get('actor_ir')]
          attempt['summaries'].append(actor_ir)
          attempt['commands'].append(actor_command)
          queue['updated_unix'] = time.time()
          write_json(queue_path, queue)
      paired = bool(
          actor_ir and frozen.get('complete') and actor_ir.get('complete') and
          frozen_resets == actor_resets)
      frozen_frames = reset_frame_hashes(pathlib.Path(frozen['logdir']))
      actor_frames = (
          reset_frame_hashes(pathlib.Path(actor_ir['logdir']))
          if actor_ir else [])
      frame_mismatches = (
          sum(left != right
              for left, right in zip(frozen_frames, actor_frames)) +
          abs(len(frozen_frames) - len(actor_frames)))
      attempt.update(
          finished_unix=time.time(),
          paired_reset_identity=paired,
          paired_initial_frames=(
              bool(frozen_frames) and frame_mismatches == 0),
          frame_mismatch_count=frame_mismatches,
          reset_mismatch_count=(
              sum(
                  left != right
                  for left, right in zip(frozen_resets, actor_resets)) +
              abs(len(frozen_resets) - len(actor_resets))),
          status='complete' if paired else 'failed',
      )
      state['status'] = attempt['status']
      queue['updated_unix'] = time.time()
      write_json(queue_path, queue)
      if paired:
        results.extend(attempt['summaries'])
        write_json(eval_root / 'results.json', results)
        write_csv(eval_root / 'results.csv', results)
      print(
          f'Pair {unit["id"]}: {attempt["status"]} '
          f'(paired={paired})', flush=True)

  incomplete = [
      unit['id'] for unit in units
      if states[unit['id']].get('status') != 'complete']
  queue.update(
      updated_unix=time.time(),
      finished_unix=time.time(),
      status='failed' if incomplete else 'complete',
      incomplete=incomplete,
  )
  write_json(queue_path, queue)
  if results:
    write_json(eval_root / 'results.json', results)
    write_csv(eval_root / 'results.csv', results)
  print(
      f'Evaluation queue {queue["status"]}: '
      f'{len(units) - len(incomplete)}/{len(units)} paired units complete.',
      flush=True)
  return 1 if incomplete else 0


if __name__ == '__main__':
  raise SystemExit(main())
