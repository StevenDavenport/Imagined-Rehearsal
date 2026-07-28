#!/usr/bin/env python3
"""Run resilient five-phase RLScape sequential FIFO training and evaluation."""

from __future__ import annotations

import argparse
import importlib.metadata
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

from embodied.run import milestones  # noqa: E402
from scripts.checkpoint_utils import resolve_checkpoint_path  # noqa: E402
from scripts.rlscape_m1_pilot import GOALS  # noqa: E402
from scripts.rlscape_m1_pilot import spec_digest  # noqa: E402
from scripts.rlscape_m1_pilot import write_json  # noqa: E402
from scripts.rlscape_m3_process import run_managed  # noqa: E402


PHASE_STEPS = 500_000
MILESTONE_EVERY = 250_000


def git_revision(repo_root: pathlib.Path) -> str:
  result = subprocess.run(
      ['git', 'rev-parse', 'HEAD'], cwd=repo_root,
      check=False, capture_output=True, text=True)
  return result.stdout.strip() if result.returncode == 0 else 'unknown'


def checkpoint_step(logdir: pathlib.Path) -> int:
  try:
    checkpoint = resolve_checkpoint_path(logdir)
  except FileNotFoundError:
    return 0
  with (checkpoint / 'step.pkl').open('rb') as handle:
    return int(pickle.load(handle))


def experiment_spec(args) -> dict[str, Any]:
  return {
      'format': 1,
      'name': 'rlscape_experiment_1_sequential_fifo_actor_ir',
      'git_commit': git_revision(args.repo_root),
      'seed': args.seed,
      'goals': list(GOALS),
      'steps_per_goal': PHASE_STEPS,
      'total_steps': PHASE_STEPS * len(GOALS),
      'milestone_every': MILESTONE_EVERY,
      'milestones': list(range(
          MILESTONE_EVERY, PHASE_STEPS * len(GOALS) + 1,
          MILESTONE_EVERY)),
      'episode_length': 200,
      'action_interface': 'click_grid',
      'grid': [28, 18],
      'reward': 'sparse_success',
      'model': '50m',
      'batch_size': 8,
      'batch_length': 32,
      'train_imag_horizon': 15,
      'train_ratio': 32,
      'compute_dtype': 'bfloat16',
      'replay': {
          'kind': 'fifo',
          'capacity': 200_000,
          'reservoir': False,
          'save_wait': True,
      },
      'goal_conditioning': {
          'encoding': 'one_hot',
          'count': 5,
          'heads': [
              'reward', 'continuation', 'actor', 'value', 'slow_value'],
          'world_model_state': 'goal_agnostic',
      },
      'training_actor_ir': False,
      'rlscape_version': '0.1.6',
      'evaluation': {
          'after_all_training': True,
          'sampled_episodes_per_condition': 100,
          'deterministic_episodes_per_condition': 50,
          'paired': True,
          'actor_ir': {
              'every_action': True,
              'updates': 1,
              'horizon': 6,
              'posterior_samples': 128,
              'lr': 4e-5,
              'entropy': 3e-4,
              'critic_frozen': True,
              'world_model_frozen': True,
          },
      },
  }


def build_train_command(
    *,
    args,
    training_logdir: pathlib.Path,
    milestone_root: pathlib.Path,
    goal: str,
    phase: int,
    target_step: int,
    digest: str,
) -> list[str]:
  command = [
      str(args.python), str(args.repo_root / 'dreamerv3' / 'main.py'),
      '--configs', 'rlscape', 'rlscape_click_grid',
      f'rlscape_{goal}', 'rlscape_m3_sequential',
      '--seed', str(args.seed),
      '--logdir', str(training_logdir),
      '--run.steps', str(target_step),
      '--run.milestone_every', str(MILESTONE_EVERY),
      '--run.milestone_dir', str(milestone_root),
      '--run.milestone_goal', goal,
      '--run.milestone_phase', str(phase),
      '--run.milestone_spec_digest', digest,
  ]
  if args.mvn_path:
    command += ['--env.rlscape.mvn_path', args.mvn_path]
  if args.java_home:
    command += ['--env.rlscape.java_home', args.java_home]
  return command


def phase_archives_complete(
    milestone_root: pathlib.Path, phase: int, *, full=False) -> bool:
  start = phase * PHASE_STEPS
  for step in (start + MILESTONE_EVERY, start + PHASE_STEPS):
    path = milestones.archive_path(milestone_root, step)
    try:
      milestones.validate(path, expected_step=step, full=full)
    except (FileNotFoundError, milestones.MilestoneError):
      return False
  return True


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Run RLScape Experiment 1 sequential training.')
  parser.add_argument('--experiment-root', type=pathlib.Path, required=True)
  parser.add_argument('--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument(
      '--python', type=pathlib.Path, default=pathlib.Path(sys.executable))
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--max-restarts', type=int, default=20)
  parser.add_argument('--max-stalled-restarts', type=int, default=4)
  parser.add_argument('--restart-delay', type=float, default=15)
  parser.add_argument('--min-free-gb', type=float, default=300)
  parser.add_argument('--poll-seconds', type=float, default=30)
  parser.add_argument('--startup-timeout', type=float, default=3600)
  parser.add_argument('--stall-timeout', type=float, default=7200)
  parser.add_argument(
      '--mvn-path', default=os.environ.get(
          'RL_SCAPE_MVN', shutil.which('mvn') or ''))
  parser.add_argument(
      '--java-home', default=os.environ.get(
          'RL_SCAPE_JAVA_HOME', os.environ.get('JAVA_HOME', '')))
  parser.add_argument('--stream-output', action='store_true')
  parser.add_argument('--skip-eval', action='store_true')
  parser.add_argument('--dry-run', action='store_true')
  return parser.parse_args(argv)


def preflight(args) -> None:
  if not args.python.is_file():
    raise FileNotFoundError(args.python)
  if not (args.repo_root / 'dreamerv3' / 'main.py').is_file():
    raise FileNotFoundError(args.repo_root / 'dreamerv3' / 'main.py')
  if args.max_restarts < 0 or args.max_stalled_restarts < 0:
    raise ValueError('restart limits must be nonnegative')
  if args.restart_delay < 0:
    raise ValueError('--restart-delay must be nonnegative')
  if min(args.poll_seconds, args.startup_timeout, args.stall_timeout) <= 0:
    raise ValueError('poll and timeout values must be positive')
  if not args.dry_run:
    installed = importlib.metadata.version('rl-scape')
    if installed != '0.1.6':
      raise RuntimeError(
          f'RLScape 0.1.6 is required, found {installed}')
  args.experiment_root.mkdir(parents=True, exist_ok=True)
  free_gb = shutil.disk_usage(args.experiment_root).free / 1024 ** 3
  if not args.dry_run and free_gb < args.min_free_gb:
    raise RuntimeError(
        f'Only {free_gb:.1f} GiB free; --min-free-gb is '
        f'{args.min_free_gb:.1f}')


def main(argv=None) -> int:
  args = parse_args(argv)
  args.repo_root = args.repo_root.resolve()
  args.experiment_root = args.experiment_root.resolve()
  args.python = args.python.resolve()
  preflight(args)
  training_logdir = args.experiment_root / 'training'
  milestone_root = args.experiment_root / 'milestones'
  training_logdir.mkdir(parents=True, exist_ok=True)
  milestone_root.mkdir(parents=True, exist_ok=True)

  spec = experiment_spec(args)
  digest = spec_digest(spec)
  spec_path = args.experiment_root / 'experiment_spec.json'
  if spec_path.exists():
    existing_spec = json.loads(spec_path.read_text())
    if spec_digest(existing_spec) != digest:
      raise RuntimeError(
          f'Existing experiment has a different spec: {spec_path}')
  else:
    write_json(spec_path, spec)

  status_path = args.experiment_root / 'supervisor_status.json'
  existing = json.loads(status_path.read_text()) if status_path.exists() else {}
  if existing.get('spec_digest') not in (None, digest):
    raise RuntimeError(
        f'Existing supervisor has a different spec: {status_path}')
  phases = existing.get('phases', {})
  status = {
      'spec_digest': digest,
      'status': 'dry_run' if args.dry_run else 'training',
      'created_unix': existing.get('created_unix', time.time()),
      'updated_unix': time.time(),
      'training_logdir': str(training_logdir),
      'milestone_root': str(milestone_root),
      'phases': phases,
      'total_restarts': int(existing.get('total_restarts', 0)),
  }
  write_json(status_path, status)
  active_path = args.experiment_root / 'active_child.json'

  def heartbeat(snapshot):
    status['active_child'] = snapshot
    status['updated_unix'] = time.time()
    write_json(status_path, status)

  for phase, goal in enumerate(GOALS):
    free_gb = shutil.disk_usage(args.experiment_root).free / 1024 ** 3
    if not args.dry_run and free_gb < args.min_free_gb:
      status.update(
          status='blocked_low_disk', free_disk_gb=free_gb,
          updated_unix=time.time())
      write_json(status_path, status)
      raise RuntimeError(
          f'Only {free_gb:.1f} GiB free before phase {phase + 1}; '
          f'--min-free-gb is {args.min_free_gb:.1f}')
    target = (phase + 1) * PHASE_STEPS
    phase_state = phases.setdefault(str(phase), {
        'goal': goal, 'target_step': target, 'attempts': []})
    if (
        checkpoint_step(training_logdir) >= target and
        phase_archives_complete(milestone_root, phase)
    ):
      phase_state['status'] = 'complete'
      write_json(status_path, status)
      print(f'Phase {phase + 1} already complete: {goal}', flush=True)
      continue
    command = build_train_command(
        args=args, training_logdir=training_logdir,
        milestone_root=milestone_root, goal=goal, phase=phase,
        target_step=target, digest=digest)
    print(
        f'\n=== Phase {phase + 1}/{len(GOALS)}: {goal}, '
        f'target={target} ===', flush=True)
    print(' '.join(command), flush=True)
    if args.dry_run:
      phase_state.update(status='dry_run', command=command)
      write_json(status_path, status)
      continue

    restarts = int(phase_state.get('restarts', 0))
    stalled = int(phase_state.get('stalled_restarts', 0))
    while True:
      before = checkpoint_step(training_logdir)
      attempt = {
          'attempt': len(phase_state['attempts']) + 1,
          'started_unix': time.time(),
          'step_before': before,
          'command': command,
      }
      phase_state['attempts'].append(attempt)
      phase_state['status'] = 'running'
      status['updated_unix'] = time.time()
      write_json(status_path, status)
      child = run_managed(
          command,
          cwd=args.repo_root,
          active_path=active_path,
          console_path=training_logdir / 'console.log',
          activity_paths=[
              training_logdir / 'metrics.jsonl',
              training_logdir / 'episodes.jsonl',
              training_logdir / 'audit_env0.jsonl',
              training_logdir / 'ckpt' / 'latest',
          ],
          stream_output=args.stream_output,
          poll_seconds=args.poll_seconds,
          startup_timeout=args.startup_timeout,
          stall_timeout=args.stall_timeout,
          heartbeat=heartbeat,
      )
      status.pop('active_child', None)
      after = checkpoint_step(training_logdir)
      complete = bool(
          child['returncode'] in (0, None) and after >= target and
          phase_archives_complete(milestone_root, phase))
      attempt.update(
          finished_unix=time.time(), returncode=child['returncode'],
          adopted=child['adopted'], timed_out=child['timed_out'],
          step_after=after, complete=complete)
      if complete:
        phase_state.update(status='complete', finished_unix=time.time())
        status['updated_unix'] = time.time()
        write_json(status_path, status)
        break
      stalled = stalled + 1 if after <= before else 0
      restarts += 1
      status['total_restarts'] += 1
      phase_state.update(
          status='recovering', restarts=restarts,
          stalled_restarts=stalled)
      status['updated_unix'] = time.time()
      write_json(status_path, status)
      if (
          status['total_restarts'] > args.max_restarts or
          stalled > args.max_stalled_restarts
      ):
        phase_state.update(status='failed', finished_unix=time.time())
        status.update(status='failed', updated_unix=time.time())
        write_json(status_path, status)
        print(
            f'Phase {goal} exhausted restart budget at step {after}.',
            flush=True)
        return child['returncode'] or 1
      print(
          f'Phase child exited rc={child["returncode"]}, step '
          f'{before}->{after}; restart in {args.restart_delay:g}s.',
          flush=True)
      time.sleep(args.restart_delay)

  if args.dry_run:
    status.update(status='dry_run', updated_unix=time.time())
    write_json(status_path, status)
    return 0

  status.update(
      status='trained', trained_unix=time.time(), updated_unix=time.time())
  write_json(status_path, status)
  if args.skip_eval:
    status.update(status='complete', finished_unix=time.time())
    write_json(status_path, status)
    return 0

  eval_command = [
      str(args.python), str(args.repo_root / 'scripts' / 'rlscape_m3_eval.py'),
      '--experiment-root', str(args.experiment_root),
      '--repo-root', str(args.repo_root),
      '--python', str(args.python),
  ]
  if args.mvn_path:
    eval_command += ['--mvn-path', args.mvn_path]
  if args.java_home:
    eval_command += ['--java-home', args.java_home]
  if args.stream_output:
    eval_command.append('--stream-output')
  status.update(
      status='evaluating', evaluation_command=eval_command,
      updated_unix=time.time())
  write_json(status_path, status)
  print('\n=== Draining Experiment 1 evaluation queue ===', flush=True)
  print(' '.join(eval_command), flush=True)
  child = run_managed(
      eval_command,
      cwd=args.repo_root,
      active_path=active_path,
      console_path=args.experiment_root / 'evaluation_console.log',
      activity_paths=[
          args.experiment_root / 'evaluations' / 'queue.json',
          args.experiment_root / 'evaluations' / 'results.json',
      ],
      stream_output=args.stream_output,
      poll_seconds=args.poll_seconds,
      startup_timeout=args.startup_timeout,
      stall_timeout=max(args.stall_timeout, 4 * 3600),
      heartbeat=heartbeat,
  )
  status.pop('active_child', None)
  evaluation_returncode = child['returncode']
  if evaluation_returncode is None:
    queue_path = args.experiment_root / 'evaluations' / 'queue.json'
    queue = json.loads(queue_path.read_text()) if queue_path.exists() else {}
    evaluation_returncode = 0 if queue.get('status') == 'complete' else 1
  status.update(
      status='complete' if evaluation_returncode == 0 else 'evaluation_failed',
      evaluation_returncode=evaluation_returncode,
      finished_unix=time.time(),
      updated_unix=time.time(),
  )
  write_json(status_path, status)
  return evaluation_returncode


if __name__ == '__main__':
  raise SystemExit(main())
