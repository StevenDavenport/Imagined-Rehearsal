#!/usr/bin/env python3
"""Run the restart-safe six-task, multi-seed MiniGrid audition matrix."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embodied.envs.minigrid_registry import TASK_NAMES  # noqa: E402
from embodied.run import milestones  # noqa: E402
from scripts.minigrid_audition_report import generate_report  # noqa: E402
from scripts.minigrid_common import checkpoint_step  # noqa: E402
from scripts.minigrid_common import evaluation_summary  # noqa: E402
from scripts.minigrid_common import git_revision  # noqa: E402
from scripts.minigrid_common import package_versions  # noqa: E402
from scripts.minigrid_common import reset_seeds  # noqa: E402
from scripts.minigrid_common import spec_digest  # noqa: E402
from scripts.minigrid_common import validate_tasks  # noqa: E402
from scripts.minigrid_common import write_json  # noqa: E402
from scripts.rlscape_m3_process import active_matches  # noqa: E402
from scripts.rlscape_m3_process import run_managed  # noqa: E402


MODEL_SIZES = ('1m', '12m', '25m')
POLICY_MODES = ('sampled', 'deterministic')


def build_train_command(
    *, args, task: str, seed: int, logdir: pathlib.Path,
    milestone_root: pathlib.Path, digest: str,
) -> list[str]:
  return [
      str(args.python), str(args.repo_root / 'dreamerv3' / 'main.py'),
      '--configs', 'minigrid', f'minigrid_size{args.model_size}',
      f'minigrid_{task}', 'minigrid_audition',
      '--seed', str(seed),
      '--logdir', str(logdir),
      '--run.steps', str(args.steps),
      '--run.milestone_every', str(args.checkpoint_every),
      '--run.milestone_dir', str(milestone_root),
      '--run.milestone_goal', task,
      '--run.milestone_phase', '0',
      '--run.milestone_spec_digest', digest,
      '--env.minigrid.episode_length', str(args.episode_length),
      '--replay.size', str(args.replay_size),
      '--run.train_ratio', str(args.train_ratio),
  ]


def build_eval_command(
    *, args, task: str, checkpoint: pathlib.Path, logdir: pathlib.Path,
    eval_seed: int, policy_mode: str, paired_seed: int,
) -> list[str]:
  return [
      str(args.python), str(args.repo_root / 'dreamerv3' / 'main.py'),
      '--configs', 'minigrid', f'minigrid_size{args.model_size}',
      f'minigrid_{task}', 'minigrid_eval',
      '--seed', str(eval_seed),
      '--logdir', str(logdir),
      '--run.from_checkpoint', str(checkpoint),
      '--run.eval_episodes', str(args.eval_episodes_per_mode),
      '--run.eval_policy_mode', policy_mode,
      '--env.minigrid.episode_length', str(args.episode_length),
      '--eval_adapt.paired_rng', 'True',
      '--eval_adapt.paired_rng_seed', str(paired_seed),
      '--eval_adapt.paired_rng_stride', str(args.paired_rng_stride),
  ]


def _milestones_complete(root: pathlib.Path, steps) -> bool:
  try:
    for step in steps:
      milestones.validate(
          milestones.archive_path(root, step), expected_step=step, full=False)
    return True
  except (FileNotFoundError, milestones.MilestoneError):
    return False


def _condition_summary(
    condition_root, *, unit, mode, episodes, paired_rng_stride):
  status_path = condition_root / 'condition_status.json'
  if not status_path.is_file():
    return None
  status = json.loads(status_path.read_text())
  summary = status.get('summary')
  if not summary:
    return None
  logdir = pathlib.Path(summary['logdir'])
  refreshed = evaluation_summary(
      logdir,
      task=unit['task'], training_seed=unit['seed'],
      eval_seed=unit['eval_seed'], checkpoint=unit['checkpoint'],
      policy_mode=mode, requested_episodes=episodes,
      paired_rng_seed=unit['paired_seed'],
      paired_rng_stride=paired_rng_stride,
      returncode=status.get('returncode'))
  return refreshed if refreshed['complete'] else None


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Run the controlled MiniGrid difficulty audition.')
  parser.add_argument('--experiment-root', type=pathlib.Path, required=True)
  parser.add_argument('--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument(
      '--python', type=pathlib.Path, default=pathlib.Path(sys.executable))
  parser.add_argument('--tasks', nargs='+', default=list(TASK_NAMES))
  parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
  parser.add_argument('--model-size', choices=MODEL_SIZES, default='12m')
  parser.add_argument('--steps', type=int, default=100_000)
  parser.add_argument('--checkpoint-every', type=int, default=25_000)
  parser.add_argument('--episode-length', type=int, default=100)
  parser.add_argument('--eval-episodes-per-mode', type=int, default=50)
  parser.add_argument('--policy-modes', nargs='+', choices=POLICY_MODES,
                      default=list(POLICY_MODES))
  parser.add_argument('--eval-seed-base', type=int, default=100_000)
  parser.add_argument('--paired-rng-seed-base', type=int, default=202_608_290)
  parser.add_argument('--paired-rng-stride', type=int, default=1000)
  parser.add_argument('--replay-size', type=int, default=100_000)
  parser.add_argument('--train-ratio', type=float, default=32.0)
  parser.add_argument('--difficulty-threshold', type=float, default=0.7)
  parser.add_argument('--difficulty-window', type=int, default=20)
  parser.add_argument('--difficulty-tolerance', type=float, default=0.25)
  parser.add_argument('--max-attempts', type=int, default=3)
  parser.add_argument('--min-free-gb', type=float, default=20.0)
  parser.add_argument('--poll-seconds', type=float, default=30.0)
  parser.add_argument('--startup-timeout', type=float, default=3600.0)
  parser.add_argument('--stall-timeout', type=float, default=7200.0)
  parser.add_argument('--stream-output', action='store_true')
  parser.add_argument('--skip-eval', action='store_true')
  parser.add_argument('--dry-run', action='store_true')
  return parser.parse_args(argv)


def preflight(args):
  args.repo_root = args.repo_root.resolve()
  args.experiment_root = args.experiment_root.resolve()
  # Preserve virtual-environment interpreter symlinks. Resolving them can turn
  # ``venv/bin/python`` back into the system interpreter and silently lose the
  # installed CUDA/project packages for every managed child.
  args.python = args.python.expanduser()
  if not args.python.is_absolute():
    args.python = (pathlib.Path.cwd() / args.python).absolute()
  args.tasks = validate_tasks(args.tasks)
  args.seeds = tuple(int(seed) for seed in args.seeds)
  args.policy_modes = tuple(dict.fromkeys(args.policy_modes))
  if not args.python.is_file():
    raise FileNotFoundError(args.python)
  if not (args.repo_root / 'dreamerv3' / 'main.py').is_file():
    raise FileNotFoundError(args.repo_root / 'dreamerv3' / 'main.py')
  if not args.seeds or len(set(args.seeds)) != len(args.seeds):
    raise ValueError('--seeds must contain unique values')
  if args.steps <= 0 or args.checkpoint_every <= 0:
    raise ValueError('steps and checkpoint interval must be positive')
  if args.steps % args.checkpoint_every:
    raise ValueError('--steps must be divisible by --checkpoint-every')
  if args.checkpoint_every % 10:
    raise ValueError('--checkpoint-every must be divisible by 10')
  if min(
      args.episode_length, args.eval_episodes_per_mode, args.replay_size,
      args.max_attempts, args.difficulty_window) <= 0:
    raise ValueError('episode, replay, attempt, and window values must be positive')
  if args.paired_rng_stride <= args.episode_length:
    raise ValueError('paired RNG stride must exceed the episode length')
  if not 0 < args.difficulty_threshold <= 1:
    raise ValueError('difficulty threshold must be in (0, 1]')
  if not 0 <= args.difficulty_tolerance <= 1:
    raise ValueError('difficulty tolerance must be in [0, 1]')
  args.experiment_root.mkdir(parents=True, exist_ok=True)
  if not args.dry_run:
    versions = package_versions()
    missing = [name for name, value in versions.items() if value == 'missing']
    if missing:
      raise RuntimeError(
          f'Missing runtime packages: {missing}. Install requirements.txt.')
    if versions['minigrid'] != '3.1.0':
      raise RuntimeError(
          f'MiniGrid 3.1.0 is required, found {versions["minigrid"]}')
    if versions['jax'] != '0.4.33':
      raise RuntimeError(
          f'JAX 0.4.33 is required by this repository, found '
          f'{versions["jax"]}')
    free = shutil.disk_usage(args.experiment_root).free / 1024 ** 3
    if free < args.min_free_gb:
      raise RuntimeError(
          f'Only {free:.1f} GiB free; require {args.min_free_gb:.1f} GiB')


def main(argv=None):
  args = parse_args(argv)
  preflight(args)
  checkpoints = tuple(range(
      args.checkpoint_every, args.steps + 1, args.checkpoint_every))
  versions = package_versions()
  spec = {
      'format': 1,
      'name': 'minigrid_fixed_mission_difficulty_audition',
      'git_commit': git_revision(args.repo_root),
      'dependency_requirements': {
          'minigrid': '==3.1.0',
          'gymnasium': '>=1.0,<2',
      },
      'tasks': list(args.tasks),
      'seeds': list(args.seeds),
      'model_size': args.model_size,
      'steps': args.steps,
      'checkpoint_every': args.checkpoint_every,
      'checkpoint_steps': list(checkpoints),
      'episode_length': args.episode_length,
      'eval_episodes_per_mode': args.eval_episodes_per_mode,
      'policy_modes': list(args.policy_modes),
      'evaluation_pairing': 'same environment reset seeds across policy modes',
      'eval_seed_base': args.eval_seed_base,
      'paired_rng_seed_base': args.paired_rng_seed_base,
      'paired_rng_stride': args.paired_rng_stride,
      'replay_size': args.replay_size,
      'replay_kind': 'fifo',
      'train_ratio': args.train_ratio,
      'observation': '64x64 partial RGB',
      'action_space': 'Discrete(7)',
      'goal_conditioning': 'six-way one-hot heads; goal-agnostic RSSM',
      'missions': 'fixed per task; randomized layouts and starts',
      'difficulty_threshold': args.difficulty_threshold,
      'difficulty_window': args.difficulty_window,
      'difficulty_tolerance': args.difficulty_tolerance,
  }
  digest = spec_digest(spec)
  spec_path = args.experiment_root / 'experiment_spec.json'
  if spec_path.is_file():
    existing = json.loads(spec_path.read_text())
    if spec_digest(existing) != digest:
      raise RuntimeError(f'Existing experiment spec differs: {spec_path}')
  else:
    write_json(spec_path, spec)

  status_path = args.experiment_root / 'supervisor_status.json'
  existing = json.loads(status_path.read_text()) if status_path.is_file() else {}
  if existing.get('spec_digest') not in (None, digest):
    raise RuntimeError(f'Existing supervisor spec differs: {status_path}')
  states = existing.get('states', {})
  status = {
      'spec_digest': digest,
      'status': 'dry_run' if args.dry_run else 'running',
      'created_unix': existing.get('created_unix', time.time()),
      'updated_unix': time.time(),
      'states': states,
      'runtime_versions': versions,
  }
  write_json(status_path, status)
  active_path = args.experiment_root / 'active_child.json'

  def heartbeat(snapshot):
    status['active_child'] = snapshot
    status['updated_unix'] = time.time()
    write_json(status_path, status)

  for task in args.tasks:
    for seed in args.seeds:
      unit_id = f'{task}__seed_{seed:04d}'
      run_root = args.experiment_root / 'runs' / task / f'seed_{seed:04d}'
      training = run_root / 'training'
      milestone_root = run_root / 'milestones'
      state = states.setdefault(unit_id, {
          'task': task, 'seed': seed, 'training_attempts': [],
          'evaluations': {}})
      command = build_train_command(
          args=args, task=task, seed=seed, logdir=training,
          milestone_root=milestone_root, digest=digest)
      complete = bool(
          checkpoint_step(training) >= args.steps and
          _milestones_complete(milestone_root, checkpoints))
      if not complete:
        if args.dry_run:
          state.update(training_status='dry_run', training_command=command)
          write_json(status_path, status)
        else:
          attempts = state['training_attempts']
          while len(attempts) < args.max_attempts and not complete:
            attempt = {
                'attempt': len(attempts) + 1,
                'started_unix': time.time(),
                'command': command,
                'step_before': checkpoint_step(training),
            }
            attempts.append(attempt)
            state['training_status'] = 'running'
            write_json(status_path, status)
            child = run_managed(
                command, cwd=args.repo_root, active_path=active_path,
                console_path=training / 'console.log',
                activity_paths=[
                    training / 'metrics.jsonl', training / 'episodes.jsonl',
                    training / 'ckpt' / 'latest'],
                stream_output=args.stream_output,
                poll_seconds=args.poll_seconds,
                startup_timeout=args.startup_timeout,
                stall_timeout=args.stall_timeout,
                heartbeat=heartbeat)
            status.pop('active_child', None)
            complete = bool(
                checkpoint_step(training) >= args.steps and
                _milestones_complete(milestone_root, checkpoints))
            attempt.update(
                finished_unix=time.time(), returncode=child['returncode'],
                timed_out=child['timed_out'],
                step_after=checkpoint_step(training), complete=complete)
            state['training_status'] = (
                'complete' if complete else 'retrying')
            write_json(status_path, status)
          if not complete:
            state['training_status'] = 'failed'
            status.update(status='failed', updated_unix=time.time())
            write_json(status_path, status)
            return 1
      else:
        state['training_status'] = 'complete'
        write_json(status_path, status)

      if args.skip_eval:
        continue
      for checkpoint_value in checkpoints:
        checkpoint = (
            milestones.archive_path(milestone_root, checkpoint_value) /
            'checkpoint')
        eval_seed = args.eval_seed_base + seed
        paired_seed = (
            args.paired_rng_seed_base + seed * 1_000_000 + checkpoint_value)
        mode_summaries = {}
        for mode in args.policy_modes:
          condition_id = f'step_{checkpoint_value:012d}__{mode}'
          condition_root = (
              run_root / 'evaluations' /
              f'step_{checkpoint_value:012d}' / mode)
          unit = {
              'task': task, 'seed': seed, 'eval_seed': eval_seed,
              'checkpoint': checkpoint_value, 'paired_seed': paired_seed}
          recovered = _condition_summary(
              condition_root, unit=unit, mode=mode,
              episodes=args.eval_episodes_per_mode,
              paired_rng_stride=args.paired_rng_stride)
          if recovered:
            mode_summaries[mode] = recovered
            continue
          condition_state = state['evaluations'].setdefault(
              condition_id, {'attempts': []})
          attempts = condition_state['attempts']
          summary = None
          if args.dry_run:
            logdir = condition_root / 'attempt_01'
            command = build_eval_command(
                args=args, task=task, checkpoint=checkpoint,
                logdir=logdir, eval_seed=eval_seed, policy_mode=mode,
                paired_seed=paired_seed)
            condition_state.update(
                status='dry_run', planned_command=command,
                planned_logdir=str(logdir))
            write_json(condition_root / 'condition_status.json', {
                'status': 'dry_run', 'attempts': attempts, 'summary': None,
                'planned_command': command, 'planned_logdir': str(logdir),
                'returncode': None})
            continue
          while summary is None:
            attempt = None
            if attempts and 'finished_unix' not in attempts[-1]:
              candidate = attempts[-1]
              logdir = pathlib.Path(candidate['logdir'])
              recovered = evaluation_summary(
                  logdir, task=task, training_seed=seed,
                  eval_seed=eval_seed, checkpoint=checkpoint_value,
                  policy_mode=mode,
                  requested_episodes=args.eval_episodes_per_mode,
                  paired_rng_seed=paired_seed,
                  paired_rng_stride=args.paired_rng_stride,
                  returncode=None)
              if recovered['complete']:
                candidate.update(
                    finished_unix=time.time(), returncode=None,
                    timed_out=False, complete=True,
                    recovery='completed_while_supervisor_offline')
                summary = recovered
                break
              command = list(candidate['command'])
              if active_matches(active_path, command, args.repo_root):
                candidate['recovery'] = 'adopting_live_child'
                attempt = candidate
              else:
                # Recheck after inspecting the process table to close the
                # small race where the child exits between the two reads.
                recovered = evaluation_summary(
                    logdir, task=task, training_seed=seed,
                    eval_seed=eval_seed, checkpoint=checkpoint_value,
                    policy_mode=mode,
                    requested_episodes=args.eval_episodes_per_mode,
                    paired_rng_seed=paired_seed,
                    paired_rng_stride=args.paired_rng_stride,
                    returncode=None)
                if recovered['complete']:
                  candidate.update(
                      finished_unix=time.time(), returncode=None,
                      timed_out=False, complete=True,
                      recovery='completed_while_supervisor_offline')
                  summary = recovered
                  break
                candidate.update(
                    finished_unix=time.time(), returncode=None,
                    timed_out=False, complete=False,
                    recovery='abandoned_dead_partial_attempt')

            if attempt is None:
              if len(attempts) >= args.max_attempts:
                break
              attempt_number = len(attempts) + 1
              logdir = condition_root / f'attempt_{attempt_number:02d}'
              command = build_eval_command(
                  args=args, task=task, checkpoint=checkpoint,
                  logdir=logdir, eval_seed=eval_seed, policy_mode=mode,
                  paired_seed=paired_seed)
              attempt = {
                  'attempt': attempt_number, 'started_unix': time.time(),
                  'command': command, 'logdir': str(logdir)}
              attempts.append(attempt)
            else:
              logdir = pathlib.Path(attempt['logdir'])
              command = list(attempt['command'])
            condition_state['status'] = 'running'
            write_json(status_path, status)
            condition_root.mkdir(parents=True, exist_ok=True)
            child = run_managed(
                command, cwd=args.repo_root, active_path=active_path,
                console_path=logdir / 'console.log',
                activity_paths=[
                    logdir / 'metrics.jsonl', logdir / 'episodes.jsonl',
                    logdir / 'audit_env0.jsonl'],
                stream_output=args.stream_output,
                poll_seconds=args.poll_seconds,
                startup_timeout=args.startup_timeout,
                stall_timeout=args.stall_timeout,
                heartbeat=heartbeat)
            status.pop('active_child', None)
            summary = evaluation_summary(
                logdir, task=task, training_seed=seed,
                eval_seed=eval_seed, checkpoint=checkpoint_value,
                policy_mode=mode,
                requested_episodes=args.eval_episodes_per_mode,
                paired_rng_seed=paired_seed,
                paired_rng_stride=args.paired_rng_stride,
                returncode=child['returncode'])
            attempt.update(
                finished_unix=time.time(), returncode=child['returncode'],
                adopted=child['adopted'], timed_out=child['timed_out'],
                complete=summary['complete'])
            if not summary['complete']:
              summary = None
          condition_status = {
              'status': (
                  'complete' if summary else 'failed'),
              'attempts': attempts,
              'summary': summary,
              'returncode': (
                  attempts[-1].get('returncode') if attempts else None),
          }
          write_json(condition_root / 'condition_status.json', condition_status)
          condition_state['status'] = condition_status['status']
          if summary:
            mode_summaries[mode] = summary
          elif not args.dry_run:
            status.update(status='failed', updated_unix=time.time())
            write_json(status_path, status)
            return 1

        if all(mode in mode_summaries for mode in args.policy_modes):
          reset_map = {
              mode: reset_seeds(pathlib.Path(mode_summaries[mode]['logdir']))
              for mode in args.policy_modes}
          reference = reset_map[args.policy_modes[0]]
          equal = all(reset_map[mode] == reference for mode in args.policy_modes)
          pairing = {
              'task': task, 'training_seed': seed,
              'checkpoint': checkpoint_value,
              'policy_modes': list(args.policy_modes),
              'episodes_per_mode': args.eval_episodes_per_mode,
              'reset_seeds_equal': equal,
              'reset_seeds': reset_map,
          }
          write_json(
              run_root / 'evaluations' /
              f'step_{checkpoint_value:012d}' / 'pairing.json', pairing)
          if not equal and not args.dry_run:
            raise RuntimeError(
                f'Paired evaluation reset seeds differ: {task} seed={seed} '
                f'step={checkpoint_value}')
      state['status'] = 'complete'
      if args.dry_run:
        state['status'] = 'dry_run'
      write_json(status_path, status)

  if args.dry_run:
    status.update(status='dry_run', updated_unix=time.time())
  elif args.skip_eval:
    status.update(status='trained', updated_unix=time.time())
  else:
    generate_report(args.experiment_root)
    status.update(status='complete', updated_unix=time.time())
  write_json(status_path, status)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
