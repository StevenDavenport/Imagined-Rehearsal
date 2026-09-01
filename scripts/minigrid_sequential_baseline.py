#!/usr/bin/env python3
"""Run the restart-safe, three-arm sequential MiniGrid baseline.

The experiment deliberately does not use imagined rehearsal or alter the
reward, continuation, critic, or actor objectives. Its purpose is to expose
where ordinary sequential Dreamer succeeds or fails before CIR is introduced.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embodied.envs.minigrid_registry import ALTERNATING_CHAIN  # noqa: E402
from embodied.envs.minigrid_registry import TASK_INDEX  # noqa: E402
from embodied.envs.minigrid_registry import TASK_NAMES  # noqa: E402
from embodied.run import milestones  # noqa: E402
from embodied.run.head_audit import create_selection  # noqa: E402
from embodied.run.head_audit import load_selection  # noqa: E402
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


ARMS = {
    'fifo_100k': {
        'kind': 'fifo',
        'description': (
            '100k-transition FIFO diagnostic control; deliberately not '
            'suitable as a future CIR memory'),
    },
    'uniform_reservoir': {
        'kind': 'episode_reservoir',
        'sampling': 'uniform_groups',
        'description': (
            'Primary CIR baseline: Algorithm-R episode reservoir per goal '
            'with uniform sampling over represented goals'),
    },
    'current_old_50_50': {
        'kind': 'episode_reservoir',
        'sampling': 'current_old',
        'description': (
            'Replay-allocation control: half current goal and half uniformly '
            'divided over represented old goals'),
    },
}
POLICY_MODES = ('sampled', 'deterministic')
MODEL_SIZES = ('1m', '12m', '25m')


def snapshot_steps(total_steps: int, every: int) -> tuple[int, ...]:
  return tuple(range(0, int(total_steps) + 1, int(every)))


def phase_for_step(step: int, phase_steps: int, phases: int) -> int:
  if step <= 0:
    return 0
  return min((int(step) - 1) // int(phase_steps), phases - 1)


def evaluation_units(spec: dict) -> list[dict]:
  units = []
  tasks = tuple(spec['tasks'])
  for step in spec['snapshot_steps']:
    phase = phase_for_step(step, spec['phase_steps'], len(tasks))
    evaluated = (
        tasks if step % spec['eval_all_every'] == 0 else (tasks[phase],))
    for task in evaluated:
      for mode in spec['policy_modes']:
        units.append({
            'step': int(step), 'phase': int(phase), 'task': task,
            'policy_mode': mode,
        })
  return units


def build_train_command(
    *, args, arm: str, task: str, phase: int, seed: int,
    training: pathlib.Path, snapshot_root: pathlib.Path,
    milestone_root: pathlib.Path, digest: str, resume_steps: int = 0,
) -> list[str]:
  end_step = (phase + 1) * args.phase_steps
  command = [
      str(args.python), str(args.repo_root / 'dreamerv3' / 'main.py'),
      '--configs', 'minigrid', f'minigrid_size{args.model_size}',
      f'minigrid_{task}', 'minigrid_sequential',
  ]
  command.extend(getattr(args, 'extra_train_configs', ()))
  definition = ARMS[arm]
  if definition['kind'] == 'episode_reservoir':
    command.append('minigrid_episode_reservoir')
  command.extend([
      '--seed', str(seed),
      '--jax.platform', str(args.jax_platform),
      '--logdir', str(training),
      '--run.steps', str(end_step),
      '--run.train_ratio', str(args.train_ratio),
      '--run.milestone_every', str(args.phase_steps),
      '--run.milestone_dir', str(milestone_root),
      '--run.milestone_goal', task,
      '--run.milestone_phase', str(phase),
      '--run.milestone_spec_digest', digest,
      '--run.snapshot_every', str(args.snapshot_every),
      '--run.snapshot_start', '0',
      '--run.snapshot_dir', str(snapshot_root),
      '--run.snapshot_goal', task,
      '--run.snapshot_phase', str(phase),
      '--run.snapshot_spec_digest', digest,
      '--env.minigrid.episode_length', str(args.episode_length),
      '--env.minigrid.seed_offset', str(
          phase * args.environment_seed_stride + int(resume_steps)),
      '--replay.kind', definition['kind'],
      '--replay.online', 'False',
      '--replay.save_wait', 'True',
  ])
  if definition['kind'] == 'fifo':
    command.extend(['--replay.size', str(args.fifo_replay_size)])
  else:
    command.extend([
        '--replay.size', str(args.episodes_per_goal * len(TASK_NAMES)),
        '--replay.groups', str(len(TASK_NAMES)),
        '--replay.group_key', 'goal_id',
        '--replay.retention', 'reservoir',
        '--replay.sampling', definition['sampling'],
        '--replay.current_group', str(TASK_INDEX[task]),
        '--replay.current_fraction', str(args.current_fraction),
    ])
  return command


def build_eval_command(
    *, args, task: str, checkpoint: pathlib.Path, logdir: pathlib.Path,
    eval_seed: int, policy_mode: str, paired_seed: int,
) -> list[str]:
  return [
      str(args.python), str(args.repo_root / 'dreamerv3' / 'main.py'),
      '--configs', 'minigrid', f'minigrid_size{args.model_size}',
      f'minigrid_{task}', 'minigrid_eval',
      '--seed', str(eval_seed),
      '--jax.platform', str(args.jax_platform),
      '--logdir', str(logdir),
      '--run.from_checkpoint', str(checkpoint),
      '--run.eval_episodes', str(args.eval_episodes_per_mode),
      '--run.eval_policy_mode', policy_mode,
      '--env.minigrid.episode_length', str(args.episode_length),
      '--eval_adapt.paired_rng', 'True',
      '--eval_adapt.paired_rng_seed', str(paired_seed),
      '--eval_adapt.paired_rng_stride', str(args.paired_rng_stride),
  ]


def build_diagnostic_command(
    *, args, kind: str, task: str, checkpoint: pathlib.Path,
    selection: pathlib.Path, logdir: pathlib.Path, seed: int,
) -> list[str]:
  if kind not in ('head', 'critic_provenance'):
    raise ValueError(kind)
  config = (
      'minigrid_head_audit' if kind == 'head' else
      'minigrid_critic_provenance')
  prefix = 'head_audit' if kind == 'head' else 'critic_provenance'
  command = [
      str(args.python), str(args.repo_root / 'dreamerv3' / 'main.py'),
      '--configs', 'minigrid', f'minigrid_size{args.model_size}',
      f'minigrid_{task}', config,
      '--seed', str(seed),
      '--jax.platform', str(args.jax_platform),
      '--logdir', str(logdir),
      '--run.from_checkpoint', str(checkpoint),
      f'--{prefix}.selection', str(selection),
      f'--{prefix}.chunk_length', str(args.diagnostic_chunk_length),
  ]
  if kind == 'critic_provenance':
    command.extend([
        '--critic_provenance.horizons',
        ','.join(str(value) for value in args.critic_horizons),
        '--critic_provenance.posterior_samples',
        str(args.critic_posterior_samples),
        '--critic_provenance.anchors_per_episode',
        str(args.critic_anchors_per_episode),
        '--critic_provenance.seed_base',
        str(args.diagnostic_seed_base + seed * 1000),
    ])
  return command


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Run the visibility-first sequential MiniGrid baseline.')
  parser.add_argument('--experiment-root', type=pathlib.Path, required=True)
  parser.add_argument(
      '--experiment-name',
      default='minigrid_visibility_first_sequential_baseline')
  parser.add_argument(
      '--scientific-role',
      default=(
          'pre-CIR baseline; no imagined rehearsal, counterfactual training, '
          'critic regularization, or objective changes'))
  parser.add_argument('--primary-objectives', nargs='+')
  parser.add_argument('--extra-train-configs', nargs='+', default=[])
  parser.add_argument('--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument(
      '--python', type=pathlib.Path, default=pathlib.Path(sys.executable))
  parser.add_argument('--tasks', nargs='+', default=list(ALTERNATING_CHAIN))
  parser.add_argument('--arms', nargs='+', choices=tuple(ARMS),
                      default=list(ARMS))
  parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
  parser.add_argument('--model-size', choices=MODEL_SIZES, default='12m')
  parser.add_argument('--jax-platform', choices=('cuda', 'cpu'), default='cuda')
  parser.add_argument('--phase-steps', type=int, default=200_000)
  parser.add_argument('--snapshot-every', type=int, default=25_000)
  parser.add_argument('--eval-all-every', type=int, default=50_000)
  parser.add_argument('--episode-length', type=int, default=100)
  parser.add_argument('--fifo-replay-size', type=int, default=100_000)
  parser.add_argument('--episodes-per-goal', type=int, default=1_000)
  parser.add_argument('--current-fraction', type=float, default=0.5)
  parser.add_argument('--train-ratio', type=float, default=32.0)
  parser.add_argument('--environment-seed-stride', type=int, default=1_000_000)
  parser.add_argument('--eval-episodes-per-mode', type=int, default=50)
  parser.add_argument('--policy-modes', nargs='+', choices=POLICY_MODES,
                      default=list(POLICY_MODES))
  parser.add_argument('--eval-seed-base', type=int, default=300_000)
  parser.add_argument('--paired-rng-seed-base', type=int,
                      default=202_608_300)
  parser.add_argument('--paired-rng-stride', type=int, default=1_000)
  parser.add_argument('--bank-successes-per-goal', type=int, default=16)
  parser.add_argument('--bank-failures-per-goal', type=int, default=16)
  parser.add_argument('--diagnostic-chunk-length', type=int, default=32)
  parser.add_argument('--critic-horizons', nargs='+', type=int,
                      default=[1, 3, 6, 15])
  parser.add_argument('--critic-posterior-samples', type=int, default=32)
  parser.add_argument('--critic-anchors-per-episode', type=int, default=2)
  parser.add_argument('--diagnostic-seed-base', type=int,
                      default=202_608_301)
  parser.add_argument('--max-attempts', type=int, default=3)
  parser.add_argument('--min-free-gb', type=float, default=50.0)
  parser.add_argument('--poll-seconds', type=float, default=30.0)
  parser.add_argument('--startup-timeout', type=float, default=3600.0)
  parser.add_argument('--stall-timeout', type=float, default=7200.0)
  parser.add_argument('--stream-output', action='store_true')
  parser.add_argument('--skip-eval', action='store_true')
  parser.add_argument('--skip-diagnostics', action='store_true')
  parser.add_argument('--dry-run', action='store_true')
  return parser.parse_args(argv)


def preflight(args) -> None:
  args.repo_root = args.repo_root.expanduser().resolve()
  args.experiment_root = args.experiment_root.expanduser().resolve()
  # Do not resolve environment interpreter symlinks into the base interpreter.
  args.python = args.python.expanduser()
  if not args.python.is_absolute():
    args.python = (pathlib.Path.cwd() / args.python).absolute()
  args.tasks = validate_tasks(args.tasks)
  args.arms = tuple(dict.fromkeys(args.arms))
  args.seeds = tuple(int(value) for value in args.seeds)
  args.policy_modes = tuple(dict.fromkeys(args.policy_modes))
  args.extra_train_configs = tuple(dict.fromkeys(args.extra_train_configs))
  args.critic_horizons = tuple(int(value) for value in args.critic_horizons)
  if not args.python.is_file():
    raise FileNotFoundError(args.python)
  if not (args.repo_root / 'dreamerv3' / 'main.py').is_file():
    raise FileNotFoundError(args.repo_root / 'dreamerv3' / 'main.py')
  if not args.seeds or len(args.seeds) != len(set(args.seeds)):
    raise ValueError('--seeds must contain unique values')
  if not args.arms:
    raise ValueError('--arms cannot be empty')
  if not args.policy_modes:
    raise ValueError('--policy-modes cannot be empty')
  if any(not re.fullmatch(r'[a-zA-Z0-9_]+', value)
         for value in args.extra_train_configs):
    raise ValueError('--extra-train-configs must contain config names')
  positive = (
      args.phase_steps, args.snapshot_every, args.eval_all_every,
      args.episode_length, args.fifo_replay_size, args.episodes_per_goal,
      args.eval_episodes_per_mode, args.paired_rng_stride,
      args.diagnostic_chunk_length, args.critic_posterior_samples,
      args.critic_anchors_per_episode, args.max_attempts)
  if any(value <= 0 for value in positive):
    raise ValueError(f'Positive experiment settings required: {positive}')
  if args.phase_steps % args.snapshot_every:
    raise ValueError('--phase-steps must be divisible by --snapshot-every')
  if args.eval_all_every % args.snapshot_every:
    raise ValueError('--eval-all-every must be divisible by --snapshot-every')
  if args.phase_steps % args.eval_all_every:
    raise ValueError('--phase-steps must be divisible by --eval-all-every')
  if args.snapshot_every % 10:
    raise ValueError('--snapshot-every must be divisible by 10')
  if args.paired_rng_stride <= args.episode_length:
    raise ValueError('paired RNG stride must exceed episode length')
  if not 0 <= args.current_fraction <= 1:
    raise ValueError('--current-fraction must be in [0, 1]')
  if not args.critic_horizons or any(x <= 0 for x in args.critic_horizons):
    raise ValueError('--critic-horizons must contain positive values')
  if len(args.critic_horizons) != len(set(args.critic_horizons)):
    raise ValueError('--critic-horizons cannot contain duplicates')
  if not args.skip_diagnostics and 'uniform_reservoir' not in args.arms:
    raise ValueError(
        'Diagnostics use the uniform-reservoir final replay as the canonical '
        'state bank; include that arm or pass --skip-diagnostics')
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
          f'JAX 0.4.33 is required, found {versions["jax"]}')
    free = shutil.disk_usage(args.experiment_root).free / 1024 ** 3
    if free < args.min_free_gb:
      raise RuntimeError(
          f'Only {free:.1f} GiB free; require {args.min_free_gb:.1f} GiB')


def _snapshot_set_complete(root: pathlib.Path, steps) -> bool:
  try:
    for step in steps:
      milestones.validate_checkpoint(
          milestones.archive_path(root, step), expected_step=step, full=False)
    return True
  except (FileNotFoundError, milestones.MilestoneError):
    return False


def _phase_complete(run_root: pathlib.Path, phase: int, args) -> bool:
  end = (phase + 1) * args.phase_steps
  snapshots = range(0, end + 1, args.snapshot_every)
  try:
    milestones.validate(
        milestones.archive_path(run_root / 'milestones', end),
        expected_step=end, full=False)
  except (FileNotFoundError, milestones.MilestoneError):
    return False
  return bool(
      checkpoint_step(run_root / 'training') >= end and
      _snapshot_set_complete(run_root / 'snapshots', snapshots))


def _execute_attempt(
    *, command, logdir, marker, attempts, args, active_path, heartbeat,
    activity_paths,
) -> bool:
  """Execute/adopt a restart-safe child and require its output marker."""
  if marker.is_file():
    return True
  attempt = None
  if attempts and 'finished_unix' not in attempts[-1]:
    candidate = attempts[-1]
    if active_matches(active_path, candidate['command'], args.repo_root):
      candidate['recovery'] = 'adopting_live_child'
      attempt = candidate
    elif marker.is_file():
      candidate.update(
          finished_unix=time.time(), returncode=None, timed_out=False,
          complete=True, recovery='completed_while_supervisor_offline')
      return True
    else:
      candidate.update(
          finished_unix=time.time(), returncode=None, timed_out=False,
          complete=False, recovery='abandoned_dead_partial_attempt')
  if attempt is None:
    if len(attempts) >= args.max_attempts:
      return False
    attempt = {
        'attempt': len(attempts) + 1, 'started_unix': time.time(),
        'command': list(command), 'logdir': str(logdir),
    }
    attempts.append(attempt)
  logdir.mkdir(parents=True, exist_ok=True)
  child = run_managed(
      list(attempt['command']), cwd=args.repo_root, active_path=active_path,
      console_path=logdir / 'console.log', activity_paths=activity_paths,
      stream_output=args.stream_output, poll_seconds=args.poll_seconds,
      startup_timeout=args.startup_timeout, stall_timeout=args.stall_timeout,
      heartbeat=heartbeat)
  complete = bool(marker.is_file() and child['returncode'] in (0, None))
  attempt.update(
      finished_unix=time.time(), returncode=child['returncode'],
      adopted=child['adopted'], timed_out=child['timed_out'],
      complete=complete)
  return complete


def _paired_seed(args, seed: int, step: int, task: str) -> int:
  return int(
      args.paired_rng_seed_base + seed * 10_000_000 +
      step * 10 + TASK_INDEX[task])


def _condition_summary(condition_root, unit, args, returncode=None):
  status_path = condition_root / 'condition_status.json'
  if not status_path.is_file():
    return None
  status = json.loads(status_path.read_text())
  summary = status.get('summary')
  if not summary:
    return None
  refreshed = evaluation_summary(
      pathlib.Path(summary['logdir']), task=unit['task'],
      training_seed=unit['seed'], eval_seed=unit['eval_seed'],
      checkpoint=unit['step'], policy_mode=unit['policy_mode'],
      requested_episodes=args.eval_episodes_per_mode,
      paired_rng_seed=unit['paired_seed'],
      paired_rng_stride=args.paired_rng_stride,
      returncode=status.get('returncode', returncode))
  return refreshed if refreshed['complete'] else None


def _write_initial_state(args, spec, digest):
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
  status = {
      'format': 1,
      'spec_digest': digest,
      'status': 'dry_run' if args.dry_run else 'running',
      'created_unix': existing.get('created_unix', time.time()),
      'updated_unix': time.time(),
      'states': existing.get('states', {}),
      'diagnostics': existing.get('diagnostics', {}),
      'runtime_versions': package_versions(),
  }
  write_json(status_path, status)
  return status_path, status


def main(argv=None) -> int:
  args = parse_args(argv)
  preflight(args)
  total_steps = args.phase_steps * len(args.tasks)
  snapshots = snapshot_steps(total_steps, args.snapshot_every)
  phase_boundaries = tuple(
      range(0, total_steps + 1, args.phase_steps))
  spec = {
      'format': 1,
      'name': args.experiment_name,
      'git_commit': git_revision(args.repo_root),
      'scientific_role': args.scientific_role,
      'primary_objectives': args.primary_objectives or [
          'complete sequential acquisition',
          'measure catastrophic forgetting',
          'localize failure across replay, world model, reward, '
          'continuation, critic, and actor',
      ],
      'tasks': list(args.tasks),
      'task_goal_ids': {task: TASK_INDEX[task] for task in args.tasks},
      'seeds': list(args.seeds),
      'arms': {arm: ARMS[arm] for arm in args.arms},
      'model_size': args.model_size,
      'jax_platform': args.jax_platform,
      'phase_steps': args.phase_steps,
      'total_steps_per_run': total_steps,
      'total_training_steps': total_steps * len(args.arms) * len(args.seeds),
      'snapshot_every': args.snapshot_every,
      'snapshot_steps': list(snapshots),
      'snapshot_contents': 'agent, optimizer, normalizers, and global step',
      'phase_boundaries': list(phase_boundaries),
      'phase_checkpoint_contents': 'agent plus exact replay state',
      'eval_all_every': args.eval_all_every,
      'evaluation_enabled': not args.skip_eval,
      'evaluation_schedule': (
          f'active task every {args.snapshot_every} steps; all tasks every '
          f'{args.eval_all_every} steps, including zero and phase boundaries'),
      'eval_episodes_per_mode': args.eval_episodes_per_mode,
      'policy_modes': list(args.policy_modes),
      'evaluation_pairing': (
          'identical environment reset seeds across arms and policy modes'),
      'eval_seed_base': args.eval_seed_base,
      'paired_rng_seed_base': args.paired_rng_seed_base,
      'paired_rng_stride': args.paired_rng_stride,
      'episode_length': args.episode_length,
      'train_ratio': args.train_ratio,
      'training_environment_seed_policy': (
          'phase_index * environment_seed_stride + recovered steps within '
          'phase; identical across arms absent a crash'),
      'environment_seed_stride': args.environment_seed_stride,
      'fifo_replay_size_transitions': args.fifo_replay_size,
      'reservoir_episodes_per_model_goal': args.episodes_per_goal,
      'reservoir_model_goal_slots': len(TASK_NAMES),
      'current_old_fraction': args.current_fraction,
      'goal_conditioning': 'six-way one-hot heads; goal-agnostic RSSM',
      'observation': '64x64 partial RGB',
      'action_space': 'Discrete(7)',
      'diagnostics': {
          'canonical_bank_source': (
              'final uniform-reservoir replay, independently fixed per seed'),
          'successes_per_goal': args.bank_successes_per_goal,
          'failures_per_goal': args.bank_failures_per_goal,
          'checkpoint_steps': list(phase_boundaries),
          'head_audit': [
              'factual and counterfactual reward/continuation',
              'online and slow critic calibration',
              'same-state goal sweeps',
              'same-state categorical actor distributions',
          ],
          'critic_provenance': {
              'horizons': list(args.critic_horizons),
              'posterior_samples': args.critic_posterior_samples,
              'anchors_per_episode': args.critic_anchors_per_episode,
              'proposals': ['actor', 'recorded control'],
              'decomposition': ['reward-only', 'bootstrap'],
          },
      },
      'diagnostics_enabled': not args.skip_diagnostics,
      'live_instrumentation': [
          'per-goal replay episodes/transitions/sample totals',
          'per-goal dynamics/reconstruction/reward/continuation losses',
          'per-goal policy/value losses',
      ],
      'dependency_requirements': {
          'minigrid': '==3.1.0', 'gymnasium': '>=1.0,<2',
          'jax': '==0.4.33'},
  }
  if args.extra_train_configs:
    spec['extra_train_configs'] = list(args.extra_train_configs)
  spec['evaluation_units_per_run'] = len(evaluation_units(spec))
  digest = spec_digest(spec)
  status_path, status = _write_initial_state(args, spec, digest)
  states = status['states']
  active_path = args.experiment_root / 'active_child.json'

  def persist():
    status['updated_unix'] = time.time()
    write_json(status_path, status)

  def heartbeat(snapshot):
    status['active_child'] = snapshot
    persist()

  # Train each run through A -> B -> C -> D in one persistent log/replay root.
  for arm in args.arms:
    for seed in args.seeds:
      run_id = f'{arm}__seed_{seed:04d}'
      run_root = args.experiment_root / 'runs' / arm / f'seed_{seed:04d}'
      state = states.setdefault(run_id, {
          'arm': arm, 'seed': seed, 'phases': {}, 'evaluations': {}})
      for phase, task in enumerate(args.tasks):
        phase_id = f'phase_{phase:02d}_{task}'
        phase_state = state['phases'].setdefault(
            phase_id, {'task': task, 'phase': phase, 'attempts': []})
        command = build_train_command(
            args=args, arm=arm, task=task, phase=phase, seed=seed,
            training=run_root / 'training',
            snapshot_root=run_root / 'snapshots',
            milestone_root=run_root / 'milestones', digest=digest)
        complete = _phase_complete(run_root, phase, args)
        if complete:
          phase_state['status'] = 'complete'
          persist()
          continue
        if args.dry_run:
          phase_state.update(status='dry_run', planned_command=command)
          persist()
          continue
        adopted_attempt = None
        if (
            phase_state['attempts'] and
            'finished_unix' not in phase_state['attempts'][-1]
        ):
          candidate = phase_state['attempts'][-1]
          if active_matches(
              active_path, candidate['command'], args.repo_root):
            candidate['recovery'] = 'adopting_live_child'
            adopted_attempt = candidate
            command = list(candidate['command'])
          else:
            candidate.update(
                finished_unix=time.time(), returncode=None,
                timed_out=False, complete=False,
                recovery='abandoned_dead_partial_attempt')
        while not complete:
          if adopted_attempt is not None:
            attempt = adopted_attempt
            adopted_attempt = None
          else:
            if len(phase_state['attempts']) >= args.max_attempts:
              break
            current_step = checkpoint_step(run_root / 'training')
            resume_steps = max(0, current_step - phase * args.phase_steps)
            command = build_train_command(
                args=args, arm=arm, task=task, phase=phase, seed=seed,
                training=run_root / 'training',
                snapshot_root=run_root / 'snapshots',
                milestone_root=run_root / 'milestones', digest=digest,
                resume_steps=resume_steps)
            attempt = {
                'attempt': len(phase_state['attempts']) + 1,
                'started_unix': time.time(), 'command': command,
                'step_before': current_step,
                'environment_resume_offset': resume_steps}
            phase_state['attempts'].append(attempt)
          phase_state['status'] = 'running'
          persist()
          child = run_managed(
              command, cwd=args.repo_root, active_path=active_path,
              console_path=run_root / 'training' / 'console.log',
              activity_paths=[
                  run_root / 'training' / 'metrics.jsonl',
                  run_root / 'training' / 'episodes.jsonl',
                  run_root / 'training' / 'ckpt' / 'latest'],
              stream_output=args.stream_output,
              poll_seconds=args.poll_seconds,
              startup_timeout=args.startup_timeout,
              stall_timeout=args.stall_timeout,
              heartbeat=heartbeat)
          status.pop('active_child', None)
          complete = _phase_complete(run_root, phase, args)
          attempt.update(
              finished_unix=time.time(), returncode=child['returncode'],
              adopted=child['adopted'], timed_out=child['timed_out'],
              step_after=checkpoint_step(run_root / 'training'),
              complete=complete)
          phase_state['status'] = 'complete' if complete else 'retrying'
          persist()
        if not complete:
          phase_state['status'] = 'failed'
          status['status'] = 'failed'
          persist()
          return 1

  # Evaluate immutable snapshots only after training. Evaluations cannot alter
  # the live sequential checkpoint and use paired environment/action RNG.
  if not args.skip_eval:
    units = evaluation_units(spec)
    for arm in args.arms:
      for seed in args.seeds:
        run_id = f'{arm}__seed_{seed:04d}'
        state = states[run_id]
        run_root = args.experiment_root / 'runs' / arm / f'seed_{seed:04d}'
        for planned in units:
          step, task, mode = (
              planned['step'], planned['task'], planned['policy_mode'])
          condition_id = f'step_{step:012d}__{task}__{mode}'
          condition = state['evaluations'].setdefault(
              condition_id, {'attempts': [], **planned})
          condition_root = (
              run_root / 'evaluations' / f'step_{step:012d}' / task / mode)
          paired_seed = _paired_seed(args, seed, step, task)
          eval_seed = args.eval_seed_base + seed
          unit = {
              **planned, 'seed': seed, 'eval_seed': eval_seed,
              'paired_seed': paired_seed}
          summary = _condition_summary(condition_root, unit, args)
          if summary:
            condition.update(status='complete', summary=summary)
            persist()
            continue
          checkpoint = milestones.archive_path(
              run_root / 'snapshots', step) / 'checkpoint'
          if not args.dry_run:
            milestones.validate_checkpoint(
                checkpoint.parent, expected_step=step, full=False)
          attempt_number = len(condition['attempts']) + 1
          logdir = condition_root / f'attempt_{attempt_number:02d}'
          command = build_eval_command(
              args=args, task=task, checkpoint=checkpoint, logdir=logdir,
              eval_seed=eval_seed, policy_mode=mode, paired_seed=paired_seed)
          if args.dry_run:
            condition.update(
                status='dry_run', planned_command=command,
                planned_logdir=str(logdir), paired_seed=paired_seed)
            continue
          summary = None
          adopted_attempt = None
          if (
              condition['attempts'] and
              'finished_unix' not in condition['attempts'][-1]
          ):
            candidate = condition['attempts'][-1]
            candidate_logdir = pathlib.Path(candidate['logdir'])
            recovered = evaluation_summary(
                candidate_logdir, task=task, training_seed=seed,
                eval_seed=eval_seed, checkpoint=step, policy_mode=mode,
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
            elif active_matches(
                active_path, candidate['command'], args.repo_root):
              candidate['recovery'] = 'adopting_live_child'
              adopted_attempt = candidate
            else:
              candidate.update(
                  finished_unix=time.time(), returncode=None,
                  timed_out=False, complete=False,
                  recovery='abandoned_dead_partial_attempt')
          while not summary:
            if adopted_attempt is not None:
              attempt = adopted_attempt
              adopted_attempt = None
              logdir = pathlib.Path(attempt['logdir'])
              command = list(attempt['command'])
            else:
              if len(condition['attempts']) >= args.max_attempts:
                break
              attempt_number = len(condition['attempts']) + 1
              logdir = condition_root / f'attempt_{attempt_number:02d}'
              command = build_eval_command(
                  args=args, task=task, checkpoint=checkpoint, logdir=logdir,
                  eval_seed=eval_seed, policy_mode=mode,
                  paired_seed=paired_seed)
              attempt = {
                  'attempt': attempt_number, 'started_unix': time.time(),
                  'command': command, 'logdir': str(logdir)}
              condition['attempts'].append(attempt)
            condition['status'] = 'running'
            persist()
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
                eval_seed=eval_seed, checkpoint=step, policy_mode=mode,
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
              'status': 'complete' if summary else 'failed',
              'attempts': condition['attempts'], 'summary': summary,
              'returncode': (
                  condition['attempts'][-1].get('returncode')
                  if condition['attempts'] else None),
          }
          write_json(condition_root / 'condition_status.json', condition_status)
          condition.update(
              status=condition_status['status'], summary=summary,
              paired_seed=paired_seed)
          if not summary:
            status['status'] = 'failed'
            persist()
            return 1
          # The two modes must use the exact same environment reset stream.
          other = next((
              value for key, value in state['evaluations'].items()
              if key.startswith(f'step_{step:012d}__{task}__') and
              value.get('status') == 'complete' and
              value.get('policy_mode') != mode), None)
          if other:
            other_logdir = pathlib.Path(other['summary']['logdir'])
            if reset_seeds(other_logdir) != reset_seeds(logdir):
              raise RuntimeError(
                  f'Paired evaluation reset mismatch: {condition_id}')
          persist()

    if not args.dry_run:
      pairing_records = []
      for seed in args.seeds:
        for planned in units:
          step, task = planned['step'], planned['task']
          # One record covers both modes and all replay arms. The same
          # task/step appears twice in ``units``; emit it on the first mode.
          if planned['policy_mode'] != args.policy_modes[0]:
            continue
          schedules = {}
          for arm in args.arms:
            state = states[f'{arm}__seed_{seed:04d}']
            for mode in args.policy_modes:
              condition_id = f'step_{step:012d}__{task}__{mode}'
              summary = state['evaluations'][condition_id]['summary']
              schedules[f'{arm}/{mode}'] = reset_seeds(
                  pathlib.Path(summary['logdir']))
          reference = next(iter(schedules.values()))
          equal = all(value == reference for value in schedules.values())
          pairing_records.append({
              'seed': seed, 'step': step, 'task': task,
              'members': sorted(schedules), 'episodes': len(reference),
              'equal': equal,
          })
          if not equal:
            raise RuntimeError(
                f'Cross-arm evaluation reset mismatch: seed={seed}, '
                f'step={step}, task={task}')
      write_json(args.experiment_root / 'evaluation_pairing.json', {
          'format': 1, 'paired_across_arms_and_modes': True,
          'records': pairing_records})

  # Fix one same-state episode bank per seed from the final uniform reservoir,
  # then probe every arm at all phase boundaries against that immutable bank.
  if not args.skip_diagnostics:
    for seed in args.seeds:
      bank_root = args.experiment_root / 'diagnostics' / 'banks' / f'seed_{seed:04d}'
      selection_path = bank_root / 'selection.json'
      replay = (
          milestones.archive_path(
              args.experiment_root / 'runs' / 'uniform_reservoir' /
              f'seed_{seed:04d}' / 'milestones', total_steps) / 'replay')
      bank_state = status['diagnostics'].setdefault(
          f'bank__seed_{seed:04d}', {})
      if args.dry_run:
        bank_state.update(
            status='dry_run', replay=str(replay),
            selection=str(selection_path))
      else:
        if selection_path.is_file():
          selection = load_selection(selection_path)
          if selection['replay_dir'] != str(replay.resolve()):
            raise RuntimeError(
                f'Existing bank has another replay: {selection_path}')
        else:
          selection = create_selection(
              replay, selection_path,
              goals=tuple(TASK_INDEX[task] for task in args.tasks),
              successes_per_goal=args.bank_successes_per_goal,
              failures_per_goal=args.bank_failures_per_goal,
              seed=args.diagnostic_seed_base + seed)
        bank_state.update(status='complete', counts=selection['counts'])
        persist()

      for arm in args.arms:
        run_root = args.experiment_root / 'runs' / arm / f'seed_{seed:04d}'
        for step in phase_boundaries:
          checkpoint = milestones.archive_path(
              run_root / 'snapshots', step) / 'checkpoint'
          for kind in ('head', 'critic_provenance'):
            diagnostic_id = (
                f'{kind}__{arm}__seed_{seed:04d}__step_{step:012d}')
            diagnostic = status['diagnostics'].setdefault(
                diagnostic_id, {'attempts': []})
            logdir = (
                args.experiment_root / 'diagnostics' / kind / arm /
                f'seed_{seed:04d}' / f'step_{step:012d}')
            marker = logdir / 'audit_complete'
            command = build_diagnostic_command(
                args=args, kind=kind, task=args.tasks[0],
                checkpoint=checkpoint, selection=selection_path,
                logdir=logdir, seed=seed)
            if marker.is_file():
              diagnostic['status'] = 'complete'
              persist()
              continue
            if args.dry_run:
              diagnostic.update(status='dry_run', planned_command=command)
              continue
            complete = False
            while not complete:
              unfinished = bool(
                  diagnostic['attempts'] and
                  'finished_unix' not in diagnostic['attempts'][-1])
              if len(diagnostic['attempts']) >= args.max_attempts and not unfinished:
                break
              diagnostic['status'] = 'running'
              persist()
              complete = _execute_attempt(
                  command=command, logdir=logdir, marker=marker,
                  attempts=diagnostic['attempts'], args=args,
                  active_path=active_path, heartbeat=heartbeat,
                  activity_paths=[logdir / 'console.log',
                                  logdir / 'summary.json'])
              status.pop('active_child', None)
              persist()
            diagnostic['status'] = 'complete' if complete else 'failed'
            if not complete:
              status['status'] = 'failed'
              persist()
              return 1
            persist()

  if args.dry_run:
    status['status'] = 'dry_run'
  else:
    from scripts.minigrid_sequential_report import generate_report
    generate_report(args.experiment_root)
    status['status'] = 'complete'
    status.pop('active_child', None)
  persist()
  print(
      f'Sequential MiniGrid baseline {status["status"]}: '
      f'{args.experiment_root}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
