#!/usr/bin/env python3
"""Run Stage 5C persistent, gated IR recovery across episode blocks."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import pathlib
import platform
import shutil
import sys
import time

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embodied.run.head_audit import _sha256  # noqa: E402
from scripts import rlscape_stage5_common as common  # noqa: E402
from scripts.rlscape_m1_pilot import spec_digest  # noqa: E402
from scripts.rlscape_m1_pilot import write_csv  # noqa: E402
from scripts.rlscape_m1_pilot import write_json  # noqa: E402
from scripts.rlscape_m3_process import active_matches  # noqa: E402
from scripts.rlscape_m3_process import run_managed  # noqa: E402
from scripts.rlscape_m3_sequential import git_revision  # noqa: E402


MODES = ('sampled', 'deterministic')


def installed_version(package):
  try:
    return importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:
    return 'not-installed'


def load_stage5b(root: pathlib.Path) -> tuple[dict, dict]:
  root = root.expanduser().resolve()
  for path in (root / 'stage5b_complete', root / 'experiment_spec.json',
               root / 'task_selection.json'):
    if not path.is_file():
      raise FileNotFoundError(path)
  spec = json.loads((root / 'experiment_spec.json').read_text())
  selection = json.loads((root / 'task_selection.json').read_text())
  if spec.get('canonical_stage') != 'stage5b':
    raise RuntimeError(f'Invalid Stage 5B experiment: {root}')
  digest = selection.pop('digest', None)
  if not digest or spec_digest(selection) != digest:
    raise RuntimeError('Stage 5B task-selection digest mismatch')
  selection['digest'] = digest
  return spec, selection


def heartbeat(args, snapshot):
  args.state['active_child'] = snapshot
  args.state['updated_unix'] = time.time()
  write_json(args.state_path, args.state)


def checkpoint_complete(path: pathlib.Path) -> bool:
  return (path / 'agent.pkl').is_file() and (path / 'done').is_file()


def persistent_manifest(path: pathlib.Path) -> dict:
  manifest_path = path / 'persistent_ir_manifest.json'
  if not checkpoint_complete(path) or not manifest_path.is_file():
    return {}
  return json.loads(manifest_path.read_text())


def _attempt(args, unit, command_builder, recover):
  """Execute one immutable queue unit and recover completed interrupted work."""
  state = args.state['units'].setdefault(unit, {
      'status': 'pending', 'attempts': []})
  if state.get('status') == 'complete':
    attempt = state['attempts'][-1]
    result = recover(pathlib.Path(attempt['logdir']), 0, attempt)
    if result['complete']:
      return result
    state['status'] = 'pending'

  attempt = state['attempts'][-1] if state['attempts'] else None
  if attempt and attempt.get('status') == 'running':
    logdir = pathlib.Path(attempt['logdir'])
    command = command_builder(logdir, attempt)
    result = recover(logdir, 0, attempt)
    if result['complete']:
      attempt.update(status='complete', finished_unix=time.time(),
                     summary=result)
      state['status'] = 'complete'
      write_json(args.state_path, args.state)
      return result
    if not active_matches(args.active_path, command, args.repo_root):
      attempt.update(status='interrupted', finished_unix=time.time())
      attempt = None

  if attempt is None or attempt.get('status') != 'running':
    number = len(state['attempts']) + 1
    if number > args.max_attempts:
      raise RuntimeError(f'Exhausted attempts for {unit}')
    logdir = args.output_root / 'runs' / unit / f'attempt_{number:02d}'
    logdir.mkdir(parents=True, exist_ok=False)
    attempt = {
        'attempt': number, 'status': 'running', 'started_unix': time.time(),
        'logdir': str(logdir)}
    state['attempts'].append(attempt)
  else:
    number = int(attempt['attempt'])
    logdir = pathlib.Path(attempt['logdir'])

  command = command_builder(logdir, attempt)
  attempt['command'] = command
  state['status'] = 'running'
  write_json(args.state_path, args.state)
  print(f'\n=== Stage 5C {unit}, attempt {number} ===', flush=True)
  print(' '.join(command), flush=True)
  child = run_managed(
      command, cwd=args.repo_root, active_path=args.active_path,
      console_path=logdir / 'console.log', activity_paths=(
          logdir / 'metrics.jsonl', logdir / 'episodes.jsonl',
          logdir / 'stage5_trace.jsonl'),
      stream_output=args.stream_output, poll_seconds=args.poll_seconds,
      startup_timeout=args.startup_timeout, stall_timeout=args.stall_timeout,
      heartbeat=lambda value: heartbeat(args, value))
  args.state.pop('active_child', None)
  result = recover(logdir, child['returncode'], attempt)
  attempt.update(
      status='complete' if result['complete'] else 'failed',
      finished_unix=time.time(), child=child, summary=result)
  state['status'] = attempt['status']
  write_json(args.state_path, args.state)
  if not result['complete']:
    if number < args.max_attempts:
      return _attempt(args, unit, command_builder, recover)
    raise RuntimeError(f'Incomplete Stage 5C unit: {unit}')
  return result


def run_recovery_block(args, condition, block, checkpoint):
  unit = f'recovery/{condition["name"]}/block_{block:03d}'
  env_seed = args.recovery_seed_base + block
  agent_seed = args.recovery_agent_seed_base + block

  def build(logdir, attempt):
    output = logdir / 'adapted_checkpoint'
    attempt['output_checkpoint'] = str(output)
    return common.build_eval_command(
        args, logdir=logdir, checkpoint=checkpoint, condition=condition,
        goal=args.target_goal, policy_mode='sampled',
        episodes=args.block_episodes, environment_seed=env_seed,
        agent_seed=agent_seed, persistence='run', output_checkpoint=output,
        resume_adapt_state=(block > 1), warmup_episodes=0)

  def recover(logdir, returncode, attempt):
    output = pathlib.Path(attempt.get(
        'output_checkpoint', logdir / 'adapted_checkpoint'))
    result = common.summarize_eval(
        logdir, condition=condition, goal=args.target_goal,
        policy_mode='sampled', episodes=args.block_episodes,
        environment_seed=env_seed, returncode=returncode,
        warmup_episodes=0, agent_seed=agent_seed,
        paired_rng_stride=args.paired_rng_stride)
    manifest = persistent_manifest(output)
    output_valid = bool(
        manifest and manifest.get('persistence') == 'run' and
        not manifest.get('frozen_component_violations'))
    result.update({
        'canonical_stage': 'stage5c', 'kind': 'recovery',
        'source_kind': args.source_kind, 'block': block,
        'trajectory_condition': condition['name'],
        'input_checkpoint': str(checkpoint),
        'output_checkpoint': str(output), 'output_valid': output_valid,
    })
    result['complete'] = bool(result['complete'] and output_valid)
    return result

  return _attempt(args, unit, build, recover)


def run_clean_eval(args, trajectory, block, checkpoint, role, goal, mode):
  unit = f'clean_eval/{trajectory}/{block:03d}/{role}/{mode}'
  condition = common.no_ir_condition()
  role_index = 0 if role == 'target' else 1
  mode_index = MODES.index(mode)
  env_seed = args.eval_seed_base + block * 100 + role_index * 10 + mode_index
  agent_seed = args.eval_agent_seed_base + block * 100 + role_index * 10 + mode_index
  episodes = args.sampled_episodes if mode == 'sampled' else args.deterministic_episodes

  def build(logdir, attempt):
    del attempt
    return common.build_eval_command(
        args, logdir=logdir, checkpoint=checkpoint, condition=condition,
        goal=goal, policy_mode=mode, episodes=episodes,
        environment_seed=env_seed, agent_seed=agent_seed,
        persistence='episode', warmup_episodes=1)

  def recover(logdir, returncode, attempt):
    del attempt
    result = common.summarize_eval(
        logdir, condition=condition, goal=goal, policy_mode=mode,
        episodes=episodes, environment_seed=env_seed,
        returncode=returncode, warmup_episodes=1, agent_seed=agent_seed,
        paired_rng_stride=args.paired_rng_stride)
    result.update({
        'canonical_stage': 'stage5c', 'kind': 'clean_eval',
        'source_kind': args.source_kind, 'block': block,
        'trajectory_condition': trajectory, 'task_role': role,
        'evaluated_checkpoint': str(checkpoint),
    })
    return result

  return _attempt(args, unit, build, recover)


def completed_output(args, condition, block):
  unit = f'recovery/{condition["name"]}/block_{block:03d}'
  state = args.state['units'].get(unit, {})
  if state.get('status') != 'complete':
    raise RuntimeError(f'Previous recovery block is incomplete: {unit}')
  output = pathlib.Path(state['attempts'][-1]['summary']['output_checkpoint'])
  if not checkpoint_complete(output):
    raise RuntimeError(f'Previous recovery checkpoint is incomplete: {output}')
  return output


def run_trajectory(args, condition):
  recovery_rows = []
  eval_rows = []
  checkpoint = args.source_checkpoint
  for block in range(args.blocks + 1):
    for role, goal in (
        ('target', args.target_goal), ('retained', args.retained_goal)):
      for mode in MODES:
        eval_rows.append(run_clean_eval(
            args, condition['name'], block, checkpoint, role, goal, mode))
        write_csv(args.output_root / 'clean_eval_results.csv', eval_rows)
    if block == args.blocks:
      break
    recovery = run_recovery_block(args, condition, block + 1, checkpoint)
    recovery_rows.append(recovery)
    write_csv(args.output_root / 'recovery_results.csv', recovery_rows)
    checkpoint = completed_output(args, condition, block + 1)
  return recovery_rows, eval_rows


def _rate(row):
  return int(row['successes']) / int(row['episodes'])


def first_stable_block(values, predicate, consecutive=2):
  for index in range(len(values) - consecutive + 1):
    if all(predicate(value) for value in values[index:index + consecutive]):
      return int(values[index]['block'])
  return None


def analyze(args, recovery_rows, eval_rows):
  summaries = []
  for condition in args.matrix:
    name = condition['name']
    recovery = sorted(
        [row for row in recovery_rows if row['trajectory_condition'] == name],
        key=lambda row: int(row['block']))
    for mode in MODES:
      target = sorted([
          row for row in eval_rows if row['trajectory_condition'] == name and
          row['task_role'] == 'target' and row['policy_mode'] == mode],
          key=lambda row: int(row['block']))
      retained = sorted([
          row for row in eval_rows if row['trajectory_condition'] == name and
          row['task_role'] == 'retained' and row['policy_mode'] == mode],
          key=lambda row: int(row['block']))
      recovered = first_stable_block(
          target, lambda row: _rate(row) >= args.recovery_threshold)
      shutoff = first_stable_block(
          recovery, lambda row: (
              float(row['gate_final_pass_sum']) /
              max(1, int(row['environment_steps']))) <= args.shutoff_threshold)
      after = [row for row in recovery
               if recovered is not None and int(row['block']) > recovered]
      summaries.append({
          'source_kind': args.source_kind, 'condition': name,
          'objective_label': condition['objective_label'],
          'gate_kind': condition['gate_kind'], 'policy_mode': mode,
          'target_goal': args.target_goal, 'retained_goal': args.retained_goal,
          'baseline_target_success': _rate(target[0]),
          'best_target_success': max(map(_rate, target)),
          'final_target_success': _rate(target[-1]),
          'baseline_retained_success': _rate(retained[0]),
          'final_retained_success': _rate(retained[-1]),
          'retained_delta': _rate(retained[-1]) - _rate(retained[0]),
          'target_auc': float(np.trapz(
              [_rate(row) for row in target],
              [int(row['block']) * args.block_episodes for row in target]) /
              (args.blocks * args.block_episodes)),
          'stable_recovery_block': recovered,
          'stable_recovery_episodes': (
              recovered * args.block_episodes if recovered is not None else None),
          'stable_shutoff_block': shutoff,
          'actor_updates_total': sum(float(
              row['compute_actor_updates_sum']) for row in recovery),
          'actor_updates_after_recovery': sum(float(
              row['compute_actor_updates_sum']) for row in after),
          'imagined_transitions_total': sum(float(
              row['compute_imagined_transitions_sum']) for row in recovery),
          'recovery_seconds_total': sum(float(
              row['evaluation_seconds']) for row in recovery),
      })
  return summaries


def make_figures(args, recovery_rows, eval_rows):
  import matplotlib.pyplot as plt
  root = args.output_root / 'figures'
  root.mkdir(exist_ok=True)
  names = [condition['name'] for condition in args.matrix]
  colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
  for mode in MODES:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for name, color in zip(names, colors):
      for axis, role in zip(axes, ('target', 'retained')):
        rows = sorted([
            row for row in eval_rows if row['trajectory_condition'] == name and
            row['task_role'] == role and row['policy_mode'] == mode],
            key=lambda row: int(row['block']))
        axis.plot(
            [int(row['block']) * args.block_episodes for row in rows],
            [_rate(row) for row in rows], marker='o', label=name, color=color)
    axes[0].axhline(args.recovery_threshold, color='black', ls='--', lw=1)
    for axis, role in zip(axes, ('Target recovery', 'Retained-task safety')):
      axis.set_title(role)
      axis.set_xlabel('Persistent recovery episodes')
      axis.set_ylabel('Clean success rate')
      axis.set_ylim(0, 1)
    axes[1].legend(fontsize=7, loc='best')
    fig.suptitle(f'Stage 5C persistent recovery ({args.source_kind}, {mode})')
    for suffix in ('png', 'pdf'):
      fig.savefig(root / f'recovery_{mode}.{suffix}', dpi=220)
    plt.close(fig)

  fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
  for name, color in zip(names, colors):
    rows = sorted([
        row for row in recovery_rows if row['trajectory_condition'] == name],
        key=lambda row: int(row['block']))
    axis.plot(
        [int(row['block']) * args.block_episodes for row in rows],
        np.cumsum([float(row['compute_actor_updates_sum']) for row in rows]),
        marker='o', label=name, color=color)
  axis.set_xlabel('Persistent recovery episodes')
  axis.set_ylabel('Cumulative actor updates')
  axis.set_title(f'Stage 5C adaptation compute ({args.source_kind})')
  axis.legend(fontsize=7)
  for suffix in ('png', 'pdf'):
    fig.savefig(root / f'cumulative_actor_updates.{suffix}', dpi=220)
  plt.close(fig)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source-kind', choices=('natural', 'corrupted'),
                      required=True)
  parser.add_argument('--source-checkpoint', type=pathlib.Path, required=True)
  parser.add_argument('--stage5a-root', type=pathlib.Path, required=True)
  parser.add_argument('--stage5b-root', type=pathlib.Path, required=True)
  parser.add_argument('--output-root', type=pathlib.Path, required=True)
  parser.add_argument('--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument('--python', type=pathlib.Path,
                      default=pathlib.Path(sys.executable))
  parser.add_argument('--blocks', type=int, default=10)
  parser.add_argument('--block-episodes', type=int, default=10)
  parser.add_argument('--sampled-episodes', type=int, default=20)
  parser.add_argument('--deterministic-episodes', type=int, default=10)
  parser.add_argument('--recovery-threshold', type=float, default=.70)
  parser.add_argument('--shutoff-threshold', type=float, default=.05)
  parser.add_argument('--episode-length', type=int, default=200)
  parser.add_argument('--grid-columns', type=int, default=28)
  parser.add_argument('--grid-rows', type=int, default=18)
  parser.add_argument('--adapt-steps', type=int, default=1)
  parser.add_argument('--adapt-lr', type=float, default=4e-5)
  parser.add_argument('--adapt-every-k', type=int, default=1)
  parser.add_argument('--start-batch', type=int, default=128)
  parser.add_argument('--paired-rng-stride', type=int, default=1000)
  parser.add_argument('--recovery-seed-base', type=int, default=111_000)
  parser.add_argument('--recovery-agent-seed-base', type=int, default=121_000)
  parser.add_argument('--eval-seed-base', type=int, default=131_000)
  parser.add_argument('--eval-agent-seed-base', type=int, default=141_000)
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


def validate(args):
  args.repo_root = args.repo_root.expanduser().resolve()
  args.python = args.python.expanduser().resolve()
  args.source_checkpoint = args.source_checkpoint.expanduser().resolve()
  args.stage5a_root = args.stage5a_root.expanduser().resolve()
  args.stage5b_root = args.stage5b_root.expanduser().resolve()
  args.output_root = args.output_root.expanduser().resolve()
  for path in (args.python, args.source_checkpoint / 'agent.pkl',
               args.source_checkpoint / 'done'):
    if not path.is_file():
      raise FileNotFoundError(path)
  args.selected_gates = common.load_gate_artifact(args.stage5a_root)
  args.stage5b_spec, args.task_selection = load_stage5b(args.stage5b_root)
  args.target_goal = args.task_selection['weak']['goal']
  args.retained_goal = args.task_selection['strong']['goal']
  args.source_sha256 = _sha256(args.source_checkpoint / 'agent.pkl')
  natural_text = str(args.stage5b_spec['checkpoint'])
  natural_sha256 = str(args.stage5b_spec.get('checkpoint_sha256', ''))
  if not natural_sha256:
    raise RuntimeError('Stage 5B specification lacks checkpoint_sha256')
  if args.source_kind == 'natural' and args.source_sha256 != natural_sha256:
    raise RuntimeError('Natural source content differs from Stage 5B checkpoint')
  if args.source_kind == 'corrupted':
    manifest_path = args.source_checkpoint / 'manifest.json'
    if not manifest_path.is_file():
      raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('source_checkpoint') != natural_text:
      raise RuntimeError('Corrupted source does not derive from Stage 5B checkpoint')
    if manifest.get('target') != 'actor':
      raise RuntimeError('Stage 5C requires an actor-only corruption')
  counts = (args.blocks, args.block_episodes, args.sampled_episodes,
            args.deterministic_episodes)
  if any(value < 1 for value in counts):
    raise ValueError('Stage 5C block and episode counts must be positive')
  if not 0 <= args.recovery_threshold <= 1:
    raise ValueError('Recovery threshold must be in [0, 1]')
  if not 0 <= args.shutoff_threshold <= 1:
    raise ValueError('Shutoff threshold must be in [0, 1]')
  if args.paired_rng_stride <= args.episode_length:
    raise ValueError('paired RNG stride must exceed episode length')


def main(argv=None):
  args = parse_args(argv)
  validate(args)
  args.output_root.mkdir(parents=True, exist_ok=True)
  args.matrix = common.conditions(args.selected_gates)
  spec = {
      'format': 1, 'canonical_stage': 'stage5c',
      'slug': 'persistent_self_terminating_recovery',
      'source_kind': args.source_kind,
      'source_checkpoint': str(args.source_checkpoint),
      'source_sha256': args.source_sha256,
      'stage5a_gate_digest': args.selected_gates['digest'],
      'stage5b_spec_digest': args.stage5b_spec['digest'],
      'stage5b_task_selection_digest': args.task_selection['digest'],
      'target_goal': args.target_goal, 'retained_goal': args.retained_goal,
      'conditions': [condition['name'] for condition in args.matrix],
      'blocks': args.blocks, 'block_episodes': args.block_episodes,
      'recovery_episode_budget': args.blocks * args.block_episodes,
      'clean_eval': {
          'sampled_episodes': args.sampled_episodes,
          'deterministic_episodes': args.deterministic_episodes,
          'frequency_blocks': 1},
      'adaptation': {
          'horizon': 15, 'posterior_samples': args.start_batch,
          'steps': args.adapt_steps, 'lr': args.adapt_lr,
          'persistence': 'run', 'optimizer_state_persistent': True,
          'world_model_frozen': True, 'critic_frozen': True},
      'analysis': {
          'recovery_threshold': args.recovery_threshold,
          'shutoff_threshold': args.shutoff_threshold,
          'consecutive_blocks': 2, 'early_stopping': False},
      'git_commit': git_revision(args.repo_root),
      'python_version': platform.python_version(),
      'rlscape_version': installed_version('rl-scape'),
  }
  spec['digest'] = spec_digest(spec)
  spec_path = args.output_root / 'experiment_spec.json'
  if spec_path.is_file() and json.loads(
      spec_path.read_text()).get('digest') != spec['digest']:
    raise RuntimeError('Existing Stage 5C root has a different specification')
  if not spec_path.is_file():
    write_json(spec_path, spec)
  if args.dry_run:
    print(json.dumps(spec, indent=2))
    return 0

  args.state_path = args.output_root / 'queue.json'
  args.active_path = args.output_root / 'active_child.json'
  args.state = (
      json.loads(args.state_path.read_text()) if args.state_path.is_file()
      else {'format': 1, 'canonical_stage': 'stage5c',
            'spec_digest': spec['digest'], 'status': 'running', 'units': {}})
  if args.state.get('spec_digest') != spec['digest']:
    raise RuntimeError('Stage 5C queue specification mismatch')
  write_json(args.state_path, args.state)

  recovery_rows = []
  eval_rows = []
  for condition in args.matrix:
    recovery, evaluations = run_trajectory(args, condition)
    recovery_rows.extend(recovery)
    eval_rows.extend(evaluations)
    write_csv(args.output_root / 'recovery_results.csv', recovery_rows)
    write_csv(args.output_root / 'clean_eval_results.csv', eval_rows)
  write_json(args.output_root / 'recovery_results.json', {
      'format': 1, 'canonical_stage': 'stage5c', 'rows': recovery_rows})
  write_json(args.output_root / 'clean_eval_results.json', {
      'format': 1, 'canonical_stage': 'stage5c', 'rows': eval_rows})
  summary = analyze(args, recovery_rows, eval_rows)
  write_csv(args.output_root / 'trajectory_summary.csv', summary)
  write_json(args.output_root / 'trajectory_summary.json', {
      'format': 1, 'canonical_stage': 'stage5c', 'rows': summary})
  make_figures(args, recovery_rows, eval_rows)
  args.state.update(status='complete', updated_unix=time.time())
  write_json(args.state_path, args.state)
  (args.output_root / 'stage5c_complete').write_bytes(b'')
  print(f'Stage 5C complete: {args.output_root}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
