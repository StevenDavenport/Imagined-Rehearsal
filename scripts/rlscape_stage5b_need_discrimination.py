#!/usr/bin/env python3
"""Run Stage 5B IR-necessity discrimination on strong and weak tasks."""

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
from scripts import rlscape_stage2_actor_recovery as stage2  # noqa: E402
from scripts import rlscape_stage4b_conservative_value as stage4b  # noqa: E402
from scripts import rlscape_stage5_common as common  # noqa: E402
from scripts.rlscape_m1_pilot import spec_digest  # noqa: E402
from scripts.rlscape_m1_pilot import write_csv  # noqa: E402
from scripts.rlscape_m1_pilot import write_json  # noqa: E402
from scripts.rlscape_m3_process import active_matches  # noqa: E402
from scripts.rlscape_m3_process import run_managed  # noqa: E402
from scripts.rlscape_m3_sequential import git_revision  # noqa: E402


GOALS = stage2.GOALS
MODES = stage2.POLICY_MODES


def installed_version(package):
  try:
    return importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:
    return 'not-installed'


def heartbeat(args, snapshot):
  args.state['active_child'] = snapshot
  args.state['updated_unix'] = time.time()
  write_json(args.state_path, args.state)


def summarize(args, *, logdir, condition, goal, role, mode, episodes,
              replicate, seed_base, returncode, phase):
  env_seed = seed_base + replicate
  agent_seed = seed_base + 10_000 + replicate
  result = common.summarize_eval(
      logdir, condition=condition, goal=goal, policy_mode=mode,
      episodes=episodes, environment_seed=env_seed, returncode=returncode,
      warmup_episodes=1, agent_seed=agent_seed,
      paired_rng_stride=args.paired_rng_stride)
  result.update({
      'canonical_stage': 'stage5b', 'phase': phase, 'task_role': role,
      'replicate': replicate, 'environment_seed': env_seed,
      'agent_seed': agent_seed, 'source_checkpoint': str(args.checkpoint),
  })
  return result


def execute(args, *, phase, condition, goal, role, mode, episodes,
            replicate, seed_base):
  unit = (
      f'{phase}/replicate_{replicate:02d}/{role}/{condition["name"]}/'
      f'{goal}/{mode}')
  state = args.state['units'].setdefault(unit, {
      'status': 'pending', 'attempts': []})

  def recover(path, code):
    return summarize(
        args, logdir=path, condition=condition, goal=goal, role=role,
        mode=mode, episodes=episodes, replicate=replicate,
        seed_base=seed_base, returncode=code, phase=phase)

  if state.get('status') == 'complete':
    result = recover(pathlib.Path(state['attempts'][-1]['logdir']), 0)
    if result['complete']:
      return result
    state['status'] = 'pending'

  attempt = state['attempts'][-1] if state['attempts'] else None
  if attempt and attempt.get('status') == 'running':
    logdir = pathlib.Path(attempt['logdir'])
    command = common.build_eval_command(
        args, logdir=logdir, checkpoint=args.checkpoint,
        condition=condition, goal=goal, policy_mode=mode,
        episodes=episodes, environment_seed=seed_base + replicate,
        agent_seed=seed_base + 10_000 + replicate)
    result = recover(logdir, 0)
    if result['complete']:
      attempt.update(status='complete', finished_unix=time.time(),
                     summary=result)
      state['status'] = 'complete'
      return result
    if not active_matches(args.active_path, command, args.repo_root):
      attempt.update(status='interrupted', finished_unix=time.time())
      attempt = None

  if attempt is None or attempt.get('status') != 'running':
    number = len(state['attempts']) + 1
    if number > args.max_attempts:
      raise RuntimeError(f'Exhausted attempts for {unit}')
    logdir = args.output_root / 'evaluations' / unit / f'attempt_{number:02d}'
    logdir.mkdir(parents=True, exist_ok=False)
    attempt = {
        'attempt': number, 'status': 'running', 'started_unix': time.time(),
        'logdir': str(logdir)}
    state['attempts'].append(attempt)
  else:
    number = int(attempt['attempt'])
    logdir = pathlib.Path(attempt['logdir'])

  command = common.build_eval_command(
      args, logdir=logdir, checkpoint=args.checkpoint,
      condition=condition, goal=goal, policy_mode=mode,
      episodes=episodes, environment_seed=seed_base + replicate,
      agent_seed=seed_base + 10_000 + replicate)
  attempt['command'] = command
  state['status'] = 'running'
  write_json(args.state_path, args.state)
  print(f'\n=== Stage 5B {unit}, attempt {number} ===', flush=True)
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
  result = recover(logdir, child['returncode'])
  attempt.update(
      status='complete' if result['complete'] else 'failed',
      finished_unix=time.time(), child=child, summary=result)
  state['status'] = attempt['status']
  write_json(args.state_path, args.state)
  if not result['complete']:
    if number < args.max_attempts:
      return execute(
          args, phase=phase, condition=condition, goal=goal, role=role,
          mode=mode, episodes=episodes, replicate=replicate,
          seed_base=seed_base)
    raise RuntimeError(f'Incomplete Stage 5B unit: {unit}')
  return result


def run_matrix(args, *, phase, conditions, tasks, replicates, episodes,
               seed_base):
  rows = []
  for replicate in range(replicates):
    for role, goal in tasks:
      for condition in conditions:
        for mode in MODES:
          rows.append(execute(
              args, phase=phase, condition=condition, goal=goal, role=role,
              mode=mode, episodes=episodes[mode], replicate=replicate,
              seed_base=seed_base))
          write_csv(args.output_root / f'{phase}_results.csv', rows)
  write_json(args.output_root / f'{phase}_results.json', {
      'format': 1, 'canonical_stage': 'stage5b', 'phase': phase,
      'rows': rows})
  return rows


def success(rows, condition, goal, mode):
  chosen = [row for row in rows if row['condition'] == condition and
            row['goal'] == goal and row['policy_mode'] == mode]
  return sum(int(row['successes']) for row in chosen) / sum(
      int(row['episodes']) for row in chosen)


def select_tasks(args, rows):
  records = []
  for goal in GOALS:
    baseline = float(np.mean([
        success(rows, 'no_ir', goal, mode) for mode in MODES]))
    benefits = {}
    for objective in common.OBJECTIVES:
      benefits[objective] = float(np.mean([
          success(rows, f'always_{objective}', goal, mode) -
          success(rows, 'no_ir', goal, mode) for mode in MODES]))
    records.append({
        'goal': goal, 'no_ir_competence': baseline,
        'benefit_standard': benefits['standard'],
        'benefit_reward_only': benefits['reward_only'],
        'best_ir_benefit': max(benefits.values()),
        'worst_ir_benefit': min(benefits.values()),
    })
  weak_pool = [row for row in records
               if row['best_ir_benefit'] >= args.min_weak_benefit]
  if not weak_pool:
    raise RuntimeError(
        'Calibration found no task with the required positive IR benefit; '
        f'inspect {args.output_root / "calibration_results.csv"}')
  weak = max(weak_pool, key=lambda row: (
      row['best_ir_benefit'], -row['no_ir_competence']))
  strong_pool = [
      row for row in records if row['goal'] != weak['goal'] and
      row['no_ir_competence'] >= args.min_strong_success]
  if not strong_pool:
    raise RuntimeError(
        'Calibration found no distinct competent task above '
        f'{args.min_strong_success:.3f}')
  strong = max(strong_pool, key=lambda row: (
      row['no_ir_competence'], -row['worst_ir_benefit']))
  selection = {
      'format': 1, 'canonical_stage': 'stage5b',
      'strong': strong, 'weak': weak, 'all_tasks': records,
      'criteria': {
          'min_strong_success': args.min_strong_success,
          'min_weak_benefit': args.min_weak_benefit}}
  selection['digest'] = spec_digest(selection)
  return selection


def paired_analysis(rows):
  mismatches = []
  for replicate in sorted({int(row['replicate']) for row in rows}):
    for role in ('strong', 'weak'):
      for mode in MODES:
        selected = [row for row in rows
                    if int(row['replicate']) == replicate and
                    row['task_role'] == role and row['policy_mode'] == mode]
        by_name = {row['condition']: row for row in selected}
        reference = stage2.reset_identities(
            pathlib.Path(by_name['no_ir']['logdir']))
        for row in selected:
          if stage2.reset_identities(pathlib.Path(row['logdir'])) != reference:
            mismatches.append((replicate, role, mode, row['condition']))
            continue
          baseline_name = (
              'no_ir' if row['objective_label'] == 'no_ir' else
              f'always_{row["objective_label"]}')
          baseline = stage4b.episode_outcomes(
              pathlib.Path(by_name[baseline_name]['logdir']))
          treatment = stage4b.episode_outcomes(pathlib.Path(row['logdir']))
          seed = int(spec_digest({
              'stage': '5b', 'replicate': replicate, 'role': role,
              'mode': mode, 'condition': row['condition']})[:8], 16)
          row.update(stage4b.paired_effect(
              baseline, treatment, seed=seed, prefix='always'))
  if mismatches:
    raise RuntimeError(f'Stage 5B reset pairing failed: {mismatches[:3]}')
  return {'format': 1, 'equal': True, 'mismatches': []}


def aggregate(rows):
  output = []
  for role in ('strong', 'weak'):
    for mode in MODES:
      for name in sorted({row['condition'] for row in rows}):
        chosen = [row for row in rows if row['task_role'] == role and
                  row['policy_mode'] == mode and row['condition'] == name]
        if not chosen:
          continue
        source = chosen[0]
        steps = sum(int(row['environment_steps']) for row in chosen)
        output.append({
            'task_role': role, 'goal': source['goal'], 'policy_mode': mode,
            'condition': name, 'objective_label': source['objective_label'],
            'gate_kind': source['gate_kind'],
            'success_rate': sum(int(row['successes']) for row in chosen) /
            sum(int(row['episodes']) for row in chosen),
            'gate_activation_rate': sum(float(
                row['gate_final_pass_sum']) for row in chosen) / steps,
            'actor_updates_per_step': sum(float(
                row['compute_actor_updates_sum']) for row in chosen) / steps,
            'imagined_transitions_per_step': sum(float(
                row['compute_imagined_transitions_sum']) for row in chosen) /
            steps,
            'seconds_per_step': sum(float(
                row['evaluation_seconds']) for row in chosen) / steps,
            'paired_delta_vs_always': float(np.mean([
                float(row.get('paired_delta_vs_always', 0.0))
                for row in chosen])),
            'episodes': sum(int(row['episodes']) for row in chosen),
        })
  return output


def make_figures(root, summary):
  import matplotlib.pyplot as plt
  figure_root = root / 'figures'
  figure_root.mkdir(exist_ok=True)
  for mode in MODES:
    chosen = [row for row in summary if row['policy_mode'] == mode]
    names = sorted({row['condition'] for row in chosen})
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    for index, role in enumerate(('strong', 'weak')):
      lookup = {(row['task_role'], row['condition']): row for row in chosen}
      values = [lookup[(role, name)]['success_rate'] for name in names]
      axes[index].bar(x, values)
      axes[index].set_xticks(x, names, rotation=40, ha='right')
      axes[index].set_ylim(0, 1)
      axes[index].set_title(f'{role.capitalize()} task')
      axes[index].set_ylabel('Success rate')
    fig.suptitle(f'Stage 5B necessity discrimination ({mode})')
    for suffix in ('png', 'pdf'):
      fig.savefig(figure_root / f'performance_{mode}.{suffix}', dpi=220)
    plt.close(fig)

  gated = [row for row in summary if row['gate_kind'] != 'none']
  fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
  for row in gated:
    marker = 'o' if row['task_role'] == 'weak' else 'x'
    axis.scatter(row['gate_activation_rate'], row['success_rate'], marker=marker)
    axis.annotate(
        f'{row["task_role"]}:{row["condition"]}',
        (row['gate_activation_rate'], row['success_rate']), fontsize=7)
  axis.set_xlabel('Gate activations per real step')
  axis.set_ylabel('Success rate')
  axis.set_title('Gate necessity separation')
  for suffix in ('png', 'pdf'):
    fig.savefig(figure_root / f'activation_separation.{suffix}', dpi=220)
  plt.close(fig)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', type=pathlib.Path, required=True)
  parser.add_argument('--stage5a-root', type=pathlib.Path, required=True)
  parser.add_argument('--output-root', type=pathlib.Path)
  parser.add_argument('--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument('--python', type=pathlib.Path,
                      default=pathlib.Path(sys.executable))
  parser.add_argument('--phase', choices=('calibrate', 'confirm', 'all'),
                      default='all')
  parser.add_argument('--calibration-sampled-episodes', type=int, default=20)
  parser.add_argument('--calibration-deterministic-episodes', type=int,
                      default=10)
  parser.add_argument('--replicates', type=int, default=3)
  parser.add_argument('--sampled-episodes', type=int, default=50)
  parser.add_argument('--deterministic-episodes', type=int, default=25)
  parser.add_argument('--min-strong-success', type=float, default=.60)
  parser.add_argument('--min-weak-benefit', type=float, default=.05)
  parser.add_argument('--episode-length', type=int, default=200)
  parser.add_argument('--grid-columns', type=int, default=28)
  parser.add_argument('--grid-rows', type=int, default=18)
  parser.add_argument('--adapt-steps', type=int, default=1)
  parser.add_argument('--adapt-lr', type=float, default=4e-5)
  parser.add_argument('--adapt-every-k', type=int, default=1)
  parser.add_argument('--start-batch', type=int, default=128)
  parser.add_argument('--paired-rng-stride', type=int, default=1000)
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
  args.checkpoint = args.checkpoint.expanduser().resolve()
  args.stage5a_root = args.stage5a_root.expanduser().resolve()
  args.python = args.python.expanduser().resolve()
  args.output_root = (
      args.output_root.expanduser().resolve() if args.output_root else
      args.checkpoint.parent / 'stage5b_need_discrimination')
  for path in (args.python, args.checkpoint / 'agent.pkl',
               args.checkpoint / 'done'):
    if not path.is_file():
      raise FileNotFoundError(path)
  args.selected = common.load_gate_artifact(args.stage5a_root)
  if args.paired_rng_stride <= args.episode_length:
    raise ValueError('paired RNG stride must exceed episode length')
  counts = (
      args.calibration_sampled_episodes,
      args.calibration_deterministic_episodes, args.replicates,
      args.sampled_episodes, args.deterministic_episodes)
  if any(value < 1 for value in counts):
    raise ValueError('Episode and replicate counts must be positive')


def main(argv=None):
  args = parse_args(argv)
  validate(args)
  args.output_root.mkdir(parents=True, exist_ok=True)
  selected_digest = args.selected['digest']
  matrix = common.conditions(args.selected)
  spec = {
      'format': 1, 'canonical_stage': 'stage5b',
      'slug': 'ir_necessity_discrimination',
      'checkpoint': str(args.checkpoint),
      'checkpoint_sha256': _sha256(args.checkpoint / 'agent.pkl'),
      'stage5a_gate_artifact': args.selected['_path'],
      'stage5a_gate_digest': selected_digest,
      'conditions': [item['name'] for item in matrix],
      'goals': list(GOALS), 'modes': list(MODES),
      'calibration': {
          'sampled_episodes': args.calibration_sampled_episodes,
          'deterministic_episodes': args.calibration_deterministic_episodes,
          'min_strong_success': args.min_strong_success,
          'min_weak_benefit': args.min_weak_benefit},
      'confirmation': {
          'replicates': args.replicates,
          'sampled_episodes': args.sampled_episodes,
          'deterministic_episodes': args.deterministic_episodes,
          'units': args.replicates * 2 * 7 * 2,
          'episodes': args.replicates * 2 * 7 * (
              args.sampled_episodes + args.deterministic_episodes)},
      'adaptation': {
          'horizon': 15, 'posterior_samples': args.start_batch,
          'steps': args.adapt_steps, 'lr': args.adapt_lr,
          'persistence': 'episode'},
      'git_commit': git_revision(args.repo_root),
      'python_version': platform.python_version(),
      'rlscape_version': installed_version('rl-scape'),
  }
  spec['digest'] = spec_digest(spec)
  spec_path = args.output_root / 'experiment_spec.json'
  if spec_path.is_file() and json.loads(
      spec_path.read_text()).get('digest') != spec['digest']:
    raise RuntimeError('Existing Stage 5B root has a different specification')
  if not spec_path.is_file():
    write_json(spec_path, spec)
  if args.dry_run:
    print(json.dumps(spec, indent=2))
    return 0

  args.state_path = args.output_root / 'queue.json'
  args.active_path = args.output_root / 'active_child.json'
  args.state = (
      json.loads(args.state_path.read_text()) if args.state_path.is_file()
      else {'format': 1, 'canonical_stage': 'stage5b',
            'spec_digest': spec['digest'], 'status': 'running', 'units': {}})
  if args.state.get('spec_digest') != spec['digest']:
    raise RuntimeError('Stage 5B queue specification mismatch')
  write_json(args.state_path, args.state)

  selection_path = args.output_root / 'task_selection.json'
  if args.phase in ('calibrate', 'all'):
    calibration_conditions = [matrix[0], matrix[1], matrix[2]]
    rows = run_matrix(
        args, phase='calibration', conditions=calibration_conditions,
        tasks=[('candidate', goal) for goal in GOALS], replicates=1,
        episodes={
            'sampled': args.calibration_sampled_episodes,
            'deterministic': args.calibration_deterministic_episodes},
        seed_base=81_000)
    selection = select_tasks(args, rows)
    write_json(selection_path, selection)
    if args.phase == 'calibrate':
      print(f'Stage 5B calibration complete: {args.output_root}')
      return 0
  if not selection_path.is_file():
    raise FileNotFoundError(selection_path)
  selection = json.loads(selection_path.read_text())

  if args.phase in ('confirm', 'all'):
    tasks = [
        ('strong', selection['strong']['goal']),
        ('weak', selection['weak']['goal'])]
    rows = run_matrix(
        args, phase='confirmation', conditions=matrix, tasks=tasks,
        replicates=args.replicates,
        episodes={'sampled': args.sampled_episodes,
                  'deterministic': args.deterministic_episodes},
        seed_base=91_000)
    pairing = paired_analysis(rows)
    write_json(args.output_root / 'pairing.json', pairing)
    write_csv(args.output_root / 'confirmation_results.csv', rows)
    summary = aggregate(rows)
    write_csv(args.output_root / 'condition_summary.csv', summary)
    write_json(args.output_root / 'aggregate_summary.json', {
        'format': 1, 'canonical_stage': 'stage5b', 'rows': summary})
    make_figures(args.output_root, summary)
    args.state.update(status='complete', updated_unix=time.time())
    write_json(args.state_path, args.state)
    (args.output_root / 'stage5b_complete').write_bytes(b'')
    print(f'Stage 5B complete: {args.output_root}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
