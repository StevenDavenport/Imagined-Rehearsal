#!/usr/bin/env python3
"""Run Stage 5A signal audit, gate qualification, and confirmation."""

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
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embodied.run.head_audit import _sha256  # noqa: E402
from scripts import rlscape_stage2_actor_recovery as stage2  # noqa: E402
from scripts import rlscape_stage4b_conservative_value as stage4b  # noqa: E402
from scripts.rlscape_m1_pilot import read_jsonl  # noqa: E402
from scripts.rlscape_m1_pilot import spec_digest  # noqa: E402
from scripts.rlscape_m1_pilot import write_csv  # noqa: E402
from scripts.rlscape_m1_pilot import write_json  # noqa: E402
from scripts.rlscape_m3_process import active_matches  # noqa: E402
from scripts.rlscape_m3_process import run_managed  # noqa: E402
from scripts.rlscape_m3_sequential import git_revision  # noqa: E402


GOALS = stage2.GOALS
POLICY_MODES = stage2.POLICY_MODES
OBJECTIVES = ('standard', 'reward_only', 'mixed_b005')
QUANTILES = (.50, .75, .90)


def installed_version(package):
  try:
    return importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:
    return 'not-installed'


def discover_diagnostic_sources(args):
  """Find prior competent, forgotten, and controlled-corruption strata."""
  sources = []
  if args.skip_diagnostic_strata:
    return sources
  experiment_root = args.stage4a_root.parent
  for label, step in (('competent_1250k', 1_250_000),
                      ('forgotten_1500k', 1_500_000)):
    checkpoint = experiment_root / 'milestones' / f'step_{step:012d}' / 'checkpoint'
    if (checkpoint / 'agent.pkl').is_file():
      sources.append((label, checkpoint))
  selection_path = experiment_root / 'stage2_actor_recovery' / 'corruption_selection.json'
  if selection_path.is_file():
    selection = json.loads(selection_path.read_text())
    value = selection.get('selected', {}).get('checkpoint', '')
    checkpoint = pathlib.Path(value).expanduser() if value else None
    if checkpoint and (checkpoint / 'agent.pkl').is_file():
      sources.append(('controlled_corruption', checkpoint.resolve()))
  return sources


def always_condition(objective):
  mapped = {
      'standard': ('standard', 0.0, 0.0),
      'reward_only': ('mixed_bounded', 0.0, 0.0),
      'mixed_b005': ('mixed_bounded', .05, 0.0),
  }
  name, beta, actent = mapped[objective]
  return dict(
      name=f'always_{objective}', family='always', objective=name,
      objective_label=objective,
      checkpoint='stage4a_counterfactual_bounded',
      enabled=True, horizon=15, mix_beta=beta,
      actent=actent, gate_kind='none', gate_enabled=False,
      gate_audit=False, entropy_threshold=1.0, js_threshold=1.0,
      min_success_rate=4 / 128, direction_threshold=1.0)


def no_ir_condition(audit=False):
  return dict(
      name='no_ir', family='no_ir', objective='standard',
      objective_label='no_ir', checkpoint='stage4a_counterfactual_bounded',
      enabled=False, horizon=15, mix_beta=0.0,
      actent=0.0, gate_kind='none', gate_enabled=False,
      gate_audit=bool(audit), entropy_threshold=1.0, js_threshold=1.0,
      min_success_rate=4 / 128, direction_threshold=1.0)


def gated_condition(objective, kind, thresholds, suffix=''):
  condition = always_condition(objective)
  condition.update(
      name=f'{kind}_{objective}{suffix}', family=kind,
      gate_kind=kind, gate_enabled=True, gate_audit=False,
      entropy_threshold=float(thresholds['entropy_threshold']),
      js_threshold=float(thresholds['js_threshold']),
      min_success_rate=float(thresholds.get('min_success_rate', 4 / 128)),
      direction_threshold=float(thresholds.get('direction_threshold', 1.0)))
  return condition


def audit_conditions():
  result = [no_ir_condition(audit=True)]
  for objective in OBJECTIVES:
    condition = always_condition(objective)
    condition['gate_audit'] = True
    result.append(condition)
  return result


def confirmation_conditions(selected):
  result = [no_ir_condition()]
  result += [always_condition(objective) for objective in OBJECTIVES]
  for kind in ('entropy', 'js', 'two_stage'):
    result += [
        gated_condition(objective, kind, selected[kind])
        for objective in OBJECTIVES]
  return result


def build_command(
    args, *, logdir, checkpoint, condition, goal, policy_mode,
    episodes, replicate, seed_base,
):
  environment_seed = seed_base + replicate
  agent_seed = seed_base + 10_000 + replicate
  command = stage2.build_command(
      python=args.python, repo_root=args.repo_root, logdir=logdir,
      checkpoint=checkpoint, reference_checkpoint=checkpoint,
      condition=condition, goal=goal, policy_mode=policy_mode,
      episodes=episodes, seed=environment_seed,
      episode_length=args.episode_length, grid_columns=args.grid_columns,
      grid_rows=args.grid_rows, adapt_steps=args.adapt_steps,
      adapt_lr=args.adapt_lr, adapt_every_k=args.adapt_every_k,
      start_batch=args.start_batch, mvn_path=args.mvn_path,
      java_home=args.java_home,
      extra_configs=('rlscape_stage4a_bounded_value',))
  command += [
      '--run.eval_warmup_episodes', '1',
      '--eval_adapt.mix_beta', str(float(condition['mix_beta'])),
      '--eval_adapt.paired_rng', 'True',
      '--eval_adapt.paired_rng_seed', str(agent_seed),
      '--eval_adapt.paired_rng_stride', str(args.paired_rng_stride),
      '--eval_adapt.trace.enabled', 'True',
      '--eval_adapt.trace.filename', 'stage5a_trace.jsonl',
      '--eval_adapt.gate.enabled', str(bool(condition['gate_enabled'])),
      '--eval_adapt.gate.kind', str(condition['gate_kind']),
      '--eval_adapt.gate.audit', str(bool(condition['gate_audit'])),
      '--eval_adapt.gate.entropy_threshold',
      str(float(condition['entropy_threshold'])),
      '--eval_adapt.gate.js_threshold',
      str(float(condition['js_threshold'])),
      '--eval_adapt.gate.min_success_rate',
      str(float(condition['min_success_rate'])),
      '--eval_adapt.gate.direction_threshold',
      str(float(condition['direction_threshold'])),
  ]
  return command


def trace_summary(logdir):
  rows = read_jsonl(logdir / 'stage5a_trace.jsonl')
  keys = (
      'gate_evaluated', 'gate_cheap_pass', 'gate_consequence_evaluated',
      'gate_consequence_pass', 'gate_final_pass', 'gate_actor_entropy',
      'gate_posterior_entropy', 'gate_js_disagreement',
      'gate_success_rate', 'gate_success_action_concentration',
      'gate_success_action_divergence', 'compute_cheap_seconds',
      'compute_consequence_seconds', 'compute_adapt_seconds',
      'compute_posterior_samples', 'compute_imagined_transitions',
      'compute_actor_updates')
  summary = {}
  for key in keys:
    values = [float(row[key]) for row in rows if key in row]
    if not values:
      summary[f'{key}_mean'] = float('nan')
      summary[f'{key}_sum'] = 0.0
    else:
      summary[f'{key}_mean'] = float(np.mean(values))
      summary[f'{key}_sum'] = float(np.sum(values))
  summary['trace_rows'] = len(rows)
  runtime_path = logdir / 'eval_runtime.json'
  runtime = json.loads(runtime_path.read_text()) if runtime_path.is_file() else {}
  summary.update({
      'evaluation_seconds': float(runtime.get('evaluation_seconds', np.nan)),
      'warmup_seconds': float(runtime.get('warmup_seconds', np.nan)),
      'environment_steps': int(runtime.get('environment_steps', 0)),
      'steps_per_second': float(runtime.get('steps_per_second', np.nan)),
  })
  return summary


def summarize(
    args, *, logdir, checkpoint, condition, goal, policy_mode,
    episodes, replicate, seed_base, phase, returncode,
):
  environment_seed = seed_base + replicate
  result = stage2.summarize_condition(
      logdir, condition=condition, goal=goal, policy_mode=policy_mode,
      episodes=episodes, seed=environment_seed, returncode=returncode)
  integrity_path = logdir / 'eval_integrity.json'
  integrity = json.loads(integrity_path.read_text()) if integrity_path.is_file() else {}
  expected_agent_seed = seed_base + 10_000 + replicate
  # Every Stage 5A unit runs one unrecorded compilation/timing warm-up
  # episode. It appears in the environment reset audit but deliberately not
  # in the requested episode or outcome logs.
  expected_resets = int(episodes) + 1
  actual_resets = len(stage2.reset_identities(logdir))
  result.update({
      'canonical_stage': 'stage5a', 'phase': phase,
      'source_checkpoint': str(checkpoint), 'replicate': replicate,
      'environment_seed': environment_seed, 'agent_seed': expected_agent_seed,
      'objective_label': condition['objective_label'],
      'warmup_episodes': 1,
      'gate_kind': condition['gate_kind'],
      'gate_enabled': bool(condition['gate_enabled']),
      'mix_beta': float(condition['mix_beta']),
      'entropy_threshold': float(condition['entropy_threshold']),
      'js_threshold': float(condition['js_threshold']),
      'min_success_rate': float(condition['min_success_rate']),
      'direction_threshold': float(condition['direction_threshold']),
      'paired_rng_verified': bool(
          integrity.get('paired_rng') is True and
          integrity.get('paired_rng_seed') == expected_agent_seed and
          integrity.get('paired_rng_stride') == args.paired_rng_stride),
  })
  result.update(trace_summary(logdir))
  result['reset_count'] = actual_resets
  result['complete'] = bool(
      returncode in (None, 0) and result['episode_count_exact'] and
      result['integrity_equal'] and actual_resets == expected_resets and
      result['paired_rng_verified'])
  return result


def heartbeat(args, snapshot):
  args.state['active_child'] = snapshot
  args.state['updated_unix'] = time.time()
  write_json(args.state_path, args.state)


def execute(
    args, *, phase, checkpoint, condition, goal, policy_mode,
    episodes, replicate, seed_base,
):
  unit_id = (
      f'{phase}/replicate_{replicate:02d}/{condition["name"]}/'
      f'{goal}/{policy_mode}')
  state = args.state['units'].setdefault(unit_id, {
      'status': 'pending', 'attempts': []})

  def recover(logdir, returncode):
    return summarize(
        args, logdir=logdir, checkpoint=checkpoint, condition=condition,
        goal=goal, policy_mode=policy_mode, episodes=episodes,
        replicate=replicate, seed_base=seed_base, phase=phase,
        returncode=returncode)

  if state.get('status') == 'complete':
    attempt = state['attempts'][-1]
    result = recover(pathlib.Path(attempt['logdir']), 0)
    if result['complete']:
      attempt['summary'] = result
      return result
    state['status'] = 'pending'

  attempt = state['attempts'][-1] if state['attempts'] else None
  if attempt and attempt.get('status') == 'running':
    number = int(attempt['attempt'])
    logdir = pathlib.Path(attempt['logdir'])
    command = build_command(
        args, logdir=logdir, checkpoint=checkpoint, condition=condition,
        goal=goal, policy_mode=policy_mode, episodes=episodes,
        replicate=replicate, seed_base=seed_base)
    result = recover(logdir, 0)
    if result['complete']:
      attempt.update(status='complete', finished_unix=time.time(), summary=result)
      state['status'] = 'complete'
      return result
    if not active_matches(args.active_path, command, args.repo_root):
      attempt.update(status='interrupted', finished_unix=time.time())
      attempt = None

  if attempt is None or attempt.get('status') != 'running':
    number = len(state['attempts']) + 1
    if number > args.max_attempts:
      raise RuntimeError(f'Exhausted attempts for {unit_id}')
    logdir = args.output_root / 'evaluations' / unit_id / f'attempt_{number:02d}'
    logdir.mkdir(parents=True, exist_ok=False)
    command = build_command(
        args, logdir=logdir, checkpoint=checkpoint, condition=condition,
        goal=goal, policy_mode=policy_mode, episodes=episodes,
        replicate=replicate, seed_base=seed_base)
    attempt = {
        'attempt': number, 'status': 'running', 'started_unix': time.time(),
        'logdir': str(logdir), 'command': command}
    state['attempts'].append(attempt)
  else:
    command = build_command(
        args, logdir=logdir, checkpoint=checkpoint, condition=condition,
        goal=goal, policy_mode=policy_mode, episodes=episodes,
        replicate=replicate, seed_base=seed_base)

  command_path = args.output_root / 'commands' / f'{unit_id}.json'
  command_path.parent.mkdir(parents=True, exist_ok=True)
  write_json(command_path, {'unit_id': unit_id, 'command': command})
  state['status'] = 'running'
  write_json(args.state_path, args.state)
  print(f'\n=== Stage 5A {unit_id}, attempt {number} ===', flush=True)
  print(' '.join(command), flush=True)
  child = run_managed(
      command, cwd=args.repo_root, active_path=args.active_path,
      console_path=logdir / 'console.log', activity_paths=(
          logdir / 'metrics.jsonl', logdir / 'episodes.jsonl',
          logdir / 'stage5a_trace.jsonl'),
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
          args, phase=phase, checkpoint=checkpoint, condition=condition,
          goal=goal, policy_mode=policy_mode, episodes=episodes,
          replicate=replicate, seed_base=seed_base)
    raise RuntimeError(f'Incomplete Stage 5A unit: {unit_id}')
  return result


def run_matrix(
    args, *, phase, conditions, episodes, replicates, seed_base,
    checkpoint=None,
):
  checkpoint = checkpoint or args.checkpoint
  rows = []
  for replicate in range(replicates):
    for condition in conditions:
      for goal in GOALS:
        for mode in POLICY_MODES:
          count = episodes[mode]
          rows.append(execute(
              args, phase=phase, checkpoint=checkpoint,
              condition=condition, goal=goal, policy_mode=mode,
              episodes=count, replicate=replicate, seed_base=seed_base))
          write_csv(args.output_root / f'{phase}_results.csv', rows)
  write_json(args.output_root / f'{phase}_results.json', {
      'format': 1, 'canonical_stage': 'stage5a', 'phase': phase, 'rows': rows})
  return rows


def trace_values(rows, key, predicate=lambda row: True):
  values = []
  for result in rows:
    for row in read_jsonl(pathlib.Path(result['logdir']) / 'stage5a_trace.jsonl'):
      if key in row and predicate(row):
        values.append(float(row[key]))
  return np.asarray(values, np.float64)


def quantile_candidates(audit_rows):
  entropy = trace_values(audit_rows, 'gate_actor_entropy')
  disagreement = trace_values(audit_rows, 'gate_js_disagreement')
  direction = trace_values(
      audit_rows, 'gate_success_action_divergence',
      lambda row: float(row.get('gate_success_rate', 0)) >= 4 / 128)
  if not len(entropy) or not len(disagreement) or not len(direction):
    raise RuntimeError('Stage 5A audit did not produce complete gate features')
  return {
      'quantiles': list(QUANTILES),
      'entropy': np.quantile(entropy, QUANTILES).tolist(),
      'js': np.quantile(disagreement, QUANTILES).tolist(),
      'direction': np.quantile(direction, QUANTILES).tolist(),
      'counts': {
          'entropy': len(entropy), 'js': len(disagreement),
          'direction': len(direction)},
  }


def make_audit_figures(output_root, audit_rows, diagnostic_rows=()):
  import matplotlib.pyplot as plt
  figure_root = output_root / 'figures'
  figure_root.mkdir(exist_ok=True)
  points = []
  for result in (*audit_rows, *diagnostic_rows):
    for row in read_jsonl(pathlib.Path(result['logdir']) / 'stage5a_trace.jsonl'):
      if 'gate_actor_entropy' in row:
        points.append(row)
  if not points:
    return
  entropy = np.asarray([row['gate_actor_entropy'] for row in points])
  posterior = np.asarray([row['gate_posterior_entropy'] for row in points])
  disagreement = np.asarray([row['gate_js_disagreement'] for row in points])
  success = np.asarray([row.get('gate_success_rate', 0) for row in points])
  direction = np.asarray([
      row.get('gate_success_action_divergence', 0) for row in points])

  fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
  axes[0].hist(entropy, bins=30, alpha=.65, label='factual actor')
  axes[0].hist(posterior, bins=30, alpha=.65, label='MCPB mean')
  axes[0].set_xlabel('Normalized entropy'); axes[0].set_ylabel('States')
  axes[0].legend(frameon=False)
  image = axes[1].scatter(entropy, disagreement, c=success, s=8,
                          cmap='viridis', alpha=.6)
  axes[1].set_xlabel('Factual actor entropy')
  axes[1].set_ylabel('MCPB JS disagreement')
  fig.colorbar(image, ax=axes[1], label='Imagined success rate')
  axes[2].scatter(success, direction, c=entropy, s=8,
                  cmap='magma', alpha=.6)
  axes[2].set_xlabel('Imagined success rate')
  axes[2].set_ylabel('Success-action divergence')
  fig.suptitle('Stage 5A gate-signal audit')
  for suffix in ('png', 'pdf'):
    fig.savefig(figure_root / f'gate_signal_audit.{suffix}', dpi=220)
  plt.close(fig)


def qualification_conditions(candidates, selected_cheap=None):
  conditions = [no_ir_condition()]
  conditions += [always_condition(objective) for objective in OBJECTIVES]
  if selected_cheap is None:
    for kind, values in (('entropy', candidates['entropy']),
                         ('js', candidates['js'])):
      for index, value in enumerate(values):
        thresholds = {
            'entropy_threshold': value if kind == 'entropy' else 1.0,
            'js_threshold': value if kind == 'js' else 1.0}
        for objective in ('reward_only', 'mixed_b005'):
          conditions.append(gated_condition(
              objective, kind, thresholds, suffix=f'_q{index}'))
  else:
    for index, value in enumerate(candidates['direction']):
      thresholds = {
          'entropy_threshold': selected_cheap['entropy']['entropy_threshold'],
          'js_threshold': selected_cheap['js']['js_threshold'],
          'min_success_rate': 4 / 128, 'direction_threshold': value}
      for objective in ('reward_only', 'mixed_b005'):
        conditions.append(gated_condition(
            objective, 'two_stage', thresholds, suffix=f'_q{index}'))
  return conditions


def aggregate_rate(rows, condition, mode, goal=None):
  selected = [
      row for row in rows if row['condition'] == condition and
      row['policy_mode'] == mode and (goal is None or row['goal'] == goal)]
  if not selected:
    raise RuntimeError((condition, mode, goal))
  return sum(int(row['successes']) for row in selected) / sum(
      int(row['episodes']) for row in selected)


def select_gate(rows, kind):
  names = sorted({
      row['condition'] for row in rows
      if row['gate_kind'] == kind and row['objective_label'] == 'reward_only'})
  candidates = []
  for reward_name in names:
    suffix = reward_name.split('reward_only', 1)[1]
    beta_name = f'{kind}_mixed_b005{suffix}'
    sampled_deltas = []
    safe = True
    worst_delta = 1.0
    for objective, name in (('reward_only', reward_name),
                            ('mixed_b005', beta_name)):
      sampled_deltas.append(
          aggregate_rate(rows, name, 'sampled') -
          aggregate_rate(rows, f'always_{objective}', 'sampled'))
      for goal in GOALS:
        delta = (
            aggregate_rate(rows, name, 'deterministic', goal) -
            aggregate_rate(rows, f'always_{objective}', 'deterministic', goal))
        worst_delta = min(worst_delta, delta)
        safe &= delta >= -.05
    source = next(row for row in rows if row['condition'] == reward_name)
    candidates.append({
        'condition': reward_name, 'safe': safe,
        'sampled_delta_mean': float(np.mean(sampled_deltas)),
        'deterministic_worst_delta': worst_delta,
        'thresholds': {
            'entropy_threshold': float(source['entropy_threshold']),
            'js_threshold': float(source['js_threshold']),
            'min_success_rate': float(source['min_success_rate']),
            'direction_threshold': float(source['direction_threshold'])}})
  eligible = [row for row in candidates if row['safe']]
  pool = eligible or candidates
  selected = max(pool, key=lambda row: row['sampled_delta_mean'])
  return {'kind': kind, 'qualified': bool(eligible), 'selected': selected,
          'candidates': candidates}


def pair_confirmation(rows):
  mismatches = []
  for replicate in range(max(int(row['replicate']) for row in rows) + 1):
    for mode in POLICY_MODES:
      for goal in GOALS:
        group = [row for row in rows if int(row['replicate']) == replicate and
                 row['policy_mode'] == mode and row['goal'] == goal]
        by_name = {row['condition']: row for row in group}
        reference = stage2.reset_identities(pathlib.Path(by_name['no_ir']['logdir']))
        for row in group:
          if stage2.reset_identities(pathlib.Path(row['logdir'])) != reference:
            mismatches.append((replicate, mode, goal, row['condition']))
            continue
          objective = row['objective_label']
          baseline_name = (
              'no_ir' if objective == 'no_ir' else f'always_{objective}')
          baseline = stage4b.episode_outcomes(
              pathlib.Path(by_name[baseline_name]['logdir']))
          treatment = stage4b.episode_outcomes(pathlib.Path(row['logdir']))
          seed = int(spec_digest({
              'stage': '5a', 'replicate': replicate, 'mode': mode,
              'goal': goal, 'condition': row['condition']})[:8], 16)
          row.update(stage4b.paired_effect(
              baseline, treatment, seed=seed, prefix='always'))
  if mismatches:
    raise RuntimeError(f'Stage 5A reset pairing failed: {mismatches[:3]}')
  return {'format': 1, 'equal': True, 'mismatches': []}


def confirmation_summary(rows):
  result = []
  for mode in POLICY_MODES:
    for condition in sorted({row['condition'] for row in rows}):
      goals = []
      source = next(row for row in rows if row['condition'] == condition)
      for goal in GOALS:
        goals.append(aggregate_rate(rows, condition, mode, goal))
      objective = source['objective_label']
      baseline = None if objective == 'no_ir' else f'always_{objective}'
      baseline_goals = (
          goals if baseline is None else
          [aggregate_rate(rows, baseline, mode, goal) for goal in GOALS])
      selected = [row for row in rows if row['condition'] == condition and
                  row['policy_mode'] == mode]
      environment_steps = sum(int(row['environment_steps']) for row in selected)
      imagined_transitions = sum(
          float(row['compute_imagined_transitions_sum']) for row in selected)
      actor_updates = sum(
          float(row['compute_actor_updates_sum']) for row in selected)
      evaluation_seconds = sum(
          float(row['evaluation_seconds']) for row in selected)
      paired = [float(row.get('paired_delta_vs_always', 0.0))
                for row in selected]
      replicate_deltas = []
      for replicate in sorted({int(row['replicate']) for row in selected}):
        replicate_deltas.append(float(np.mean([
            row.get('paired_delta_vs_always', 0.0) for row in selected
            if int(row['replicate']) == replicate])))
      result.append({
          'policy_mode': mode, 'condition': condition,
          'objective_label': objective, 'gate_kind': source['gate_kind'],
          'mean_goal_success': float(np.mean(goals)),
          'worst_goal_success': float(np.min(goals)),
          'mean_delta_vs_always': float(np.mean(np.asarray(goals) - baseline_goals)),
          'worst_delta_vs_always': float(np.min(np.asarray(goals) - baseline_goals)),
          'paired_delta_vs_always': float(np.mean(paired)),
          'replicate_positive_delta_count': sum(
              value > 0 for value in replicate_deltas),
          'episodes': sum(int(row['episodes']) for row in selected),
          'environment_steps': environment_steps,
          'imagined_transitions': imagined_transitions,
          'imagined_transitions_per_step': (
              imagined_transitions / environment_steps),
          'actor_updates': actor_updates,
          'actor_updates_per_step': actor_updates / environment_steps,
          'gate_seconds': sum(
              float(row['compute_cheap_seconds_sum']) +
              float(row['compute_consequence_seconds_sum']) for row in selected),
          'adapt_seconds': sum(
              float(row['compute_adapt_seconds_sum']) for row in selected),
          'evaluation_seconds': evaluation_seconds,
          'seconds_per_step': evaluation_seconds / environment_steps,
      })
  lookup = {(row['policy_mode'], row['condition']): row for row in result}
  for row in result:
    objective = row['objective_label']
    if objective == 'no_ir':
      row['imagined_compute_saved_vs_always'] = float('nan')
      row['wall_overhead_saved_vs_always'] = float('nan')
      row['advances'] = False
      continue
    baseline = lookup[(row['policy_mode'], f'always_{objective}')]
    no_ir = lookup[(row['policy_mode'], 'no_ir')]
    denominator = (
        baseline['imagined_transitions_per_step'] -
        no_ir['imagined_transitions_per_step'])
    row['imagined_compute_saved_vs_always'] = (
        1 - (row['imagined_transitions_per_step'] -
             no_ir['imagined_transitions_per_step']) /
        denominator if denominator > 0 else float('nan'))
    wall_denominator = baseline['seconds_per_step'] - no_ir['seconds_per_step']
    row['wall_overhead_saved_vs_always'] = (
        1 - (row['seconds_per_step'] - no_ir['seconds_per_step']) /
        wall_denominator if wall_denominator > 0 else float('nan'))
    row['advances'] = bool(
        row['gate_kind'] != 'none' and row['policy_mode'] == 'sampled' and
        row['paired_delta_vs_always'] > 0 and
        row['worst_delta_vs_always'] >= -.05 and
        lookup[('deterministic', row['condition'])][
            'worst_delta_vs_always'] >= -.05 and
        row['replicate_positive_delta_count'] >= 2)
  return result


def make_figures(output_root, rows, summary):
  import matplotlib.pyplot as plt
  root = output_root / 'figures'
  root.mkdir(exist_ok=True)
  labels = [row['condition'] for row in summary if row['policy_mode'] == 'sampled']
  sampled = {row['condition']: row for row in summary if row['policy_mode'] == 'sampled'}
  deterministic = {
      row['condition']: row for row in summary if row['policy_mode'] == 'deterministic'}
  x = np.arange(len(labels))
  fig, axis = plt.subplots(figsize=(13, 5), constrained_layout=True)
  axis.bar(x - .2, [sampled[x]['mean_goal_success'] for x in labels], .4,
           label='sampled')
  axis.bar(x + .2, [deterministic[x]['mean_goal_success'] for x in labels], .4,
           label='deterministic')
  axis.set_xticks(x, labels, rotation=40, ha='right')
  axis.set_ylim(0, 1); axis.set_ylabel('Macro success'); axis.legend()
  axis.set_title('Stage 5A gated IR performance')
  for suffix in ('png', 'pdf'):
    fig.savefig(root / f'performance.{suffix}', dpi=220)
  plt.close(fig)

  fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
  for name in labels:
    row = sampled[name]
    axis.scatter(row['imagined_transitions_per_step'], row['mean_goal_success'], s=55)
    axis.annotate(name, (row['imagined_transitions_per_step'], row['mean_goal_success']),
                  fontsize=7, xytext=(3, 3), textcoords='offset points')
  axis.set_xlabel('Imagined transitions per real step')
  axis.set_ylabel('Sampled macro success')
  axis.set_title('Performance--compute frontier')
  for suffix in ('png', 'pdf'):
    fig.savefig(root / f'performance_compute.{suffix}', dpi=220)
  plt.close(fig)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--stage4a-root', type=pathlib.Path, required=True)
  parser.add_argument('--output-root', type=pathlib.Path)
  parser.add_argument('--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument('--python', type=pathlib.Path, default=pathlib.Path(sys.executable))
  parser.add_argument('--phase', choices=('audit', 'qualify', 'confirm', 'all'), default='all')
  parser.add_argument('--audit-sampled-episodes', type=int, default=20)
  parser.add_argument('--audit-deterministic-episodes', type=int, default=10)
  parser.add_argument('--qualification-sampled-episodes', type=int, default=20)
  parser.add_argument('--qualification-deterministic-episodes', type=int, default=10)
  parser.add_argument('--replicates', type=int, default=3)
  parser.add_argument('--sampled-episodes', type=int, default=50)
  parser.add_argument('--deterministic-episodes', type=int, default=25)
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
  parser.add_argument('--skip-diagnostic-strata', action='store_true')
  parser.add_argument('--dry-run', action='store_true')
  return parser.parse_args(argv)


def validate(args):
  args.repo_root = args.repo_root.expanduser().resolve()
  args.stage4a_root = args.stage4a_root.expanduser().resolve()
  args.python = args.python.expanduser().resolve()
  args.output_root = (
      args.output_root.expanduser().resolve() if args.output_root else
      args.stage4a_root.parent / 'stage5a_gating')
  args.checkpoint = args.stage4a_root / 'checkpoints' / 'counterfactual_bounded'
  for path in (args.python, args.checkpoint / 'done', args.checkpoint / 'agent.pkl',
               args.checkpoint / 'repair_manifest.json'):
    if not path.is_file():
      raise FileNotFoundError(path)
  repair = json.loads((args.checkpoint / 'repair_manifest.json').read_text())
  if not (
      repair.get('condition') == 'counterfactual_bounded' and
      repair.get('counterfactual_reward_continuation') is True and
      repair.get('bounded_value') is True and
      int(repair.get('value_updates', 0)) > 0 and
      not repair.get('frozen_component_violations')):
    raise RuntimeError(
        'Stage 5A requires the complete integrity-clean Stage 4A '
        'counterfactual_bounded checkpoint')
  if args.paired_rng_stride <= args.episode_length:
    raise ValueError('paired RNG stride must exceed episode length')
  counts = (
      args.audit_sampled_episodes, args.audit_deterministic_episodes,
      args.qualification_sampled_episodes,
      args.qualification_deterministic_episodes, args.replicates,
      args.sampled_episodes, args.deterministic_episodes, args.start_batch)
  if any(value < 1 for value in counts):
    raise ValueError('Stage 5A counts must be positive')
  args.diagnostic_sources = discover_diagnostic_sources(args)


def main(argv=None):
  args = parse_args(argv)
  validate(args)
  args.output_root.mkdir(parents=True, exist_ok=True)
  spec = {
      'format': 1, 'canonical_stage': 'stage5a',
      'slug': 'interpretable_ir_gating',
      'hypothesis': 'State-dependent gates improve IR and reduce unnecessary compute.',
      'git_commit': git_revision(args.repo_root),
      'python_version': platform.python_version(),
      'rlscape_version': installed_version('rl-scape'),
      'checkpoint': str(args.checkpoint),
      'checkpoint_manifest_sha256': _sha256(args.checkpoint / 'repair_manifest.json'),
      'diagnostic_sources': [
          {'label': label, 'checkpoint': str(checkpoint)}
          for label, checkpoint in args.diagnostic_sources],
      'goals': list(GOALS), 'objectives': list(OBJECTIVES),
      'gate_families': ['entropy', 'js', 'two_stage'],
      'features': [
          'actor_entropy', 'posterior_entropy', 'js_disagreement',
          'success_rate', 'success_action_concentration',
          'success_action_divergence'],
      'audit_episodes': {
          'sampled': args.audit_sampled_episodes,
          'deterministic': args.audit_deterministic_episodes},
      'qualification_episodes': {
          'sampled': args.qualification_sampled_episodes,
          'deterministic': args.qualification_deterministic_episodes},
      'confirmation': {
          'replicates': args.replicates,
          'sampled_episodes': args.sampled_episodes,
          'deterministic_episodes': args.deterministic_episodes,
          'units': args.replicates * 13 * len(GOALS) * len(POLICY_MODES),
          'episodes': args.replicates * 13 * len(GOALS) * (
              args.sampled_episodes + args.deterministic_episodes)},
      'adaptation': {'horizon': 15, 'start_batch': args.start_batch,
                     'steps': args.adapt_steps, 'lr': args.adapt_lr,
                     'entropy': 0.0},
      'selection': {
          'primary': 'sampled_macro_success',
          'deterministic_worst_goal_floor': -0.05,
          'compute_is_constraint': False},
  }
  spec['digest'] = spec_digest(spec)
  spec_path = args.output_root / 'experiment_spec.json'
  if spec_path.is_file() and json.loads(spec_path.read_text()).get('digest') != spec['digest']:
    raise RuntimeError('Existing Stage 5A root has a different specification')
  if not spec_path.is_file():
    write_json(spec_path, spec)
  write_json(args.output_root / 'experiment_spec_digest.json', {'sha256': spec['digest']})
  if args.dry_run:
    print(json.dumps(spec, indent=2))
    example = {
        key: {'entropy_threshold': .5, 'js_threshold': .1,
              'min_success_rate': 4 / 128, 'direction_threshold': .1}
        for key in ('entropy', 'js', 'two_stage')}
    print('Confirmation conditions:', [
        item['name'] for item in confirmation_conditions(example)])
    return 0

  args.state_path = args.output_root / 'queue.json'
  args.active_path = args.output_root / 'active_child.json'
  args.state = json.loads(args.state_path.read_text()) if args.state_path.is_file() else {
      'format': 1, 'canonical_stage': 'stage5a', 'spec_digest': spec['digest'],
      'status': 'running', 'units': {}}
  if args.state.get('spec_digest') != spec['digest']:
    raise RuntimeError('Stage 5A queue belongs to another specification')
  write_json(args.state_path, args.state)

  candidates_path = args.output_root / 'threshold_candidates.json'
  selected_path = args.output_root / 'selected_gates.json'
  if args.phase in ('audit', 'all'):
    audit_rows = run_matrix(
        args, phase='audit', conditions=audit_conditions(), replicates=1,
        episodes={'sampled': args.audit_sampled_episodes,
                  'deterministic': args.audit_deterministic_episodes},
        seed_base=51_000)
    diagnostic_rows = []
    diagnostic_conditions = [
        no_ir_condition(audit=True),
        {**always_condition('standard'), 'gate_audit': True},
        {**always_condition('reward_only'), 'gate_audit': True},
    ]
    for source_index, (label, checkpoint) in enumerate(args.diagnostic_sources):
      diagnostic_rows += run_matrix(
          args, phase=f'audit_{label}', conditions=diagnostic_conditions,
          replicates=1, checkpoint=checkpoint,
          episodes={'sampled': args.audit_sampled_episodes,
                    'deterministic': args.audit_deterministic_episodes},
          seed_base=52_000 + source_index * 1_000)
    if diagnostic_rows:
      write_csv(args.output_root / 'audit_diagnostic_results.csv', diagnostic_rows)
      write_json(args.output_root / 'audit_diagnostic_results.json', {
          'format': 1, 'canonical_stage': 'stage5a', 'rows': diagnostic_rows})
    make_audit_figures(args.output_root, audit_rows, diagnostic_rows)
    candidates = quantile_candidates(audit_rows)
    write_json(candidates_path, candidates)
    if args.phase == 'audit':
      print(f'Stage 5A audit complete: {args.output_root}')
      return 0
  if not candidates_path.is_file():
    raise FileNotFoundError(candidates_path)
  candidates = json.loads(candidates_path.read_text())

  if args.phase in ('qualify', 'all'):
    first = run_matrix(
        args, phase='qualification_cheap',
        conditions=qualification_conditions(candidates), replicates=1,
        episodes={'sampled': args.qualification_sampled_episodes,
                  'deterministic': args.qualification_deterministic_episodes},
        seed_base=61_000)
    entropy = select_gate(first, 'entropy')
    js = select_gate(first, 'js')
    selected_cheap = {
        'entropy': entropy['selected']['thresholds'],
        'js': js['selected']['thresholds']}
    second = run_matrix(
        args, phase='qualification_consequence',
        conditions=qualification_conditions(candidates, selected_cheap),
        replicates=1,
        episodes={'sampled': args.qualification_sampled_episodes,
                  'deterministic': args.qualification_deterministic_episodes},
        seed_base=61_000)
    two_stage = select_gate(second, 'two_stage')
    selected = {
        'format': 1, 'canonical_stage': 'stage5a',
        'entropy': entropy['selected']['thresholds'],
        'js': js['selected']['thresholds'],
        'two_stage': two_stage['selected']['thresholds'],
        'qualification': {'entropy': entropy, 'js': js, 'two_stage': two_stage}}
    selected['digest'] = spec_digest(selected)
    write_json(selected_path, selected)
    if args.phase == 'qualify':
      print(f'Stage 5A qualification complete: {args.output_root}')
      return 0
  if not selected_path.is_file():
    raise FileNotFoundError(selected_path)
  selected = json.loads(selected_path.read_text())

  if args.phase in ('confirm', 'all'):
    rows = run_matrix(
        args, phase='confirmation',
        conditions=confirmation_conditions(selected),
        replicates=args.replicates,
        episodes={'sampled': args.sampled_episodes,
                  'deterministic': args.deterministic_episodes},
        seed_base=71_000)
    pairing = pair_confirmation(rows)
    write_json(args.output_root / 'pairing.json', pairing)
    write_csv(args.output_root / 'confirmation_results.csv', rows)
    write_json(args.output_root / 'confirmation_results.json', {
        'format': 1, 'canonical_stage': 'stage5a', 'rows': rows})
    summary = confirmation_summary(rows)
    write_csv(args.output_root / 'condition_summary.csv', summary)
    write_json(args.output_root / 'aggregate_summary.json', {
        'format': 1, 'canonical_stage': 'stage5a', 'rows': summary})
    make_figures(args.output_root, rows, summary)
    args.state.update(status='complete', updated_unix=time.time())
    write_json(args.state_path, args.state)
    (args.output_root / 'stage5a_complete').write_bytes(b'')
    print(f'Stage 5A complete: {args.output_root}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
