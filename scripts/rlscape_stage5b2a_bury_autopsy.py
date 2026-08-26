#!/usr/bin/env python3
"""Run Stage 5B.2a: the sampled bury-bones virtual-gate autopsy."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.metadata
import json
import math
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
from scripts import rlscape_stage5_common as common  # noqa: E402
from scripts.rlscape_m1_pilot import read_jsonl  # noqa: E402
from scripts.rlscape_m1_pilot import spec_digest  # noqa: E402
from scripts.rlscape_m1_pilot import write_csv  # noqa: E402
from scripts.rlscape_m1_pilot import write_json  # noqa: E402
from scripts.rlscape_m3_process import active_matches  # noqa: E402
from scripts.rlscape_m3_process import run_managed  # noqa: E402
from scripts.rlscape_m3_sequential import git_revision  # noqa: E402


GOAL = 'bury_bones'
MODE = 'sampled'


def installed_version(package):
  try:
    return importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:
    return 'not-installed'


def _condition(
    name, family, *, enabled=True, objective='reward_only',
    objective_label='reward_only', mix_beta=0.0, update_budget=0,
    gate_enabled=False, gate_kind='none', gate_audit=False,
    adapt_every_k=1, **kwargs):
  result = dict(
      name=name, family=family, checkpoint='source', enabled=enabled,
      objective=objective, objective_label=objective_label, horizon=15,
      actent=0.0, mix_beta=float(mix_beta),
      max_actor_updates_per_episode=int(update_budget),
      gate_enabled=bool(gate_enabled), gate_kind=gate_kind,
      gate_audit=bool(gate_audit), adapt_every_k=int(adapt_every_k),
      entropy_threshold=1.0, js_threshold=1.0,
      min_success_rate=4 / 128, direction_threshold=1.0)
  result.update(kwargs)
  return result


def _virtual(name, *, objective='reward_only', objective_label='reward_only',
             mix_beta=0.0, accept_mode='positive'):
  return _condition(
      name, 'virtual_update', objective=objective,
      objective_label=objective_label, mix_beta=mix_beta,
      gate_enabled=True, gate_kind='virtual_update',
      virtual_score='reward_return', virtual_min_improvement=0.0,
      virtual_max_baseline_score=1e30, virtual_dormant_every=25,
      virtual_active_every=1, virtual_deactivate_after=3,
      virtual_accept_mode=accept_mode)


def _random(name, *, objective='reward_only', objective_label='reward_only',
            mix_beta=0.0, updates=46, window=200):
  return _condition(
      name, 'random_schedule', objective=objective,
      objective_label=objective_label, mix_beta=mix_beta,
      update_budget=updates, gate_enabled=True,
      gate_kind='random_schedule', random_updates=updates,
      random_window=window, random_seed_offset=500000)


def conditions(args):
  """Return the prespecified diagnostic matrix in execution order."""
  result = [
      _condition(
          'no_ir', 'no_ir', enabled=False, objective='standard',
          objective_label='no_ir'),
      _condition(
          'no_ir_world_audit', 'world_audit', enabled=False,
          objective='standard', objective_label='no_ir', gate_audit=True,
          adapt_every_k=args.audit_every),
      _condition('always_reward_only', 'always'),
  ]
  result += [
      _condition(
          f'reward_cap_{dose}', 'dose', update_budget=dose)
      for dose in args.reward_doses]
  result += [
      _random(
          f'reward_random_{args.matched_updates}',
          updates=args.matched_updates, window=args.episode_length),
      _virtual('virtual_positive_reward_d25', accept_mode='positive'),
      _virtual('virtual_negative_reward_d25', accept_mode='negative'),
      _condition(
          'always_mixed_beta_0p005', 'always_mixed',
          objective='mixed_bounded', objective_label='mixed_beta_0p005',
          mix_beta=args.mixed_beta),
      _condition(
          f'mixed_beta_0p005_cap_{args.matched_updates}', 'dose_mixed',
          objective='mixed_bounded', objective_label='mixed_beta_0p005',
          mix_beta=args.mixed_beta, update_budget=args.matched_updates),
      _random(
          f'mixed_beta_0p005_random_{args.matched_updates}',
          objective='mixed_bounded', objective_label='mixed_beta_0p005',
          mix_beta=args.mixed_beta, updates=args.matched_updates,
          window=args.episode_length),
      _virtual(
          'virtual_positive_mixed_beta_0p005_d25',
          objective='mixed_bounded', objective_label='mixed_beta_0p005',
          mix_beta=args.mixed_beta, accept_mode='positive'),
  ]
  names = [item['name'] for item in result]
  if len(names) != len(set(names)):
    raise AssertionError(names)
  return result


def episodes_for(args, condition):
  return (
      args.audit_episodes if condition['family'] == 'world_audit'
      else args.episodes)


def heartbeat(args, snapshot):
  args.state['active_child'] = snapshot
  args.state['updated_unix'] = time.time()
  write_json(args.state_path, args.state)


def build_command(args, *, logdir, condition, replicate):
  return common.build_eval_command(
      args, logdir=logdir, checkpoint=args.checkpoint,
      condition=condition, goal=GOAL, policy_mode=MODE,
      episodes=episodes_for(args, condition),
      environment_seed=args.eval_seed_base + replicate,
      agent_seed=args.agent_seed_base + replicate,
      persistence='episode', warmup_episodes=1)


def summarize(args, *, logdir, condition, replicate, returncode):
  episodes = episodes_for(args, condition)
  result = common.summarize_eval(
      logdir, condition=condition, goal=GOAL, policy_mode=MODE,
      episodes=episodes,
      environment_seed=args.eval_seed_base + replicate,
      returncode=returncode, warmup_episodes=1,
      agent_seed=args.agent_seed_base + replicate,
      paired_rng_stride=args.paired_rng_stride)
  result.update({
      'canonical_stage': 'stage5b2a', 'replicate': replicate,
      'source_checkpoint': str(args.checkpoint),
      'mix_beta': float(condition.get('mix_beta', 0.0)),
      'update_budget': int(condition.get(
          'max_actor_updates_per_episode', 0)),
      'virtual_accept_mode': condition.get('virtual_accept_mode', ''),
      'random_updates': int(condition.get('random_updates', 0)),
      'requested_episodes': episodes,
  })
  return result


def execute(args, *, condition, replicate):
  unit = f'replicate_{replicate:02d}/{condition["name"]}'
  state = args.state['units'].setdefault(
      unit, {'status': 'pending', 'attempts': []})

  def recover(path, code):
    return summarize(
        args, logdir=path, condition=condition, replicate=replicate,
        returncode=code)

  if state.get('status') == 'complete':
    result = recover(pathlib.Path(state['attempts'][-1]['logdir']), 0)
    if result['complete']:
      return result
    state['status'] = 'pending'

  attempt = state['attempts'][-1] if state['attempts'] else None
  if attempt and attempt.get('status') == 'running':
    logdir = pathlib.Path(attempt['logdir'])
    command = build_command(
        args, logdir=logdir, condition=condition, replicate=replicate)
    result = recover(logdir, 0)
    if result['complete']:
      attempt.update(
          status='complete', finished_unix=time.time(), summary=result)
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
        'attempt': number, 'status': 'running',
        'started_unix': time.time(), 'logdir': str(logdir)}
    state['attempts'].append(attempt)
  else:
    number = int(attempt['attempt'])
    logdir = pathlib.Path(attempt['logdir'])

  command = build_command(
      args, logdir=logdir, condition=condition, replicate=replicate)
  attempt['command'] = command
  state['status'] = 'running'
  write_json(args.state_path, args.state)
  print(f'\n=== Stage 5B.2a {unit}, attempt {number} ===', flush=True)
  print(' '.join(command), flush=True)
  child = run_managed(
      command, cwd=args.repo_root, active_path=args.active_path,
      console_path=logdir / 'console.log', activity_paths=(
          logdir / 'metrics.jsonl', logdir / 'episodes.jsonl',
          logdir / 'stage5_trace.jsonl', logdir / 'audit_env0.jsonl'),
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
      return execute(args, condition=condition, replicate=replicate)
    raise RuntimeError(f'Incomplete Stage 5B.2a unit: {unit}')
  return result


def run_matrix(args, matrix):
  rows = []
  for replicate in range(args.replicates):
    for condition in matrix:
      rows.append(execute(
          args, condition=condition, replicate=replicate))
      write_csv(args.output_root / 'results.csv', rows)
  write_json(args.output_root / 'results.json', {
      'format': 1, 'canonical_stage': 'stage5b2a', 'rows': rows})
  return rows


TRACE_FEATURES = (
    'gate_score_before', 'gate_score_after', 'gate_score_improvement',
    'gate_before_reward_return_mean', 'gate_after_reward_return_mean',
    'gate_before_reward_peak_mean', 'gate_after_reward_peak_mean',
    'gate_before_reward_peak_p95', 'gate_after_reward_peak_p95',
    'gate_before_reward_event_step_rate',
    'gate_after_reward_event_step_rate',
    'gate_before_reward_peak_timestep_mean',
    'gate_after_reward_peak_timestep_mean',
    'gate_before_reward_preceding_action_action_mode',
    'gate_after_reward_preceding_action_action_mode',
    'gate_before_reward_preceding_action_action_concentration',
    'gate_after_reward_preceding_action_action_concentration',
    'gate_before_imagined_action_action_mode',
    'gate_after_imagined_action_action_mode',
    'gate_before_imagined_action_action_concentration',
    'gate_after_imagined_action_action_concentration',
    'gate_reward_return_mean', 'gate_reward_peak_mean',
    'gate_reward_peak_p95', 'gate_reward_event_step_rate',
    'gate_reward_preceding_action_action_mode',
    'gate_reward_preceding_action_action_concentration',
    'gate_imagined_action_action_mode',
    'gate_imagined_action_action_concentration',
    'adapt_actor_reward_only', 'adapt_actor_reward_particle_rate',
    'adapt_actor_rew', 'adapt_actor_ret', 'adapt_actor_adv',
)


def trace_episode_rows(logdir):
  grouped = defaultdict(list)
  for row in read_jsonl(logdir / 'stage5_trace.jsonl'):
    grouped[int(row['episode_index'])].append(row)
  output = []
  for index in sorted(grouped):
    rows = grouped[index]
    committed = [
        row for row in rows if float(row.get('adapt_trigger', 0)) > 0]
    probes = [row for row in rows if float(
        row.get('gate_consequence_evaluated', 0)) > 0]
    item = {
        'episode_index': index,
        'trace_steps': len(rows),
        'committed_updates': sum(float(
            row.get('adapt_steps', 0)) for row in committed),
        'commit_events': len(committed),
        'first_commit_step': (
            int(committed[0]['episode_step']) if committed else -1),
        'last_commit_step': (
            int(committed[-1]['episode_step']) if committed else -1),
        'probes': len(probes),
        'accepted_probe_fraction': (
            len(committed) / len(probes) if probes else float('nan')),
    }
    for key in TRACE_FEATURES:
      values = [float(row[key]) for row in rows
                if key in row and np.isfinite(float(row[key]))]
      item[f'first_{key}'] = values[0] if values else float('nan')
      item[f'mean_{key}'] = (
          float(np.mean(values)) if values else float('nan'))
      accepted_values = [float(row[key]) for row in committed
                         if key in row and np.isfinite(float(row[key]))]
      item[f'accepted_mean_{key}'] = (
          float(np.mean(accepted_values))
          if accepted_values else float('nan'))
    output.append(item)
  return output


def compact_probe_rows(logdir, condition, replicate):
  output = []
  for row in read_jsonl(logdir / 'stage5_trace.jsonl'):
    chosen = {key: row[key] for key in TRACE_FEATURES if key in row}
    if not chosen:
      continue
    output.append({
        'canonical_stage': 'stage5b2a', 'condition': condition['name'],
        'objective_label': condition['objective_label'],
        'replicate': replicate, 'episode_index': int(row['episode_index']),
        'episode_step': int(row['episode_step']),
        'adapt_trigger': float(row.get('adapt_trigger', 0)),
        **chosen,
    })
  return output


def _entropy(counter):
  total = sum(counter.values())
  if not total:
    return float('nan')
  return float(-sum(
      (count / total) * math.log(count / total)
      for count in counter.values() if count))


def real_action_data(logdir, *, rows, columns):
  records = read_jsonl(logdir / 'audit_env0.jsonl')
  resets = [row for row in records if row.get('kind') == 'reset'][1:]
  grouped = defaultdict(list)
  for row in records:
    if row.get('kind') == 'step' and row.get('requested_action'):
      grouped[row.get('episode_id')].append(row)
  episodes = read_jsonl(logdir / 'episodes.jsonl')
  traces = trace_episode_rows(logdir)
  if not len(resets) == len(episodes) == len(traces):
    raise RuntimeError(
        f'Action/episode alignment failed in {logdir}: '
        f'{len(resets)}, {len(episodes)}, {len(traces)}')

  episode_rows = []
  sequences = []
  histogram = Counter()
  for index, (reset, episode, trace) in enumerate(
      zip(resets, episodes, traces)):
    actions = grouped.get(reset.get('episode_id'), [])
    cells = [(
        int(row['requested_action']['row']),
        int(row['requested_action']['column'])) for row in actions]
    indices = [row * columns + column for row, column in cells]
    counts = Counter(indices)
    top_index, top_count = counts.most_common(1)[0] if counts else (-1, 0)
    success = int(
        float(episode.get('episode/score', 0)) > 0 or
        float(episode.get('episode/log/task_success/max', 0)) > 0)
    progress = max(
        [float(row.get('task_progress', 0)) for row in actions], default=0)
    buried = sum(
        event.get('type') == 'item_buried' for row in actions
        for event in row.get('events', []))
    episode_rows.append({
        **trace, 'episode_index': index,
        'reset_seed': reset.get('reset_seed'), 'success': success,
        'action_length': len(indices), 'task_progress_max': progress,
        'item_buried_events': buried, 'real_action_entropy': _entropy(counts),
        'real_unique_action_fraction': (
            len(counts) / len(indices) if indices else float('nan')),
        'real_top_action_index': top_index,
        'real_top_action_row': (
            top_index // columns if top_index >= 0 else -1),
        'real_top_action_column': (
            top_index % columns if top_index >= 0 else -1),
        'real_top_action_fraction': (
            top_count / len(indices) if indices else float('nan')),
    })
    sequences.append(indices)
    for step, action in enumerate(indices):
      phase = (
          'early_0_24' if step < 25 else
          'middle_25_74' if step < 75 else 'late_75_plus')
      for outcome in ('all', 'success' if success else 'failure'):
        histogram[(phase, outcome, action)] += 1
        histogram[('all_steps', outcome, action)] += 1

  hist_rows = []
  totals = Counter()
  for (phase, outcome, _), count in histogram.items():
    totals[(phase, outcome)] += count
  for (phase, outcome, action), count in sorted(histogram.items()):
    hist_rows.append({
        'phase': phase, 'outcome': outcome, 'action_index': action,
        'row': action // columns, 'column': action % columns,
        'count': count, 'fraction': count / totals[(phase, outcome)],
        'grid_rows': rows, 'grid_columns': columns,
    })
  return episode_rows, hist_rows, sequences, resets


def collect_diagnostics(args, results, matrix):
  by_name = {item['name']: item for item in matrix}
  episode_output = []
  histogram_output = []
  probe_output = []
  sequences = {}
  resets = {}
  traces = {}
  for result in results:
    condition = by_name[result['condition']]
    replicate = int(result['replicate'])
    logdir = pathlib.Path(result['logdir'])
    episodes, hist, seq, reset = real_action_data(
        logdir, rows=args.grid_rows, columns=args.grid_columns)
    for row in episodes:
      episode_output.append({
          'canonical_stage': 'stage5b2a', 'condition': condition['name'],
          'family': condition['family'],
          'objective_label': condition['objective_label'],
          'mix_beta': condition.get('mix_beta', 0.0),
          'update_budget': condition.get(
              'max_actor_updates_per_episode', 0),
          'replicate': replicate, **row,
      })
    for row in hist:
      histogram_output.append({
          'canonical_stage': 'stage5b2a', 'condition': condition['name'],
          'replicate': replicate, **row})
    probe_output.extend(compact_probe_rows(logdir, condition, replicate))
    sequences[(replicate, condition['name'])] = seq
    resets[(replicate, condition['name'])] = reset
    traces[(replicate, condition['name'])] = episodes
  return (
      episode_output, histogram_output, probe_output,
      sequences, resets, traces)


def pairing_audit(args, matrix, sequences, resets, traces):
  rows = []
  mismatches = []
  names = [item['name'] for item in matrix]
  for replicate in range(args.replicates):
    reference_resets = resets[(replicate, 'no_ir')]
    reference_seq = sequences[(replicate, 'no_ir')]
    for name in names:
      candidate_resets = resets[(replicate, name)]
      count = min(len(reference_resets), len(candidate_resets))
      seed_equal = all(
          reference_resets[index].get('reset_seed') ==
          candidate_resets[index].get('reset_seed')
          for index in range(count))
      if not seed_equal:
        mismatches.append({'replicate': replicate, 'condition': name})
      candidate_seq = sequences[(replicate, name)]
      candidate_trace = traces[(replicate, name)]
      for index in range(min(len(reference_seq), len(candidate_seq))):
        first_commit = int(candidate_trace[index]['first_commit_step'])
        limit = (
            min(first_commit, len(reference_seq[index]),
                len(candidate_seq[index]))
            if first_commit >= 0 else
            min(len(reference_seq[index]), len(candidate_seq[index])))
        matches = sum(
            reference_seq[index][step] == candidate_seq[index][step]
            for step in range(limit))
        rows.append({
            'replicate': replicate, 'condition': name,
            'episode_index': index, 'reset_seed_equal': seed_equal,
            'first_commit_step': first_commit,
            'pre_commit_compared_actions': limit,
            'pre_commit_matching_actions': matches,
            'pre_commit_action_match_fraction': (
                matches / limit if limit else float('nan')),
            'first_action_match': (
                int(reference_seq[index][0] == candidate_seq[index][0])
                if limit and first_commit != 0 else -1),
        })
  if mismatches:
    raise RuntimeError(f'Stage 5B.2a reset pairing failed: {mismatches[:3]}')
  audit_rows = [row for row in rows
                if row['condition'] == 'no_ir_world_audit']
  compared = sum(row['pre_commit_compared_actions'] for row in audit_rows)
  matched = sum(row['pre_commit_matching_actions'] for row in audit_rows)
  summary = {
      'format': 1, 'canonical_stage': 'stage5b2a',
      'reset_seeds_equal': True,
      'probe_noninterference_compared_actions': compared,
      'probe_noninterference_matching_actions': matched,
      'probe_noninterference_action_match_fraction': (
          matched / compared if compared else float('nan')),
  }
  return rows, summary


def aggregate(results, episodes, matrix):
  output = []
  for condition in matrix:
    name = condition['name']
    chosen = [row for row in episodes if row['condition'] == name]
    units = [row for row in results if row['condition'] == name]
    successful = [row for row in chosen if row['success']]
    entry = {
        'condition': name, 'family': condition['family'],
        'objective_label': condition['objective_label'],
        'mix_beta': condition.get('mix_beta', 0.0),
        'update_budget': condition.get(
            'max_actor_updates_per_episode', 0),
        'episodes': len(chosen),
        'success_rate': float(np.mean([row['success'] for row in chosen])),
        'mean_action_length': float(np.mean([
            row['action_length'] for row in chosen])),
        'mean_success_length': (
            float(np.mean([row['action_length'] for row in successful]))
            if successful else float('nan')),
        'mean_task_progress': float(np.mean([
            row['task_progress_max'] for row in chosen])),
        'mean_committed_updates': float(np.mean([
            row['committed_updates'] for row in chosen])),
        'median_committed_updates': float(np.median([
            row['committed_updates'] for row in chosen])),
        'mean_real_action_entropy': float(np.mean([
            row['real_action_entropy'] for row in chosen])),
        'mean_real_top_action_fraction': float(np.mean([
            row['real_top_action_fraction'] for row in chosen])),
        'evaluation_seconds': sum(float(
            row['evaluation_seconds']) for row in units),
        'environment_steps': sum(int(row['environment_steps']) for row in units),
    }
    for key in (
        'mean_gate_score_improvement',
        'mean_gate_before_reward_peak_mean',
        'mean_gate_after_reward_peak_mean',
        'mean_gate_before_reward_event_step_rate',
        'mean_gate_after_reward_event_step_rate',
        'mean_adapt_actor_reward_particle_rate'):
      values = [float(row[key]) for row in chosen
                if np.isfinite(float(row.get(key, np.nan)))]
      entry[key] = float(np.mean(values)) if values else float('nan')
    output.append(entry)
  return output


def _heatmap(histogram, condition, phase, rows, columns):
  array = np.zeros((rows, columns), np.float64)
  for item in histogram:
    if (item['condition'] == condition and item['phase'] == phase and
        item['outcome'] == 'all'):
      array[int(item['row']), int(item['column'])] += float(item['count'])
  return array / max(array.sum(), 1)


def make_figures(args, summary, episodes, histogram, probes, matrix):
  import matplotlib.pyplot as plt
  figure_root = args.output_root / 'figures'
  figure_root.mkdir(exist_ok=True)
  names = [item['name'] for item in matrix]
  lookup = {row['condition']: row for row in summary}

  fig, axis = plt.subplots(figsize=(15, 5.5), constrained_layout=True)
  values = [lookup[name]['success_rate'] for name in names]
  axis.bar(np.arange(len(names)), values, color='tab:blue', alpha=.75)
  for x, name in enumerate(names):
    reps = []
    for replicate in range(args.replicates):
      chosen = [row['success'] for row in episodes
                if row['condition'] == name and
                int(row['replicate']) == replicate]
      if chosen:
        reps.append(float(np.mean(chosen)))
    axis.scatter([x] * len(reps), reps, color='black', s=18, zorder=3)
  axis.set_xticks(np.arange(len(names)), names, rotation=40, ha='right')
  axis.set_ylim(0, 1); axis.set_ylabel('Sampled success rate')
  axis.set_title('Stage 5B.2a bury-bones autopsy')
  for suffix in ('png', 'pdf'):
    fig.savefig(figure_root / f'performance.{suffix}', dpi=220)
  plt.close(fig)

  fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
  reward_points = [(0, lookup['no_ir']['success_rate'])]
  for dose in args.reward_doses:
    reward_points.append((
        dose, lookup[f'reward_cap_{dose}']['success_rate']))
  reward_points.append((
      args.episode_length, lookup['always_reward_only']['success_rate']))
  axis.plot(*zip(*reward_points), marker='o', label='reward-only')
  mixed_points = [
      (0, lookup['no_ir']['success_rate']),
      (args.matched_updates, lookup[
          f'mixed_beta_0p005_cap_{args.matched_updates}']['success_rate']),
      (args.episode_length,
       lookup['always_mixed_beta_0p005']['success_rate'])]
  axis.plot(*zip(*mixed_points), marker='s', label=f'mixed beta={args.mixed_beta}')
  axis.set_xlabel('Maximum actor updates per full episode')
  axis.set_ylabel('Sampled success rate'); axis.set_ylim(0, 1)
  axis.set_title('Does partial IR pass through a recovery valley?')
  axis.legend()
  for suffix in ('png', 'pdf'):
    fig.savefig(figure_root / f'dose_response.{suffix}', dpi=220)
  plt.close(fig)

  key_names = [
      'no_ir', 'always_reward_only',
      f'reward_cap_{args.matched_updates}',
      f'reward_random_{args.matched_updates}',
      'virtual_positive_reward_d25', 'virtual_negative_reward_d25']
  fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
  maximum = max(float(_heatmap(
      histogram, name, 'late_75_plus', args.grid_rows,
      args.grid_columns).max()) for name in key_names)
  image = None
  for axis, name in zip(axes.flat, key_names):
    image = axis.imshow(_heatmap(
        histogram, name, 'late_75_plus', args.grid_rows,
        args.grid_columns), vmin=0, vmax=maximum, cmap='magma')
    axis.set_title(name); axis.set_xlabel('grid column')
    axis.set_ylabel('grid row')
  fig.colorbar(image, ax=axes, shrink=.8, label='Late-click fraction')
  fig.suptitle('Real late-episode click attractors')
  for suffix in ('png', 'pdf'):
    fig.savefig(figure_root / f'real_click_heatmaps.{suffix}', dpi=220)
  plt.close(fig)

  virtual = [row for row in episodes if row['family'] == 'virtual_update']
  fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
  for name in sorted({row['condition'] for row in virtual}):
    chosen = [row for row in virtual if row['condition'] == name]
    axis.scatter(
        [row['mean_gate_score_improvement'] for row in chosen],
        [row['success'] for row in chosen], alpha=.3, s=14, label=name)
  axis.axvline(0, color='black', linewidth=1)
  axis.set_xlabel('Mean imagined before/after score change')
  axis.set_ylabel('Real episode success')
  axis.set_title('Imagined approval versus real outcome')
  axis.legend(fontsize=7)
  for suffix in ('png', 'pdf'):
    fig.savefig(figure_root / f'imagined_vs_real.{suffix}', dpi=220)
  plt.close(fig)

  virtual_probe = [row for row in probes if row.get('adapt_trigger', 0) > 0]
  virtual_names = sorted({row['condition'] for row in virtual_probe
                          if 'gate_after_reward_preceding_action_action_mode'
                          in row})
  if virtual_names:
    fig, axes = plt.subplots(
        1, len(virtual_names), figsize=(5 * len(virtual_names), 4.5),
        constrained_layout=True, squeeze=False)
    for axis, name in zip(axes.flat, virtual_names):
      array = np.zeros((args.grid_rows, args.grid_columns), np.float64)
      for row in virtual_probe:
        if row['condition'] != name:
          continue
        value = row.get(
            'gate_after_reward_preceding_action_action_mode', np.nan)
        concentration = row.get(
            'gate_after_reward_preceding_action_action_concentration', 0.0)
        if np.isfinite(float(value)) and float(concentration) > 0:
          index = int(round(float(value)))
          if 0 <= index < args.grid_rows * args.grid_columns:
            array[index // args.grid_columns, index % args.grid_columns] += 1
      array /= max(array.sum(), 1)
      image = axis.imshow(array, cmap='viridis')
      axis.set_title(name); axis.set_xlabel('grid column')
      axis.set_ylabel('grid row')
      fig.colorbar(image, ax=axis, shrink=.75)
    fig.suptitle('Actions preceding peak imagined reward after accepted updates')
    for suffix in ('png', 'pdf'):
      fig.savefig(
          figure_root / f'imagined_reward_action_heatmaps.{suffix}', dpi=220)
    plt.close(fig)


def write_findings(args, summary, pairing):
  lines = [
      '# Stage 5B.2a Bury-Bones Gate Autopsy', '',
      'Status: **complete; interpretation pending**.', '',
      '| Condition | Success | Updates/episode | Top-click fraction | Mean length |',
      '|---|---:|---:|---:|---:|']
  for row in summary:
    lines.append(
        f'| {row["condition"]} | {row["success_rate"]:.3f} | '
        f'{row["mean_committed_updates"]:.1f} | '
        f'{row["mean_real_top_action_fraction"]:.3f} | '
        f'{row["mean_action_length"]:.1f} |')
  lines += [
      '', '## Prespecified interpretation', '',
      f'- If both the `{args.matched_updates}`-update cap and matched random '
      'schedule collapse, the leading explanation is a partial-dose recovery valley.',
      '- If random scheduling succeeds but virtual-positive selection collapses, '
      'the imagined score is selecting model-exploiting updates.',
      '- If virtual-negative exceeds virtual-positive, the learned model ranks '
      'candidate updates in the wrong real-world direction.',
      f'- If exact mixed beta `{args.mixed_beta}` avoids the reward-only attractor, '
      'a small independent value signal regularizes reward-model exploitation.',
      '- No-IR world-audit behavior must match no-IR behavior exactly before its '
      'imagined diagnostics can be treated as non-intervening.', '',
      '## Pairing audit', '',
      f'- Probe non-interference action match: '
      f'`{pairing["probe_noninterference_action_match_fraction"]:.6f}` over '
      f'`{pairing["probe_noninterference_compared_actions"]}` actions.', '',
      'Figures and compact diagnostics are stored beside this file. Real outcomes '
      'are never visible to any gate or update scheduler during evaluation.',
  ]
  (args.output_root / 'STAGE5B2A_FINDINGS.md').write_text(
      '\n'.join(lines) + '\n')


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', type=pathlib.Path, required=True)
  parser.add_argument('--stage5b2-root', type=pathlib.Path, required=True)
  parser.add_argument('--output-root', type=pathlib.Path)
  parser.add_argument('--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument('--python', type=pathlib.Path,
                      default=pathlib.Path(sys.executable))
  parser.add_argument('--replicates', type=int, default=3)
  parser.add_argument('--episodes', type=int, default=50)
  parser.add_argument('--audit-episodes', type=int, default=20)
  parser.add_argument('--audit-every', type=int, default=25)
  parser.add_argument('--reward-doses', type=int, nargs='+',
                      default=(16, 46, 80, 127))
  parser.add_argument('--matched-updates', type=int, default=46)
  # This is exactly 0.005. Historical Stage 4B `mixed_b005` denoted beta=.05.
  parser.add_argument('--mixed-beta', type=float, default=.005)
  parser.add_argument('--episode-length', type=int, default=200)
  parser.add_argument('--grid-columns', type=int, default=28)
  parser.add_argument('--grid-rows', type=int, default=18)
  parser.add_argument('--adapt-steps', type=int, default=1)
  parser.add_argument('--adapt-lr', type=float, default=4e-5)
  parser.add_argument('--adapt-every-k', type=int, default=1)
  parser.add_argument('--start-batch', type=int, default=128)
  parser.add_argument('--paired-rng-stride', type=int, default=1000)
  # Exact Stage 5B.2 seeds make the three shared conditions direct
  # reproductions of Table 15 before the autopsy controls branch away.
  parser.add_argument('--eval-seed-base', type=int, default=121000)
  parser.add_argument('--agent-seed-base', type=int, default=131000)
  parser.add_argument('--max-attempts', type=int, default=4)
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
  args.checkpoint = args.checkpoint.expanduser().resolve()
  args.stage5b2_root = args.stage5b2_root.expanduser().resolve()
  args.output_root = (
      args.output_root.expanduser().resolve() if args.output_root else
      args.stage5b2_root.parent / 'stage5b2a_bury_autopsy')
  for path in (
      args.python, args.checkpoint / 'agent.pkl', args.checkpoint / 'done',
      args.stage5b2_root / 'stage5b2_complete',
      args.stage5b2_root / 'experiment_spec.json'):
    if not path.is_file():
      raise FileNotFoundError(path)
  upstream = json.loads(
      (args.stage5b2_root / 'experiment_spec.json').read_text())
  if upstream.get('canonical_stage') != 'stage5b2':
    raise RuntimeError('Invalid Stage 5B.2 source artifact')
  if args.paired_rng_stride <= args.episode_length:
    raise ValueError('paired RNG stride must exceed episode length')
  positive = (
      args.replicates, args.episodes, args.audit_episodes, args.audit_every,
      args.matched_updates, args.episode_length, args.adapt_steps,
      args.adapt_every_k, args.start_batch, *args.reward_doses)
  if min(positive) < 1:
    raise ValueError('Episode, update, dose, and cadence values must be positive')
  if args.matched_updates > args.episode_length:
    raise ValueError('matched updates cannot exceed episode length')
  if not 0 <= args.mixed_beta <= 1:
    raise ValueError('mixed beta must lie in [0, 1]')


def main(argv=None):
  args = parse_args(argv)
  validate(args)
  args.output_root.mkdir(parents=True, exist_ok=True)
  matrix = conditions(args)
  upstream_spec = args.stage5b2_root / 'experiment_spec.json'
  spec = {
      'format': 1, 'canonical_stage': 'stage5b2a',
      'slug': 'bury_bones_gate_autopsy',
      'question': (
          'Does virtual gating collapse bury through update dose, imagined '
          'selection, or reward-model exploitation?'),
      'checkpoint': str(args.checkpoint),
      'checkpoint_sha256': _sha256(args.checkpoint / 'agent.pkl'),
      'stage5b2_root': str(args.stage5b2_root),
      'stage5b2_spec_sha256': _sha256(upstream_spec),
      'goal': GOAL, 'policy_mode': MODE,
      'deterministic_primary': False,
      'replicates': args.replicates, 'episodes': args.episodes,
      'audit_episodes': args.audit_episodes,
      'environment_seed_base': args.eval_seed_base,
      'agent_seed_base': args.agent_seed_base,
      'reproduces_stage5b2_seed_schedule': bool(
          args.eval_seed_base == 121000 and args.agent_seed_base == 131000),
      'conditions': [item['name'] for item in matrix],
      'adaptation': {
          'horizon': 15, 'posterior_samples': args.start_batch,
          'steps': args.adapt_steps, 'lr': args.adapt_lr,
          'reward_doses': list(args.reward_doses),
          'matched_updates': args.matched_updates,
          'mixed_beta_exact': args.mixed_beta,
          'historical_mixed_b005_was_beta': .05,
          'persistence': 'episode'},
      'diagnostics': {
          'real_click_heatmaps': True,
          'imagined_reward_location_and_timing': True,
          'probe_noninterference_pairing': True,
          'real_outcomes_visible_to_gate': False},
      'git_commit': git_revision(args.repo_root),
      'python_version': platform.python_version(),
      'rlscape_version': installed_version('rl-scape'),
  }
  spec['digest'] = spec_digest(spec)
  spec_path = args.output_root / 'experiment_spec.json'
  if spec_path.is_file() and json.loads(
      spec_path.read_text()).get('digest') != spec['digest']:
    raise RuntimeError(
        'Existing Stage 5B.2a root has a different specification')
  if not spec_path.is_file():
    write_json(spec_path, spec)
  if args.dry_run:
    print(json.dumps(spec, indent=2))
    return 0

  args.state_path = args.output_root / 'queue.json'
  args.active_path = args.output_root / 'active_child.json'
  args.state = (
      json.loads(args.state_path.read_text()) if args.state_path.is_file()
      else {'format': 1, 'canonical_stage': 'stage5b2a',
            'spec_digest': spec['digest'], 'status': 'running', 'units': {}})
  if args.state.get('spec_digest') != spec['digest']:
    raise RuntimeError('Stage 5B.2a queue specification mismatch')
  write_json(args.state_path, args.state)

  results = run_matrix(args, matrix)
  (episodes, histogram, probes,
   sequences, resets, traces) = collect_diagnostics(args, results, matrix)
  write_csv(args.output_root / 'episode_diagnostics.csv', episodes)
  write_csv(args.output_root / 'real_action_histograms.csv', histogram)
  write_csv(args.output_root / 'imagined_reward_diagnostics.csv', probes)
  pairing_rows, pairing = pairing_audit(
      args, matrix, sequences, resets, traces)
  write_csv(args.output_root / 'action_pairing_audit.csv', pairing_rows)
  write_json(args.output_root / 'pairing.json', pairing)
  summary = aggregate(results, episodes, matrix)
  write_csv(args.output_root / 'condition_summary.csv', summary)
  write_json(args.output_root / 'aggregate_summary.json', {
      'format': 1, 'canonical_stage': 'stage5b2a', 'rows': summary})
  make_figures(args, summary, episodes, histogram, probes, matrix)
  write_findings(args, summary, pairing)
  args.state.update(status='complete', updated_unix=time.time())
  write_json(args.state_path, args.state)
  (args.output_root / 'stage5b2a_complete').write_bytes(b'')
  print(f'Stage 5B.2a complete: {args.output_root}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
