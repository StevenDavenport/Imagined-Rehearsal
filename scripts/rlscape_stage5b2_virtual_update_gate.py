#!/usr/bin/env python3
"""Run Stage 5B.2 paired virtual-update gating in RLScape."""

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
from embodied.run.ir_gate import incremental_compute  # noqa: E402
from scripts import rlscape_stage2_actor_recovery as stage2  # noqa: E402
from scripts import rlscape_stage4b_conservative_value as stage4b  # noqa: E402
from scripts import rlscape_stage5_common as common  # noqa: E402
from scripts.rlscape_m1_pilot import read_jsonl  # noqa: E402
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


def conditions(args):
  selected = {'format': 1, 'canonical_stage': 'stage5a'}
  result = [
      common.no_ir_condition(),
      common.ir_condition('reward_only', 'always', selected),
  ]
  result += [
      common.virtual_reward_condition(
          f'virtual_reward_dormant_{cadence}',
          dormant_every=cadence,
          active_every=args.active_every,
          deactivate_after=args.deactivate_after,
          min_improvement=args.min_improvement,
          max_baseline_score=args.max_baseline_score)
      for cadence in args.dormant_cadences]
  assert len({item['name'] for item in result}) == len(result)
  return result


def heartbeat(args, snapshot):
  args.state['active_child'] = snapshot
  args.state['updated_unix'] = time.time()
  write_json(args.state_path, args.state)


def trace_episode_summary(logdir):
  grouped = {}
  for row in read_jsonl(logdir / 'stage5_trace.jsonl'):
    index = int(row['episode_index'])
    grouped.setdefault(index, []).append(row)
  result = []
  for index in sorted(grouped):
    rows = grouped[index]
    probes = [row for row in rows if 'gate_score_improvement' in row]
    committed = sum(float(row.get('adapt_trigger', 0)) for row in rows)
    tentative = sum(float(
        row.get('compute_tentative_actor_updates', 0)) for row in rows)
    rejected = sum(float(
        row.get('compute_rejected_actor_updates', 0)) for row in rows)
    deltas = [float(row['gate_score_improvement']) for row in probes]
    before_scores = [
        float(row.get('gate_score_before', np.nan)) for row in probes]
    result.append({
        'episode_index': index,
        'trace_steps': len(rows),
        'probes': len(probes),
        'committed_updates': committed,
        'tentative_updates': tentative,
        'rejected_updates': rejected,
        'first_probe_step': (
            int(probes[0]['episode_step']) if probes else -1),
        'last_probe_step': (
            int(probes[-1]['episode_step']) if probes else -1),
        'mean_predicted_improvement': (
            float(np.mean(deltas)) if deltas else float('nan')),
        'max_predicted_improvement': (
            float(np.max(deltas)) if deltas else float('nan')),
        'positive_probe_fraction': (
            float(np.mean(np.asarray(deltas) > 0))
            if deltas else float('nan')),
        'first_score_before': (
            before_scores[0] if before_scores else float('nan')),
        'mean_score_before': (
            float(np.mean(before_scores)) if before_scores else float('nan')),
        'first_predicted_improvement': (
            deltas[0] if deltas else float('nan')),
        'first_before_success_rate': (
            float(probes[0].get('gate_before_success_rate', np.nan))
            if probes else float('nan')),
        'first_before_js': (
            float(probes[0].get('gate_before_js_disagreement', np.nan))
            if probes else float('nan')),
        'first_before_actor_entropy': (
            float(probes[0].get('gate_before_actor_entropy', np.nan))
            if probes else float('nan')),
    })
  return result


def summarize(args, *, logdir, condition, goal, mode, episodes,
              replicate, returncode):
  env_seed = args.eval_seed_base + replicate
  agent_seed = args.agent_seed_base + replicate
  result = common.summarize_eval(
      logdir, condition=condition, goal=goal, policy_mode=mode,
      episodes=episodes, environment_seed=env_seed, returncode=returncode,
      warmup_episodes=1, agent_seed=agent_seed,
      paired_rng_stride=args.paired_rng_stride)
  traces = trace_episode_summary(logdir)
  result.update({
      'canonical_stage': 'stage5b2', 'replicate': replicate,
      'environment_seed': env_seed, 'agent_seed': agent_seed,
      'source_checkpoint': str(args.checkpoint),
      'virtual_dormant_every': int(condition.get(
          'virtual_dormant_every', 0)),
      'virtual_active_every': int(condition.get(
          'virtual_active_every', 0)),
      'virtual_deactivate_after': int(condition.get(
          'virtual_deactivate_after', 0)),
      'episode_trace_count': len(traces),
  })
  result['complete'] = bool(
      result['complete'] and len(traces) == int(episodes))
  return result


def build_command(args, *, logdir, condition, goal, mode, episodes,
                  replicate):
  return common.build_eval_command(
      args, logdir=logdir, checkpoint=args.checkpoint,
      condition=condition, goal=goal, policy_mode=mode,
      episodes=episodes,
      environment_seed=args.eval_seed_base + replicate,
      agent_seed=args.agent_seed_base + replicate,
      persistence='episode', warmup_episodes=1)


def execute(args, *, condition, goal, mode, episodes, replicate):
  unit = (
      f'replicate_{replicate:02d}/{condition["name"]}/{goal}/{mode}')
  state = args.state['units'].setdefault(
      unit, {'status': 'pending', 'attempts': []})

  def recover(path, code):
    return summarize(
        args, logdir=path, condition=condition, goal=goal, mode=mode,
        episodes=episodes, replicate=replicate, returncode=code)

  if state.get('status') == 'complete':
    result = recover(pathlib.Path(state['attempts'][-1]['logdir']), 0)
    if result['complete']:
      return result
    state['status'] = 'pending'

  attempt = state['attempts'][-1] if state['attempts'] else None
  if attempt and attempt.get('status') == 'running':
    logdir = pathlib.Path(attempt['logdir'])
    command = build_command(
        args, logdir=logdir, condition=condition, goal=goal, mode=mode,
        episodes=episodes, replicate=replicate)
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

  command = build_command(
      args, logdir=logdir, condition=condition, goal=goal, mode=mode,
      episodes=episodes, replicate=replicate)
  attempt['command'] = command
  state['status'] = 'running'
  write_json(args.state_path, args.state)
  print(f'\n=== Stage 5B.2 {unit}, attempt {number} ===', flush=True)
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
          args, condition=condition, goal=goal, mode=mode,
          episodes=episodes, replicate=replicate)
    raise RuntimeError(f'Incomplete Stage 5B.2 unit: {unit}')
  return result


def run_matrix(args, matrix):
  rows = []
  for replicate in range(args.replicates):
    for goal in GOALS:
      for condition in matrix:
        for mode in MODES:
          episodes = (
              args.sampled_episodes if mode == 'sampled'
              else args.deterministic_episodes)
          rows.append(execute(
              args, condition=condition, goal=goal, mode=mode,
              episodes=episodes, replicate=replicate))
          write_csv(args.output_root / 'results.csv', rows)
  write_json(args.output_root / 'results.json', {
      'format': 1, 'canonical_stage': 'stage5b2', 'rows': rows})
  return rows


def episode_rows(logdir):
  output = []
  resets = stage2.reset_identities(logdir)[1:]
  episodes = read_jsonl(logdir / 'episodes.jsonl')
  traces = trace_episode_summary(logdir)
  if not len(resets) == len(episodes) == len(traces):
    raise RuntimeError(
        f'Episode alignment failed in {logdir}: '
        f'{len(resets)}, {len(episodes)}, {len(traces)}')
  for index, (reset, episode, trace) in enumerate(
      zip(resets, episodes, traces)):
    success = int(
        float(episode.get('episode/score', 0)) > 0 or
        float(episode.get('episode/log/task_success/max', 0)) > 0)
    output.append({
        **trace, 'episode_index': index, 'success': success,
        'action_length': float(episode.get('episode/length', np.nan)) - 1,
        'reset_seed': reset.get('reset_seed'),
    })
  return output


def necessity_class(no_ir, always):
  if not no_ir and always:
    return 'ir_needed'
  if no_ir and always:
    return 'ir_unnecessary'
  if no_ir and not always:
    return 'ir_harmful'
  return 'not_repaired'


def paired_diagnostics(rows):
  output = []
  mismatches = []
  keys = sorted({
      (int(row['replicate']), row['goal'], row['policy_mode'])
      for row in rows})
  for replicate, goal, mode in keys:
    group = [
        row for row in rows if int(row['replicate']) == replicate and
        row['goal'] == goal and row['policy_mode'] == mode]
    by_name = {row['condition']: row for row in group}
    reference = stage2.reset_identities(pathlib.Path(
        by_name['no_ir']['logdir']))
    for row in group:
      if stage2.reset_identities(pathlib.Path(row['logdir'])) != reference:
        mismatches.append({
            'replicate': replicate, 'goal': goal, 'policy_mode': mode,
            'condition': row['condition']})
    episodes = {
        name: episode_rows(pathlib.Path(row['logdir']))
        for name, row in by_name.items()}
    count = len(episodes['no_ir'])
    if any(len(value) != count for value in episodes.values()):
      raise RuntimeError(f'Paired episode count mismatch: {replicate}/{goal}/{mode}')
    for index in range(count):
      factual = episodes['no_ir'][index]
      treated = episodes['always_reward_only'][index]
      label = necessity_class(factual['success'], treated['success'])
      for condition, values in episodes.items():
        item = values[index]
        output.append({
            'canonical_stage': 'stage5b2', 'replicate': replicate,
            'goal': goal, 'policy_mode': mode, 'condition': condition,
            'episode_index': index, 'reset_seed': item['reset_seed'],
            'necessity_class': label,
            'no_ir_success': factual['success'],
            'always_reward_success': treated['success'],
            'success': item['success'],
            'action_length': item['action_length'],
            'probes': item['probes'],
            'committed_updates': item['committed_updates'],
            'tentative_updates': item['tentative_updates'],
            'rejected_updates': item['rejected_updates'],
            'mean_predicted_improvement': item['mean_predicted_improvement'],
            'max_predicted_improvement': item['max_predicted_improvement'],
            'positive_probe_fraction': item['positive_probe_fraction'],
            'first_score_before': item['first_score_before'],
            'mean_score_before': item['mean_score_before'],
            'first_predicted_improvement': item[
                'first_predicted_improvement'],
            'first_before_success_rate': item['first_before_success_rate'],
            'first_before_js': item['first_before_js'],
            'first_before_actor_entropy': item[
                'first_before_actor_entropy'],
            'commit_on_no_ir_success': int(
                factual['success'] and item['committed_updates'] > 0),
            'missed_repair_opportunity': int(
                label == 'ir_needed' and item['committed_updates'] == 0),
        })
  if mismatches:
    raise RuntimeError(f'Stage 5B.2 reset pairing failed: {mismatches[:3]}')
  return output, {'format': 1, 'equal': True, 'mismatches': []}


def roc_auc(labels, scores):
  labels = np.asarray(labels, bool)
  scores = np.asarray(scores, np.float64)
  valid = np.isfinite(scores)
  labels, scores = labels[valid], scores[valid]
  positive, negative = scores[labels], scores[~labels]
  if not len(positive) or not len(negative):
    return float('nan')
  return float(np.mean(
      (positive[:, None] > negative[None, :]) +
      .5 * (positive[:, None] == negative[None, :])))


def gate_diagnostics(diagnostics):
  rows = []
  virtual_names = sorted({
      row['condition'] for row in diagnostics
      if row['condition'].startswith('virtual_')})
  cells = [('all', 'all')] + [
      (goal, mode) for goal in GOALS for mode in MODES]
  predictors = {
      'predicted_failure': lambda row: -float(row['first_score_before']),
      'first_virtual_improvement': lambda row: float(
          row['first_predicted_improvement']),
      'max_virtual_improvement': lambda row: float(
          row['max_predicted_improvement']),
      'first_js': lambda row: float(row['first_before_js']),
      'first_actor_entropy': lambda row: float(
          row['first_before_actor_entropy']),
  }
  for condition in virtual_names:
    for goal, mode in cells:
      chosen = [
          row for row in diagnostics if row['condition'] == condition and
          (goal == 'all' or row['goal'] == goal) and
          (mode == 'all' or row['policy_mode'] == mode)]
      for target in ('underperform', 'ir_needed'):
        labels = [
            not bool(row['no_ir_success']) if target == 'underperform'
            else row['necessity_class'] == 'ir_needed'
            for row in chosen]
        for name, predictor in predictors.items():
          scores = [predictor(row) for row in chosen]
          rows.append({
              'condition': condition, 'goal': goal, 'policy_mode': mode,
              'target': target, 'predictor': name,
              'episodes': len(chosen), 'positives': sum(labels),
              'roc_auc': roc_auc(labels, scores),
          })
  return rows


def aggregate(rows, diagnostics):
  output = []
  for goal in GOALS:
    for mode in MODES:
      cell = [row for row in rows if row['goal'] == goal and
              row['policy_mode'] == mode]
      lookup = {}
      for name in sorted({row['condition'] for row in cell}):
        chosen = [row for row in cell if row['condition'] == name]
        steps = sum(int(row['environment_steps']) for row in chosen)
        entry = {
            'goal': goal, 'policy_mode': mode, 'condition': name,
            'success_rate': sum(int(row['successes']) for row in chosen) /
            sum(int(row['episodes']) for row in chosen),
            'episodes': sum(int(row['episodes']) for row in chosen),
            'environment_steps': steps,
            'seconds_per_step': sum(float(
                row['evaluation_seconds']) for row in chosen) / steps,
            'probes_per_step': sum(float(
                row.get('gate_evaluated_sum', 0)) for row in chosen) / steps,
            'committed_updates_per_step': sum(float(
                row.get('compute_actor_updates_sum', 0))
                for row in chosen) / steps,
            'tentative_updates_per_step': sum(float(
                row.get('compute_tentative_actor_updates_sum', 0))
                for row in chosen) / steps,
            'rejected_updates_per_step': sum(float(
                row.get('compute_rejected_actor_updates_sum', 0))
                for row in chosen) / steps,
            'imagined_transitions_per_step': sum(float(
                row.get('compute_imagined_transitions_sum', 0))
                for row in chosen) / steps,
        }
        diag = [row for row in diagnostics if row['goal'] == goal and
                row['policy_mode'] == mode and row['condition'] == name]
        successful = [row['action_length'] for row in diag if row['success']]
        entry['mean_success_length'] = (
            float(np.mean(successful)) if successful else float('nan'))
        lookup[name] = entry
        output.append(entry)
      no_ir = lookup['no_ir']['seconds_per_step']
      always = lookup['always_reward_only']['seconds_per_step']
      for entry in lookup.values():
        entry['incremental_compute_saved'] = incremental_compute(
            no_ir, always, entry['seconds_per_step'])
  return output


def make_figures(root, summary, diagnostics):
  import matplotlib.pyplot as plt
  figure_root = root / 'figures'
  figure_root.mkdir(exist_ok=True)
  names = sorted({row['condition'] for row in summary})
  for mode in MODES:
    chosen = [row for row in summary if row['policy_mode'] == mode]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for axis, goal in zip(axes, GOALS):
      lookup = {(row['goal'], row['condition']): row for row in chosen}
      values = [lookup[(goal, name)]['success_rate'] for name in names]
      axis.bar(np.arange(len(names)), values)
      axis.set_xticks(np.arange(len(names)), names, rotation=35, ha='right')
      axis.set_ylim(0, 1); axis.set_title(goal)
      axis.set_ylabel('Success rate')
    fig.suptitle(f'Stage 5B.2 virtual-update gate ({mode})')
    for suffix in ('png', 'pdf'):
      fig.savefig(figure_root / f'performance_{mode}.{suffix}', dpi=220)
    plt.close(fig)

  virtual = [row for row in summary
             if row['condition'].startswith('virtual_')]
  fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
  for row in virtual:
    axis.scatter(
        row['tentative_updates_per_step'], row['success_rate'],
        marker='o' if row['policy_mode'] == 'sampled' else 'x')
    axis.annotate(
        f'{row["goal"]}:{row["policy_mode"]}:{row["condition"].split("_")[-1]}',
        (row['tentative_updates_per_step'], row['success_rate']), fontsize=7)
  axis.set_xlabel('Tentative updates per real step')
  axis.set_ylabel('Success rate')
  axis.set_title('Performance versus virtual-gate compute')
  for suffix in ('png', 'pdf'):
    fig.savefig(figure_root / f'performance_compute.{suffix}', dpi=220)
  plt.close(fig)

  virtual_diag = [row for row in diagnostics
                  if row['condition'].startswith('virtual_')]
  for condition in sorted({row['condition'] for row in virtual_diag}):
    chosen = [row for row in virtual_diag if row['condition'] == condition]
    for mode in MODES:
      mode_rows = [row for row in chosen if row['policy_mode'] == mode]
      fig, axes = plt.subplots(
          1, len(GOALS), figsize=(15, 4.2), constrained_layout=True)
      markers = {
          'ir_needed': ('o', 'tab:red'),
          'ir_unnecessary': ('^', 'tab:green'),
          'ir_harmful': ('x', 'tab:orange'),
          'not_repaired': ('s', 'tab:gray'),
      }
      for axis, goal in zip(axes, GOALS):
        for label, (marker, color) in markers.items():
          points = [row for row in mode_rows if row['goal'] == goal and
                    row['necessity_class'] == label]
          axis.scatter(
              [row['first_score_before'] for row in points],
              [row['first_predicted_improvement'] for row in points],
              marker=marker, color=color, alpha=.7, label=label)
        axis.axhline(0, color='black', linewidth=1)
        axis.set_title(goal); axis.set_xlabel('Initial imagined return')
        axis.set_ylabel('Predicted benefit of first IR update')
      handles, labels = axes[-1].get_legend_handles_labels()
      fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False)
      fig.suptitle(f'{condition}: prospective gate plane ({mode})')
      for suffix in ('png', 'pdf'):
        fig.savefig(
            figure_root / f'gate_plane_{condition}_{mode}.{suffix}', dpi=220)
      plt.close(fig)

  classes = ('ir_needed', 'ir_unnecessary', 'ir_harmful', 'not_repaired')
  fig, axes = plt.subplots(
      1, len(classes), figsize=(15, 4), constrained_layout=True,
      sharey=True)
  for axis, label in zip(axes, classes):
    values = [float(row['mean_predicted_improvement'])
              for row in virtual_diag if row['necessity_class'] == label and
              np.isfinite(float(row['mean_predicted_improvement']))]
    axis.hist(values, bins=20)
    axis.axvline(0, color='black', linewidth=1)
    axis.set_title(label.replace('_', ' ')); axis.set_xlabel('Predicted delta')
  axes[0].set_ylabel('Episodes')
  fig.suptitle('Virtual-update predictions by actual IR necessity')
  for suffix in ('png', 'pdf'):
    fig.savefig(figure_root / f'predicted_delta_by_class.{suffix}', dpi=220)
  plt.close(fig)


def write_findings(root, summary, diagnostics, diagnostic_auc):
  virtual_names = sorted({row['condition'] for row in summary
                          if row['condition'].startswith('virtual_')})
  lines = [
      '# Stage 5B.2 Virtual-Update Gate', '',
      'This document is generated from the completed paired experiment. ',
      '', '## Aggregate results', '',
      '| Goal | Mode | Condition | Success | Tentative/step | Commit/step | Seconds/step |',
      '|---|---|---:|---:|---:|---:|---:|']
  for row in summary:
    lines.append(
        f'| {row["goal"]} | {row["policy_mode"]} | '
        f'{row["condition"]} | {row["success_rate"]:.3f} | '
        f'{row["tentative_updates_per_step"]:.4f} | '
        f'{row["committed_updates_per_step"]:.4f} | '
        f'{row["seconds_per_step"]:.4f} |')
  lines += ['', '## Decision audit', '']
  for name in virtual_names:
    chosen = [row for row in diagnostics if row['condition'] == name]
    lines.append(
        f'- `{name}`: {sum(row["committed_updates"] > 0 for row in chosen)} '
        f'episodes received IR; '
        f'{sum(row["missed_repair_opportunity"] for row in chosen)} known '
        f'repair opportunities received no update; '
        f'{sum(row["commit_on_no_ir_success"] for row in chosen)} competent '
        'episodes received at least one update.')
    auc = [row for row in diagnostic_auc if row['condition'] == name and
           row['goal'] == 'all' and row['policy_mode'] == 'all']
    lookup = {(row['target'], row['predictor']): row['roc_auc'] for row in auc}
    lines.append(
        f'  Pooled AUC: initial predicted failure -> real underperformance '
        f'`{lookup[("underperform", "predicted_failure")]:.3f}`; first '
        f'virtual improvement -> actual IR need '
        f'`{lookup[("ir_needed", "first_virtual_improvement")]:.3f}`.')
  lines += [
      '', 'Interpret all decisions prospectively, but necessity labels ',
      'retrospectively. The latter are used only for audit and never by the gate.',
  ]
  (root / 'STAGE5B2_FINDINGS.md').write_text('\n'.join(lines) + '\n')


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', type=pathlib.Path, required=True)
  parser.add_argument('--stage5b1-root', type=pathlib.Path, required=True)
  parser.add_argument('--output-root', type=pathlib.Path)
  parser.add_argument('--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument('--python', type=pathlib.Path,
                      default=pathlib.Path(sys.executable))
  parser.add_argument('--replicates', type=int, default=3)
  parser.add_argument('--sampled-episodes', type=int, default=50)
  parser.add_argument('--deterministic-episodes', type=int, default=25)
  parser.add_argument('--dormant-cadences', type=int, nargs='+',
                      default=(25, 100))
  parser.add_argument('--active-every', type=int, default=1)
  parser.add_argument('--deactivate-after', type=int, default=3)
  parser.add_argument('--min-improvement', type=float, default=0.0)
  parser.add_argument('--max-baseline-score', type=float, default=1e30)
  parser.add_argument('--episode-length', type=int, default=200)
  parser.add_argument('--grid-columns', type=int, default=28)
  parser.add_argument('--grid-rows', type=int, default=18)
  parser.add_argument('--adapt-steps', type=int, default=1)
  parser.add_argument('--adapt-lr', type=float, default=4e-5)
  parser.add_argument('--adapt-every-k', type=int, default=1)
  parser.add_argument('--start-batch', type=int, default=128)
  parser.add_argument('--paired-rng-stride', type=int, default=1000)
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
  args.stage5b1_root = args.stage5b1_root.expanduser().resolve()
  args.output_root = (
      args.output_root.expanduser().resolve() if args.output_root else
      args.stage5b1_root.parent / 'stage5b2_virtual_update_gate')
  for path in (
      args.python, args.checkpoint / 'agent.pkl', args.checkpoint / 'done',
      args.stage5b1_root / 'stage5b1_complete',
      args.stage5b1_root / 'audit_spec.json'):
    if not path.is_file():
      raise FileNotFoundError(path)
  audit = json.loads((args.stage5b1_root / 'audit_spec.json').read_text())
  if audit.get('canonical_stage') != 'stage5b1':
    raise RuntimeError('Invalid Stage 5B.1 audit artifact')
  if args.paired_rng_stride <= args.episode_length:
    raise ValueError('paired RNG stride must exceed episode length')
  if min(
      args.replicates, args.sampled_episodes,
      args.deterministic_episodes, args.active_every,
      args.deactivate_after, *args.dormant_cadences) < 1:
    raise ValueError('Episode counts, cadences, and replicates must be positive')


def main(argv=None):
  args = parse_args(argv)
  validate(args)
  args.output_root.mkdir(parents=True, exist_ok=True)
  matrix = conditions(args)
  audit_path = args.stage5b1_root / 'audit_spec.json'
  spec = {
      'format': 1, 'canonical_stage': 'stage5b2',
      'slug': 'paired_virtual_update_gate',
      'checkpoint': str(args.checkpoint),
      'checkpoint_sha256': _sha256(args.checkpoint / 'agent.pkl'),
      'stage5b1_root': str(args.stage5b1_root),
      'stage5b1_audit_sha256': _sha256(audit_path),
      'conditions': [item['name'] for item in matrix],
      'goals': list(GOALS), 'modes': list(MODES),
      'replicates': args.replicates,
      'sampled_episodes': args.sampled_episodes,
      'deterministic_episodes': args.deterministic_episodes,
      'adaptation': {
          'objective': 'reward_only', 'horizon': 15,
          'posterior_samples': args.start_batch, 'steps': args.adapt_steps,
          'lr': args.adapt_lr, 'persistence': 'episode',
          'score': 'reward_return', 'min_improvement': args.min_improvement,
          'max_baseline_score': args.max_baseline_score,
          'dormant_cadences': list(args.dormant_cadences),
          'active_every': args.active_every,
          'deactivate_after': args.deactivate_after,
          'paired_before_after': True},
      'retrospective_labels_visible_to_gate': False,
      'git_commit': git_revision(args.repo_root),
      'python_version': platform.python_version(),
      'rlscape_version': installed_version('rl-scape'),
  }
  spec['digest'] = spec_digest(spec)
  spec_path = args.output_root / 'experiment_spec.json'
  if spec_path.is_file() and json.loads(
      spec_path.read_text()).get('digest') != spec['digest']:
    raise RuntimeError('Existing Stage 5B.2 root has a different specification')
  if not spec_path.is_file():
    write_json(spec_path, spec)
  if args.dry_run:
    print(json.dumps(spec, indent=2))
    return 0

  args.state_path = args.output_root / 'queue.json'
  args.active_path = args.output_root / 'active_child.json'
  args.state = (
      json.loads(args.state_path.read_text()) if args.state_path.is_file()
      else {'format': 1, 'canonical_stage': 'stage5b2',
            'spec_digest': spec['digest'], 'status': 'running', 'units': {}})
  if args.state.get('spec_digest') != spec['digest']:
    raise RuntimeError('Stage 5B.2 queue specification mismatch')
  write_json(args.state_path, args.state)

  rows = run_matrix(args, matrix)
  diagnostics, pairing = paired_diagnostics(rows)
  write_csv(args.output_root / 'episode_diagnostics.csv', diagnostics)
  write_json(args.output_root / 'pairing.json', pairing)
  diagnostic_auc = gate_diagnostics(diagnostics)
  write_csv(args.output_root / 'gate_diagnostics.csv', diagnostic_auc)
  summary = aggregate(rows, diagnostics)
  write_csv(args.output_root / 'condition_summary.csv', summary)
  write_json(args.output_root / 'aggregate_summary.json', {
      'format': 1, 'canonical_stage': 'stage5b2', 'rows': summary})
  make_figures(args.output_root, summary, diagnostics)
  write_findings(args.output_root, summary, diagnostics, diagnostic_auc)
  args.state.update(status='complete', updated_unix=time.time())
  write_json(args.state_path, args.state)
  (args.output_root / 'stage5b2_complete').write_bytes(b'')
  print(f'Stage 5B.2 complete: {args.output_root}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
