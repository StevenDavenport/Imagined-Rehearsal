#!/usr/bin/env python3
"""Offline Stage 5B.1 audit of prospective IR-necessity signals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import rlscape_stage2_actor_recovery as stage2  # noqa: E402
from scripts.rlscape_m1_pilot import read_jsonl  # noqa: E402
from scripts.rlscape_m1_pilot import spec_digest  # noqa: E402
from scripts.rlscape_m1_pilot import write_csv  # noqa: E402
from scripts.rlscape_m1_pilot import write_json  # noqa: E402


FEATURES = (
    'actor_entropy', 'posterior_entropy', 'js_disagreement',
    'success_rate', 'success_action_concentration',
    'success_action_divergence')
WINDOWS = (1, 5, 10, 25)
MODES = ('sampled', 'deterministic')


def file_sha256(path):
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    while chunk := handle.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def load_csv(path):
  with path.open(newline='') as handle:
    return list(csv.DictReader(handle))


def validated_json(path, stage, digest_required=True):
  data = json.loads(path.read_text())
  if data.get('canonical_stage') != stage:
    raise RuntimeError(f'Expected {stage} artifact: {path}')
  if digest_required:
    recorded = data.get('digest')
    payload = dict(data)
    payload.pop('digest', None)
    if not recorded or spec_digest(payload) != recorded:
      raise RuntimeError(f'Artifact digest mismatch: {path}')
  return data


def outcome_rows(logdir):
  rows = read_jsonl(logdir / 'episodes.jsonl')
  result = []
  for row in rows:
    success = bool(
        float(row.get('episode/score', 0.0)) > 0 or
        float(row.get('episode/log/task_success/max', 0.0)) > 0)
    result.append({
        'success': success,
        'action_length': max(0.0, float(row['episode/length']) - 1)})
  return result


def trace_episodes(logdir):
  groups = {}
  for row in read_jsonl(logdir / 'stage5a_trace.jsonl'):
    if 'gate_js_disagreement' not in row:
      continue
    groups.setdefault(int(row['episode_index']), []).append(row)
  for rows in groups.values():
    rows.sort(key=lambda row: int(row['episode_step']))
  return groups


def reset_identities(logdir):
  return stage2.reset_identities(logdir)


def unique_row(rows, condition, goal, mode):
  chosen = [row for row in rows if row['condition'] == condition and
            row['goal'] == goal and row['policy_mode'] == mode]
  if len(chosen) != 1:
    raise RuntimeError(
        f'Expected one {condition}/{goal}/{mode} row, got {len(chosen)}')
  return chosen[0]


def aggregate_prefix(rows, window):
  chosen = rows[:min(window, len(rows))]
  result = {'observed_states': len(chosen)}
  for feature in FEATURES:
    values = np.asarray([
        float(row.get(f'gate_{feature}', 0.0)) for row in chosen],
        np.float64)
    result[f'mean_{feature}_w{window}'] = float(values.mean())
    result[f'max_{feature}_w{window}'] = float(values.max())
    result[f'p90_{feature}_w{window}'] = float(np.quantile(values, .9))
  js = result[f'mean_js_disagreement_w{window}']
  posterior = result[f'mean_posterior_entropy_w{window}']
  result[f'js_fraction_w{window}'] = js / max(posterior, 1e-8)
  result[f'predicted_failure_w{window}'] = (
      1.0 - result[f'mean_success_rate_w{window}'])
  result[f'js_x_predicted_failure_w{window}'] = (
      js * result[f'predicted_failure_w{window}'])
  return result


def necessity_class(no_ir_success, ir_success):
  if not no_ir_success and ir_success:
    return 'ir_needed'
  if no_ir_success and ir_success:
    return 'ir_unnecessary'
  if no_ir_success and not ir_success:
    return 'ir_harmful'
  return 'not_repaired'


def extract_episodes(stage5a_root, audit_rows):
  episode_rows = []
  pairing = []
  goals = sorted({row['goal'] for row in audit_rows})
  for goal in goals:
    for mode in MODES:
      no_ir = unique_row(audit_rows, 'no_ir', goal, mode)
      reward = unique_row(audit_rows, 'always_reward_only', goal, mode)
      no_ir_dir = pathlib.Path(no_ir['logdir'])
      reward_dir = pathlib.Path(reward['logdir'])
      resets_equal = reset_identities(no_ir_dir) == reset_identities(reward_dir)
      pairing.append({
          'goal': goal, 'policy_mode': mode, 'equal': resets_equal,
          'no_ir_logdir': str(no_ir_dir), 'reward_logdir': str(reward_dir)})
      if not resets_equal:
        raise RuntimeError(f'Unpaired audit cell: {goal}/{mode}')
      factual = outcome_rows(no_ir_dir)
      treated = outcome_rows(reward_dir)
      traces = trace_episodes(no_ir_dir)
      if not len(factual) == len(treated) == len(traces):
        raise RuntimeError(
            f'Outcome/trace count mismatch for {goal}/{mode}: '
            f'{len(factual)}, {len(treated)}, {len(traces)}')
      for index, (base, adapted) in enumerate(zip(factual, treated)):
        trace = traces[index]
        row = {
            'canonical_stage': 'stage5b1', 'source_stage': 'stage5a_audit',
            'goal': goal, 'policy_mode': mode, 'episode_index': index,
            'reset_seed': reset_identities(no_ir_dir)[index + 1]['reset_seed'],
            'no_ir_success': int(base['success']),
            'no_ir_action_length': base['action_length'],
            'always_reward_success': int(adapted['success']),
            'always_reward_action_length': adapted['action_length'],
            'underperform': int(not base['success']),
            'ir_needed': int(not base['success'] and adapted['success']),
            'necessity_class': necessity_class(
                base['success'], adapted['success']),
            'episode_states': len(trace),
        }
        for window in WINDOWS:
          row.update(aggregate_prefix(trace, window))
        episode_rows.append(row)
  return goals, episode_rows, pairing


def roc_auc(labels, scores):
  labels = np.asarray(labels, bool)
  scores = np.asarray(scores, np.float64)
  positive = scores[labels]
  negative = scores[~labels]
  if not len(positive) or not len(negative):
    return float('nan')
  comparisons = positive[:, None] - negative[None, :]
  return float((np.sum(comparisons > 0) + .5 * np.sum(comparisons == 0)) /
               comparisons.size)


def rates(labels, predictions):
  labels = np.asarray(labels, bool)
  predictions = np.asarray(predictions, bool)
  positives = max(1, int(labels.sum()))
  negatives = max(1, int((~labels).sum()))
  tpr = float(np.sum(predictions & labels) / positives)
  fpr = float(np.sum(predictions & ~labels) / negatives)
  return tpr, fpr, .5 * (tpr + 1 - fpr)


def optimal_threshold(labels, scores):
  labels = np.asarray(labels, bool)
  scores = np.asarray(scores, np.float64)
  if not labels.any() or labels.all():
    return {
        'threshold': float('nan'), 'balanced_accuracy': float('nan'),
        'true_positive_rate': float('nan'), 'false_positive_rate': float('nan')}
  values = np.unique(scores)
  candidates = np.concatenate((
      [np.nextafter(values.min(), -np.inf)],
      (values[:-1] + values[1:]) / 2,
      [np.nextafter(values.max(), np.inf)]))
  results = []
  for threshold in candidates:
    tpr, fpr, balanced = rates(labels, scores >= threshold)
    results.append((balanced, -fpr, threshold, tpr, fpr))
  balanced, _, threshold, tpr, fpr = max(results)
  return {
      'threshold': float(threshold), 'balanced_accuracy': float(balanced),
      'true_positive_rate': float(tpr), 'false_positive_rate': float(fpr)}


def separability(episodes):
  rows = []
  predictors = {
      'mean_js': lambda row, window: row[
          f'mean_js_disagreement_w{window}'],
      'max_js': lambda row, window: row[
          f'max_js_disagreement_w{window}'],
      'actor_entropy': lambda row, window: row[
          f'mean_actor_entropy_w{window}'],
      'posterior_entropy': lambda row, window: row[
          f'mean_posterior_entropy_w{window}'],
      'predicted_failure': lambda row, window: row[
          f'predicted_failure_w{window}'],
      'js_fraction': lambda row, window: row[f'js_fraction_w{window}'],
      'js_x_predicted_failure': lambda row, window: row[
          f'js_x_predicted_failure_w{window}'],
  }
  groups = [('all', 'all', episodes)]
  groups += [(goal, mode, [row for row in episodes
                           if row['goal'] == goal and
                           row['policy_mode'] == mode])
             for goal in sorted({row['goal'] for row in episodes})
             for mode in MODES]
  for goal, mode, selected in groups:
    for target in ('underperform', 'ir_needed'):
      labels = [bool(row[target]) for row in selected]
      for window in WINDOWS:
        for predictor, fn in predictors.items():
          scores = [fn(row, window) for row in selected]
          threshold = optimal_threshold(labels, scores)
          rows.append({
              'goal': goal, 'policy_mode': mode, 'target': target,
              'window': window, 'predictor': predictor,
              'episodes': len(selected), 'positives': sum(labels),
              'roc_auc': roc_auc(labels, scores), **threshold})
  return rows


def threshold_transfer(episodes, window, selected_js_threshold):
  rows = []
  goals = sorted({row['goal'] for row in episodes})
  for mode in MODES:
    by_goal = {
        goal: [row for row in episodes if row['goal'] == goal and
               row['policy_mode'] == mode] for goal in goals}
    for source in goals:
      source_rows = by_goal[source]
      labels = [bool(row['underperform']) for row in source_rows]
      values = [row[f'mean_js_disagreement_w{window}']
                for row in source_rows]
      learned = optimal_threshold(labels, values)['threshold']
      for target in goals:
        target_rows = by_goal[target]
        target_labels = [bool(row['underperform']) for row in target_rows]
        target_values = np.asarray([
            row[f'mean_js_disagreement_w{window}'] for row in target_rows])
        if math.isnan(learned):
          tpr = fpr = balanced = float('nan')
        else:
          tpr, fpr, balanced = rates(
              target_labels, target_values >= learned)
        rows.append({
            'policy_mode': mode, 'window': window,
            'source_goal': source, 'target_goal': target,
            'threshold': learned, 'true_positive_rate': tpr,
            'false_positive_rate': fpr, 'balanced_accuracy': balanced})
    for target in goals:
      target_rows = by_goal[target]
      labels = [bool(row['underperform']) for row in target_rows]
      values = np.asarray([
          row[f'mean_js_disagreement_w{window}'] for row in target_rows])
      tpr, fpr, balanced = rates(labels, values >= selected_js_threshold)
      rows.append({
          'policy_mode': mode, 'window': window,
          'source_goal': 'stage5a_global', 'target_goal': target,
          'threshold': selected_js_threshold, 'true_positive_rate': tpr,
          'false_positive_rate': fpr, 'balanced_accuracy': balanced})
  return rows


def calibration(episodes, window, bins=5):
  rows = []
  for mode in MODES:
    for goal in sorted({row['goal'] for row in episodes}):
      selected = [row for row in episodes if row['goal'] == goal and
                  row['policy_mode'] == mode]
      selected.sort(key=lambda row: row[f'mean_js_disagreement_w{window}'])
      for index, indices in enumerate(np.array_split(
          np.arange(len(selected)), min(bins, len(selected)))):
        chosen = [selected[int(i)] for i in indices]
        successful_lengths = [row['no_ir_action_length'] for row in chosen
                              if row['no_ir_success']]
        rows.append({
            'goal': goal, 'policy_mode': mode, 'window': window,
            'bin': index, 'episodes': len(chosen),
            'mean_js': float(np.mean([
                row[f'mean_js_disagreement_w{window}'] for row in chosen])),
            'underperformance_rate': float(np.mean([
                row['underperform'] for row in chosen])),
            'ir_needed_rate': float(np.mean([
                row['ir_needed'] for row in chosen])),
            'mean_predicted_success': float(np.mean([
                row[f'mean_success_rate_w{window}'] for row in chosen])),
            'median_success_length': (
                float(np.median(successful_lengths))
                if successful_lengths else float('nan')),
        })
  return rows


def make_figures(root, episodes, transfer, calibration_rows, window):
  import matplotlib.pyplot as plt
  figure_root = root / 'figures'
  figure_root.mkdir(exist_ok=True)
  goals = sorted({row['goal'] for row in episodes})
  colors = {'success': '#2a9d8f', 'underperform': '#e76f51'}

  for mode in MODES:
    fig, axes = plt.subplots(
        1, len(goals), figsize=(5 * len(goals), 4), constrained_layout=True,
        sharex=True, sharey=True)
    for axis, goal in zip(np.atleast_1d(axes), goals):
      chosen = [row for row in episodes if row['goal'] == goal and
                row['policy_mode'] == mode]
      for label, predicate in (
          ('success', lambda row: not row['underperform']),
          ('underperform', lambda row: row['underperform'])):
        values = [row[f'mean_js_disagreement_w{window}'] for row in chosen
                  if predicate(row)]
        if values:
          axis.hist(values, bins=10, alpha=.6, density=True,
                    label=label, color=colors[label])
      axis.set_title(goal)
      axis.set_xlabel(f'Mean JS, first {window} states')
      axis.set_ylabel('Density')
      axis.legend(frameon=False)
    fig.suptitle(f'JS distributions by eventual performance ({mode})')
    for suffix in ('png', 'pdf'):
      fig.savefig(figure_root / f'js_distributions_{mode}.{suffix}', dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(
        1, len(goals), figsize=(5 * len(goals), 4), constrained_layout=True,
        sharey=True)
    for axis, goal in zip(np.atleast_1d(axes), goals):
      chosen = [row for row in episodes if row['goal'] == goal and
                row['policy_mode'] == mode]
      for klass, marker, color in (
          ('ir_needed', 'o', '#d62828'),
          ('ir_unnecessary', '^', '#2a9d8f'),
          ('ir_harmful', 'x', '#f4a261'),
          ('not_repaired', 's', '#6c757d')):
        points = [row for row in chosen if row['necessity_class'] == klass]
        if points:
          axis.scatter(
              [row[f'mean_js_disagreement_w{window}'] for row in points],
              [row[f'mean_success_rate_w{window}'] for row in points],
              marker=marker, color=color, alpha=.75, label=klass)
      axis.set_title(goal)
      axis.set_xlabel(f'Mean JS, first {window} states')
      axis.set_ylabel('Imagined H=15 success fraction')
      axis.legend(frameon=False, fontsize=7)
    fig.suptitle(f'Prospective signal plane and paired IR necessity ({mode})')
    for suffix in ('png', 'pdf'):
      fig.savefig(figure_root / f'performance_plane_{mode}.{suffix}', dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for goal in goals:
      chosen = [row for row in calibration_rows if row['goal'] == goal and
                row['policy_mode'] == mode]
      axis.plot([row['mean_js'] for row in chosen],
                [row['underperformance_rate'] for row in chosen],
                marker='o', label=goal)
    axis.set_xlabel(f'Mean JS, first {window} states (quantile bins)')
    axis.set_ylabel('Observed underperformance rate')
    axis.set_ylim(-.03, 1.03)
    axis.set_title(f'JS performance calibration ({mode})')
    axis.legend(frameon=False)
    for suffix in ('png', 'pdf'):
      fig.savefig(figure_root / f'js_calibration_{mode}.{suffix}', dpi=220)
    plt.close(fig)

    chosen = [row for row in transfer if row['policy_mode'] == mode and
              row['source_goal'] != 'stage5a_global']
    lookup = {(row['source_goal'], row['target_goal']): row
              for row in chosen}
    matrix = np.asarray([
        [lookup[(source, target)]['balanced_accuracy'] for target in goals]
        for source in goals], np.float64)
    fig, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap='viridis')
    axis.set_xticks(range(len(goals)), goals, rotation=30, ha='right')
    axis.set_yticks(range(len(goals)), goals)
    axis.set_xlabel('Threshold evaluated on task')
    axis.set_ylabel('Threshold selected on task')
    axis.set_title(f'JS threshold transfer ({mode})')
    for y in range(len(goals)):
      for x in range(len(goals)):
        value = matrix[y, x]
        axis.text(x, y, 'NA' if np.isnan(value) else f'{value:.2f}',
                  ha='center', va='center', color='white')
    fig.colorbar(image, ax=axis, label='Balanced accuracy')
    for suffix in ('png', 'pdf'):
      fig.savefig(figure_root / f'threshold_transfer_{mode}.{suffix}', dpi=220)
    plt.close(fig)


def write_findings(root, episodes, separability_rows, transfer, window):
  goals = sorted({row['goal'] for row in episodes})
  classes = {name: sum(row['necessity_class'] == name for row in episodes)
             for name in ('ir_needed', 'ir_unnecessary', 'ir_harmful',
                          'not_repaired')}
  lines = [
      '# Stage 5B.1 Offline Performance-Gate Audit', '',
      'This is a retrospective diagnostic over prospectively recorded no-IR '
      'signals. It does not qualify a deployable gate.', '',
      f'Episodes: **{len(episodes)}**. Paired classes: `{classes}`.', '',
      f'Core decision window: first **{window}** factual states.', '',
      '## Feature separability', '',
      '| Goal | Mode | JS AUC: underperform | Predicted-failure AUC | '
      'JS AUC: IR needed |',
      '|---|---:|---:|---:|---:|']
  for goal in goals:
    for mode in MODES:
      def auc(target, predictor):
        selected = [row for row in separability_rows
                    if row['goal'] == goal and row['policy_mode'] == mode and
                    row['target'] == target and row['window'] == window and
                    row['predictor'] == predictor]
        return selected[0]['roc_auc'] if selected else float('nan')
      lines.append(
          f'| {goal} | {mode} | {auc("underperform", "mean_js"):.3f} | '
          f'{auc("underperform", "predicted_failure"):.3f} | '
          f'{auc("ir_needed", "mean_js"):.3f} |')
  lines += [
      '', '## Interpretation boundary', '',
      '- AUC measures ranking, not a safe operating threshold.',
      '- Task-specific optimal thresholds are deliberately diagnostic and '
      'must not be deployed without held-out cross-task validation.',
      '- The H=15 imagined success fraction measures whether the current actor '
      'already discovers success; zero predicted successes may indicate need, '
      'lack of a learnable direction, or both.',
      '- A cross-validated pre/post imagined-performance improvement signal is '
      'not present in these historical traces and is the principal candidate '
      'for the next small intervention.', '',
      'See `threshold_transfer.csv` and `figures/` for the task-dependence '
      'diagnosis.', '']
  (root / 'STAGE5B1_FINDINGS.md').write_text('\n'.join(lines))


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--stage5a-root', type=pathlib.Path, required=True)
  parser.add_argument('--stage5b-root', type=pathlib.Path, required=True)
  parser.add_argument('--output-root', type=pathlib.Path)
  parser.add_argument('--window', type=int, choices=WINDOWS, default=10)
  return parser.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)
  stage5a_root = args.stage5a_root.expanduser().resolve()
  stage5b_root = args.stage5b_root.expanduser().resolve()
  output_root = (
      args.output_root.expanduser().resolve() if args.output_root else
      stage5b_root.parent / 'stage5b1_performance_gate_audit')
  for marker in (stage5a_root / 'stage5a_complete',
                 stage5b_root / 'stage5b_complete'):
    if not marker.is_file():
      raise FileNotFoundError(marker)
  stage5a_spec = validated_json(
      stage5a_root / 'experiment_spec.json', 'stage5a')
  stage5b_spec = validated_json(
      stage5b_root / 'experiment_spec.json', 'stage5b')
  selected = validated_json(stage5a_root / 'selected_gates.json', 'stage5a')
  selection = validated_json(stage5b_root / 'task_selection.json', 'stage5b')
  pairing = json.loads((stage5b_root / 'pairing.json').read_text())
  if not pairing.get('equal'):
    raise RuntimeError('Stage 5B pairing audit did not pass')
  if pathlib.Path(stage5a_spec['checkpoint']) != pathlib.Path(
      stage5b_spec['checkpoint']):
    raise RuntimeError('Stage 5A and 5B used different checkpoints')

  audit_path = stage5a_root / 'audit_results.csv'
  confirmation_path = stage5b_root / 'confirmation_results.csv'
  audit_rows = load_csv(audit_path)
  if not audit_rows or not all(row.get('complete') == 'True'
                               for row in audit_rows):
    raise RuntimeError('Stage 5A audit results are incomplete')
  goals, episodes, audit_pairing = extract_episodes(stage5a_root, audit_rows)
  separate = separability(episodes)
  transfer = threshold_transfer(
      episodes, args.window, float(selected['js']['js_threshold']))
  calibrated = calibration(episodes, args.window)

  spec = {
      'format': 1, 'canonical_stage': 'stage5b1',
      'slug': 'offline_performance_gate_audit',
      'stage5a_root': str(stage5a_root),
      'stage5a_spec_digest': stage5a_spec['digest'],
      'stage5a_gate_digest': selected['digest'],
      'stage5a_audit_sha256': file_sha256(audit_path),
      'stage5b_root': str(stage5b_root),
      'stage5b_spec_digest': stage5b_spec['digest'],
      'stage5b_selection_digest': selection['digest'],
      'stage5b_confirmation_sha256': file_sha256(confirmation_path),
      'goals': goals, 'policy_modes': list(MODES),
      'windows': list(WINDOWS), 'core_window': args.window,
      'targets': ['underperform', 'ir_needed'],
      'retrospective_labels_only': True,
      'environment_interaction': False,
  }
  spec['digest'] = spec_digest(spec)
  output_root.mkdir(parents=True, exist_ok=True)
  spec_path = output_root / 'audit_spec.json'
  if spec_path.is_file() and json.loads(
      spec_path.read_text()).get('digest') != spec['digest']:
    raise RuntimeError('Existing Stage 5B.1 root has a different specification')
  write_json(spec_path, spec)
  write_csv(output_root / 'episode_features.csv', episodes)
  write_csv(output_root / 'feature_separability.csv', separate)
  write_csv(output_root / 'threshold_transfer.csv', transfer)
  write_csv(output_root / 'js_calibration.csv', calibrated)
  write_json(output_root / 'audit_pairing.json', {
      'format': 1, 'canonical_stage': 'stage5b1',
      'equal': all(row['equal'] for row in audit_pairing),
      'cells': audit_pairing})
  summary = {
      'format': 1, 'canonical_stage': 'stage5b1',
      'spec_digest': spec['digest'], 'episodes': len(episodes),
      'necessity_class_counts': {
          name: sum(row['necessity_class'] == name for row in episodes)
          for name in ('ir_needed', 'ir_unnecessary', 'ir_harmful',
                       'not_repaired')},
      'feature_rows': separate, 'threshold_transfer_rows': transfer,
  }
  write_json(output_root / 'audit_summary.json', summary)
  make_figures(output_root, episodes, transfer, calibrated, args.window)
  write_findings(output_root, episodes, separate, transfer, args.window)
  (output_root / 'stage5b1_complete').write_bytes(b'')
  print(f'Stage 5B.1 audit complete: {output_root}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
