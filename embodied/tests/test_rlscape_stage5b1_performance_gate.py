import csv
import json

from scripts import rlscape_stage5b1_performance_gate_audit as audit
from scripts.rlscape_m1_pilot import spec_digest


def write_jsonl(path, rows):
  path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def digested(payload):
  payload = dict(payload)
  payload['digest'] = spec_digest(payload)
  return payload


def test_auc_and_threshold_recover_perfect_prospective_ranking():
  labels = [0, 0, 1, 1]
  scores = [.1, .2, .8, .9]
  assert audit.roc_auc(labels, scores) == 1.0
  selected = audit.optimal_threshold(labels, scores)
  assert .2 < selected['threshold'] < .8
  assert selected['balanced_accuracy'] == 1.0
  assert selected['true_positive_rate'] == 1.0
  assert selected['false_positive_rate'] == 0.0


def make_logdir(path, goal, mode, condition):
  path.mkdir(parents=True)
  outcomes = (
      [(1, 12), (1, 25)] if condition == 'always_reward_only' else
      [(1, 15), (0, 201)])
  write_jsonl(path / 'episodes.jsonl', [
      {'episode/score': score, 'episode/length': length}
      for score, length in outcomes])
  write_jsonl(path / 'audit_env0.jsonl', [
      {'kind': 'reset', 'goal': goal, 'reset_seed': seed}
      for seed in (999, 100, 101)])
  if condition == 'no_ir':
    traces = []
    for episode, (js, predicted) in enumerate(((.1, .8), (.9, .1))):
      for step in range(2):
        traces.append({
            'episode_index': episode, 'episode_step': step,
            'gate_actor_entropy': js, 'gate_posterior_entropy': js + .05,
            'gate_js_disagreement': js,
            'gate_success_rate': predicted,
            'gate_success_action_concentration': .5,
            'gate_success_action_divergence': .2})
    write_jsonl(path / 'stage5a_trace.jsonl', traces)


def write_csv(path, rows):
  with path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)


def test_offline_audit_writes_paired_features_figures_and_findings(tmp_path):
  stage5a = tmp_path / 'stage5a'
  stage5b = tmp_path / 'stage5b'
  stage5a.mkdir(); stage5b.mkdir()
  checkpoint = tmp_path / 'checkpoint'
  checkpoint.mkdir()
  stage5a_spec = digested({
      'format': 1, 'canonical_stage': 'stage5a',
      'checkpoint': str(checkpoint)})
  stage5b_spec = digested({
      'format': 1, 'canonical_stage': 'stage5b',
      'checkpoint': str(checkpoint)})
  selected = digested({
      'format': 1, 'canonical_stage': 'stage5a',
      'js': {'js_threshold': .5}})
  selection = digested({
      'format': 1, 'canonical_stage': 'stage5b',
      'strong': {'goal': 'chop_logs'}, 'weak': {'goal': 'bury_bones'}})
  for root, name in ((stage5a, 'stage5a_complete'),
                     (stage5b, 'stage5b_complete')):
    (root / name).write_bytes(b'')
  (stage5a / 'experiment_spec.json').write_text(json.dumps(stage5a_spec))
  (stage5b / 'experiment_spec.json').write_text(json.dumps(stage5b_spec))
  (stage5a / 'selected_gates.json').write_text(json.dumps(selected))
  (stage5b / 'task_selection.json').write_text(json.dumps(selection))
  (stage5b / 'pairing.json').write_text(json.dumps({'equal': True}))
  (stage5b / 'confirmation_results.csv').write_text('complete\nTrue\n')

  rows = []
  for goal in ('kill_goblin', 'bury_bones', 'chop_logs'):
    for mode in audit.MODES:
      for condition in ('no_ir', 'always_reward_only'):
        logdir = stage5a / 'logs' / goal / mode / condition
        make_logdir(logdir, goal, mode, condition)
        rows.append({
            'condition': condition, 'goal': goal, 'policy_mode': mode,
            'complete': 'True', 'logdir': str(logdir)})
  write_csv(stage5a / 'audit_results.csv', rows)
  output = tmp_path / 'output'

  assert audit.main([
      '--stage5a-root', str(stage5a), '--stage5b-root', str(stage5b),
      '--output-root', str(output)]) == 0
  assert (output / 'stage5b1_complete').is_file()
  assert (output / 'episode_features.csv').is_file()
  assert (output / 'figures' / 'performance_plane_sampled.png').is_file()
  assert (output / 'STAGE5B1_FINDINGS.md').is_file()
  separate = list(csv.DictReader(
      (output / 'feature_separability.csv').open()))
  chosen = [row for row in separate if row['goal'] == 'bury_bones' and
            row['policy_mode'] == 'sampled' and
            row['target'] == 'underperform' and row['window'] == '10' and
            row['predictor'] == 'mean_js']
  assert float(chosen[0]['roc_auc']) == 1.0
