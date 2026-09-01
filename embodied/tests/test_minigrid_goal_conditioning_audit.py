import csv
import json

import numpy as np
import pytest

from scripts import minigrid_goal_conditioning_audit as audit


def _write_fixture(root):
  spec = {
      'name': 'synthetic_goal_leakage',
      'arms': {'control': {}},
      'seeds': [0],
      'phase_steps': 20,
      'phase_boundaries': [0],
      'tasks': ['first', 'second'],
      # Deliberately non-contiguous to ensure trained-goal indexing is robust.
      'task_goal_ids': {'first': 1, 'second': 3},
  }
  (root / 'experiment_spec.json').write_text(json.dumps(spec))
  goal_names = np.asarray(['unused_a', 'first', 'unused_b', 'second'])
  actual = np.asarray([1, 1, 3, 3], np.int32)
  reward = np.asarray([
      [0.10, 0.20, 0.15, 0.10],
      [0.20, 0.90, 0.10, 0.30],
      [0.05, 0.10, 0.10, 0.20],
      [0.10, 0.40, 0.20, 0.80],
  ], np.float32)
  value = np.asarray([
      [0.2, 0.4, 0.3, 0.2],
      [0.5, 2.0, 0.4, 1.0],
      [0.1, 0.2, 0.2, 0.5],
      [0.2, 1.1, 0.3, 2.1],
  ], np.float32)
  continuation = np.full((4, 4), 0.90, np.float32)
  continuation[1, 1] = 0.05
  continuation[3, 3] = 0.05
  actor = np.full((4, 4, 3), 0.05, np.float32)
  for goal in range(4):
    actor[:, goal, goal % 3] = 0.90

  head = (
      root / 'diagnostics' / 'head' / 'control' / 'seed_0000' /
      'step_000000000000')
  head.mkdir(parents=True)
  np.savez_compressed(
      head / 'predictions.npz', goal_names=goal_names, actual_goal=actual,
      goal_complete=np.asarray([False, True, False, True]),
      is_physical_terminal=np.zeros(4, bool),
      is_first=np.asarray([True, False, True, False]),
      reward_prediction=reward,
      continuation_prediction=continuation,
      value_prediction=value,
      slowvalue_prediction=value * 0.95,
      return_inclusive=np.asarray([0.2, 1.0, 0.4, 1.0], np.float32),
      actor_probability=actor,
      discount=np.asarray(0.99, np.float32))
  (head / 'summary.json').write_text(json.dumps({
      'goal_names': goal_names.tolist(), 'integrity_equal': True}))
  (head / 'audit_complete').touch()

  critic = (
      root / 'diagnostics' / 'critic_provenance' / 'control' /
      'seed_0000' / 'step_000000000000')
  critic.mkdir(parents=True)
  fieldnames = [
      'episode_id', 'anchor_ordinal', 'actual_goal', 'probed_goal',
      'proposal', 'horizon', 'recorded_control_available',
      'online_value_start_mean', 'slow_value_start_mean', 'return_mean',
      'reward_only_mean', 'bootstrap_mean', 'imagined_reward_sum_mean',
      'imagined_reward_event_posterior_sample_rate',
      'imagined_stop_posterior_sample_rate']
  rows = []
  for episode_id, actual_goal in ((101, 1), (102, 3)):
    for proposal in ('actor', 'recorded'):
      for horizon in (1, 15):
        for probed_goal in range(4):
          matching = probed_goal == actual_goal
          reward_only = horizon * (0.20 if matching else 0.05)
          value_start = 1.50 if matching else 1.20
          bootstrap = 0.8 * value_start
          rows.append({
              'episode_id': episode_id, 'anchor_ordinal': 0,
              'actual_goal': actual_goal, 'probed_goal': probed_goal,
              'proposal': proposal, 'horizon': horizon,
              'recorded_control_available': True,
              'online_value_start_mean': value_start,
              'slow_value_start_mean': value_start * 0.95,
              'return_mean': reward_only + bootstrap,
              'reward_only_mean': reward_only,
              'bootstrap_mean': bootstrap,
              'imagined_reward_sum_mean': reward_only,
              'imagined_reward_event_posterior_sample_rate': (
                  0.8 if matching else 0.2),
              'imagined_stop_posterior_sample_rate': (
                  0.7 if matching else 0.1),
          })
  with (critic / 'rollout_summary.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
  (critic / 'summary.json').write_text(json.dumps({
      'goal_names': goal_names.tolist(), 'integrity_equal': True}))
  (critic / 'audit_complete').touch()


def test_goal_conditioning_audit_measures_same_state_leakage(tmp_path):
  _write_fixture(tmp_path)

  result = audit.generate_audit(tmp_path)

  assert result['all_inputs_unchanged'] is True
  assert result['head_jobs'] == 1
  assert result['critic_jobs'] == 1
  output = tmp_path / 'analysis' / 'goal_conditioning_audit'
  with (output / 'head_goal_conditioning.csv').open(newline='') as handle:
    head_rows = list(csv.DictReader(handle))
  completion = next(
      row for row in head_rows
      if row['scope'] == 'trained' and
      row['state_kind'] == 'goal_completion')
  assert float(completion['reward_factual_mean']) == pytest.approx(0.85)
  assert float(completion['reward_nonmatching_mean']) == pytest.approx(0.35)
  assert float(completion['reward_goal_top1_accuracy']) == 1.0
  assert float(completion['stop_factual_minus_nonmatching']) > 0.8
  assert float(completion[
      'reward_value_within_state_correlation']) > 0.9

  with (output / 'imagination_goal_conditioning.csv').open(
      newline='') as handle:
    imagination_rows = list(csv.DictReader(handle))
  actor_h15 = next(
      row for row in imagination_rows
      if row['scope'] == 'trained' and row['proposal'] == 'actor' and
      row['comparison_scope'] == 'all' and row['horizon'] == '15')
  assert float(actor_h15['reward_only_mean_factual_mean']) == 3.0
  assert float(actor_h15['reward_only_mean_nonmatching_mean']) == 0.75
  assert float(actor_h15[
      'reward_only_value_within_anchor_correlation']) > 0.99
  assert (output / 'FINDINGS.md').is_file()
  assert (output / 'audit_complete').is_file()


def test_goal_conditioning_audit_never_writes_into_diagnostics(tmp_path):
  _write_fixture(tmp_path)

  with pytest.raises(ValueError, match='must not modify diagnostics'):
    audit.generate_audit(
        tmp_path, output=tmp_path / 'diagnostics' / 'derived')
