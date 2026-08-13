import numpy as np

from scripts import rlscape_stage3b_natural_forgetting as stage3b


def test_stage3b_condition_matrix_is_prespecified():
  conditions = stage3b.build_conditions((6, 15, 20))
  assert [item['name'] for item in conditions] == [
      'frozen', 'standard_ir_h6', 'reward_only_ir_h6',
      'reward_only_ir_h15', 'reward_only_ir_h20']
  assert conditions[0]['enabled'] is False
  assert conditions[1]['objective'] == 'standard'
  assert conditions[1]['actent'] == 3e-4
  assert all(item['objective'] == 'reward_only' for item in conditions[2:])
  assert all(item['actent'] == 0 for item in conditions[2:])


def test_stage3b_historical_classification_is_fixed_before_new_eval():
  assert stage3b.historical_classification(
      1_250_000, 'kill_goblin', .5) == (.56, 'competent')
  assert stage3b.historical_classification(
      1_250_000, 'bury_bones', .5) == (.49, 'weak')
  assert stage3b.historical_classification(
      1_500_000, 'chop_logs', .5) == (.05, 'weak')


def test_stage3b_paired_effect_reports_direction_and_discordance():
  baseline = np.asarray([0, 0, 1, 1, 0], np.float64)
  treatment = np.asarray([1, 0, 1, 0, 1], np.float64)
  result = stage3b.paired_effect(baseline, treatment, seed=7, samples=500)
  assert np.isclose(result['paired_delta_vs_frozen'], .2)
  assert result['paired_discordant_improved'] == 2
  assert result['paired_discordant_harmed'] == 1
  assert result['paired_delta_ci_low'] <= .2 <= result['paired_delta_ci_high']


def test_stage3b_checkpoint_labels_sort_lexically():
  labels = [stage3b.checkpoint_label(x) for x in (1_500_000, 1_250_000)]
  assert sorted(labels) == [
      'step_0001250000', 'step_0001500000']
