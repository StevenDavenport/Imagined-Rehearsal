import json
from types import SimpleNamespace

from scripts import rlscape_stage5b2a_bury_autopsy as stage5b2a
from scripts import rlscape_stage5_status as stage5_status


def make_checkpoint(path):
  path.mkdir(parents=True)
  (path / 'agent.pkl').write_bytes(b'agent')
  (path / 'done').write_bytes(b'')


def make_stage5b2(path):
  path.mkdir()
  (path / 'stage5b2_complete').write_bytes(b'')
  (path / 'experiment_spec.json').write_text(json.dumps({
      'format': 1, 'canonical_stage': 'stage5b2'}))


def test_autopsy_matrix_separates_dose_selection_and_mixed_signal():
  args = SimpleNamespace(
      reward_doses=(16, 46, 80, 127), matched_updates=46,
      episode_length=200, mixed_beta=.005, audit_every=25)
  matrix = stage5b2a.conditions(args)
  assert [item['name'] for item in matrix] == [
      'no_ir', 'no_ir_world_audit', 'always_reward_only',
      'reward_cap_16', 'reward_cap_46', 'reward_cap_80',
      'reward_cap_127', 'reward_random_46',
      'virtual_positive_reward_d25', 'virtual_negative_reward_d25',
      'always_mixed_beta_0p005', 'mixed_beta_0p005_cap_46',
      'mixed_beta_0p005_random_46',
      'virtual_positive_mixed_beta_0p005_d25']
  lookup = {item['name']: item for item in matrix}
  assert lookup['reward_cap_46']['max_actor_updates_per_episode'] == 46
  assert lookup['reward_random_46']['gate_kind'] == 'random_schedule'
  assert lookup['virtual_negative_reward_d25'][
      'virtual_accept_mode'] == 'negative'
  assert lookup['always_mixed_beta_0p005']['mix_beta'] == .005


def test_autopsy_commands_encode_budget_random_and_inverted_controls(tmp_path):
  args = SimpleNamespace(
      reward_doses=(16, 46, 80, 127), matched_updates=46,
      episode_length=200, mixed_beta=.005, audit_every=25,
      episodes=50, audit_episodes=20, checkpoint=tmp_path / 'checkpoint',
      python=tmp_path / 'python', repo_root=tmp_path, grid_columns=28,
      grid_rows=18, adapt_steps=1, adapt_lr=4e-5, adapt_every_k=1,
      start_batch=128, mvn_path='', java_home='', paired_rng_stride=1000,
      eval_seed_base=141000, agent_seed_base=151000)
  lookup = {item['name']: item for item in stage5b2a.conditions(args)}
  random_command = stage5b2a.build_command(
      args, logdir=tmp_path / 'random',
      condition=lookup['reward_random_46'], replicate=0)
  value = lambda command, key: command[command.index(key) + 1]
  assert value(
      random_command, '--eval_adapt.max_actor_updates_per_episode') == '46'
  assert value(random_command, '--eval_adapt.gate.kind') == 'random_schedule'
  assert value(random_command, '--eval_adapt.gate.random_updates') == '46'

  mixed_command = stage5b2a.build_command(
      args, logdir=tmp_path / 'mixed',
      condition=lookup['virtual_positive_mixed_beta_0p005_d25'],
      replicate=0)
  assert value(mixed_command, '--eval_adapt.objective') == 'mixed_bounded'
  assert value(mixed_command, '--eval_adapt.mix_beta') == '0.005'

  inverted_command = stage5b2a.build_command(
      args, logdir=tmp_path / 'negative',
      condition=lookup['virtual_negative_reward_d25'], replicate=0)
  assert value(
      inverted_command, '--eval_adapt.gate.virtual_accept_mode') == 'negative'


def test_autopsy_dry_run_is_sampled_bury_only_and_immutable(
    tmp_path, monkeypatch):
  checkpoint = tmp_path / 'checkpoint'
  upstream = tmp_path / 'stage5b2'
  output = tmp_path / 'stage5b2a'
  python = tmp_path / 'python'
  make_checkpoint(checkpoint)
  make_stage5b2(upstream)
  python.write_bytes(b'')
  monkeypatch.setattr(stage5b2a, 'git_revision', lambda path: 'revision')
  argv = [
      '--checkpoint', str(checkpoint), '--stage5b2-root', str(upstream),
      '--output-root', str(output), '--python', str(python), '--dry-run']
  assert stage5b2a.main(argv) == 0
  spec = json.loads((output / 'experiment_spec.json').read_text())
  assert spec['canonical_stage'] == 'stage5b2a'
  assert spec['goal'] == 'bury_bones'
  assert spec['policy_mode'] == 'sampled'
  assert spec['deterministic_primary'] is False
  assert spec['environment_seed_base'] == 121000
  assert spec['agent_seed_base'] == 131000
  assert spec['reproduces_stage5b2_seed_schedule'] is True
  assert spec['adaptation']['mixed_beta_exact'] == .005
  assert stage5b2a.main(argv) == 0


def test_real_action_data_records_click_collapse_and_reward_event(tmp_path):
  audit = [
      {'kind': 'reset', 'episode_id': 1, 'reset_seed': 1},
      {'kind': 'reset', 'episode_id': 2, 'reset_seed': 2},
      {'kind': 'step', 'episode_id': 2, 'task_progress': 0,
       'requested_action': {'row': 6, 'column': 23}, 'events': []},
      {'kind': 'step', 'episode_id': 2, 'task_progress': 1,
       'requested_action': {'row': 6, 'column': 23},
       'events': [{'type': 'item_buried'}]},
  ]
  with (tmp_path / 'audit_env0.jsonl').open('w') as handle:
    for row in audit:
      handle.write(json.dumps(row) + '\n')
  (tmp_path / 'episodes.jsonl').write_text(json.dumps({
      'episode/score': 1, 'episode/length': 3}) + '\n')
  traces = [
      {'episode_index': 0, 'episode_step': 0, 'adapt_trigger': 1,
       'adapt_steps': 1},
      {'episode_index': 0, 'episode_step': 1, 'adapt_trigger': 0,
       'adapt_steps': 0},
  ]
  with (tmp_path / 'stage5_trace.jsonl').open('w') as handle:
    for row in traces:
      handle.write(json.dumps(row) + '\n')
  episodes, histogram, sequences, resets = stage5b2a.real_action_data(
      tmp_path, rows=18, columns=28)
  assert episodes[0]['success'] == 1
  assert episodes[0]['item_buried_events'] == 1
  assert episodes[0]['real_top_action_fraction'] == 1
  assert episodes[0]['first_commit_step'] == 0
  assert sequences == [[6 * 28 + 23, 6 * 28 + 23]]
  assert resets[0]['reset_seed'] == 2
  assert any(row['phase'] == 'early_0_24' for row in histogram)


def test_autopsy_figures_cover_dose_real_clicks_and_imagined_reward(tmp_path):
  args = SimpleNamespace(
      reward_doses=(16, 46, 80, 127), matched_updates=46,
      episode_length=200, mixed_beta=.005, audit_every=25,
      output_root=tmp_path, replicates=1, grid_rows=18, grid_columns=28)
  matrix = stage5b2a.conditions(args)
  summary = [{
      'condition': item['name'], 'success_rate': .5,
  } for item in matrix]
  episodes = [{
      'condition': item['name'], 'family': item['family'], 'replicate': 0,
      'success': 1, 'mean_gate_score_improvement': .01,
  } for item in matrix]
  key_names = {
      'no_ir', 'always_reward_only', 'reward_cap_46', 'reward_random_46',
      'virtual_positive_reward_d25', 'virtual_negative_reward_d25'}
  histogram = [{
      'condition': name, 'phase': 'late_75_plus', 'outcome': 'all',
      'row': 6, 'column': 23, 'count': 10,
  } for name in key_names]
  probes = [{
      'condition': 'virtual_positive_reward_d25', 'adapt_trigger': 1,
      'gate_after_reward_preceding_action_action_mode': 6 * 28 + 23,
      'gate_after_reward_preceding_action_action_concentration': .5,
  }]
  stage5b2a.make_figures(
      args, summary, episodes, histogram, probes, matrix)
  for name in (
      'performance', 'dose_response', 'real_click_heatmaps',
      'imagined_vs_real', 'imagined_reward_action_heatmaps'):
    assert (tmp_path / 'figures' / f'{name}.png').is_file()


def test_stage5_status_reports_partial_autopsy_results(tmp_path, capsys):
  (tmp_path / 'experiment_spec.json').write_text(json.dumps({
      'canonical_stage': 'stage5b2a', 'goal': 'bury_bones',
      'policy_mode': 'sampled', 'replicates': 3,
      'conditions': ['no_ir', 'reward_cap_46']}))
  (tmp_path / 'queue.json').write_text(json.dumps({
      'canonical_stage': 'stage5b2a', 'status': 'running',
      'units': {
          'replicate_00/no_ir': {
              'status': 'complete', 'attempts': [{
                  'summary': {'condition': 'no_ir', 'successes': 4,
                              'episodes': 10}}]},
      }}))
  assert stage5_status.main([str(tmp_path)]) == 0
  output = capsys.readouterr().out
  assert 'Planned units: 6' in output
  assert 'no_ir: 4/10 = 0.400 (1/3 replicates)' in output
