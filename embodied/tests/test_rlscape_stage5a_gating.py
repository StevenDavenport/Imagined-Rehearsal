import json
from types import SimpleNamespace

from scripts import rlscape_stage5a_gating as stage5a


def _selected():
  return {
      kind: {
          'entropy_threshold': .6, 'js_threshold': .2,
          'min_success_rate': 4 / 128, 'direction_threshold': .3}
      for kind in ('entropy', 'js', 'two_stage')}


def test_confirmation_matrix_is_fixed_and_has_three_objectives_per_gate():
  conditions = stage5a.confirmation_conditions(_selected())
  assert len(conditions) == 13
  assert [condition['name'] for condition in conditions[:4]] == [
      'no_ir', 'always_standard', 'always_reward_only',
      'always_mixed_b005']
  for kind in ('entropy', 'js', 'two_stage'):
    selected = [condition for condition in conditions
                if condition['gate_kind'] == kind]
    assert len(selected) == 3
    assert {condition['objective_label'] for condition in selected} == {
        'standard', 'reward_only', 'mixed_b005'}
  assert all(condition['checkpoint'] == 'stage4a_counterfactual_bounded'
             for condition in conditions)


def test_build_command_encodes_gate_and_compute_trace_contract(tmp_path):
  args = SimpleNamespace(
      python=tmp_path / 'python', repo_root=tmp_path,
      episode_length=200, grid_columns=28, grid_rows=18,
      adapt_steps=1, adapt_lr=4e-5, adapt_every_k=1, start_batch=128,
      mvn_path='', java_home='', paired_rng_stride=1000)
  condition = stage5a.gated_condition(
      'mixed_b005', 'two_stage', _selected()['two_stage'])
  command = stage5a.build_command(
      args, logdir=tmp_path / 'log', checkpoint=tmp_path / 'ckpt',
      condition=condition, goal='bury_bones', policy_mode='sampled',
      episodes=50, replicate=2, seed_base=71_000)

  assert command[command.index('--seed') + 1] == '71002'
  assert command[command.index('--eval_adapt.paired_rng_seed') + 1] == '81002'
  assert command[command.index('--eval_adapt.gate.kind') + 1] == 'two_stage'
  assert command[command.index('--eval_adapt.mix_beta') + 1] == '0.05'
  assert command[command.index('--eval_adapt.trace.enabled') + 1] == 'True'
  assert command[command.index('--run.eval_warmup_episodes') + 1] == '1'


def test_dry_run_writes_immutable_8775_episode_confirmation_spec(
    tmp_path, monkeypatch):
  checkpoint = tmp_path / 'stage4a' / 'checkpoints' / 'counterfactual_bounded'
  checkpoint.mkdir(parents=True)
  (checkpoint / 'done').write_bytes(b'')
  (checkpoint / 'agent.pkl').write_bytes(b'agent')
  (checkpoint / 'repair_manifest.json').write_text(json.dumps({
      'condition': 'counterfactual_bounded',
      'counterfactual_reward_continuation': True,
      'bounded_value': True, 'value_updates': 2000,
      'frozen_component_violations': {}}))
  python = tmp_path / 'python'
  python.write_bytes(b'')
  monkeypatch.setattr(stage5a, 'git_revision', lambda path: 'revision')
  output = tmp_path / 'stage5a'
  common = [
      '--stage4a-root', str(tmp_path / 'stage4a'), '--output-root', str(output),
      '--python', str(python), '--dry-run']

  assert stage5a.main(common) == 0
  spec = json.loads((output / 'experiment_spec.json').read_text())
  assert spec['canonical_stage'] == 'stage5a'
  assert spec['confirmation']['units'] == 234
  assert spec['confirmation']['episodes'] == 8775
  assert spec['selection']['compute_is_constraint'] is False
  assert stage5a.main(common) == 0
