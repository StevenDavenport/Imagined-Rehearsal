import json
import hashlib
from types import SimpleNamespace

import pytest

from scripts import rlscape_stage5_common as common
from scripts import rlscape_stage5b_need_discrimination as stage5b
from scripts import rlscape_stage5c_persistent_recovery as stage5c
from scripts.rlscape_m1_pilot import spec_digest


def selected_gates():
  return {
      'format': 1, 'canonical_stage': 'stage5a',
      'entropy': {'entropy_threshold': .8},
      'js': {'js_threshold': .2},
      'two_stage': {
          'js_threshold': .25, 'min_success_rate': .1,
          'direction_threshold': .3},
  }


def write_stage5a(root):
  root.mkdir()
  selected = selected_gates()
  selected['digest'] = spec_digest(selected)
  (root / 'selected_gates.json').write_text(json.dumps(selected))
  (root / 'stage5a_complete').write_bytes(b'')


def make_checkpoint(path):
  path.mkdir(parents=True)
  (path / 'agent.pkl').write_bytes(b'agent')
  (path / 'done').write_bytes(b'')


def test_stage5_matrix_has_exact_requested_seven_conditions():
  matrix = common.conditions(selected_gates())
  assert [item['name'] for item in matrix] == [
      'no_ir', 'always_standard', 'always_reward_only',
      'js_standard', 'js_reward_only',
      'two_stage_standard', 'two_stage_reward_only']
  assert all(item['horizon'] == 15 for item in matrix)
  assert all(item['actent'] == 0 for item in matrix)


def test_stage5_persistent_command_is_explicit_and_opt_in(tmp_path):
  args = SimpleNamespace(
      python=tmp_path / 'python', repo_root=tmp_path,
      episode_length=200, grid_columns=28, grid_rows=18,
      adapt_steps=1, adapt_lr=4e-5, adapt_every_k=1, start_batch=128,
      paired_rng_stride=1000, mvn_path='', java_home='')
  condition = common.conditions(selected_gates())[-1]
  command = common.build_eval_command(
      args, logdir=tmp_path / 'log', checkpoint=tmp_path / 'source',
      condition=condition, goal='kill_goblin', policy_mode='sampled',
      episodes=10, environment_seed=3, agent_seed=4,
      persistence='run', output_checkpoint=tmp_path / 'output',
      resume_adapt_state=True, warmup_episodes=0)
  value = lambda key: command[command.index(key) + 1]
  assert value('--eval_adapt.persistence') == 'run'
  assert value('--eval_adapt.output_checkpoint') == str(tmp_path / 'output')
  assert value('--eval_adapt.resume_adapt_state') == 'True'
  assert value('--eval_adapt.gate.kind') == 'two_stage'
  assert value('--run.eval_warmup_episodes') == '0'
  configs = command[command.index('--configs') + 1:command.index('--seed')]
  assert 'rlscape_stage4a_bounded_value' in configs


def test_stage5b_dry_run_is_immutable_and_prespecified(tmp_path, monkeypatch):
  checkpoint = tmp_path / 'checkpoint'
  make_checkpoint(checkpoint)
  gate_root = tmp_path / 'stage5a'
  write_stage5a(gate_root)
  python = tmp_path / 'python'
  python.write_bytes(b'')
  output = tmp_path / 'stage5b'
  monkeypatch.setattr(stage5b, 'git_revision', lambda path: 'revision')
  argv = [
      '--checkpoint', str(checkpoint), '--stage5a-root', str(gate_root),
      '--output-root', str(output), '--python', str(python), '--dry-run']
  assert stage5b.main(argv) == 0
  spec = json.loads((output / 'experiment_spec.json').read_text())
  assert spec['conditions'] == [item['name'] for item in
                                common.conditions(selected_gates())]
  assert spec['confirmation']['units'] == 84
  assert spec['confirmation']['episodes'] == 3150
  assert stage5b.main(argv) == 0
  with pytest.raises(RuntimeError, match='different specification'):
    stage5b.main([*argv[:-1], '--replicates', '2', '--dry-run'])


def test_stage5c_dry_run_chains_natural_source_and_fixed_budget(
    tmp_path, monkeypatch):
  natural = tmp_path / 'natural'
  make_checkpoint(natural)
  gate_root = tmp_path / 'stage5a'
  write_stage5a(gate_root)
  stage5b_root = tmp_path / 'stage5b'
  stage5b_root.mkdir()
  stage5b_spec = {
      'format': 1, 'canonical_stage': 'stage5b',
      'checkpoint': str(natural),
      'checkpoint_sha256': hashlib.sha256(b'agent').hexdigest(),
      'digest': 'stage5b-spec'}
  (stage5b_root / 'experiment_spec.json').write_text(json.dumps(stage5b_spec))
  selection = {
      'format': 1, 'canonical_stage': 'stage5b',
      'strong': {'goal': 'chop_logs'}, 'weak': {'goal': 'kill_goblin'}}
  selection['digest'] = spec_digest(selection)
  (stage5b_root / 'task_selection.json').write_text(json.dumps(selection))
  (stage5b_root / 'stage5b_complete').write_bytes(b'')
  python = tmp_path / 'python'
  python.write_bytes(b'')
  output = tmp_path / 'stage5c'
  monkeypatch.setattr(stage5c, 'git_revision', lambda path: 'revision')
  argv = [
      '--source-kind', 'natural', '--source-checkpoint', str(natural),
      '--stage5a-root', str(gate_root), '--stage5b-root', str(stage5b_root),
      '--output-root', str(output), '--python', str(python), '--dry-run']
  assert stage5c.main(argv) == 0
  spec = json.loads((output / 'experiment_spec.json').read_text())
  assert spec['target_goal'] == 'kill_goblin'
  assert spec['retained_goal'] == 'chop_logs'
  assert spec['recovery_episode_budget'] == 100
  assert spec['adaptation']['optimizer_state_persistent'] is True
  assert spec['analysis']['early_stopping'] is False
