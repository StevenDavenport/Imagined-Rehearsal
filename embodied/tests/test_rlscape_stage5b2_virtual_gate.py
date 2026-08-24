import json
from types import SimpleNamespace

from scripts import rlscape_stage5_common as common
from scripts import rlscape_stage5b2_virtual_update_gate as stage5b2


def make_checkpoint(path):
  path.mkdir(parents=True)
  (path / 'agent.pkl').write_bytes(b'agent')
  (path / 'done').write_bytes(b'')


def make_audit(path):
  path.mkdir()
  (path / 'stage5b1_complete').write_bytes(b'')
  (path / 'audit_spec.json').write_text(json.dumps({
      'format': 1, 'canonical_stage': 'stage5b1'}))


def test_virtual_condition_and_command_are_explicit(tmp_path):
  condition = common.virtual_reward_condition(
      'virtual', dormant_every=25, active_every=1, deactivate_after=3)
  args = SimpleNamespace(
      python=tmp_path / 'python', repo_root=tmp_path,
      episode_length=200, grid_columns=28, grid_rows=18,
      adapt_steps=1, adapt_lr=4e-5, adapt_every_k=1, start_batch=128,
      paired_rng_stride=1000, mvn_path='', java_home='')
  command = common.build_eval_command(
      args, logdir=tmp_path / 'log', checkpoint=tmp_path / 'checkpoint',
      condition=condition, goal='kill_goblin', policy_mode='sampled',
      episodes=10, environment_seed=3, agent_seed=4)
  value = lambda key: command[command.index(key) + 1]
  assert value('--eval_adapt.objective') == 'reward_only'
  assert value('--eval_adapt.gate.kind') == 'virtual_update'
  assert value('--eval_adapt.gate.virtual_score') == 'reward_return'
  assert value('--eval_adapt.gate.virtual_dormant_every') == '25'
  assert value('--eval_adapt.gate.virtual_active_every') == '1'
  assert value('--eval_adapt.gate.virtual_deactivate_after') == '3'


def test_stage5b2_dry_run_is_immutable_and_prespecified(
    tmp_path, monkeypatch):
  checkpoint = tmp_path / 'checkpoint'
  audit = tmp_path / 'stage5b1'
  output = tmp_path / 'stage5b2'
  python = tmp_path / 'python'
  make_checkpoint(checkpoint)
  make_audit(audit)
  python.write_bytes(b'')
  monkeypatch.setattr(stage5b2, 'git_revision', lambda path: 'revision')
  argv = [
      '--checkpoint', str(checkpoint), '--stage5b1-root', str(audit),
      '--output-root', str(output), '--python', str(python), '--dry-run']
  assert stage5b2.main(argv) == 0
  spec = json.loads((output / 'experiment_spec.json').read_text())
  assert spec['canonical_stage'] == 'stage5b2'
  assert spec['conditions'] == [
      'no_ir', 'always_reward_only',
      'virtual_reward_dormant_25', 'virtual_reward_dormant_100']
  assert spec['adaptation']['paired_before_after'] is True
  assert spec['adaptation']['posterior_samples'] == 128
  assert stage5b2.main(argv) == 0


def test_trace_episode_summary_counts_proposed_and_committed_updates(tmp_path):
  rows = [
      {'episode_index': 0, 'episode_step': 0, 'adapt_trigger': 1,
       'gate_score_improvement': .2,
       'compute_tentative_actor_updates': 1,
       'compute_rejected_actor_updates': 0},
      {'episode_index': 0, 'episode_step': 1, 'adapt_trigger': 0},
      {'episode_index': 0, 'episode_step': 2, 'adapt_trigger': 0,
       'gate_score_improvement': -.1,
       'compute_tentative_actor_updates': 1,
       'compute_rejected_actor_updates': 1},
  ]
  with (tmp_path / 'stage5_trace.jsonl').open('w') as handle:
    for row in rows:
      handle.write(json.dumps(row) + '\n')
  summary = stage5b2.trace_episode_summary(tmp_path)[0]
  assert summary['probes'] == 2
  assert summary['committed_updates'] == 1
  assert summary['tentative_updates'] == 2
  assert summary['rejected_updates'] == 1
