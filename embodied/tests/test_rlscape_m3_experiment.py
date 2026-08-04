from types import SimpleNamespace
import json
import pathlib

import ruamel.yaml as yaml

from scripts import rlscape_m3_eval as evaluation
from scripts import rlscape_m3_sequential as sequential
from dreamerv3 import main as dreamer_main


def test_triangular_matrix_has_30_cells_and_60_paired_units():
  units = evaluation.build_units()
  cells = {(unit['step'], unit['goal']) for unit in units}

  assert len(cells) == 30
  assert len(units) == 60
  assert {
      unit['episodes'] for unit in units
      if unit['policy_mode'] == 'sampled'} == {100}
  assert {
      unit['episodes'] for unit in units
      if unit['policy_mode'] == 'deterministic'} == {50}
  for step in range(250_000, 2_500_001, 250_000):
    phase = (step - 1) // 500_000
    goals = {
        unit['goal'] for unit in units
        if unit['step'] == step}
    assert goals == set(sequential.GOALS[:phase + 1])


def test_three_goal_matrix_preserves_eval_regime_at_six_milestones():
  goals = tuple(sequential.ALL_GOALS[:3])
  units = evaluation.build_units(goals)
  cells = {(unit['step'], unit['goal']) for unit in units}

  assert len(cells) == 12
  assert len(units) == 24
  assert max(unit['step'] for unit in units) == 1_500_000
  assert {unit['episodes'] for unit in units
          if unit['policy_mode'] == 'sampled'} == {100}
  assert {unit['episodes'] for unit in units
          if unit['policy_mode'] == 'deterministic'} == {50}


def test_pair_identity_uses_seed_while_retaining_frame_diagnostic(tmp_path):
  rows = [
      {'kind': 'reset', 'goal': 'kill_goblin', 'reset_seed': 7,
       'frame_sha256': 'abc'},
      {'kind': 'step', 'goal': 'kill_goblin'},
  ]
  (tmp_path / 'audit_env0.jsonl').write_text(
      ''.join(json.dumps(row) + '\n' for row in rows))

  assert evaluation.reset_identities(tmp_path) == [{
      'goal': 'kill_goblin', 'reset_seed': 7}]
  assert evaluation.reset_frame_hashes(tmp_path) == ['abc']


def test_eval_command_encodes_actor_ir_and_policy_mode(tmp_path):
  command = evaluation.build_command(
      python=pathlib.Path('/env/bin/python'),
      repo_root=sequential.ROOT,
      logdir=tmp_path,
      checkpoint=pathlib.Path('/archive/checkpoint'),
      goal='bury_bones',
      seed=1000,
      episodes=100,
      policy_mode='sampled',
      actor_ir=True,
  )
  configs = command[command.index('--configs') + 1:command.index('--seed')]

  assert configs == [
      'rlscape', 'rlscape_click_grid', 'rlscape_bury_bones',
      'rlscape_m3_eval', 'rlscape_m3_actor_ir']
  assert command[command.index('--run.eval_policy_mode') + 1] == 'sampled'
  assert command[command.index('--run.eval_episodes') + 1] == '100'
  assert command[command.index('--run.from_checkpoint') + 1] == (
      '/archive/checkpoint')


def test_train_command_uses_one_logdir_and_exact_milestones(tmp_path):
  args = SimpleNamespace(
      python=pathlib.Path('/env/bin/python'),
      repo_root=pathlib.Path('/repo'),
      seed=0,
      goals=tuple(sequential.ALL_GOALS),
      replay_kind='fifo',
      fifo_replay_size=200_000,
      reservoir_episodes_per_goal=1_000,
      mvn_path='',
      java_home='',
  )
  command = sequential.build_train_command(
      args=args,
      training_logdir=tmp_path / 'training',
      milestone_root=tmp_path / 'milestones',
      goal='chop_logs',
      phase=2,
      target_step=1_500_000,
      digest='abc123',
  )
  configs = command[command.index('--configs') + 1:command.index('--seed')]

  assert configs == [
      'rlscape', 'rlscape_click_grid', 'rlscape_chop_logs',
      'rlscape_m3_sequential']
  assert command[command.index('--run.steps') + 1] == '1500000'
  assert command[command.index('--run.milestone_every') + 1] == '250000'
  assert command[command.index('--run.milestone_goal') + 1] == 'chop_logs'
  assert command[command.index('--run.milestone_phase') + 1] == '2'
  assert command[command.index('--replay.kind') + 1] == 'fifo'
  assert command[command.index('--replay.size') + 1] == '200000'


def test_three_goal_reservoir_workflow_changes_only_replay_and_goal_count(
    tmp_path):
  args = SimpleNamespace(
      python=pathlib.Path('/env/bin/python'),
      repo_root=sequential.ROOT,
      seed=0,
      goals=tuple(sequential.ALL_GOALS[:3]),
      replay_kind='episode_reservoir',
      fifo_replay_size=200_000,
      reservoir_episodes_per_goal=1_000,
      mvn_path='',
      java_home='',
  )
  spec = sequential.experiment_spec(args)
  command = sequential.build_train_command(
      args=args,
      training_logdir=tmp_path / 'training',
      milestone_root=tmp_path / 'milestones',
      goal='chop_logs',
      phase=2,
      target_step=1_500_000,
      digest='abc123',
  )

  assert spec['goals'] == ['kill_goblin', 'bury_bones', 'chop_logs']
  assert spec['total_steps'] == 1_500_000
  assert spec['milestones'] == [
      250_000, 500_000, 750_000, 1_000_000, 1_250_000, 1_500_000]
  assert spec['replay'] == {
      'kind': 'episode_reservoir',
      'capacity': 3_000,
      'capacity_unit': 'episodes',
      'capacity_per_goal': 1_000,
      'persistent_across_phases': True,
      'retention': 'algorithm_r_within_goal',
      'sampling': 'goal_then_episode_then_sequence',
      'statistical_stream_reservoir': True,
      'short_episode_rule': 'concatenate_fragments_with_reset',
      'save_wait': True,
  }
  configs = command[command.index('--configs') + 1:command.index('--seed')]
  assert configs == [
      'rlscape', 'rlscape_click_grid', 'rlscape_chop_logs',
      'rlscape_m3_sequential', 'rlscape_m3_episode_reservoir']
  assert command[command.index('--replay.kind') + 1] == 'episode_reservoir'
  assert command[command.index('--replay.size') + 1] == '3000'
  assert command[command.index('--replay.groups') + 1] == '3'
  assert command[command.index('--replay.retention') + 1] == 'reservoir'


def test_replay_factory_selects_episode_reservoir(tmp_path):
  config = SimpleNamespace(
      batch_length=32,
      report_length=32,
      consec_train=1,
      consec_report=1,
      replay_context=0,
      batch_size=8,
      logdir=str(tmp_path),
      replicas=1,
      replica=0,
      seed=0,
      replay=SimpleNamespace(
          kind='episode_reservoir',
          size=3_000,
          online=False,
          chunksize=1_024,
          save_wait=True,
          groups=3,
          group_key='goal_id',
          retention='reservoir',
          fracs=SimpleNamespace(uniform=1.0),
      ),
  )

  replay = dreamer_main.make_replay(config, 'replay')

  assert replay.__class__.__name__ == 'EpisodeReplay'
  assert replay.capacity == 3_000
  assert replay.group_capacity == 1_000


def test_m3_configs_pin_scientific_training_and_ir_settings():
  path = pathlib.Path(__file__).parents[2] / 'dreamerv3' / 'configs.yaml'
  configs = yaml.YAML(typ='safe').load(path.read_text())
  train = configs['rlscape_m3_sequential']
  eval_config = configs['rlscape_m3_eval']
  actor_ir = configs['rlscape_m3_actor_ir']['eval_adapt']

  assert train['.*\\.rssm']['deter'] == 4096
  assert train['.*\\.rssm']['classes'] == 32
  assert train['replay.size'] == 200000
  assert train['replay.save_wait'] is True
  reservoir = configs['rlscape_m3_episode_reservoir']
  assert reservoir['replay.kind'] == 'episode_reservoir'
  assert reservoir['replay.retention'] == 'reservoir'
  assert train['run']['train_ratio'] == 32
  assert train['run']['milestone_every'] == 250000
  assert eval_config['run']['eval_policy_mode'] == 'sampled'
  assert actor_ir == {
      'enabled': True,
      'train_critic': False,
      'steps': 1,
      'imag_length': 6,
      'lr': 4e-5,
      'critic_lr': 4e-5,
      'every_k': 1,
      'actent': 3e-4,
      'start_batch': 128,
  }
