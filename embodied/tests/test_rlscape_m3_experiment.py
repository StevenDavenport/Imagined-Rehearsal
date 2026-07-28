from types import SimpleNamespace
import pathlib

import ruamel.yaml as yaml

from scripts import rlscape_m3_eval as evaluation
from scripts import rlscape_m3_sequential as sequential


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


def test_eval_command_encodes_actor_ir_and_policy_mode(tmp_path):
  command = evaluation.build_command(
      python=pathlib.Path('/env/bin/python'),
      repo_root=pathlib.Path('/repo'),
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
