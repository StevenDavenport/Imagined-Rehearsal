import json
import pathlib
import pickle

import ruamel.yaml as yaml

from scripts import rlscape_m2_eval as evaluation
from scripts import rlscape_m2_train as training


def write_jsonl(path, rows):
  path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_multitask_config_uses_random_goals_and_gpu_scale():
  path = pathlib.Path(__file__).parents[2] / 'dreamerv3' / 'configs.yaml'
  configs = yaml.YAML(typ='safe').load(path.read_text())
  config = configs['rlscape_m2_multitask']
  assert config['env.rlscape.goal_schedule'] == 'random'
  assert config['env.rlscape.episode_length'] == 200
  assert config['run']['steps'] == 2000000
  assert config['run']['train_ratio'] == 32
  assert config['replay.size'] == 350000
  assert config['batch_length'] == 32
  assert config['.*\\.rssm']['deter'] == 4096
  assert config['.*\\.rssm']['classes'] == 32


def test_single_task_learnability_config_uses_click_grid_gpu_budget():
  path = pathlib.Path(__file__).parents[2] / 'dreamerv3' / 'configs.yaml'
  configs = yaml.YAML(typ='safe').load(path.read_text())

  assert configs['defaults']['env']['rlscape']['package_version'] == '0.1.6'
  assert configs['rlscape_click_grid']['env.rlscape.action_interface'] == (
      'click_grid')
  config = configs['rlscape_m1_learnability']
  assert config['run']['steps'] == 200000
  assert config['run']['train_ratio'] == 32
  assert config['replay.size'] == 200000
  assert config['batch_size'] == 8
  assert config['.*\\.rssm']['deter'] == 4096


def test_training_command_selects_multitask_profile(tmp_path):
  command = training.build_command(
      python=pathlib.Path('/env/bin/python'),
      repo_root=pathlib.Path('/repo'),
      logdir=tmp_path,
      seed=7,
      steps=2000000,
      batch_size=8,
      replay_size=350000,
      mvn_path='/tools/mvn',
      java_home='/tools/java',
  )
  configs = command[command.index('--configs') + 1:command.index('--seed')]
  assert configs == ['rlscape', 'rlscape_m2_multitask']
  assert command[command.index('--run.steps') + 1] == '2000000'
  assert command[command.index('--batch_size') + 1] == '8'
  assert command[command.index('--replay.size') + 1] == '350000'
  assert command[command.index('--env.rlscape.mvn_path') + 1] == '/tools/mvn'


def test_training_supervisor_reads_exact_checkpoint_step(tmp_path):
  checkpoint = tmp_path / 'ckpt' / 'checkpoint-1'
  checkpoint.mkdir(parents=True)
  (checkpoint / 'done').touch()
  (checkpoint / 'step.pkl').write_bytes(pickle.dumps(1234))
  (checkpoint.parent / 'latest').write_text(checkpoint.name)
  assert training.checkpoint_step(tmp_path) == 1234


def test_eval_command_is_fixed_goal_frozen_and_episode_bounded(tmp_path):
  command = evaluation.build_command(
      python=pathlib.Path('/env/bin/python'),
      repo_root=pathlib.Path('/repo'),
      logdir=tmp_path,
      checkpoint=pathlib.Path('/run/ckpt/exact'),
      goal='catch_fish',
      seed=1000,
      episodes=20,
      episode_length=200,
      mvn_path='/tools/mvn',
      java_home='/tools/java',
  )
  configs = command[command.index('--configs') + 1:command.index('--seed')]
  assert configs == [
      'rlscape', 'rlscape_catch_fish', 'rlscape_m2_eval']
  assert command[command.index('--run.eval_episodes') + 1] == '20'
  assert command[command.index('--run.from_checkpoint') + 1] == (
      '/run/ckpt/exact')
  assert command[command.index('--env.rlscape.episode_length') + 1] == '200'


def test_eval_command_can_restore_click_grid_checkpoint(tmp_path):
  command = evaluation.build_command(
      python=pathlib.Path('/env/bin/python'),
      repo_root=pathlib.Path('/repo'),
      logdir=tmp_path,
      checkpoint=pathlib.Path('/run/ckpt/exact'),
      goal='chop_logs',
      seed=1000,
      episodes=20,
      episode_length=200,
      action_interface='click_grid',
      grid_columns=32,
      grid_rows=21,
  )

  configs = command[command.index('--configs') + 1:command.index('--seed')]
  assert configs == [
      'rlscape', 'rlscape_click_grid', 'rlscape_chop_logs',
      'rlscape_m2_eval']
  assert command[command.index('--env.rlscape.grid_columns') + 1] == '32'
  assert command[command.index('--env.rlscape.grid_rows') + 1] == '21'


def test_eval_summary_requires_exact_episode_count(tmp_path):
  write_jsonl(tmp_path / 'episodes.jsonl', [
      {
          'step': 11,
          'episode/score': 1,
          'episode/length': 11,
          'episode/log/task_success/max': 1,
          'episode/log/player_death/max': 0,
      },
      {
          'step': 212,
          'episode/score': 0,
          'episode/length': 201,
          'episode/log/task_success/max': 0,
          'episode/log/player_death/max': 0,
      },
  ])
  summary = evaluation.summarize_run(
      tmp_path,
      goal='catch_fish',
      seed=1000,
      requested_episodes=2,
      returncode=0,
  )
  assert summary['episode_count_exact']
  assert summary['successes'] == 1
  assert summary['success_rate'] == 0.5
  assert summary['mean_action_length'] == 105
  assert summary['median_action_length'] == 105
