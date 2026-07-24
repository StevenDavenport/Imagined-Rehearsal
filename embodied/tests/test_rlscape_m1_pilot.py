import json
import pathlib

from scripts import rlscape_m1_pilot as pilot


def write_jsonl(path, rows):
  path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_build_command_layers_goal_and_pilot_configs(tmp_path):
  command = pilot.build_command(
      python=pathlib.Path('/env/bin/python'),
      repo_root=pathlib.Path('/repo'),
      logdir=tmp_path,
      goal='light_fire',
      seed=3,
      steps=1200,
      episode_length=240,
      train_ratio=4,
      log_every=100,
      save_every=250,
      mvn_path='/tools/mvn',
      java_home='/tools/java',
  )
  configs = command[command.index('--configs') + 1:command.index('--seed')]
  assert configs == [
      'rlscape', 'rlscape_light_fire', 'rlscape_m1_pilot']
  assert command[command.index('--run.steps') + 1] == '1200'
  assert command[command.index('--env.rlscape.episode_length') + 1] == '240'
  assert command[command.index('--env.rlscape.mvn_path') + 1] == '/tools/mvn'
  assert command[command.index('--env.rlscape.java_home') + 1] == '/tools/java'


def test_summarize_run_extracts_learning_and_action_signals(tmp_path):
  write_jsonl(tmp_path / 'episodes.jsonl', [
      {
          'step': 201,
          'episode/score': 0,
          'episode/length': 201,
          'episode/log/task_success/max': 0,
          'episode/log/task_progress/max': 0,
          'episode/log/player_death/max': 0,
          'episode/log/action_mode_0/sum': 50,
          'episode/log/action_mode_1/sum': 51,
          'episode/log/action_mode_2/sum': 49,
          'episode/log/action_mode_3/sum': 50,
          'episode/log/action_position_abs/avg': 0.4,
          'episode/log/action_position_abs/sum': 80,
      },
      {
          'step': 322,
          'episode/score': 1,
          'episode/length': 121,
          'episode/log/task_success/max': 1,
          'episode/log/task_progress/max': 1,
          'episode/log/player_death/max': 0,
          'episode/log/action_mode_2/sum': 120,
          'episode/log/action_position_abs/avg': 0.6,
          'episode/log/action_position_abs/sum': 72,
      },
  ])
  write_jsonl(tmp_path / 'metrics.jsonl', [{
      'step': 300,
      'train/opt/updates': 12,
      'train/loss/con': 0.8,
      'fps/policy': 3.5,
      'fps/train': 9.0,
  }])
  summary = pilot.summarize_run(
      tmp_path, goal='light_fire', seed=3, returncode=0)
  assert summary['episodes'] == 2
  assert summary['successes'] == 1
  assert summary['success_rate'] == 0.5
  assert summary['first_success_step'] == 322
  assert summary['max_progress'] == 1
  assert summary['action_mode_2_count'] == 169
  assert summary['action_steps'] == 320
  assert summary['expected_action_steps'] == 320
  assert summary['action_log_complete']
  assert summary['action_position_abs_mean'] == 152 / 322
  assert summary['latest_mean_update_index'] == 12
  assert 'loss_comp' not in summary
