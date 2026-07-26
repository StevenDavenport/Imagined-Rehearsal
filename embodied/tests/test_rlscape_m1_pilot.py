import json
import pathlib
from types import SimpleNamespace

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
      'rlscape', 'rlscape_click_grid', 'rlscape_light_fire',
      'rlscape_m1_learnability']
  assert command[command.index('--run.steps') + 1] == '1200'
  assert command[command.index('--env.rlscape.episode_length') + 1] == '240'
  assert command[command.index('--env.rlscape.grid_columns') + 1] == '28'
  assert command[command.index('--env.rlscape.grid_rows') + 1] == '18'
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


def test_summarize_grid_run_uses_valid_action_count_and_final_window(tmp_path):
  rows = []
  for index in range(120):
    rows.append({
        'step': (index + 1) * 11,
        'episode/score': float(index >= 100),
        'episode/length': 11,
        'episode/log/task_success/max': float(index >= 100),
        'episode/log/task_progress/max': float(index >= 100),
        'episode/log/player_death/max': 0,
        'episode/log/action_mode_2/sum': 10,
        'episode/log/action_grid_valid/sum': 10,
        'episode/log/action_grid_index/sum': 50,
    })
  write_jsonl(tmp_path / 'episodes.jsonl', rows)

  summary = pilot.summarize_run(
      tmp_path, goal='chop_logs', seed=0, returncode=0)

  assert summary['action_interface'] == 'click_grid'
  assert summary['action_steps'] == 1200
  assert summary['action_log_complete']
  assert summary['action_grid_index_mean'] == 5
  assert summary['successes'] == 20
  assert summary['final_100_success_rate'] == 0.2


def test_evaluation_log_roots_are_step_seed_and_count_specific(tmp_path):
  root = pilot.evaluation_log_root(
      tmp_path, step=200000, goal='catch_fish', seed=1000, episodes=20)

  assert root == (
      tmp_path / 'evaluations' / 'step_200000_seed1000_n20')


def test_failed_evaluation_is_retried_without_reusing_artifacts(tmp_path):
  base = tmp_path / 'evaluations' / 'step_200000_seed1000_n20'
  pilot.write_json(base / 'summary.json', [{
      'goal': 'catch_fish',
      'seed': 1000,
      'returncode': 1,
      'requested_episodes': 20,
      'episodes': 20,
      'episode_count_exact': True,
  }])

  root = pilot.evaluation_log_root(
      tmp_path, step=200000, goal='catch_fish', seed=1000, episodes=20)

  assert root == pathlib.Path(f'{base}_attempt2')


def test_training_spec_allows_only_target_step_extension():
  suite = {
      key: f'value-{index}'
      for index, key in enumerate(pilot.TRAINING_SPEC_KEYS)
  }
  suite['target_steps'] = 200000
  first = pilot.training_spec(suite, goal='bury_bones', seed=0)
  suite['target_steps'] = 400000
  extended = pilot.training_spec(suite, goal='bury_bones', seed=0)

  assert pilot.spec_digest(first) == pilot.spec_digest(extended)
  assert 'target_steps' not in first


def test_main_trains_evaluates_and_extends_same_run(
    tmp_path, monkeypatch):
  repo_root = pathlib.Path(__file__).parents[2]
  checkpoint = tmp_path / 'checkpoint'
  checkpoint.mkdir()
  steps = iter((0, 200000, 200000, 400000))
  commands = []

  monkeypatch.setattr(pilot, 'checkpoint_step', lambda _: next(steps))
  monkeypatch.setattr(
      pilot, 'resolve_checkpoint_path', lambda _: checkpoint)

  def run(command, **kwargs):
    del kwargs
    commands.append(command)
    if command[1].endswith('rlscape_m2_eval.py'):
      log_root = pathlib.Path(command[command.index('--log-root') + 1])
      goal = command[command.index('--goals') + 1]
      seed = int(command[command.index('--seed') + 1])
      episodes = int(command[command.index('--episodes') + 1])
      pilot.write_json(log_root / 'summary.json', [{
          'goal': goal,
          'seed': seed,
          'returncode': 0,
          'requested_episodes': episodes,
          'episodes': episodes,
          'episode_count_exact': True,
          'successes': episodes,
          'success_rate': 1.0,
      }])
    return SimpleNamespace(returncode=0)

  monkeypatch.setattr(pilot.subprocess, 'run', run)
  pilot.write_json(tmp_path / 'runs' / 'summary.json', [{
      'goal': 'bury_bones',
      'seed': 0,
      'success_rate': 0.5,
  }])
  common = [
      '--repo-root', str(repo_root),
      '--python', str(pathlib.Path(pilot.sys.executable)),
      '--log-root', str(tmp_path / 'runs'),
      '--goals', 'kill_goblin',
      '--stream-output',
  ]

  assert pilot.main([*common, '--steps', '200000']) == 0
  assert pilot.main([*common, '--steps', '400000']) == 0

  status = json.loads((
      tmp_path / 'runs' / 'kill_goblin_seed0' /
      'pilot_status.json').read_text())
  assert status['status'] == 'complete'
  assert status['final_step'] == 400000
  assert len(status['training_attempts']) == 2
  assert len(commands) == 4
  assert '--stream-output' in commands[1]
  assert '--stream-output' in commands[3]
  assert any('step_200000_seed1000_n20' in arg for arg in commands[1])
  assert any('step_400000_seed1000_n20' in arg for arg in commands[3])
  summaries = json.loads(
      (tmp_path / 'runs' / 'summary.json').read_text())
  assert {row['goal'] for row in summaries} == {
      'bury_bones', 'kill_goblin'}
