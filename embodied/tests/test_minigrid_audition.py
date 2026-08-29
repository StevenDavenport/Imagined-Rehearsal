import json
import pathlib
from types import SimpleNamespace

import ruamel.yaml as yaml

from scripts import minigrid_audition as audition
from scripts import minigrid_audition_report as report
from scripts import minigrid_audition_status as status
from scripts.minigrid_common import evaluation_summary


def _write_jsonl(path, rows):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_configs_define_six_tasks_reduced_sizes_and_paired_evaluation():
  configs = yaml.YAML(typ='safe').load(
      (audition.ROOT / 'dreamerv3' / 'configs.yaml').read_text())

  assert configs['minigrid']['task'] == 'minigrid_go_to_door'
  assert configs['minigrid']['agent']['goal_conditioning']['count'] == 6
  assert configs['minigrid']['batch_size'] == 8
  assert configs['minigrid_eval']['run']['eval_episodes'] == 50
  assert configs['minigrid_eval']['eval_adapt.paired_rng'] is True
  assert {
      configs[f'minigrid_{task}']['task'].removeprefix('minigrid_')
      for task in audition.TASK_NAMES
  } == set(audition.TASK_NAMES)
  assert all(
      f'minigrid_size{size}' in configs for size in audition.MODEL_SIZES)


def test_commands_pin_task_size_checkpoint_and_paired_modes(tmp_path):
  args = SimpleNamespace(
      python=pathlib.Path('/env/bin/python'),
      repo_root=pathlib.Path('/repo'), model_size='12m', steps=100_000,
      checkpoint_every=25_000, episode_length=100, replay_size=100_000,
      train_ratio=32.0, eval_episodes_per_mode=50,
      paired_rng_stride=1000)
  train = audition.build_train_command(
      args=args, task='put_near', seed=2, logdir=tmp_path / 'training',
      milestone_root=tmp_path / 'milestones', digest='spec123')
  evaluate = audition.build_eval_command(
      args=args, task='put_near', checkpoint=pathlib.Path('/ckpt'),
      logdir=tmp_path / 'evaluation', eval_seed=100_002,
      policy_mode='sampled', paired_seed=202_000)

  configs = train[train.index('--configs') + 1:train.index('--seed')]
  assert configs == [
      'minigrid', 'minigrid_size12m', 'minigrid_put_near',
      'minigrid_audition']
  assert train[train.index('--run.milestone_every') + 1] == '25000'
  assert train[train.index('--run.milestone_spec_digest') + 1] == 'spec123'
  assert evaluate[evaluate.index('--run.eval_policy_mode') + 1] == 'sampled'
  assert evaluate[evaluate.index('--run.eval_episodes') + 1] == '50'
  assert evaluate[evaluate.index('--eval_adapt.paired_rng_seed') + 1] == (
      '202000')


def test_evaluation_summary_requires_exact_audit_and_integrity(tmp_path):
  _write_jsonl(tmp_path / 'episodes.jsonl', [
      {'episode/score': 0.0, 'episode/length': 101},
      {'episode/score': 0.8, 'episode/length': 21},
  ])
  _write_jsonl(tmp_path / 'audit_env0.jsonl', [
      {'kind': 'reset', 'reset_seed': 4},
      {'kind': 'reset', 'reset_seed': 5},
  ])
  (tmp_path / 'eval_integrity.json').write_text(json.dumps({
      'equal': True, 'policy_mode': 'sampled', 'paired_rng': True,
      'paired_rng_seed': 700, 'paired_rng_stride': 1000,
  }))
  summary = evaluation_summary(
      tmp_path, task='fetch', training_seed=0, eval_seed=100,
      checkpoint=1000, policy_mode='sampled', requested_episodes=2,
      paired_rng_seed=700, paired_rng_stride=1000, returncode=0)

  assert summary['complete']
  assert summary['success_rate'] == 0.5
  assert summary['mean_action_length'] == 60
  assert summary['reset_count_exact']
  assert summary['integrity_verified']


def test_report_keeps_policy_modes_separate_and_uses_success_auc(tmp_path):
  spec = {
      'tasks': ['go_to_door', 'fetch'], 'seeds': [0, 1], 'steps': 100,
      'checkpoint_steps': [100], 'policy_modes': ['sampled', 'deterministic'],
      'difficulty_threshold': 0.7, 'difficulty_window': 2,
      'difficulty_tolerance': 0.25,
  }
  (tmp_path / 'experiment_spec.json').write_text(json.dumps(spec))
  for task in spec['tasks']:
    for seed in spec['seeds']:
      run_root = tmp_path / 'runs' / task / f'seed_{seed:04d}'
      _write_jsonl(run_root / 'training' / 'episodes.jsonl', [
          {'step': 10, 'episode/score': 0.9},
          {'step': 20, 'episode/score': 0.8},
          {'step': 100, 'episode/score': 0.0},
      ])
      for mode, success in (('sampled', 0.75), ('deterministic', 0.5)):
        condition = (
            run_root / 'evaluations' / f'step_{100:012d}' / mode)
        condition.mkdir(parents=True)
        summary = {
            'task': task, 'training_seed': seed, 'checkpoint': 100,
            'policy_mode': mode, 'mean_return': success,
            'success_rate': success,
        }
        (condition / 'condition_status.json').write_text(json.dumps({
            'status': 'complete', 'summary': summary}))

  result = report.generate_report(tmp_path)
  first = result['task_summaries'][0]
  assert first['median_training_steps_to_threshold'] == 20
  assert first['mean_training_auc'] > 0
  assert first['mean_final_sampled_success'] == 0.75
  assert first['mean_final_deterministic_success'] == 0.5
  assert (tmp_path / 'analysis' / 'pairwise_difficulty.csv').is_file()


def test_dry_run_plans_training_and_every_evaluation_without_spending_attempts(
    tmp_path):
  code = audition.main([
      '--experiment-root', str(tmp_path), '--dry-run',
      '--tasks', 'fetch', '--seeds', '0', '--steps', '20',
      '--checkpoint-every', '10', '--eval-episodes-per-mode', '2',
  ])
  payload = json.loads((tmp_path / 'supervisor_status.json').read_text())
  unit = payload['states']['fetch__seed_0000']

  assert code == 0
  assert payload['status'] == 'dry_run'
  assert unit['training_status'] == 'dry_run'
  assert len(unit['evaluations']) == 4
  assert unit['training_attempts'] == []
  assert all(condition['attempts'] == []
             for condition in unit['evaluations'].values())
  progress = status.snapshot(tmp_path)
  assert progress['training'] == '0/1'
  assert progress['evaluations'] == '0/4'


def test_restart_adopts_the_exact_live_evaluation_attempt(
    tmp_path, monkeypatch):
  common = [
      '--experiment-root', str(tmp_path), '--tasks', 'fetch',
      '--seeds', '0', '--steps', '10', '--checkpoint-every', '10',
      '--eval-episodes-per-mode', '2', '--policy-modes', 'sampled',
      '--min-free-gb', '0',
  ]
  assert audition.main([*common, '--dry-run']) == 0
  status_path = tmp_path / 'supervisor_status.json'
  payload = json.loads(status_path.read_text())
  condition = payload['states']['fetch__seed_0000']['evaluations'][
      'step_000000000010__sampled']
  command = condition.pop('planned_command')
  logdir = pathlib.Path(condition.pop('planned_logdir'))
  condition.update(status='running', attempts=[{
      'attempt': 1, 'started_unix': 1.0,
      'command': command, 'logdir': str(logdir),
  }])
  status_path.write_text(json.dumps(payload))

  monkeypatch.setattr(audition, 'package_versions', lambda: {
      'minigrid': '3.1.0', 'gymnasium': '1.3.0', 'jax': '0.4.33'})
  monkeypatch.setattr(audition, 'checkpoint_step', lambda _path: 10)
  monkeypatch.setattr(audition, '_milestones_complete', lambda *_args: True)
  monkeypatch.setattr(audition, 'active_matches', lambda *args: True)
  launched = []

  def run_managed(adopted_command, **_kwargs):
    launched.append(adopted_command)
    assert adopted_command == command
    _write_jsonl(logdir / 'episodes.jsonl', [
        {'episode/score': 1.0, 'episode/length': 5},
        {'episode/score': 0.0, 'episode/length': 11},
    ])
    _write_jsonl(logdir / 'audit_env0.jsonl', [
        {'kind': 'reset', 'reset_seed': 7},
        {'kind': 'reset', 'reset_seed': 8},
    ])
    (logdir / 'eval_integrity.json').write_text(json.dumps({
        'equal': True, 'policy_mode': 'sampled', 'paired_rng': True,
        'paired_rng_seed': 202_608_300, 'paired_rng_stride': 1000,
    }))
    return {
        'returncode': 0, 'adopted': True, 'timed_out': False,
        'pid': 1, 'pgid': 1, 'started_unix': 1, 'finished_unix': 2,
    }

  monkeypatch.setattr(audition, 'run_managed', run_managed)
  assert audition.main(common) == 0

  final = json.loads(status_path.read_text())
  attempt = final['states']['fetch__seed_0000']['evaluations'][
      'step_000000000010__sampled']['attempts'][0]
  assert launched == [command]
  assert attempt['adopted'] is True
  assert attempt['complete'] is True
  assert attempt['recovery'] == 'adopting_live_child'
  assert not (logdir.parent / 'attempt_02').exists()
