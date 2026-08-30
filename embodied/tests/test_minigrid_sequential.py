import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import ruamel.yaml as yaml

from scripts import minigrid_sequential_baseline as sequential
from scripts import minigrid_sequential_report as report
from scripts import minigrid_sequential_status as status


def test_evaluation_schedule_is_dense_for_current_and_complete_every_other():
  spec = {
      'tasks': ['go_to_door', 'fetch', 'go_to_object', 'put_near'],
      'phase_steps': 20,
      'snapshot_steps': list(range(0, 81, 10)),
      'eval_all_every': 20,
      'policy_modes': ['sampled', 'deterministic'],
  }
  units = sequential.evaluation_units(spec)

  assert len(units) == 48
  at_zero = [unit for unit in units if unit['step'] == 0]
  at_ten = [unit for unit in units if unit['step'] == 10]
  at_thirty = [unit for unit in units if unit['step'] == 30]
  assert {unit['task'] for unit in at_zero} == set(spec['tasks'])
  assert {unit['task'] for unit in at_ten} == {'go_to_door'}
  assert {unit['task'] for unit in at_thirty} == {'fetch'}
  assert {unit['policy_mode'] for unit in at_ten} == {
      'sampled', 'deterministic'}


def _args(tmp_path):
  return SimpleNamespace(
      python=pathlib.Path('/env/bin/python'), repo_root=pathlib.Path('/repo'),
      model_size='12m', jax_platform='cuda', phase_steps=200_000,
      train_ratio=32.0,
      snapshot_every=25_000, episode_length=100,
      environment_seed_stride=1_000_000, fifo_replay_size=100_000,
      episodes_per_goal=1_000, current_fraction=0.5,
      eval_episodes_per_mode=50, paired_rng_stride=1_000,
      diagnostic_chunk_length=32, critic_horizons=(1, 3, 6, 15),
      critic_posterior_samples=32, critic_anchors_per_episode=2,
      diagnostic_seed_base=202_608_301,
  )


def test_training_commands_keep_uniform_primary_and_5050_as_separate_arm(
    tmp_path):
  args = _args(tmp_path)
  common = dict(
      args=args, task='fetch', phase=1, seed=2,
      training=tmp_path / 'training', snapshot_root=tmp_path / 'snapshots',
      milestone_root=tmp_path / 'milestones', digest='fixed')
  uniform = sequential.build_train_command(
      arm='uniform_reservoir', **common)
  biased = sequential.build_train_command(
      arm='current_old_50_50', **common)
  fifo = sequential.build_train_command(arm='fifo_100k', **common)

  assert uniform[uniform.index('--run.steps') + 1] == '400000'
  assert uniform[uniform.index('--run.snapshot_every') + 1] == '25000'
  assert uniform[uniform.index('--run.milestone_every') + 1] == '200000'
  assert uniform[uniform.index('--replay.size') + 1] == '6000'
  assert uniform[uniform.index('--replay.current_group') + 1] == '1'
  assert uniform[uniform.index('--replay.sampling') + 1] == 'uniform_groups'
  assert biased[biased.index('--replay.sampling') + 1] == 'current_old'
  assert biased[biased.index('--replay.current_fraction') + 1] == '0.5'
  assert fifo[fifo.index('--replay.kind') + 1] == 'fifo'
  assert fifo[fifo.index('--replay.size') + 1] == '100000'
  assert 'minigrid_episode_reservoir' not in fifo


def test_diagnostic_commands_use_same_bank_and_phase_snapshot(tmp_path):
  args = _args(tmp_path)
  head = sequential.build_diagnostic_command(
      args=args, kind='head', task='go_to_door',
      checkpoint=pathlib.Path('/snapshot/checkpoint'),
      selection=pathlib.Path('/bank/selection.json'),
      logdir=tmp_path / 'head', seed=1)
  critic = sequential.build_diagnostic_command(
      args=args, kind='critic_provenance', task='go_to_door',
      checkpoint=pathlib.Path('/snapshot/checkpoint'),
      selection=pathlib.Path('/bank/selection.json'),
      logdir=tmp_path / 'critic', seed=1)

  assert 'minigrid_head_audit' in head
  assert head[head.index('--head_audit.selection') + 1] == (
      '/bank/selection.json')
  assert 'minigrid_critic_provenance' in critic
  assert critic[critic.index('--critic_provenance.horizons') + 1] == (
      '1,3,6,15')
  assert critic[critic.index(
      '--critic_provenance.posterior_samples') + 1] == '32'


def test_configs_expose_sequential_and_read_only_diagnostic_modes():
  configs = yaml.YAML(typ='safe').load(
      (sequential.ROOT / 'dreamerv3' / 'configs.yaml').read_text())
  train = configs['minigrid_sequential']

  assert train['run']['milestone_every'] == 200_000
  assert train['run']['snapshot_every'] == 25_000
  assert train['agent.goal_loss_metrics'] is True
  assert configs['minigrid_episode_reservoir']['replay.kind'] == (
      'episode_reservoir')
  assert configs['minigrid_head_audit']['script'] == 'head_audit'
  assert configs['minigrid_critic_provenance']['script'] == (
      'critic_provenance')
  assert len(configs['minigrid_head_audit']['head_audit.goal_names']) == 6


def test_full_dry_run_pins_three_arms_four_phases_and_all_visibility_jobs(
    tmp_path):
  assert sequential.main([
      '--experiment-root', str(tmp_path), '--dry-run',
      '--seeds', '0', '--phase-steps', '20', '--snapshot-every', '10',
      '--eval-all-every', '20', '--eval-episodes-per-mode', '2',
      '--min-free-gb', '0',
  ]) == 0
  spec = json.loads((tmp_path / 'experiment_spec.json').read_text())
  payload = json.loads((tmp_path / 'supervisor_status.json').read_text())

  assert spec['tasks'] == list(sequential.ALTERNATING_CHAIN)
  assert spec['total_steps_per_run'] == 80
  assert spec['total_training_steps'] == 240
  assert spec['evaluation_units_per_run'] == 48
  assert len(payload['states']) == 3
  assert all(len(state['phases']) == 4
             for state in payload['states'].values())
  assert all(len(state['evaluations']) == 48
             for state in payload['states'].values())
  # Three arms x five phase boundaries x two diagnostic families, plus bank.
  assert len(payload['diagnostics']) == 31
  progress = status.snapshot(tmp_path)
  assert progress['training_phases'] == '0/12'
  assert progress['evaluations'] == '0/144'
  assert progress['diagnostics'] == '0/30'


def test_report_computes_forward_transfer_forgetting_and_bwt(tmp_path):
  spec = {
      'name': 'synthetic', 'arms': {'uniform_reservoir': {}}, 'seeds': [0],
      'tasks': ['go_to_door', 'fetch'], 'policy_modes': ['sampled'],
      'phase_steps': 20, 'total_steps_per_run': 40,
      'phase_boundaries': [0, 20, 40], 'evaluation_units_per_run': 6,
  }
  state = {'evaluations': {}}
  values = {
      ('go_to_door', 0): 0.0, ('go_to_door', 20): 0.9,
      ('go_to_door', 40): 0.6, ('fetch', 0): 0.1,
      ('fetch', 20): 0.3, ('fetch', 40): 0.8,
  }
  for (task, step), success in values.items():
    state['evaluations'][f'{task}_{step}'] = {
        'task': task, 'step': step, 'phase': 0,
        'policy_mode': 'sampled', 'status': 'complete',
        'summary': {
            'task': task, 'checkpoint': step, 'policy_mode': 'sampled',
            'episodes': 10, 'success_rate': success, 'mean_return': success,
            'integrity_verified': True,
        },
    }
  (tmp_path / 'experiment_spec.json').write_text(json.dumps(spec))
  (tmp_path / 'supervisor_status.json').write_text(json.dumps({
      'states': {'uniform_reservoir__seed_0000': state},
  }))

  summary = report.generate_report(tmp_path)
  transfers = json.loads((tmp_path / 'analysis' / 'summary.json').read_text())
  assert summary['evaluation_rows'] == 6
  assert transfers['evaluation_complete']
  rows = report.transfer_rows(report.evaluation_rows(
      tmp_path, spec, {'states': {
          'uniform_reservoir__seed_0000': state}}), spec)
  door = next(row for row in rows if row['task'] == 'go_to_door')
  fetch = next(row for row in rows if row['task'] == 'fetch')
  assert door['forgetting_from_peak'] == pytest.approx(0.3)
  assert door['backward_transfer_from_acquisition'] == pytest.approx(-0.3)
  assert fetch['forward_transfer_vs_initial'] == pytest.approx(0.2)


def test_actor_drift_uses_identical_states_and_factual_goal_distribution(
    tmp_path):
  spec = {
      'arms': {'uniform_reservoir': {}}, 'seeds': [0],
      'tasks': ['go_to_door'], 'task_goal_ids': {'go_to_door': 0},
      'phase_boundaries': [0, 20],
  }
  base = (
      tmp_path / 'diagnostics' / 'head' / 'uniform_reservoir' /
      'seed_0000')
  for step, action in ((0, 0), (20, 1)):
    folder = base / f'step_{step:012d}'
    folder.mkdir(parents=True)
    probability = np.full((3, 6, 7), 1e-6, np.float32)
    probability[..., action] = 1 - 6e-6
    np.savez_compressed(
        folder / 'predictions.npz', episode_id=np.asarray([1, 1, 1]),
        timestep=np.asarray([0, 1, 2]), actual_goal=np.asarray([0, 0, 0]),
        actor_probability=probability)

  rows = report.actor_drift_rows(tmp_path, spec)

  assert len(rows) == 1
  assert rows[0]['states'] == 3
  assert rows[0]['modal_action_agreement'] == 0
  assert rows[0]['js_divergence_mean'] > 0.68
