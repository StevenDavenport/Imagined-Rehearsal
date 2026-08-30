import json

import elements
import embodied
import numpy as np

from embodied.run import milestones
from embodied.run.train import train


class Saveable:

  def __init__(self, value):
    self.value = value

  def save(self):
    return {'value': self.value}

  def load(self, data):
    self.value = data['value']


def test_milestone_is_self_contained_and_restores_exact_fifo(tmp_path):
  source = tmp_path / 'live_replay'
  replay = embodied.replay.Replay(
      length=3, capacity=10, directory=source, chunksize=4,
      save_wait=True)
  for step in range(30):
    replay.add({'step': np.int32(step)})
  assert len(replay) == 10

  counter = elements.Counter()
  counter.increment(30)
  checkpoint = elements.Checkpoint()
  checkpoint.step = counter
  checkpoint.agent = Saveable(7)
  checkpoint.replay = replay
  archive = milestones.create(
      checkpoint, replay, tmp_path / 'milestones', 30,
      metadata={'goal': 'kill_goblin'})

  manifest = milestones.validate(archive, expected_step=30, full=True)
  assert manifest['metadata']['goal'] == 'kill_goblin'
  assert len(manifest['replay_files']) > 0
  assert all(
      (archive / 'replay' / record['name']).is_file()
      for record in manifest['replay_files'])

  for path in source.glob('*.npz'):
    path.unlink()
  restored = embodied.replay.Replay(
      length=3, capacity=10, directory=archive / 'replay', chunksize=4)
  restored.load()
  assert len(restored) == 10
  for _ in range(10):
    batch = restored.sample(1)
    assert batch['step'].shape == (1, 3)


def test_milestone_validation_detects_changed_file(tmp_path):
  source = tmp_path / 'live_replay'
  replay = embodied.replay.Replay(
      length=2, capacity=10, directory=source, chunksize=4,
      save_wait=True)
  for step in range(8):
    replay.add({'step': np.int32(step)})
  counter = elements.Counter()
  counter.increment(8)
  checkpoint = elements.Checkpoint()
  checkpoint.step = counter
  checkpoint.agent = Saveable(1)
  checkpoint.replay = replay
  archive = milestones.create(
      checkpoint, replay, tmp_path / 'milestones', 8)
  manifest = json.loads((archive / 'manifest.json').read_text())
  record = manifest['checkpoint_files'][0]
  path = archive / 'checkpoint' / record['path']
  path.write_bytes(path.read_bytes() + b'changed')

  try:
    milestones.validate(archive, expected_step=8, full=False)
  except milestones.MilestoneError as error:
    assert 'size changed' in str(error)
  else:
    raise AssertionError('Changed archive unexpectedly validated')


def test_checkpoint_snapshot_contains_agent_but_no_replay(tmp_path):
  counter = elements.Counter()
  counter.increment(25)
  checkpoint = elements.Checkpoint()
  checkpoint.step = counter
  checkpoint.agent = Saveable(9)

  archive = milestones.create_checkpoint(
      checkpoint, tmp_path / 'snapshots', 25,
      metadata={'goal': 'fetch', 'phase': 1})
  manifest = milestones.validate_checkpoint(
      archive, expected_step=25, full=True)

  assert manifest['kind'] == 'checkpoint_snapshot'
  assert manifest['metadata'] == {'goal': 'fetch', 'phase': 1}
  assert manifest['checkpoint_files']
  assert 'replay_files' not in manifest
  assert not (archive / 'replay').exists()


def test_training_loop_publishes_exact_requested_milestones(tmp_path):
  from embodied.envs.dummy import Dummy

  env = Dummy('disc', size=(8, 8), length=100)
  agent = embodied.RandomAgent(env.obs_space, env.act_space)

  def make_stream(replay, mode):
    del mode
    while True:
      yield replay.sample(1)

  logger = elements.Logger(
      elements.Counter(), [elements.logger.TerminalOutput()])
  args = type('Args', (), dict(
      logdir=str(tmp_path / 'training'),
      usage={},
      batch_size=1,
      batch_length=2,
      train_ratio=0,
      log_every=1000,
      report_every=1000,
      save_every=1000,
      envs=1,
      debug=True,
      report_batches=1,
      consec_report=1,
      steps=20,
      ckpt_keep=2,
      from_checkpoint='',
      from_checkpoint_regex='',
      milestone_every=10,
      milestone_dir=str(tmp_path / 'milestones'),
      milestone_goal='kill_goblin',
      milestone_phase=0,
      milestone_spec_digest='fixed',
      snapshot_every=10,
      snapshot_start=0,
      snapshot_dir=str(tmp_path / 'snapshots'),
      snapshot_goal='kill_goblin',
      snapshot_phase=0,
      snapshot_spec_digest='fixed',
  ))()

  train(
      lambda: agent,
      lambda: embodied.replay.Replay(
          length=2, capacity=100, directory=tmp_path / 'live_replay',
          save_wait=True),
      lambda index: env,
      make_stream,
      lambda: logger,
      args,
  )

  for step in (10, 20):
    manifest = milestones.validate(
        milestones.archive_path(tmp_path / 'milestones', step),
        expected_step=step, full=True)
    assert manifest['metadata']['goal'] == 'kill_goblin'
  for step in (0, 10, 20):
    manifest = milestones.validate_checkpoint(
        milestones.archive_path(tmp_path / 'snapshots', step),
        expected_step=step, full=True)
    assert manifest['metadata']['goal'] == 'kill_goblin'


def test_training_resumes_complete_milestone_state_into_new_root(tmp_path):
  from embodied.envs.dummy import Dummy

  env = Dummy('disc', size=(8, 8), length=100)
  source_agent = embodied.RandomAgent(env.obs_space, env.act_space)
  source_replay = embodied.replay.Replay(
      length=2, capacity=100, directory=tmp_path / 'source_replay',
      chunksize=4, save_wait=True)
  for index in range(30):
    source_replay.add({'step': np.int32(index)})
  source_step = elements.Counter()
  source_step.increment(30)
  source_checkpoint = elements.Checkpoint()
  source_checkpoint.step = source_step
  source_checkpoint.agent = source_agent
  source_checkpoint.replay = source_replay
  archive = milestones.create(
      source_checkpoint, source_replay, tmp_path / 'source_milestones', 30,
      metadata={'spec_digest': 'source-spec'})
  source_manifest = milestones.validate(archive, full=True)

  target = tmp_path / 'extended_training'
  target_milestones = tmp_path / 'extended_milestones'
  target_agent = embodied.RandomAgent(env.obs_space, env.act_space)

  def make_stream(replay, mode):
    del mode
    while True:
      yield replay.sample(1)

  logger = elements.Logger(
      elements.Counter(), [elements.logger.TerminalOutput()])
  args = type('Args', (), dict(
      logdir=str(target), usage={}, batch_size=1, batch_length=2,
      train_ratio=0, log_every=1000, report_every=1000,
      save_every=1000, envs=1, debug=True, report_batches=1,
      consec_report=1, steps=40, ckpt_keep=2, from_checkpoint='',
      from_checkpoint_regex='', resume_from_milestone=str(archive),
      milestone_every=10, milestone_start=40,
      milestone_dir=str(target_milestones), milestone_goal='fetch',
      milestone_phase=0, milestone_spec_digest='extension-spec',
  ))()

  train(
      lambda: target_agent,
      lambda: embodied.replay.Replay(
          length=2, capacity=100, directory=target / 'replay',
          chunksize=4, save_wait=True),
      lambda index: env,
      make_stream,
      lambda: logger,
      args,
  )

  provenance = json.loads((target / 'resume_provenance.json').read_text())
  assert provenance['source_step'] == 30
  assert provenance['source_manifest_metadata']['spec_digest'] == 'source-spec'
  assert provenance['replay_files'] == len(source_manifest['replay_files'])
  assert milestones.validate(
      milestones.archive_path(target_milestones, 40),
      expected_step=40, full=True)
  assert milestones.validate(archive, expected_step=30, full=True)
