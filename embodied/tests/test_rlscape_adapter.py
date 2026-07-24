import json
from types import SimpleNamespace

import embodied
import numpy as np
import pytest

from embodied.envs.rlscape import GOAL_NAMES
from embodied.envs.rlscape import RLScape
from embodied.envs.rlscape import TASK_IDS
from embodied.envs.rlscape import canonical_action


class FakeRLScape:

  def __init__(self, outcomes=()):
    self.observation_space = SimpleNamespace(shape=(16, 16, 3))
    self.outcomes = list(outcomes)
    self.reset_calls = []
    self.actions = []
    self.closed = False
    self.goal = None
    self.episode = 0
    self.step_index = 0

  def reset(self, seed=None, options=None):
    options = dict(options or {})
    self.goal = options['goal']
    self.episode += 1
    self.step_index = 0
    self.reset_calls.append((seed, options))
    return self._frame(), self._info(task_id=True)

  def step(self, action):
    self.actions.append({
        'mode': np.asarray(action['mode']).copy(),
        'position': np.asarray(action['position']).copy(),
    })
    self.step_index += 1
    outcome = self.outcomes.pop(0) if self.outcomes else {}
    success = bool(outcome.get('success', False))
    death = bool(outcome.get('death', False))
    terminated = success or death
    truncated = bool(outcome.get('truncated', False))
    reason = (
        'task_success' if success else
        'player_death' if death else None)
    info = self._info(task_id=False)
    info.update(
        task_success=success,
        task_progress=float(outcome.get('progress', success)),
        termination_reason=reason,
        truncation_reason='episode_length' if truncated else None,
        executed_action=self.actions[-1],
    )
    return (
        self._frame(), float(success), terminated, truncated, info)

  def create_snapshot(self):
    return f'snapshot-{self.episode}-{self.step_index}'

  def restore_snapshot(self, snapshot_id):
    del snapshot_id
    return self._frame(), self._info(task_id=True)

  def close(self):
    self.closed = True

  def _frame(self):
    value = (self.episode * 10 + self.step_index) % 256
    return np.full(self.observation_space.shape, value, np.uint8)

  def _info(self, task_id):
    goal_id = GOAL_NAMES.index(self.goal)
    info = {
        'goal': self.goal,
        'lifetime_id': 'life-1',
        'episode_id': f'episode-{self.episode}',
        'action_sequence_id': self.step_index,
        'logical_tick_id': self.step_index,
        'frame_identity': {'tick': self.step_index},
        'frame_barrier_tag': f'frame-{self.step_index}',
        'events': [],
        'privileged_state': {'never': 'expose this'},
    }
    if task_id:
      info['task_id'] = TASK_IDS[goal_id]
    return info


def action(mode=0, position=(0.75, -0.5), reset=False):
  return {
      'mode': np.int32(mode),
      'position': np.asarray(position, np.float32),
      'reset': bool(reset),
  }


def make_adapter(fake, **kwargs):
  return RLScape(
      'multigoal', env=fake, check_version=False, seed=7,
      goals=GOAL_NAMES, goal_schedule='round_robin', **kwargs)


def test_canonical_action_is_pure_and_zeroes_noop_position():
  original = action()
  result = canonical_action(original)
  np.testing.assert_array_equal(result['position'], np.zeros(2, np.float32))
  np.testing.assert_array_equal(
      original['position'], np.asarray((0.75, -0.5), np.float32))
  moved = canonical_action(action(mode=2))
  np.testing.assert_array_equal(
      moved['position'], np.asarray((0.75, -0.5), np.float32))


def test_round_robin_goals_and_public_metadata_are_stable():
  fake = FakeRLScape()
  env = make_adapter(fake)
  observations = [env.step(action(reset=True)) for _ in GOAL_NAMES]
  assert [int(obs['goal_id']) for obs in observations] == list(range(5))
  assert [call[1]['goal'] for call in fake.reset_calls] == list(GOAL_NAMES)
  assert all(obs['is_first'] for obs in observations)
  assert all(obs['image'].dtype == np.uint8 for obs in observations)
  assert 'privileged_state' not in env.obs_space
  rows = env.pop_audit()
  assert len(rows) == 5
  assert all('privileged_state' not in row for row in rows)
  assert [row['goal'] for row in rows] == list(GOAL_NAMES)


def test_random_goal_schedule_is_seeded_and_varied():
  first_fake = FakeRLScape()
  second_fake = FakeRLScape()
  first = RLScape(
      'multigoal', env=first_fake, check_version=False, seed=11,
      goals=GOAL_NAMES, goal_schedule='random')
  second = RLScape(
      'multigoal', env=second_fake, check_version=False, seed=11,
      goals=GOAL_NAMES, goal_schedule='random')

  first_goals = [
      int(first.step(action(reset=True))['goal_id']) for _ in range(50)]
  second_goals = [
      int(second.step(action(reset=True))['goal_id']) for _ in range(50)]

  assert first_goals == second_goals
  assert set(first_goals) == set(range(len(GOAL_NAMES)))
  first.close()
  second.close()


@pytest.mark.parametrize(
    'outcome, expected',
    [
        ({'success': True}, (True, False, True)),
        ({'death': True}, (True, True, False)),
        ({'truncated': True}, (False, False, False)),
    ],
)
def test_goal_completion_and_physical_termination_are_separate(
    outcome, expected):
  fake = FakeRLScape([outcome])
  env = make_adapter(fake)
  env.step(action(reset=True))
  obs = env.step(action(mode=1))
  terminal, physical, complete = expected
  assert obs['is_last']
  assert obs['is_terminal'] is terminal
  assert obs['is_physical_terminal'] is physical
  assert obs['goal_complete'] is complete


def test_adapter_executes_canonical_action_and_audits_it():
  fake = FakeRLScape([{'truncated': True}])
  env = make_adapter(fake)
  env.step(action(reset=True))
  obs = env.step(action(mode=0, position=(0.9, -0.8)))
  np.testing.assert_array_equal(
      fake.actions[-1]['position'], np.zeros(2, np.float32))
  assert obs['log/action_mode_0'] == 1
  assert obs['log/action_mode_1'] == 0
  assert obs['log/action_position_abs'] == 0
  row = env.pop_audit()[-1]
  assert row['requested_action'] == {'mode': 0, 'position': [0.0, 0.0]}


def test_adapter_streams_audit_without_retaining_it(tmp_path):
  fake = FakeRLScape([{'truncated': True}])
  path = tmp_path / 'audit.jsonl'
  env = make_adapter(fake, audit_path=path)
  env.step(action(reset=True))
  env.step(action(mode=2, position=(0.25, -0.5)))
  env.close()

  rows = [json.loads(line) for line in path.read_text().splitlines()]
  assert [row['kind'] for row in rows] == ['reset', 'step']
  assert isinstance(rows[0]['reset_seed'], int)
  assert rows[0]['reset_kind'] == 'task'
  assert rows[-1]['requested_action'] == {
      'mode': 2, 'position': [0.25, -0.5]}
  assert all('privileged_state' not in row for row in rows)
  assert env.pop_audit() == []


def test_driver_and_replay_preserve_goal_and_terminal_fields():
  fake = FakeRLScape([{'truncated': True}] * 5)
  env = make_adapter(fake)
  driver = embodied.Driver([lambda: env], parallel=False)
  replay = embodied.replay.Replay(length=2, capacity=20)
  transitions = []
  driver.on_step(lambda tran, worker: (
      transitions.append(tran), replay.add(tran, worker)))

  def policy(carry, obs, mode='train'):
    del obs, mode
    batch = 1
    act = {
        'mode': np.zeros(batch, np.int32),
        'position': np.zeros((batch, 2), np.float32),
    }
    return carry, act, {}

  driver.reset(lambda batch_size: ())
  driver(policy, episodes=5)
  driver.close()
  assert {int(x['goal_id']) for x in transitions} == set(range(5))
  assert any(bool(x['is_last']) for x in transitions)
  assert any(bool(x['is_physical_terminal']) is False for x in transitions)
  sample = replay.sample(1)
  assert sample['goal_id'].shape == (1, 2)
  assert sample['goal_complete'].shape == (1, 2)
  assert sample['is_physical_terminal'].shape == (1, 2)
  assert fake.closed
