import json
from types import SimpleNamespace

import numpy as np
import pytest

from embodied.envs.minigrid import MiniGrid
from embodied.envs.minigrid import _resize_nearest
from embodied.envs.minigrid_registry import ALTERNATING_CHAIN
from embodied.envs.minigrid_registry import PROGRESSIVE_CHAIN
from embodied.envs.minigrid_registry import TASK_INDEX
from embodied.envs.minigrid_registry import TASK_NAMES


class FakeMiniGrid:

  action_space = SimpleNamespace(n=7)
  mission = 'fixed test mission'

  def __init__(self):
    self.unwrapped = self
    self.reset_seeds = []
    self.actions = []

  def reset(self, *, seed):
    self.reset_seeds.append(seed)
    return {'mission': self.mission}, {'seed': seed}

  def step(self, action):
    self.actions.append(action)
    return {}, float(action == 3), action == 3, False, {'action': action}

  def get_frame(self, *, highlight, tile_size, agent_pov):
    assert highlight is False
    assert tile_size == 8
    assert agent_pov is True
    return np.arange(7 * 7 * 3, dtype=np.uint8).reshape(7, 7, 3)

  def close(self):
    pass


def test_registry_has_stable_six_way_goal_ids_and_candidate_chains():
  assert tuple(TASK_INDEX) == TASK_NAMES
  assert tuple(TASK_INDEX.values()) == tuple(range(6))
  assert ALTERNATING_CHAIN == (
      'go_to_door', 'fetch', 'go_to_object', 'put_near')
  assert PROGRESSIVE_CHAIN == ('fetch', 'put_near', 'unlock_pickup')


def test_adapter_is_seeded_flat_and_auditable(tmp_path):
  first_raw = FakeMiniGrid()
  second_raw = FakeMiniGrid()
  first = MiniGrid(
      'fetch', seed=17, env=first_raw,
      audit_path=tmp_path / 'first.jsonl')
  second = MiniGrid('fetch', seed=17, env=second_raw)

  initial = first.step({'reset': True, 'action': np.int32(0)})
  paired = second.step({'reset': True, 'action': np.int32(0)})
  assert first_raw.reset_seeds == second_raw.reset_seeds
  assert np.array_equal(initial['image'], paired['image'])
  assert initial['image'].shape == (64, 64, 3)
  assert initial['image'].dtype == np.uint8
  assert initial['is_first'] and not initial['is_last']
  assert initial['goal_id'] == TASK_INDEX['fetch']

  terminal = first.step({'reset': False, 'action': np.int32(3)})
  assert terminal['reward'] == 1
  assert terminal['goal_complete']
  assert terminal['is_last'] and terminal['is_terminal']
  assert terminal['log/action_3'] == 1
  first.close()
  second.close()

  audit = [json.loads(line) for line in (
      tmp_path / 'first.jsonl').read_text().splitlines()]
  assert audit[0]['kind'] == 'reset'
  assert audit[0]['episode_index'] == 1
  assert audit[0]['mission'] == 'fixed test mission'
  assert len(audit[0]['frame_sha256']) == 64
  assert audit[1]['kind'] == 'step'
  assert audit[1]['episode_index'] == 1


def test_adapter_rejects_non_scalar_and_out_of_range_actions():
  env = MiniGrid('go_to_door', env=FakeMiniGrid())
  env.step({'reset': True, 'action': np.int32(0)})
  with pytest.raises(ValueError, match='integer scalar'):
    env.step({'reset': False, 'action': np.array([1], np.int32)})
  with pytest.raises(ValueError, match=r'\[0, 7\)'):
    env.step({'reset': False, 'action': np.int32(7)})
  env.close()


def test_nearest_resize_has_requested_width_height_order():
  image = np.zeros((3, 5, 3), np.uint8)
  assert _resize_nearest(image, (11, 7)).shape == (7, 11, 3)


def test_all_controlled_tasks_reset_and_render_partial_rgb():
  pytest.importorskip('minigrid')
  from embodied.envs.minigrid_tasks import make_task

  for task in TASK_NAMES:
    env = make_task(task, episode_length=100)
    for seed in (0, 1):
      _observation, _info = env.reset(seed=seed)
      assert env.action_space.n == 7
      assert env.unwrapped.mission
      frame = env.unwrapped.get_frame(
          highlight=False, tile_size=8, agent_pov=True)
      assert frame.shape == (56, 56, 3)
      assert frame.dtype == np.uint8
    env.close()
