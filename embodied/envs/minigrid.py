"""Embodied adapter for the controlled MiniGrid CIR audition suite."""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib

import elements
import embodied
import numpy as np

from .minigrid_registry import TASK_INDEX
from .minigrid_registry import TASK_NAMES


def _resize_nearest(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
  """Dependency-free deterministic nearest-neighbour image resize."""
  image = np.asarray(image)
  width, height = (int(size[0]), int(size[1]))
  if image.ndim != 3 or image.shape[-1] != 3:
    raise ValueError(f'Expected RGB image, got {image.shape}')
  if min(width, height) <= 0:
    raise ValueError(f'Image size must be positive, got {size}')
  if image.shape[:2] == (height, width):
    return np.asarray(image, np.uint8)
  ys = np.rint(np.linspace(0, image.shape[0] - 1, height)).astype(np.intp)
  xs = np.rint(np.linspace(0, image.shape[1] - 1, width)).astype(np.intp)
  return np.asarray(image[ys[:, None], xs[None, :]], np.uint8)


class MiniGrid(embodied.Env):
  """Expose fixed-mission MiniGrid tasks through the flat Embodied contract."""

  def __init__(
      self,
      task='go_to_door',
      *,
      seed=0,
      size=(64, 64),
      episode_length=100,
      tile_size=8,
      audit_path='',
      env=None,
  ):
    task = str(task)
    if task not in TASK_INDEX:
      raise ValueError(
          f'Unknown MiniGrid task {task!r}; expected one of {TASK_NAMES}')
    self._task = task
    self._goal_id = TASK_INDEX[task]
    self._seed = int(seed)
    self._rng = np.random.default_rng(self._seed)
    self._size = tuple(int(value) for value in size)
    if len(self._size) != 2 or min(self._size) <= 0:
      raise ValueError(f'MiniGrid size must be [width, height], got {size}')
    self._shape = (self._size[1], self._size[0], 3)
    self._episode_length = int(episode_length)
    self._tile_size = int(tile_size)
    if min(self._episode_length, self._tile_size) <= 0:
      raise ValueError('episode_length and tile_size must be positive')

    if env is None:
      try:
        from .minigrid_tasks import make_task
      except ImportError as exc:
        raise ImportError(
            'MiniGrid support requires the pinned minigrid and gymnasium '
            'dependencies. Install this repository requirements first.') from exc
      env = make_task(task, episode_length=self._episode_length)
    self._env = env
    self._raw_env = getattr(env, 'unwrapped', env)
    action_count = getattr(getattr(env, 'action_space', None), 'n', None)
    if int(action_count or -1) != 7:
      raise ValueError(
          f'Controlled MiniGrid tasks require Discrete(7), got '
          f'{getattr(env, "action_space", None)!r}')

    self._done = True
    self._episode_index = 0
    self._last_info = None
    self._last_mission = ''
    self._audit_handle = None
    if audit_path:
      path = pathlib.Path(audit_path)
      path.parent.mkdir(parents=True, exist_ok=True)
      self._audit_handle = path.open('a', buffering=1)

  @property
  def obs_space(self):
    return {
        'image': elements.Space(np.uint8, self._shape, 0, 256),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'is_physical_terminal': elements.Space(bool),
        'goal_complete': elements.Space(bool),
        'goal_id': elements.Space(np.int32, (), 0, len(TASK_NAMES)),
        'log/task_success': elements.Space(np.float32),
        'log/task_progress': elements.Space(np.float32),
        'log/goal_id': elements.Space(np.float32),
        'log/action': elements.Space(np.float32),
        **{
            f'log/action_{index}': elements.Space(np.float32)
            for index in range(7)
        },
    }

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, 7),
        'reset': elements.Space(bool),
    }

  @property
  def raw_env(self):
    return self._raw_env

  @property
  def last_info(self):
    return copy.deepcopy(self._last_info)

  @property
  def current_goal(self):
    return self._task

  def step(self, action):
    if bool(np.asarray(action['reset'])) or self._done:
      return self._reset()
    value = np.asarray(action['action'])
    if value.shape != () or not np.issubdtype(value.dtype, np.integer):
      raise ValueError(
          f'MiniGrid action must be an integer scalar, got '
          f'{value.dtype}{value.shape}')
    value = int(value)
    if not 0 <= value < 7:
      raise ValueError(f'MiniGrid action must be in [0, 7), got {value}')
    _observation, reward, terminated, truncated, info = self._env.step(value)
    self._done = bool(terminated or truncated)
    self._last_info = copy.deepcopy(info or {})
    success = bool(float(reward) > 0)
    frame = self._frame()
    self._audit('step', action=value, reward=float(reward), success=success)
    return self._obs(
        frame, reward, action=value, is_last=self._done,
        is_terminal=bool(terminated), goal_complete=success)

  def close(self):
    try:
      self._env.close()
    finally:
      if self._audit_handle is not None:
        self._audit_handle.close()
        self._audit_handle = None

  def _reset(self):
    episode_seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
    observation, info = self._env.reset(seed=episode_seed)
    del observation
    self._done = False
    self._last_info = copy.deepcopy(info or {})
    self._last_mission = str(getattr(self._raw_env, 'mission', ''))
    frame = self._frame()
    self._episode_index += 1
    self._audit(
        'reset', reset_seed=episode_seed, mission=self._last_mission,
        frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest())
    return self._obs(frame, 0.0, is_first=True)

  def _frame(self):
    getter = getattr(self._raw_env, 'get_frame', None)
    if getter is None:
      frame = self._env.render()
    else:
      frame = getter(
          highlight=False, tile_size=self._tile_size, agent_pov=True)
    frame = _resize_nearest(np.asarray(frame), self._size)
    if frame.shape != self._shape or frame.dtype != np.uint8:
      raise RuntimeError(
          f'Invalid MiniGrid frame: expected uint8{self._shape}, got '
          f'{frame.dtype}{frame.shape}')
    return frame

  def _obs(
      self,
      image,
      reward,
      *,
      action=None,
      is_first=False,
      is_last=False,
      is_terminal=False,
      goal_complete=False,
  ):
    action_value = -1 if action is None else int(action)
    progress = float(goal_complete)
    return {
        'image': np.asarray(image, np.uint8),
        'reward': np.float32(reward),
        'is_first': bool(is_first),
        'is_last': bool(is_last),
        'is_terminal': bool(is_terminal),
        'is_physical_terminal': False,
        'goal_complete': bool(goal_complete),
        'goal_id': np.int32(self._goal_id),
        'log/task_success': np.float32(goal_complete),
        'log/task_progress': np.float32(progress),
        'log/goal_id': np.float32(self._goal_id),
        'log/action': np.float32(action_value),
        **{
            f'log/action_{index}': np.float32(action_value == index)
            for index in range(7)
        },
    }

  def _audit(self, kind, **values):
    if self._audit_handle is None:
      return
    row = {
        'kind': kind,
        'task': self._task,
        'goal_id': self._goal_id,
        'episode_index': self._episode_index,
        **values,
    }
    self._audit_handle.write(json.dumps(row, sort_keys=True) + '\n')
