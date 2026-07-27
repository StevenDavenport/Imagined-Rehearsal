"""Embodied adapter for the public RLScape Gymnasium environment."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import pathlib

import elements
import embodied
import numpy as np


EXPECTED_VERSION = '0.1.6'
ACTION_INTERFACES = ('mixed', 'click_grid')
GOAL_NAMES = (
    'kill_goblin',
    'bury_bones',
    'chop_logs',
    'light_fire',
    'catch_fish',
)
TASK_IDS = tuple(f'lumbridge_v1/{name}@1' for name in GOAL_NAMES)
GOAL_INDEX = {
    identifier: index
    for index, (name, task_id) in enumerate(zip(GOAL_NAMES, TASK_IDS))
    for identifier in (name, task_id)
}


def canonical_action(action):
  """Return the canonical public action without mutating the input."""
  mode = np.asarray(action['mode'], np.int32)
  position = np.asarray(action['position'], np.float32)
  if mode.shape != ():
    raise ValueError(f'RLScape action mode must be scalar, got {mode.shape}')
  if position.shape != (2,):
    raise ValueError(
        f'RLScape action position must have shape (2,), got {position.shape}')
  if not 0 <= int(mode) < 4:
    raise ValueError(f'RLScape action mode must be in [0, 4), got {mode}')
  if not np.isfinite(position).all():
    raise ValueError('RLScape action position must be finite')
  if (position < -1).any() or (position > 1).any():
    raise ValueError(
        f'RLScape action position must be within [-1, 1], got {position}')
  if int(mode) == 0:
    position = np.zeros((2,), np.float32)
  return {'mode': np.int32(mode), 'position': position}


class RLScape(embodied.Env):
  """Translate RLScape v0.1.6 into the flat Embodied environment contract."""

  def __init__(
      self,
      task='multigoal',
      *,
      goals=(),
      goal_schedule='round_robin',
      seed=0,
      package_version=EXPECTED_VERSION,
      launch=True,
      server_dir='',
      server_runtime_dir='',
      client_dir='',
      java_home='',
      mvn_path='',
      username='agent',
      resize=(240, 160),
      render_mode='rgb_array',
      camera_mode='birdseye',
      resize_filter='area',
      action_interface='mixed',
      grid_columns=28,
      grid_rows=18,
      episode_length=200,
      reward_mode='sparse_success',
      server_barrier=True,
      sync_to_tick=True,
      tick_divisor=1,
      auto_calibrate_tick=True,
      reset_kind='task',
      audit_path='',
      env=None,
      check_version=True,
  ):
    self._seed = int(seed)
    self._rng = np.random.default_rng(self._seed)
    self._action_interface = str(action_interface)
    if self._action_interface not in ACTION_INTERFACES:
      raise ValueError(
          f'action_interface must be one of {ACTION_INTERFACES}, '
          f'got {self._action_interface!r}')
    self._grid_columns = int(grid_columns)
    self._grid_rows = int(grid_rows)
    if self._grid_columns <= 0 or self._grid_rows <= 0:
      raise ValueError('grid_columns and grid_rows must be positive')
    self._schedule = str(goal_schedule)
    if self._schedule not in ('fixed', 'round_robin', 'random'):
      raise ValueError(
          'goal_schedule must be fixed, round_robin, or random, got '
          f'{self._schedule!r}')
    self._reset_kind = str(reset_kind)
    if self._reset_kind not in ('task', 'lifetime'):
      raise ValueError(
          f'reset_kind must be task or lifetime, got {self._reset_kind!r}')

    if goals:
      selected = tuple(str(goal) for goal in goals)
    elif task in ('all', 'multigoal'):
      selected = GOAL_NAMES
    else:
      selected = (str(task),)
    unknown = [goal for goal in selected if goal not in GOAL_INDEX]
    if unknown:
      raise ValueError(
          f'Unknown RLScape goals {unknown}; expected a subset of {GOAL_NAMES}')
    self._goals = tuple(
        GOAL_NAMES[GOAL_INDEX[goal]] for goal in selected)
    if not self._goals:
      raise ValueError('RLScape requires at least one configured goal')
    if len(self._goals) == 1:
      self._schedule = 'fixed'

    if env is None:
      if check_version:
        installed = importlib.metadata.version('rl-scape')
        if installed != str(package_version):
          raise RuntimeError(
              f'RLScape version mismatch: expected {package_version}, '
              f'found {installed}')
      from rl_scape import ClickGridActionWrapper
      from rl_scape import RLScapeEnv
      kwargs = dict(
          launch=bool(launch),
          username=str(username),
          resize=tuple(int(x) for x in resize),
          render_mode=str(render_mode),
          camera_mode=str(camera_mode),
          resize_filter=str(resize_filter),
          episode_length=int(episode_length),
          reward_mode=str(reward_mode),
          server_barrier=bool(server_barrier),
          sync_to_tick=bool(sync_to_tick),
          tick_divisor=int(tick_divisor),
          auto_calibrate_tick=bool(auto_calibrate_tick),
      )
      optional = dict(
          server_dir=server_dir,
          server_runtime_dir=server_runtime_dir,
          client_dir=client_dir,
          java_home=java_home,
      )
      kwargs.update({key: value for key, value in optional.items() if value})
      previous_mvn = os.environ.get('RL_SCAPE_MVN')
      try:
        if mvn_path:
          os.environ['RL_SCAPE_MVN'] = str(mvn_path)
        env = RLScapeEnv(**kwargs)
        if self._action_interface == 'click_grid':
          env = ClickGridActionWrapper(
              env, columns=self._grid_columns, rows=self._grid_rows)
      finally:
        if mvn_path:
          if previous_mvn is None:
            os.environ.pop('RL_SCAPE_MVN', None)
          else:
            os.environ['RL_SCAPE_MVN'] = previous_mvn
    self._env = env
    self._raw_env = getattr(self._env, 'unwrapped', self._env)
    if self._action_interface == 'click_grid':
      action_space = getattr(self._env, 'action_space', None)
      grid_actions = getattr(action_space, 'n', None)
      expected = self._grid_columns * self._grid_rows
      if grid_actions is None or int(grid_actions) != expected:
        raise ValueError(
            'click_grid requires an environment exposing '
            f'Discrete({expected}), got {action_space!r}')
      self._grid_actions = expected
    else:
      self._grid_actions = None

    shape = tuple(int(x) for x in self._env.observation_space.shape)
    if len(shape) != 3 or shape[-1] != 3:
      raise ValueError(f'RLScape observation must be RGB, got {shape}')
    self._shape = shape
    self._done = True
    self._started = False
    self._episode_index = 0
    self._goal_name = None
    self._goal_id = None
    self._task_id = None
    self._last_info = None
    self._audit = []
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
        'goal_id': elements.Space(np.int32, (), 0, len(GOAL_NAMES)),
        'log/task_success': elements.Space(np.float32),
        'log/task_progress': elements.Space(np.float32),
        'log/player_death': elements.Space(np.float32),
        'log/goal_id': elements.Space(np.float32),
        'log/action_mode_0': elements.Space(np.float32),
        'log/action_mode_1': elements.Space(np.float32),
        'log/action_mode_2': elements.Space(np.float32),
        'log/action_mode_3': elements.Space(np.float32),
        'log/action_position_abs': elements.Space(np.float32),
        'log/action_grid_valid': elements.Space(np.float32),
        'log/action_grid_index': elements.Space(np.float32),
        'log/action_grid_column': elements.Space(np.float32),
        'log/action_grid_row': elements.Space(np.float32),
    }

  @property
  def act_space(self):
    if self._action_interface == 'click_grid':
      return {
          'action': elements.Space(
              np.int32, (), 0, self._grid_actions),
          'reset': elements.Space(bool),
      }
    return {
        'mode': elements.Space(np.int32, (), 0, 4),
        'position': elements.Space(np.float32, (2,), -1, 1),
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
    return self._goal_name

  def pop_audit(self):
    rows, self._audit = self._audit, []
    return rows

  def step(self, action):
    if bool(np.asarray(action['reset'])) or self._done:
      return self._reset()
    requested = self._canonical_policy_action(action)
    observation, reward, terminated, truncated, info = self._env.step(requested)
    self._last_info = copy.deepcopy(info)
    self._done = bool(terminated or truncated)
    success = bool(info.get('task_success', False))
    reason = info.get('termination_reason')
    physical = bool(reason == 'player_death')
    self._append_audit('step', info, requested)
    return self._obs(
        observation,
        reward,
        is_first=False,
        is_last=self._done,
        is_terminal=bool(terminated),
        is_physical_terminal=physical,
        goal_complete=success,
        task_progress=info.get('task_progress', 0),
        action=requested,
        executed_action=info.get('executed_action'),
    )

  def close(self):
    try:
      self._env.close()
    finally:
      if self._audit_handle is not None:
        self._audit_handle.close()
        self._audit_handle = None

  def create_snapshot(self):
    return self._raw_env.create_snapshot()

  def restore_snapshot(self, snapshot_id):
    observation, info = self._raw_env.restore_snapshot(snapshot_id)
    self._last_info = copy.deepcopy(info)
    task_id = info.get('task_id', self._task_id)
    if task_id is not None:
      self._set_goal(task_id)
    self._done = False
    self._append_audit('restore', info, snapshot_id=snapshot_id)
    return self._obs(observation, 0.0, is_first=True)

  def _reset(self):
    goal = self._select_goal()
    options = {
        'goal': goal,
        'reset_seed': int(self._rng.integers(0, np.iinfo(np.int64).max)),
    }
    seed = None
    if not self._started:
      seed = self._seed
      if self._reset_kind == 'lifetime':
        options['reset_kind'] = 'lifetime'
    elif self._reset_kind == 'lifetime':
      seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
      options['reset_kind'] = 'lifetime'
    observation, info = self._env.reset(seed=seed, options=options)
    self._started = True
    self._done = False
    self._last_info = copy.deepcopy(info)
    self._set_goal(info['task_id'])
    if info.get('goal') != self._goal_name:
      raise RuntimeError(
          f'RLScape reset goal mismatch: requested {goal!r}, '
          f'received {info.get("goal")!r}')
    self._append_audit(
        'reset', info, reset_seed=options['reset_seed'],
        reset_kind=options.get('reset_kind', 'task'))
    self._episode_index += 1
    return self._obs(observation, 0.0, is_first=True)

  def _select_goal(self):
    if self._schedule == 'fixed':
      return self._goals[0]
    if self._schedule == 'round_robin':
      return self._goals[self._episode_index % len(self._goals)]
    return self._goals[int(self._rng.integers(len(self._goals)))]

  def _set_goal(self, identifier):
    try:
      goal_id = GOAL_INDEX[str(identifier)]
    except KeyError as exc:
      raise RuntimeError(f'Unknown task ID returned by RLScape: {identifier!r}') from exc
    self._goal_id = int(goal_id)
    self._goal_name = GOAL_NAMES[goal_id]
    self._task_id = TASK_IDS[goal_id]

  def _obs(
      self,
      observation,
      reward,
      *,
      is_first=False,
      is_last=False,
      is_terminal=False,
      is_physical_terminal=False,
      goal_complete=False,
      task_progress=0,
      action=None,
      executed_action=None,
  ):
    observation = np.asarray(observation)
    if observation.dtype != np.uint8 or observation.shape != self._shape:
      raise RuntimeError(
          f'Invalid RLScape frame: expected uint8{self._shape}, '
          f'got {observation.dtype}{observation.shape}')
    if self._goal_id is None:
      raise RuntimeError('RLScape goal is unavailable before reset metadata')
    action_mode = -1
    action_position = np.zeros((2,), np.float32)
    grid_valid = False
    grid_index = 0
    grid_column = 0
    grid_row = 0
    if action is not None:
      if self._action_interface == 'click_grid':
        grid_valid = True
        grid_index = int(action)
        grid_column, grid_row = self._grid_coordinates(grid_index)
        if executed_action is not None:
          action_mode = int(executed_action['mode'])
          action_position = np.asarray(
              executed_action['position'], np.float32)
      else:
        action_mode = int(action['mode'])
        action_position = np.asarray(action['position'], np.float32)
    return {
        'image': observation,
        'reward': np.float32(reward),
        'is_first': bool(is_first),
        'is_last': bool(is_last),
        'is_terminal': bool(is_terminal),
        'is_physical_terminal': bool(is_physical_terminal),
        'goal_complete': bool(goal_complete),
        'goal_id': np.int32(self._goal_id),
        'log/task_success': np.float32(goal_complete),
        'log/task_progress': np.float32(task_progress),
        'log/player_death': np.float32(is_physical_terminal),
        'log/goal_id': np.float32(self._goal_id),
        **{
            f'log/action_mode_{index}': np.float32(action_mode == index)
            for index in range(4)
        },
        'log/action_position_abs': np.float32(np.abs(action_position).mean()),
        'log/action_grid_valid': np.float32(grid_valid),
        'log/action_grid_index': np.float32(grid_index),
        'log/action_grid_column': np.float32(grid_column),
        'log/action_grid_row': np.float32(grid_row),
    }

  def _canonical_policy_action(self, action):
    if self._action_interface == 'mixed':
      return canonical_action(action)
    value = np.asarray(action['action'])
    if value.shape != ():
      raise ValueError(
          f'RLScape click-grid action must be scalar, got {value.shape}')
    if not np.issubdtype(value.dtype, np.integer):
      raise ValueError(
          f'RLScape click-grid action must be integer, got {value.dtype}')
    index = int(value)
    if not 0 <= index < self._grid_actions:
      raise ValueError(
          f'RLScape click-grid action must be in [0, {self._grid_actions}), '
          f'got {index}')
    return index

  def _grid_coordinates(self, action):
    if hasattr(self._env, 'grid_coordinates'):
      column, row = self._env.grid_coordinates(action)
      return int(column), int(row)
    row, column = divmod(int(action), self._grid_columns)
    return column, row

  def _append_audit(self, kind, info, action=None, **extra):
    frame_identity = copy.deepcopy(info.get('frame_identity'))
    row = {
        'kind': kind,
        'goal': self._goal_name,
        'goal_id': self._goal_id,
        'task_id': info.get('task_id', self._task_id),
        'lifetime_id': info.get('lifetime_id'),
        'episode_id': info.get('episode_id'),
        'action_sequence_id': info.get('action_sequence_id'),
        'logical_tick_id': info.get('logical_tick_id'),
        'frame_identity': frame_identity,
        'frame_barrier_tag': info.get('frame_barrier_tag'),
        'task_success': info.get('task_success', False),
        'task_progress': info.get('task_progress', 0),
        'termination_reason': info.get('termination_reason'),
        'truncation_reason': info.get('truncation_reason'),
        'events': copy.deepcopy(info.get('events', [])),
        'executed_action': copy.deepcopy(info.get('executed_action')),
        **copy.deepcopy(extra),
    }
    if action is not None:
      if self._action_interface == 'click_grid':
        column, grid_row = self._grid_coordinates(action)
        row['requested_action'] = {
            'interface': 'click_grid',
            'index': int(action),
            'column': column,
            'row': grid_row,
            'columns': self._grid_columns,
            'rows': self._grid_rows,
        }
      else:
        row['requested_action'] = {
            'mode': int(action['mode']),
            'position': np.asarray(action['position']).tolist(),
        }
    if self._audit_handle is None:
      self._audit.append(row)
    else:
      self._audit_handle.write(json.dumps(
          row, default=self._json_default, sort_keys=True) + '\n')

  @staticmethod
  def _json_default(value):
    if isinstance(value, np.ndarray):
      return value.tolist()
    if isinstance(value, np.generic):
      return value.item()
    raise TypeError(f'Cannot serialize audit value of type {type(value)}')
