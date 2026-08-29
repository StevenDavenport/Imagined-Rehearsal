"""Controlled fixed-mission tasks built on the MiniGrid primitives.

The stock GoTo, Fetch, PutNear, and BabyAI environments communicate a
randomly sampled episode goal through natural language.  The first CIR
audition deliberately keeps language outside the experiment: every task has
one stable, visually identifiable goal while layouts and starting states are
still randomized.  This module is imported lazily by the Embodied adapter so
MiniGrid remains an optional dependency for all other suites.
"""

from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Door, Key
from minigrid.envs.unlockpickup import UnlockPickupEnv
from minigrid.minigrid_env import MiniGridEnv


def _mission_space(text: str) -> MissionSpace:
  return MissionSpace(mission_func=lambda: text)


def _adjacent(left, right) -> bool:
  return bool(
      (left[0] == right[0] and abs(left[1] - right[1]) == 1) or
      (left[1] == right[1] and abs(left[0] - right[0]) == 1))


class _SingleRoom(MiniGridEnv):

  def __init__(self, mission: str, *, size: int, max_steps: int, **kwargs):
    self.size = int(size)
    if self.size < 5:
      raise ValueError('Controlled MiniGrid rooms must be at least 5x5')
    super().__init__(
        mission_space=_mission_space(mission),
        width=self.size,
        height=self.size,
        max_steps=int(max_steps),
        see_through_walls=True,
        **kwargs,
    )
    self._fixed_mission = mission

  def _empty_room(self, width, height):
    self.grid = Grid(width, height)
    self.grid.wall_rect(0, 0, width, height)


class GoToDoorFixedEnv(_SingleRoom):
  """Find the uniquely red door and use the ``done`` action beside it."""

  def __init__(self, *, size=8, max_steps=100, **kwargs):
    super().__init__(
        'go to the red door', size=size, max_steps=max_steps, **kwargs)

  def _gen_grid(self, width, height):
    self._empty_room(width, height)
    positions = [
        (self._rand_int(2, width - 2), 0),
        (self._rand_int(2, width - 2), height - 1),
        (0, self._rand_int(2, height - 2)),
        (width - 1, self._rand_int(2, height - 2)),
    ]
    colors = ['red', 'green', 'blue', 'purple']
    colors = self._rand_subset(colors, len(colors))
    for position, color in zip(positions, colors):
      self.grid.set(*position, Door(color))
      if color == 'red':
        self.target_pos = position
    self.place_agent(size=(width, height))
    self.mission = self._fixed_mission

  def step(self, action):
    obs, reward, terminated, truncated, info = super().step(action)
    if action == self.actions.toggle:
      terminated = True
    if action == self.actions.done:
      if _adjacent(tuple(self.agent_pos), self.target_pos):
        reward = self._reward()
      terminated = True
    return obs, reward, terminated, truncated, info


class GoToObjectFixedEnv(_SingleRoom):
  """Find the uniquely green key and use ``done`` beside it."""

  def __init__(self, *, size=8, max_steps=100, **kwargs):
    super().__init__(
        'go to the green key', size=size, max_steps=max_steps, **kwargs)

  def _gen_grid(self, width, height):
    self._empty_room(width, height)
    target = Key('green')
    self.target_pos = self.place_obj(target)
    self.place_obj(Box('purple'))
    self.place_agent()
    self.mission = self._fixed_mission

  def step(self, action):
    obs, reward, terminated, truncated, info = super().step(action)
    if action in (self.actions.toggle, self.actions.pickup):
      terminated = True
    if action == self.actions.done:
      if _adjacent(tuple(self.agent_pos), self.target_pos):
        reward = self._reward()
      terminated = True
    return obs, reward, terminated, truncated, info


class _PickupFixed(_SingleRoom):

  mission = ''
  target_factory = None
  distractor_factories = ()

  def __init__(self, *, size=8, max_steps=100, **kwargs):
    super().__init__(
        self.mission, size=size, max_steps=max_steps, **kwargs)

  def _gen_grid(self, width, height):
    self._empty_room(width, height)
    self.target = self.target_factory()
    self.target_pos = self.place_obj(self.target)
    for factory in self.distractor_factories:
      self.place_obj(factory())
    self.place_agent()
    self.mission = self._fixed_mission

  def step(self, action):
    obs, reward, terminated, truncated, info = super().step(action)
    if action == self.actions.pickup and self.carrying is not None:
      if self.carrying is self.target:
        reward = self._reward()
      else:
        reward = 0.0
      terminated = True
    return obs, reward, terminated, truncated, info


class FetchFixedEnv(_PickupFixed):
  """Fetch a blue ball among two visually distinct distractors."""

  mission = 'fetch the blue ball'
  target_factory = staticmethod(lambda: Ball('blue'))
  distractor_factories = (
      lambda: Key('red'),
      lambda: Ball('green'),
  )


class PickupDistFixedEnv(_PickupFixed):
  """Pickup-with-distractors analogue of BabyAI PickupDist."""

  mission = 'pick up the grey box'
  target_factory = staticmethod(lambda: Box('grey'))
  distractor_factories = (
      lambda: Ball('red'),
      lambda: Key('blue'),
      lambda: Box('green'),
      lambda: Ball('purple'),
  )


class PutNearFixedEnv(_SingleRoom):
  """Move the yellow ball next to the blue box."""

  def __init__(self, *, size=6, max_steps=100, **kwargs):
    super().__init__(
        'put the yellow ball near the blue box',
        size=size, max_steps=max_steps, **kwargs)

  def _gen_grid(self, width, height):
    self._empty_room(width, height)
    positions = []

    def reject_near(_env, position):
      return any(
          abs(position[0] - other[0]) <= 1 and
          abs(position[1] - other[1]) <= 1
          for other in positions)

    self.move_obj = Ball('yellow')
    self.move_pos = self.place_obj(self.move_obj)
    positions.append(self.move_pos)
    self.target_obj = Box('blue')
    self.target_pos = self.place_obj(
        self.target_obj, reject_fn=reject_near)
    self.place_agent()
    self.mission = self._fixed_mission

  def step(self, action):
    before = self.carrying
    obs, reward, terminated, truncated, info = super().step(action)
    if (
        action == self.actions.pickup and self.carrying is not None and
        self.carrying is not self.move_obj
    ):
      terminated = True
    if action == self.actions.drop and before is not None:
      if before is self.move_obj and self.carrying is None:
        position = tuple(self.move_obj.cur_pos)
        target = tuple(self.target_obj.cur_pos)
        if (
            abs(position[0] - target[0]) <= 1 and
            abs(position[1] - target[1]) <= 1
        ):
          reward = self._reward()
      terminated = True
    return obs, reward, terminated, truncated, info


class UnlockPickupFixedEnv(UnlockPickupEnv):
  """Stock UnlockPickup geometry with a functionally fixed mission."""

  def _gen_grid(self, width, height):
    super()._gen_grid(width, height)
    self.mission = 'pick up the only box behind the locked door'


def make_task(task: str, *, episode_length: int):
  """Construct one controlled task with the shared seven-action interface."""
  task = str(task)
  kwargs = {'max_steps': int(episode_length), 'render_mode': 'rgb_array'}
  constructors = {
      'go_to_door': GoToDoorFixedEnv,
      'fetch': FetchFixedEnv,
      'go_to_object': GoToObjectFixedEnv,
      'put_near': PutNearFixedEnv,
      'unlock_pickup': UnlockPickupFixedEnv,
      'pickup_dist': PickupDistFixedEnv,
  }
  try:
    constructor = constructors[task]
  except KeyError as exc:
    raise ValueError(
        f'Unknown controlled MiniGrid task {task!r}; expected one of '
        f'{tuple(constructors)}') from exc
  return constructor(**kwargs)
