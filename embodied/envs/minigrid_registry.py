"""Pure-data registry for the controlled MiniGrid audition."""

from __future__ import annotations


TASK_SPECS = (
    {
        'name': 'go_to_door',
        'goal_id': 0,
        'family': 'navigation',
        'mission': 'go to the red door',
        'source': 'MiniGrid-GoToDoor-8x8-v0',
    },
    {
        'name': 'fetch',
        'goal_id': 1,
        'family': 'pickup',
        'mission': 'fetch the blue ball',
        'source': 'MiniGrid-Fetch-8x8-N3-v0',
    },
    {
        'name': 'go_to_object',
        'goal_id': 2,
        'family': 'navigation',
        'mission': 'go to the green key',
        'source': 'MiniGrid-GoToObject-8x8-N2-v0',
    },
    {
        'name': 'put_near',
        'goal_id': 3,
        'family': 'pickup_and_place',
        'mission': 'put the yellow ball near the blue box',
        'source': 'MiniGrid-PutNear-6x6-N2-v0',
    },
    {
        'name': 'unlock_pickup',
        'goal_id': 4,
        'family': 'unlock_and_pickup',
        'mission': 'pick up the only box behind the locked door',
        'source': 'MiniGrid-UnlockPickup-v0',
    },
    {
        'name': 'pickup_dist',
        'goal_id': 5,
        'family': 'pickup',
        'mission': 'pick up the grey box',
        'source': 'BabyAI-PickupDist-v0 analogue',
    },
)

TASK_NAMES = tuple(spec['name'] for spec in TASK_SPECS)
TASK_INDEX = {spec['name']: spec['goal_id'] for spec in TASK_SPECS}
TASK_FAMILIES = {spec['name']: spec['family'] for spec in TASK_SPECS}
TASK_BY_NAME = {spec['name']: spec for spec in TASK_SPECS}

ALTERNATING_CHAIN = (
    'go_to_door', 'fetch', 'go_to_object', 'put_near')
PROGRESSIVE_CHAIN = ('fetch', 'put_near', 'unlock_pickup')


assert tuple(TASK_INDEX.values()) == tuple(range(len(TASK_SPECS)))
