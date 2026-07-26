import jax.numpy as jnp
import numpy as np

from dreamerv3.agent import Agent
from dreamerv3.agent import select_policy_action


class _Distribution:

  def __init__(self, prediction):
    self.prediction = prediction

  def pred(self):
    return self.prediction


def test_eval_policy_uses_deterministic_distribution_predictions():
  policy = {
      'action': _Distribution(jnp.asarray([3, 1], jnp.int32)),
  }

  action = select_policy_action(policy, 'eval')

  np.testing.assert_array_equal(action['action'], [3, 1])


def test_head_input_is_deter_stoch_goal_onehot():
  agent = object.__new__(Agent)
  agent.goal_enabled = True
  agent.goal_count = 5
  agent.goal_key = 'goal_id'
  agent.feat2tensor = lambda feat: jnp.concatenate([
      feat['deter'],
      feat['stoch'].reshape((*feat['stoch'].shape[:-2], -1)),
  ], -1)
  feat = {
      'deter': jnp.asarray([[1, 2, 3], [4, 5, 6]], jnp.float32),
      'stoch': jnp.asarray([
          [[7, 8], [9, 10]],
          [[11, 12], [13, 14]],
      ], jnp.float32),
  }

  actual = np.asarray(agent._head_input(feat, jnp.asarray([1, 4])))
  expected = np.asarray([
      [1, 2, 3, 7, 8, 9, 10, 0, 1, 0, 0, 0],
      [4, 5, 6, 11, 12, 13, 14, 0, 0, 0, 0, 1],
  ])

  np.testing.assert_array_equal(actual, expected)


def test_head_input_without_goal_matches_original_dreamer_features():
  agent = object.__new__(Agent)
  agent.goal_enabled = False
  agent.feat2tensor = lambda feat: feat['deter']
  deter = jnp.asarray([[1, 2, 3]], jnp.float32)

  actual = agent._head_input({'deter': deter})

  np.testing.assert_array_equal(actual, deter)
