import jax
import jax.numpy as jnp
import numpy as np
import optax

from scripts import minigrid_twin_q_qualification as twin_q

twin_q.jax = jax
twin_q.jnp = jnp
twin_q.optax = optax


def test_counterfactual_event_targets_keep_nonmatching_goal_alive():
  reward, continuation = twin_q.counterfactual_targets(
      actual_goal=np.asarray([[0], [1]]),
      query_goal=np.asarray([[0, 1], [0, 1]]),
      reward=np.asarray([[1.0], [0.0]]),
      goal_complete=np.asarray([[True], [False]]),
      is_physical_terminal=np.asarray([[False], [True]]),
      is_terminal=np.asarray([[True], [True]]),
      is_last=np.asarray([[True], [True]]))

  np.testing.assert_array_equal(reward, [[1, 0], [0, 0]])
  np.testing.assert_array_equal(continuation, [[0, 1], [0, 0]])


def test_return_to_go_starts_with_reward_after_the_action():
  result = twin_q.factual_return_to_go(
      reward=np.asarray([0, -0.1, 1.0], np.float32),
      is_terminal=np.asarray([False, False, True]),
      is_last=np.asarray([False, False, True]), gamma=0.9)

  np.testing.assert_allclose(result, [0.8, 1.0, 0.0])
  np.testing.assert_array_equal(
      twin_q.distance_to_next_reward([0, -0.1, 1.0]), [2, 1, -1])


def test_factual_queries_are_dose_matched_and_counterfactual_queries_both_goals():
  actual = jnp.asarray([0, 1, 1], jnp.int32)

  np.testing.assert_array_equal(
      twin_q.queried_goals(actual, 2, False), [[0, 0], [1, 1], [1, 1]])
  np.testing.assert_array_equal(
      twin_q.queried_goals(actual, 2, True), [[0, 1], [0, 1], [0, 1]])


def test_twin_q_step_bootstraps_without_existing_value_input():
  states, transitions, actions = 8, 6, 3
  data = {
      'deter': jnp.zeros((states, 4), jnp.float16),
      'stoch': jnp.zeros((states, 1, 2), jnp.uint8),
      'actor_probability': jnp.full((states, 2, actions), 1 / actions),
      'state_index': jnp.arange(transitions, dtype=jnp.int32),
      'next_state_index': jnp.arange(1, transitions + 1, dtype=jnp.int32),
      'action': jnp.arange(transitions, dtype=jnp.int32) % actions,
      'actual_goal': jnp.arange(transitions, dtype=jnp.int32) % 2,
      'reward': jnp.asarray([0, 0, 1, 0, 0, 1], jnp.float32),
      'goal_complete': jnp.asarray([0, 0, 1, 0, 0, 1], bool),
      'is_physical_terminal': jnp.zeros(transitions, bool),
      'is_terminal': jnp.asarray([0, 0, 1, 0, 0, 1], bool),
      'is_last': jnp.asarray([0, 0, 1, 0, 0, 1], bool),
  }
  key = jax.random.PRNGKey(0)
  key, first, second = jax.random.split(key, 3)
  params = (
      twin_q.init_q_params(first, 8, 8, actions),
      twin_q.init_q_params(second, 8, 8, actions))
  targets = jax.tree.map(lambda value: value.copy(), params)
  optimizer = optax.adam(1e-3)
  opt_state = optimizer.init(params)
  step = twin_q.make_train_step(
      optimizer, batch_size=4, goal_count=2, gamma=.99,
      target_tau=.01, counterfactual=True)

  updated, _, _, _, metrics = step(
      params, targets, opt_state, key, data)

  assert np.isfinite(np.asarray(metrics['loss']))
  assert not np.array_equal(
      np.asarray(updated[0]['wout']), np.asarray(params[0]['wout']))
