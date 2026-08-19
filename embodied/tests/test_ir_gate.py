import math

import jax.numpy as jnp
import numpy as np

from dreamerv3.agent import categorical_js
from dreamerv3.agent import normalized_categorical_entropy
from embodied.run.ir_gate import cheap_decision
from embodied.run.ir_gate import consequence_decision
from embodied.run.ir_gate import incremental_compute


def test_entropy_and_js_separate_shared_uncertainty_from_disagreement():
  uniform = jnp.ones((1, 4, 8), jnp.float32) / 8
  assert np.isclose(float(normalized_categorical_entropy(uniform).mean()), 1)
  assert np.allclose(np.asarray(categorical_js(uniform, axis=1)), 0, atol=1e-6)

  onehots = jnp.eye(4, dtype=jnp.float32)[None]
  assert np.isclose(float(normalized_categorical_entropy(onehots).mean()), 0)
  assert float(np.asarray(categorical_js(onehots, axis=1)).item()) > 1.3


def test_two_stage_gate_short_circuits_and_requires_consequence_evidence():
  low = {'actor_entropy': .2, 'js_disagreement': .1}
  decision = cheap_decision(
      'two_stage', low, entropy_threshold=.5, js_threshold=.4)
  assert not decision.cheap_pass
  assert not decision.needs_consequence
  assert not decision.adapt

  high = {'actor_entropy': .7, 'js_disagreement': .1}
  decision = cheap_decision(
      'two_stage', high, entropy_threshold=.5, js_threshold=.4)
  assert decision.needs_consequence
  rejected = consequence_decision(
      decision, {'success_rate': .01, 'success_action_divergence': .8},
      min_success_rate=.03, direction_threshold=.2)
  accepted = consequence_decision(
      decision, {'success_rate': .2, 'success_action_divergence': .4},
      min_success_rate=.03, direction_threshold=.2)
  assert not rejected.adapt
  assert accepted.adapt


def test_gate_thresholds_are_inclusive_and_compute_saving_is_incremental():
  features = {'actor_entropy': .5, 'js_disagreement': .4}
  assert cheap_decision(
      'entropy', features, entropy_threshold=.5, js_threshold=.9).adapt
  assert cheap_decision(
      'js', features, entropy_threshold=.9, js_threshold=.4).adapt
  assert np.isclose(incremental_compute(10, 30, 20), .5)
  assert math.isnan(incremental_compute(10, 10, 9))
