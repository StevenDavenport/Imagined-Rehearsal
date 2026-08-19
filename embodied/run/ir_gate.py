"""Pure decision and accounting helpers for Stage 5A IR gates."""

from __future__ import annotations

from dataclasses import dataclass


GATE_KINDS = ('none', 'entropy', 'js', 'two_stage')


@dataclass(frozen=True)
class GateDecision:

  cheap_pass: bool
  needs_consequence: bool
  consequence_pass: bool | None
  adapt: bool


def cheap_decision(kind, features, *, entropy_threshold, js_threshold):
  if kind not in GATE_KINDS:
    raise ValueError(f'Unknown IR gate kind: {kind!r}')
  entropy = float(features['actor_entropy'])
  disagreement = float(features['js_disagreement'])
  if kind == 'none':
    return GateDecision(True, False, None, True)
  if kind == 'entropy':
    passed = entropy >= float(entropy_threshold)
    return GateDecision(passed, False, None, passed)
  if kind == 'js':
    passed = disagreement >= float(js_threshold)
    return GateDecision(passed, False, None, passed)
  passed = (
      entropy >= float(entropy_threshold) or
      disagreement >= float(js_threshold))
  return GateDecision(passed, passed, None, False)


def consequence_decision(
    cheap, features, *, min_success_rate, direction_threshold):
  if not cheap.needs_consequence:
    return cheap
  passed = (
      float(features['success_rate']) >= float(min_success_rate) and
      float(features['success_action_divergence']) >=
      float(direction_threshold))
  return GateDecision(
      cheap_pass=cheap.cheap_pass,
      needs_consequence=True,
      consequence_pass=passed,
      adapt=passed)


def incremental_compute(no_ir, always_ir, gated):
  """Return the fraction of avoidable always-on overhead saved by gating."""
  denominator = float(always_ir) - float(no_ir)
  if denominator <= 0:
    return float('nan')
  return 1.0 - ((float(gated) - float(no_ir)) / denominator)
