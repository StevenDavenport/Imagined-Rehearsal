"""Pure decision and accounting helpers for Stage 5A IR gates."""

from __future__ import annotations

from dataclasses import dataclass


GATE_KINDS = ('none', 'entropy', 'js', 'two_stage', 'virtual_update')


@dataclass(frozen=True)
class GateDecision:

  cheap_pass: bool
  needs_consequence: bool
  consequence_pass: bool | None
  adapt: bool


@dataclass(frozen=True)
class VirtualUpdateDecision:

  baseline_score: float
  candidate_score: float
  improvement: float
  baseline_pass: bool
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
  if kind == 'virtual_update':
    raise ValueError(
        'virtual_update requires a paired tentative actor update')
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


def virtual_update_decision(
    baseline_score, candidate_score, *, min_improvement=0.0,
    max_baseline_score=float('inf')):
  """Decide whether to commit a tentative IR update.

  The baseline screen represents the prospective "will I underperform?"
  question. Setting it to infinity keeps the screen in audit-only mode. The
  paired score difference then asks the intervention question directly:
  "does this exact IR update improve imagined performance?"
  """
  baseline = float(baseline_score)
  candidate = float(candidate_score)
  improvement = candidate - baseline
  baseline_pass = baseline <= float(max_baseline_score)
  return VirtualUpdateDecision(
      baseline_score=baseline,
      candidate_score=candidate,
      improvement=improvement,
      baseline_pass=baseline_pass,
      adapt=(baseline_pass and improvement > float(min_improvement)))


def incremental_compute(no_ir, always_ir, gated):
  """Return the fraction of avoidable always-on overhead saved by gating."""
  denominator = float(always_ir) - float(no_ir)
  if denominator <= 0:
    return float('nan')
  return 1.0 - ((float(gated) - float(no_ir)) / denominator)
