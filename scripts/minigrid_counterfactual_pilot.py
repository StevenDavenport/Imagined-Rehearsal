#!/usr/bin/env python3
"""Run the pinned one-seed MiniGrid counterfactual-head pilot.

This is intentionally a Phase-A-only causal check before committing compute
to a full sequential re-baseline. It changes normal training only by adding
off-goal oracle labels to the reward and continuation heads. The actor, critic,
encoder, decoder, and RSSM objectives remain unchanged, and only one
environment is used.
"""

from __future__ import annotations

import argparse
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import minigrid_sequential_baseline as sequential  # noqa: E402


EXPERIMENT_NAME = 'minigrid_counterfactual_heads_phase_a_pilot'
SCIENTIFIC_ROLE = (
    'one-seed Phase-A causal pilot against the existing standard-training '
    'baseline; add counterfactual reward and continuation head supervision '
    'only, with no critic regularization or actor/critic objective changes')
OBJECTIVES = (
    'test whether oracle off-goal labels make reward and continuation '
    'predictions strongly goal-selective',
    'verify that factual GoToDoor acquisition is not materially degraded',
    'decide whether the intervention merits a full multi-seed sequential '
    're-baseline',
)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Run the pinned counterfactual-head Phase-A pilot.')
  parser.add_argument('--experiment-root', type=pathlib.Path, required=True)
  parser.add_argument('--repo-root', type=pathlib.Path, default=ROOT)
  parser.add_argument(
      '--python', type=pathlib.Path, default=pathlib.Path(sys.executable))
  parser.add_argument('--jax-platform', choices=('cuda', 'cpu'), default='cuda')
  parser.add_argument('--min-free-gb', type=float, default=50.0)
  parser.add_argument('--poll-seconds', type=float, default=30.0)
  parser.add_argument('--startup-timeout', type=float, default=3600.0)
  parser.add_argument('--stall-timeout', type=float, default=7200.0)
  parser.add_argument('--max-attempts', type=int, default=3)
  parser.add_argument('--stream-output', action='store_true')
  parser.add_argument('--dry-run', action='store_true')
  return parser.parse_args(argv)


def forwarded_args(args) -> list[str]:
  values = [
      '--experiment-root', str(args.experiment_root),
      '--repo-root', str(args.repo_root),
      '--python', str(args.python),
      '--experiment-name', EXPERIMENT_NAME,
      '--scientific-role', SCIENTIFIC_ROLE,
      '--primary-objectives', *OBJECTIVES,
      '--extra-train-configs', 'minigrid_counterfactual_heads',
      '--tasks', 'go_to_door',
      '--arms', 'uniform_reservoir',
      '--seeds', '0',
      '--model-size', '12m',
      '--jax-platform', str(args.jax_platform),
      '--phase-steps', '200000',
      '--snapshot-every', '25000',
      '--eval-all-every', '50000',
      '--episode-length', '100',
      '--episodes-per-goal', '1000',
      '--train-ratio', '32',
      '--eval-episodes-per-mode', '50',
      '--policy-modes', 'sampled', 'deterministic',
      '--min-free-gb', str(args.min_free_gb),
      '--poll-seconds', str(args.poll_seconds),
      '--startup-timeout', str(args.startup_timeout),
      '--stall-timeout', str(args.stall_timeout),
      '--max-attempts', str(args.max_attempts),
  ]
  if args.stream_output:
    values.append('--stream-output')
  if args.dry_run:
    values.append('--dry-run')
  return values


def main(argv=None) -> int:
  return sequential.main(forwarded_args(parse_args(argv)))


if __name__ == '__main__':
  raise SystemExit(main())
