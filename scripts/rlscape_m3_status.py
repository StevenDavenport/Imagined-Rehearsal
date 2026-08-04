#!/usr/bin/env python3
"""Print a concise, single-shot status for RLScape Experiment 1."""

from __future__ import annotations

import argparse
from collections import deque
import json
import pathlib
import shutil
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rlscape_m3_sequential import checkpoint_step  # noqa: E402


TOTAL_STEPS = 2_500_000
TOTAL_MILESTONES = 10
TOTAL_EVAL_PAIRS = 60


def read_json(path: pathlib.Path) -> dict:
  return json.loads(path.read_text()) if path.exists() else {}


def recent_episodes(path: pathlib.Path, limit=100) -> list[dict]:
  if not path.exists():
    return []
  with path.open() as handle:
    lines = deque((line for line in handle if line.strip()), maxlen=limit)
  return [json.loads(line) for line in lines]


def collect(root: pathlib.Path) -> dict:
  experiment = read_json(root / 'experiment_spec.json')
  supervisor = read_json(root / 'supervisor_status.json')
  queue = read_json(root / 'evaluations' / 'queue.json')
  states = queue.get('states', {})
  complete_pairs = sum(
      state.get('status') == 'complete' for state in states.values())
  failed_pairs = sum(
      state.get('status') == 'failed' for state in states.values())
  milestones = list((root / 'milestones').glob(
      'step_*/archive_complete')) if (root / 'milestones').exists() else []
  episodes = recent_episodes(root / 'training' / 'episodes.jsonl')
  successes = [
      bool(
          row.get('episode/score', 0) > 0 or
          row.get('episode/log/task_success/max', 0) > 0)
      for row in episodes
  ]
  if (root / 'active_child.json').exists():
    active = supervisor.get('active_child') or {}
  elif (root / 'evaluations' / 'active_child.json').exists():
    active = queue.get('active_child') or {}
  else:
    active = {}
  free_gb = (
      shutil.disk_usage(root).free / 1024 ** 3 if root.exists() else None)
  training_target = int(experiment.get('total_steps', TOTAL_STEPS))
  milestone_target = len(experiment.get('milestones', ())) or TOTAL_MILESTONES
  evaluation_target = (
      len(queue.get('spec', {}).get('units', ())) or
      sum(4 * (phase + 1)
          for phase in range(len(experiment.get('goals', ())))) or
      TOTAL_EVAL_PAIRS)
  return {
      'status': supervisor.get('status', 'not_started'),
      'training_step': checkpoint_step(root / 'training'),
      'training_target': training_target,
      'milestones': len(milestones),
      'milestone_target': milestone_target,
      'recent_episodes': len(episodes),
      'recent_success_rate': (
          sum(successes) / len(successes) if successes else None),
      'total_restarts': supervisor.get('total_restarts', 0),
      'evaluation_status': queue.get('status', 'not_started'),
      'evaluation_pairs_complete': complete_pairs,
      'evaluation_pairs_failed': failed_pairs,
      'evaluation_pairs_target': evaluation_target,
      'active_pid': active.get('pid'),
      'active_adopted': active.get('adopted'),
      'last_activity_age_seconds': (
          time.time() - active['last_activity_unix']
          if active.get('last_activity_unix') else None),
      'free_disk_gb': free_gb,
      'updated_unix': supervisor.get(
          'updated_unix', queue.get('updated_unix')),
  }


def main(argv=None) -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment-root', type=pathlib.Path, required=True)
  parser.add_argument('--json', action='store_true')
  args = parser.parse_args(argv)
  result = collect(args.experiment_root.expanduser().resolve())
  if args.json:
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
  step = result['training_step']
  target = result['training_target']
  rate = result['recent_success_rate']
  print(
      f"status={result['status']}  "
      f'train={step:,}/{target:,} ({100 * step / target:.1f}%)  '
      f"milestones={result['milestones']}/{result['milestone_target']}")
  print(
      f"last{result['recent_episodes']}_success="
      f'{rate:.1%}' if rate is not None else 'recent_success=unavailable',
      f"restarts={result['total_restarts']}  "
      f"disk_free={result['free_disk_gb']:.1f}GiB")
  print(
      f"eval={result['evaluation_status']}  "
      f"pairs={result['evaluation_pairs_complete']}/"
      f"{result['evaluation_pairs_target']}  "
      f"failed={result['evaluation_pairs_failed']}")
  if result['active_pid']:
    age = result['last_activity_age_seconds']
    print(
        f"active_pid={result['active_pid']}  adopted="
        f"{result['active_adopted']}  last_activity={age:.0f}s_ago")
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
