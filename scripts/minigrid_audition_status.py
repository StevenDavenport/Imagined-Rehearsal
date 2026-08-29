#!/usr/bin/env python3
"""Print a compact progress snapshot for a MiniGrid audition run."""

from __future__ import annotations

import argparse
import json
import pathlib


def snapshot(experiment_root: pathlib.Path) -> dict:
  root = pathlib.Path(experiment_root).expanduser().resolve()
  spec = json.loads((root / 'experiment_spec.json').read_text())
  status = json.loads((root / 'supervisor_status.json').read_text())
  states = status.get('states', {})
  training_total = len(spec['tasks']) * len(spec['seeds'])
  eval_total = (
      training_total * len(spec['checkpoint_steps']) *
      len(spec['policy_modes']))
  training_complete = sum(
      state.get('training_status') == 'complete'
      for state in states.values())
  evaluation_states = [
      condition
      for state in states.values()
      for condition in state.get('evaluations', {}).values()]
  evaluation_complete = sum(
      condition.get('status') == 'complete'
      for condition in evaluation_states)
  failed = [
      unit for unit, state in states.items()
      if state.get('training_status') == 'failed' or
      any(condition.get('status') == 'failed'
          for condition in state.get('evaluations', {}).values())]
  active = status.get('active_child')
  active_path = root / 'active_child.json'
  if active_path.is_file():
    record = json.loads(active_path.read_text())
    active = {**record, **(active or {})}
  return {
      'status': status.get('status', 'unknown'),
      'training': f'{training_complete}/{training_total}',
      'evaluations': f'{evaluation_complete}/{eval_total}',
      'failed_units': failed,
      'active_pid': active.get('pid') if active else None,
      'active_command': active.get('command') if active else None,
      'updated_unix': status.get('updated_unix'),
      'analysis_ready': (root / 'analysis' / 'summary.json').is_file(),
  }


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Show controlled MiniGrid audition progress.')
  parser.add_argument('--experiment-root', type=pathlib.Path, required=True)
  return parser.parse_args(argv)


def main(argv=None):
  result = snapshot(parse_args(argv).experiment_root)
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
