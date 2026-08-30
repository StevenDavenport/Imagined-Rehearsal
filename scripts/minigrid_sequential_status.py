#!/usr/bin/env python3
"""Print a compact health/progress snapshot for the sequential baseline."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.minigrid_common import checkpoint_step
from scripts.rlscape_m3_process import record_is_live


def snapshot(experiment_root: pathlib.Path) -> dict:
  root = pathlib.Path(experiment_root).expanduser().resolve()
  spec = json.loads((root / 'experiment_spec.json').read_text())
  status = json.loads((root / 'supervisor_status.json').read_text())
  states = status.get('states', {})
  runs = len(spec['arms']) * len(spec['seeds'])
  total_steps = int(spec['total_steps_per_run']) * runs
  observed_steps = 0
  training_phases_complete = 0
  training_phases_total = runs * len(spec['tasks'])
  eval_complete = 0
  eval_total = (
      int(spec['evaluation_units_per_run']) * runs
      if spec.get('evaluation_enabled', True) else 0)
  failures = []
  for arm in spec['arms']:
    for seed in spec['seeds']:
      name = f'{arm}__seed_{seed:04d}'
      state = states.get(name, {})
      training = root / 'runs' / arm / f'seed_{seed:04d}' / 'training'
      observed_steps += min(checkpoint_step(training), spec['total_steps_per_run'])
      training_phases_complete += sum(
          phase.get('status') == 'complete'
          for phase in state.get('phases', {}).values())
      eval_complete += sum(
          condition.get('status') == 'complete'
          for condition in state.get('evaluations', {}).values())
      if any(
          phase.get('status') == 'failed'
          for phase in state.get('phases', {}).values()) or any(
              condition.get('status') == 'failed'
              for condition in state.get('evaluations', {}).values()):
        failures.append(name)
  diagnostic_states = status.get('diagnostics', {})
  diagnostic_jobs = [
      value for key, value in diagnostic_states.items()
      if not key.startswith('bank__')]
  diagnostic_total = (
      2 * runs * len(spec['phase_boundaries'])
      if spec.get('diagnostics_enabled', True) else 0)
  diagnostic_complete = sum(
      value.get('status') == 'complete' for value in diagnostic_jobs)
  active = status.get('active_child')
  active_path = root / 'active_child.json'
  stale_active_record = False
  if active_path.is_file():
    record = json.loads(active_path.read_text())
    if record_is_live(record):
      active = {**record, **(active or {})}
    else:
      active = None
      stale_active_record = True
  else:
    # A heartbeat without the process-identity record is historical.
    active = None
  elapsed = max(0.0, time.time() - float(status.get('created_unix', time.time())))
  completed_units = (
      training_phases_complete + eval_complete + diagnostic_complete)
  total_units = training_phases_total + eval_total + diagnostic_total
  remaining_seconds = (
      elapsed / completed_units * (total_units - completed_units)
      if completed_units and total_units > completed_units else 0.0)
  return {
      'status': status.get('status', 'unknown'),
      'health': 'failed' if failures else ('active' if active else 'idle'),
      'training_steps': f'{observed_steps}/{total_steps}',
      'training_percent': 100 * observed_steps / max(1, total_steps),
      'training_phases': f'{training_phases_complete}/{training_phases_total}',
      'evaluations': f'{eval_complete}/{eval_total}',
      'diagnostics': f'{diagnostic_complete}/{diagnostic_total}',
      'overall_unit_percent': 100 * completed_units / max(1, total_units),
      'elapsed_hours': elapsed / 3600,
      'naive_remaining_hours': remaining_seconds / 3600,
      'failed_runs': failures,
      'stale_active_record': stale_active_record,
      'active_pid': active.get('pid') if active else None,
      'active_command': active.get('command') if active else None,
      'updated_unix': status.get('updated_unix'),
      'analysis_ready': (root / 'analysis' / 'summary.json').is_file(),
  }


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Show sequential MiniGrid baseline progress.')
  parser.add_argument('--experiment-root', type=pathlib.Path, required=True)
  return parser.parse_args(argv)


def main(argv=None):
  print(json.dumps(snapshot(parse_args(argv).experiment_root),
                   indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
