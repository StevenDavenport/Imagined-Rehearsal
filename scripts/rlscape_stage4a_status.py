#!/usr/bin/env python3
"""Print a concise progress report for the resumable Stage 4A workflow."""

from __future__ import annotations

import argparse
import json
import pathlib
import time


def read_json(path: pathlib.Path, default=None):
  return json.loads(path.read_text()) if path.is_file() else default


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('output_root', type=pathlib.Path)
  args = parser.parse_args(argv)
  root = args.output_root.expanduser().resolve()
  spec = read_json(root / 'experiment_spec.json')
  queue = read_json(root / 'queue.json')
  if not spec:
    raise FileNotFoundError(root / 'experiment_spec.json')
  print(f'Stage 4A: {root}')
  print(
      f'checkpoint={spec["checkpoint_step"]:,} '
      f'git={spec.get("git_commit", "unknown")[:12]} '
      f'status={(queue or {}).get("status", "not started")}')
  if not queue:
    return 0

  units = queue.get('units', {})
  counts = {name: 0 for name in ('complete', 'running', 'failed', 'pending')}
  for unit in units.values():
    status = unit.get('status', 'pending')
    counts[status] = counts.get(status, 0) + 1
  print('units: ' + ' / '.join(
      f'{name}={counts.get(name, 0)}'
      for name in ('complete', 'running', 'failed', 'pending')))

  active = queue.get('active_child') or read_json(root / 'active_child.json', {})
  if active:
    age = time.time() - float(active.get('heartbeat_unix', time.time()))
    print(
        f'active: pid={active.get("pid")} heartbeat_age={age:.0f}s '
        f'activity={active.get("saw_activity", False)}')
  running = [name for name, value in units.items()
             if value.get('status') == 'running']
  if running:
    print(f'current unit: {running[-1]}')

  eval_units = {
      name: value for name, value in units.items() if name.startswith('eval/')}
  complete_evals = [
      (name, value['attempts'][-1].get('summary', {}))
      for name, value in eval_units.items() if value.get('status') == 'complete']
  expected_evals = len(spec['evaluation']['conditions']) * 3 * 2
  completed_episodes = sum(
      int(summary.get('episodes', 0)) for _, summary in complete_evals)
  expected_episodes = len(spec['evaluation']['conditions']) * 3 * (
      int(spec['evaluation']['sampled_episodes']) +
      int(spec['evaluation']['deterministic_episodes']))
  print(
      f'evaluation: units={len(complete_evals)}/{expected_evals} '
      f'episodes={completed_episodes}/{expected_episodes}')
  if complete_evals:
    print('latest evaluation results:')
    for name, summary in complete_evals[-6:]:
      print(
          f'  {name.removeprefix("eval/")}: '
          f'{int(summary.get("successes", 0))}/'
          f'{int(summary.get("episodes", 0))} '
          f'({100 * float(summary.get("success_rate", 0)):.1f}%)')
  if (root / 'stage4a_complete').is_file():
    print('terminal marker: stage4a_complete')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
