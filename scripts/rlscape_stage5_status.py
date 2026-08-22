#!/usr/bin/env python3
"""Print concise progress for a Stage 5B or Stage 5C workflow."""

import argparse
import json
import pathlib


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('root', type=pathlib.Path)
  args = parser.parse_args(argv)
  root = args.root.expanduser().resolve()
  queue_path = root / 'queue.json'
  spec_path = root / 'experiment_spec.json'
  if not queue_path.is_file():
    raise FileNotFoundError(queue_path)
  queue = json.loads(queue_path.read_text())
  spec = json.loads(spec_path.read_text()) if spec_path.is_file() else {}
  units = queue.get('units', {})
  counts = {}
  for unit in units.values():
    status = unit.get('status', 'unknown')
    counts[status] = counts.get(status, 0) + 1
  print(f'Stage: {queue.get("canonical_stage", "unknown")}')
  print(f'Status: {queue.get("status", "unknown")}')
  print(f'Units created: {len(units)} {counts}')
  active = queue.get('active_child')
  if active:
    print('Active child:', active.get('pid', active))
  if spec.get('canonical_stage') == 'stage5b':
    selection = root / 'task_selection.json'
    if selection.is_file():
      data = json.loads(selection.read_text())
      print('Strong task:', data['strong']['goal'])
      print('Weak task:', data['weak']['goal'])
    print('Planned confirmation units:',
          spec.get('confirmation', {}).get('units'))
  elif spec.get('canonical_stage') == 'stage5c':
    print('Source:', spec.get('source_kind'))
    print('Target / retained:',
          spec.get('target_goal'), '/', spec.get('retained_goal'))
    print('Recovery budget:', spec.get('recovery_episode_budget'),
          'episodes per trajectory')
  for filename in ('condition_summary.csv', 'trajectory_summary.csv'):
    if (root / filename).is_file():
      print('Current summary:', root / filename)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
