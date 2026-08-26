#!/usr/bin/env python3
"""Print concise progress for a Stage 5B or Stage 5C workflow."""

import argparse
from collections import defaultdict
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
  elif spec.get('canonical_stage') == 'stage5b2a':
    conditions = spec.get('conditions', [])
    replicates = int(spec.get('replicates', 0))
    print('Goal / mode:', spec.get('goal'), '/', spec.get('policy_mode'))
    print('Planned units:', len(conditions) * replicates)
    partial = defaultdict(lambda: [0, 0, 0])
    for unit in units.values():
      if unit.get('status') != 'complete' or not unit.get('attempts'):
        continue
      summary = unit['attempts'][-1].get('summary', {})
      condition = summary.get('condition')
      if not condition:
        continue
      partial[condition][0] += int(summary.get('successes', 0))
      partial[condition][1] += int(summary.get('episodes', 0))
      partial[condition][2] += 1
    if partial:
      print('Partial sampled results:')
      for condition in conditions:
        if condition not in partial:
          continue
        successes, episodes, completed = partial[condition]
        rate = successes / episodes if episodes else float('nan')
        print(
            f'  {condition}: {successes}/{episodes} = {rate:.3f} '
            f'({completed}/{replicates} replicates)')
  for filename in ('condition_summary.csv', 'trajectory_summary.csv'):
    if (root / filename).is_file():
      print('Current summary:', root / filename)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
