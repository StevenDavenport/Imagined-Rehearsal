#!/usr/bin/env python3
"""Print concise progress or final endpoints for an RLScape Stage 4B run."""

import argparse
import json
import pathlib


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('run_root', type=pathlib.Path)
  args = parser.parse_args(argv)
  root = args.run_root.expanduser().resolve()
  queue_path = root / 'queue.json'
  if not queue_path.is_file():
    raise FileNotFoundError(queue_path)
  queue = json.loads(queue_path.read_text())
  spec = json.loads((root / 'experiment_spec.json').read_text())
  expected = (
      int(spec['replicates']) * len(spec['conditions']) *
      len(spec['goals']) * len(spec['policy_modes']))
  units = queue.get('units', {})
  counts = {}
  for unit in units.values():
    status = unit.get('status', 'unknown')
    counts[status] = counts.get(status, 0) + 1
  print(f'Stage 4B: {queue.get("status", "unknown")}')
  print(f'Units: {sum(counts.values())}/{expected} {counts}')
  active = queue.get('active_child')
  if active:
    print('Active:', active.get('command', active))
  summary_path = root / 'aggregate_summary.json'
  if summary_path.is_file():
    rows = json.loads(summary_path.read_text())['condition_rows']
    print('\ncondition,mode,mean,worst,delta_vs_reward,worst_delta')
    for row in rows:
      print(','.join((
          row['condition'], row['policy_mode'],
          f'{row["mean_goal_success"]:.3f}',
          f'{row["worst_goal_success"]:.3f}',
          f'{row["mean_goal_delta_vs_reward_only"]:+.3f}',
          f'{row["worst_goal_delta_vs_reward_only"]:+.3f}')))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
