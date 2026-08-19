#!/usr/bin/env python3
"""Print a compact status summary for the Stage 5A gating workflow."""

import argparse
import json
import pathlib


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('root', type=pathlib.Path)
  args = parser.parse_args(argv)
  queue_path = args.root.expanduser().resolve() / 'queue.json'
  if not queue_path.is_file():
    raise FileNotFoundError(queue_path)
  queue = json.loads(queue_path.read_text())
  states = [value.get('status', 'unknown') for value in queue.get('units', {}).values()]
  counts = {status: states.count(status) for status in sorted(set(states))}
  print(f"Stage: {queue.get('canonical_stage', 'unknown')}")
  print(f"Status: {queue.get('status', 'unknown')}")
  print(f'Units: {len(states)} {counts}')
  active = queue.get('active_child')
  if active:
    print('Active child:', active.get('pid', active))
  selected = args.root / 'selected_gates.json'
  if selected.is_file():
    data = json.loads(selected.read_text())
    print('Selected gate digest:', data.get('digest'))
    for kind in ('entropy', 'js', 'two_stage'):
      print(f'  {kind}: {data.get(kind)}')
  summary = args.root / 'condition_summary.csv'
  if summary.is_file():
    print('Confirmation summary:', summary)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
