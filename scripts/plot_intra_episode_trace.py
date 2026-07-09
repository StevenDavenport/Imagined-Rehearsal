#!/usr/bin/env python3
"""Plot intra-episode rehearsal traces with variance bands."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--trace_glob', required=True)
  parser.add_argument('--out', required=True)
  parser.add_argument('--title', default='')
  parser.add_argument('--max_episode_index', type=int, default=0)
  parser.add_argument('--band', choices=('none', 'std', 'ci95', 'iqr'), default='std')
  parser.add_argument('--metric', choices=('reward', 'return_so_far'), default='reward')
  parser.add_argument('--show_seeds', action='store_true')
  return parser.parse_args()


def load_trace(path: Path, max_episode_index: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
  episode_steps = []
  values = []
  with path.open() as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      row = json.loads(line)
      if int(row.get('episode_index', 0)) != max_episode_index:
        continue
      episode_steps.append(int(row['episode_step']))
      values.append(float(row[metric]))
  return np.asarray(episode_steps, dtype=int), np.asarray(values, dtype=float)


def main() -> int:
  args = parse_args()
  paths = sorted(Path().glob(args.trace_glob))
  if not paths:
    raise SystemExit(f'No trace files matched: {args.trace_glob}')

  traces = []
  min_len = math.inf
  for path in paths:
    steps, values = load_trace(path, args.max_episode_index, args.metric)
    if len(steps) == 0:
      continue
    traces.append((path, steps, values))
    min_len = min(min_len, len(steps))
  if not traces:
    raise SystemExit('Matched trace files, but no episode data found.')

  min_len = int(min_len)
  x = traces[0][1][:min_len]
  ys = np.stack([ret[:min_len] for _, _, ret in traces], axis=0)
  mean = ys.mean(axis=0)
  lower = upper = band_label = None
  if args.band == 'std':
    spread = ys.std(axis=0, ddof=1) if ys.shape[0] > 1 else np.zeros_like(mean)
    lower = mean - spread
    upper = mean + spread
    band_label = 'mean ± std'
  elif args.band == 'ci95':
    spread = (1.96 * ys.std(axis=0, ddof=1) / np.sqrt(ys.shape[0])
              if ys.shape[0] > 1 else np.zeros_like(mean))
    lower = mean - spread
    upper = mean + spread
    band_label = 'mean ± 95% CI'
  elif args.band == 'iqr':
    lower = np.quantile(ys, 0.25, axis=0)
    upper = np.quantile(ys, 0.75, axis=0)
    band_label = 'IQR'

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)

  fig, ax = plt.subplots(figsize=(8.2, 4.8))
  if args.show_seeds:
    for _, _, vals in traces:
      ax.plot(x, vals[:min_len], color='#9ecae1', alpha=0.35, linewidth=1.0)
  ax.plot(x, mean, color='#08519c', linewidth=2.5, label='seed mean')
  if lower is not None and upper is not None:
    ax.fill_between(x, lower, upper, color='#6baed6', alpha=0.25, label=band_label)
  ax.set_xlabel('Episode step')
  ylabel = 'Reward' if args.metric == 'reward' else 'Cumulative return'
  ax.set_ylabel(ylabel)
  ax.set_title(args.title or f'Intra-episode {ylabel.lower()} trace')
  ax.grid(True, alpha=0.25)
  ax.legend()
  fig.tight_layout()
  fig.savefig(out, dpi=170)
  plt.close(fig)

  print(f'Wrote: {out}')
  print(f'Traces: {len(traces)}')
  print(f'Points per trace: {min_len}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
