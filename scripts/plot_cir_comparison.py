#!/usr/bin/env python3
"""Compare matched A/B evaluation milestones across three CIR variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


PHASES = ('Post-WM', 'M1', 'M3', 'M5', 'M10')
LABELS = {
    ('Post-WM', 'A'): 'post_model_repair_a',
    ('Post-WM', 'B'): 'post_model_repair_b',
    **{
        (f'M{step}', task): f'milestone_{step}_{task.lower()}'
        for step in (1, 3, 5, 10) for task in ('A', 'B')
    },
}
STYLES = {
    'WM only': ('#059669', 'o'),
    'Actor only': ('#7c3aed', 's'),
    'Critic first': ('#dc2626', '^'),
}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--wm-only', type=Path, required=True)
  parser.add_argument('--actor-only', type=Path, required=True)
  parser.add_argument('--critic-first', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  return parser.parse_args()


def scores(path):
  with path.open() as file:
    rows = [json.loads(line) for line in file if line.strip()]
  return {
      row['label']: float(row['score']) for row in rows
      if row.get('kind') == 'episode'}


def main():
  args = parse_args()
  runs = {
      'WM only': scores(args.wm_only),
      'Actor only': scores(args.actor_only),
      'Critic first': scores(args.critic_first),
  }

  fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8))
  fig.patch.set_facecolor('#fafafa')
  for ax, task in zip(axes, ('B', 'A')):
    ax.set_facecolor('#ffffff')
    for name, values in runs.items():
      color, marker = STYLES[name]
      ys = [values[LABELS[(phase, task)]] for phase in PHASES]
      ax.plot(
          PHASES, ys, color=color, marker=marker, markersize=8,
          linewidth=2.5, label=name, zorder=2)
      for x, y in zip(PHASES, ys):
        ax.annotate(
            f'{y:.0f}', (x, y), xytext=(0, 8),
            textcoords='offset points', ha='center', fontsize=8.5,
            color=color)
    ax.set_title(f'Task {task}', fontsize=14, weight='bold')
    ax.set_ylabel('Episode score')
    ax.set_xlabel('Frozen evaluation point')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', color='#d1d5db', linewidth=0.8, alpha=0.75)
    ax.spines[['top', 'right']].set_visible(False)

  axes[0].legend(loc='upper right', frameon=True)
  fig.suptitle(
      'CIR Behaviour-Update Ablation — Matched Milestone Scores',
      fontsize=17, weight='bold', y=0.985)
  fig.text(
      0.5, 0.925,
      'All variants use 448 supervised WM updates · individual evaluations '
      'are stochastic single episodes',
      ha='center', color='#4b5563', fontsize=10.5)
  fig.tight_layout(rect=(0, 0, 1, 0.86))
  args.output.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.output, dpi=180, bbox_inches='tight')
  fig.savefig(args.output.with_suffix('.svg'), bbox_inches='tight')
  plt.close(fig)


if __name__ == '__main__':
  main()
