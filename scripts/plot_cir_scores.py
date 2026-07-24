#!/usr/bin/env python3
"""Plot chronological A/B episode scores from a CIR events file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


COLORS = {'A': '#2563eb', 'B': '#ea580c'}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('events', type=Path)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument(
      '--title', default='CIR pilot: score by chronological episode')
  return parser.parse_args()


def domain(label):
  return 'A' if label.startswith('baseline_a_') or label.endswith('_a') else 'B'


def short_label(label):
  replacements = (
      ('baseline_a_', 'A baseline '),
      ('shock_b_', 'B shock '),
      ('post_model_repair_', 'Post-WM '),
      ('cir_b_train_', 'B train '),
      ('wm_b_collect_', 'B collect '),
      ('milestone_', 'M'),
  )
  for source, target in replacements:
    if label.startswith(source):
      result = target + label[len(source):]
      if source == 'milestone_':
        step, task = result[1:].rsplit('_', 1)
        result = f'M{step} {task.upper()}'
      return result
  return label


def main():
  args = parse_args()
  with args.events.open() as file:
    rows = [json.loads(line) for line in file if line.strip()]
  episodes = [row for row in rows if row.get('kind') == 'episode']
  if not episodes:
    raise RuntimeError(f'No episode events found in {args.events}')
  wm_only = any(
      row['label'].startswith('wm_b_collect_') for row in episodes)

  xs = list(range(1, len(episodes) + 1))
  labels = [short_label(row['label']) for row in episodes]

  fig, ax = plt.subplots(figsize=(18, 8.5))
  fig.patch.set_facecolor('#fafafa')
  ax.set_facecolor('#ffffff')

  spans = (
      (0.5, 3.5, '#dbeafe', 'Baseline A'),
      (3.5, 5.5, '#ffedd5', 'Shock B'),
      (5.5, 7.5, '#dcfce7', 'Post-WM evaluation'),
      (7.5, len(episodes) + 0.5, '#f3e8ff',
       ('WM-only collection + episodic WM training' if wm_only else
        'Actor-only IR + episodic WM training')),
  )
  for left, right, color, text in spans:
    ax.axvspan(left, right, color=color, alpha=0.52, zorder=0)
    ax.text(
        (left + right) / 2, 1.015, text, transform=ax.get_xaxis_transform(),
        ha='center', va='bottom', fontsize=10, color='#374151')

  for task in ('A', 'B'):
    points = [
        (x, float(row['score'])) for x, row in zip(xs, episodes)
        if domain(row['label']) == task]
    ax.plot(
        [point[0] for point in points], [point[1] for point in points],
        color=COLORS[task], linewidth=2.2, alpha=0.72, zorder=2)

  for x, row in zip(xs, episodes):
    task = domain(row['label'])
    evaluation = row['label'].startswith(('post_model_repair_', 'milestone_'))
    marker = 'D' if evaluation else 'o'
    size = 80 if evaluation else 66
    score = float(row['score'])
    ax.scatter(
        x, score, s=size, marker=marker, color=COLORS[task],
        edgecolor='white', linewidth=1.2, zorder=3)
    ax.annotate(
        f'{score:.0f}', (x, score), xytext=(0, 9),
        textcoords='offset points', ha='center', va='bottom',
        fontsize=8.5, color='#111827', zorder=4)

  ax.axvline(5.5, color='#15803d', linestyle='--', linewidth=1.4, alpha=0.8)
  ax.text(
      5.62, 0.965, '128 WM updates', transform=ax.get_xaxis_transform(),
      ha='left', va='top', fontsize=9, color='#166534', rotation=90)
  ax.axvline(7.5, color='#7e22ce', linestyle='--', linewidth=1.4, alpha=0.8)
  ax.text(
      7.62, 0.965,
      'WM-only phase begins' if wm_only else 'Actor-only IR begins',
      transform=ax.get_xaxis_transform(), ha='left', va='top',
      fontsize=9, color='#6b21a8', rotation=90)

  fig.suptitle(args.title, fontsize=17, weight='bold', y=0.985)
  fig.text(
      0.5, 0.948,
      'DMC Cheetah Run · A action scale 1.0 · B action scale 0.7 · seed 0',
      ha='center', va='center', fontsize=10.5, color='#4b5563')
  ax.set_ylabel('Episode score', fontsize=12)
  ax.set_xlabel('Episode in execution order', fontsize=12, labelpad=14)
  ax.set_xticks(xs)
  ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=8.5)
  ax.set_xlim(0.5, len(episodes) + 0.5)
  ax.set_ylim(bottom=0)
  ax.grid(axis='y', color='#d1d5db', linewidth=0.8, alpha=0.7)
  ax.spines[['top', 'right']].set_visible(False)

  legend = (
      Line2D([0], [0], color=COLORS['A'], marker='o', linewidth=2,
             label='Task A score'),
      Line2D([0], [0], color=COLORS['B'], marker='o', linewidth=2,
             label='Task B score'),
      Line2D([0], [0], color='#4b5563', marker='o', linestyle='None',
             markeredgecolor='white', markersize=8,
             label='Collection/training episode'),
      Line2D([0], [0], color='#4b5563', marker='D', linestyle='None',
             markeredgecolor='white', markersize=8,
             label='Frozen evaluation'),
  )
  ax.legend(handles=legend, loc='upper right', frameon=True, ncol=2)

  fig.tight_layout(rect=(0, 0, 1, 0.91))
  args.output.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.output, dpi=180, bbox_inches='tight')
  fig.savefig(args.output.with_suffix('.svg'), bbox_inches='tight')
  plt.close(fig)


if __name__ == '__main__':
  main()
