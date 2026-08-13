#!/usr/bin/env python3
"""Generate the compact behavior-versus-critic Stage 1 comparison."""

from __future__ import annotations

import csv
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parents[3]
EVAL = REPO / 'reports/rlscape_reservoir_report/data/evaluation_matrix.csv'
GOALS = ('Kill goblin', 'Bury bones', 'Chop logs')
STEPS = {'strong': 1250000, 'weak': 1500000}
COLORS = {'strong': '#4c78a8', 'weak': '#e45756'}


def evaluation_success():
  result = {label: {} for label in STEPS}
  with EVAL.open() as handle:
    for row in csv.DictReader(handle):
      if row['replay'] != 'reservoir' or row['policy_mode'] != 'sampled':
        continue
      for label, step in STEPS.items():
        if int(row['checkpoint_steps']) == step:
          result[label][row['goal']] = float(row['frozen_success'])
  return result


def initial_values():
  result = {}
  for label in STEPS:
    summary = json.loads((ROOT / label / 'summary.json').read_text())
    result[label] = {
        GOALS[row['observed_goal']]: row['prediction_mean']
        for row in summary['initial_value_cross_goal']
        if row['observed_goal'] == row['probed_goal']
    }
  return result


def main():
  success = evaluation_success()
  values = initial_values()
  x = np.arange(len(GOALS))
  width = .36
  fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), constrained_layout=True)
  for offset, label in zip((-.5, .5), ('strong', 'weak')):
    axes[0].bar(
        x + offset * width,
        [success[label][goal] for goal in GOALS],
        width, color=COLORS[label], label=f'{label} checkpoint')
    axes[1].bar(
        x + offset * width,
        [values[label][goal] for goal in GOALS],
        width, color=COLORS[label], label=f'{label} checkpoint')
  axes[0].set_title('Actual sampled-policy evaluation')
  axes[0].set_ylabel('Success rate')
  axes[0].set_ylim(0, 1)
  axes[1].set_title('Critic on identical episode starts')
  axes[1].set_ylabel('Predicted value')
  axes[1].axhline(1, color='0.35', linestyle='--', linewidth=1,
                  label='maximum observed return')
  for axis in axes:
    axis.set_xticks(x, GOALS, rotation=20, ha='right')
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='y', alpha=.2)
  axes[0].legend(frameon=False)
  axes[1].legend(frameon=False)
  fig.suptitle('Behavior collapses while the critic becomes more optimistic')
  target = ROOT / 'figures/behavior_vs_critic'
  fig.savefig(target.with_suffix('.png'), dpi=220, bbox_inches='tight')
  fig.savefig(target.with_suffix('.pdf'), bbox_inches='tight')


if __name__ == '__main__':
  main()
