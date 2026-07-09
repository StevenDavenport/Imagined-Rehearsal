#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('eval_suite/paper/aggregates/main_recovery_master.csv')
OUTDIR = Path('figures/main_benchmark')
OUTDIR.mkdir(parents=True, exist_ok=True)

MAIN_TASKS = [
    'dmc_walker_walk',
    'dmc_walker_run',
    'dmc_cheetah_run',
    'dmc_cartpole_swingup',
    'dmc_finger_turn_hard',
    'dmc_quadruped_run',
]
SEVERITIES = ['mild', 'moderate', 'severe']
METHODS = ['corrupt_baseline', 'rehearsal_sb1', 'rehearsal_sb512']
METHOD_LABELS = {
    'corrupt_baseline': 'Corrupt',
    'rehearsal_sb1': 'IR',
    'rehearsal_sb512': 'IR with MCPB',
}
METHOD_COLORS = {
    'clean': '#d9d9d9',
    'corrupt_baseline': '#fb6a4a',
    'rehearsal_sb1': '#9e9ac8',
    'rehearsal_sb512': '#a1d99b',
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 15,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def style_axes(ax):
    ax.spines['left'].set_linewidth(1.15)
    ax.spines['bottom'].set_linewidth(1.15)
    ax.tick_params(axis='both', width=1.0, length=4)


def ffloat(x: str) -> float:
    if x in ('', 'nan', 'NaN', None):
        return math.nan
    return float(x)


def severity_of(row: dict[str, str]) -> str:
    sev = (row.get('severity') or row.get('corruption_severity') or '').strip()
    if sev:
        return sev
    # Backfill old walker rows from suite label or strength.
    suite = (row.get('suite_label') or '').strip().lower()
    if suite in {'mild', 'moderate', 'severe'}:
        return suite
    strength = row.get('corruption_strength', '')
    if row.get('task') == 'dmc_walker_walk':
        mapping = {'0.37': 'mild', '0.46': 'moderate', '0.47': 'severe'}
        return mapping.get(strength, '')
    return ''


def stderr(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    arr = np.asarray(xs, dtype=float)
    return float(arr.std(ddof=1) / math.sqrt(len(arr)))


rows = []
with ROOT.open() as f:
    for row in csv.DictReader(f):
        row['task'] = row['task'].strip()
        if row['task'] not in MAIN_TASKS:
            continue
        row['condition'] = row['condition'].strip()
        row['severity_fixed'] = severity_of(row)
        row['clean_reference_score_f'] = ffloat(row['clean_reference_score'])
        row['score_mean_f'] = ffloat(row['score_mean'])
        row['fraction_lost_return_recovered_f'] = ffloat(row['fraction_lost_return_recovered'])
        row['return_retained_vs_clean_f'] = ffloat(row['return_retained_vs_clean'])
        row['wall_seconds_mean_f'] = ffloat(row['wall_seconds_mean'])
        row['fps_policy_mean_f'] = ffloat(row['fps_policy_mean'])
        rows.append(row)

clean_scores = {}
for row in rows:
    if row['condition'] == 'clean_baseline':
        clean_scores[row['task']] = row['score_mean_f']

corrupt_scores = {}
for row in rows:
    if row['condition'] == 'corrupt_baseline' and row['severity_fixed'] in SEVERITIES:
        corrupt_scores[(row['task'], row['severity_fixed'])] = row['score_mean_f']

# Normalize raw return vs clean for each task and backfill missing recovery fractions.
for row in rows:
    clean = clean_scores.get(row['task'], math.nan)
    row['norm_return_vs_clean'] = row['score_mean_f'] / clean if clean and not math.isnan(clean) else math.nan
    if math.isnan(row['fraction_lost_return_recovered_f']):
        sev = row['severity_fixed']
        corrupt = corrupt_scores.get((row['task'], sev), math.nan)
        if row['condition'] == 'corrupt_baseline' and sev in SEVERITIES:
            row['fraction_lost_return_recovered_f'] = 0.0
        elif (
            sev in SEVERITIES
            and not math.isnan(clean)
            and not math.isnan(corrupt)
            and clean != corrupt
            and not math.isnan(row['score_mean_f'])
        ):
            row['fraction_lost_return_recovered_f'] = (
                (row['score_mean_f'] - corrupt) / (clean - corrupt)
            )

# Aggregates by severity and method.
agg_norm = defaultdict(list)
agg_rec = defaultdict(list)
heat_vals = defaultdict(dict)
for row in rows:
    sev = row['severity_fixed']
    cond = row['condition']
    task = row['task']
    if sev in SEVERITIES and cond in METHODS:
        agg_norm[(sev, cond)].append(row['norm_return_vs_clean'])
        agg_rec[(sev, cond)].append(row['fraction_lost_return_recovered_f'])
        heat_vals[task][(sev, cond)] = row['fraction_lost_return_recovered_f']

# 1. Aggregate normalized-return bar chart, similar to the old Walker figure.
fig, ax = plt.subplots(figsize=(8.8, 5.2))
style_axes(ax)
centers = np.arange(len(SEVERITIES)) * 1.9
w = 0.18
# repeated clean bar at 1.0 for visual parity with older figure
clean_x = centers - 1.5 * w
ax.bar(clean_x, [1.0] * len(SEVERITIES), width=w, color=METHOD_COLORS['clean'], edgecolor='#9e9e9e', linewidth=0.7, label='Clean')
for idx, cond in enumerate(METHODS):
    xs = centers + (-0.5 + idx) * w
    ys = [np.nanmean(agg_norm[(sev, cond)]) for sev in SEVERITIES]
    es = [stderr(agg_norm[(sev, cond)]) for sev in SEVERITIES]
    ax.bar(xs, ys, width=w, color=METHOD_COLORS[cond], edgecolor='#666666', linewidth=0.7, label=METHOD_LABELS[cond])
    ax.errorbar(xs, ys, yerr=es, fmt='none', ecolor='#444444', capsize=3, linewidth=1)
ax.set_xticks(centers)
ax.set_xticklabels([s.title() for s in SEVERITIES])
ax.set_ylabel('Normalised return vs clean')
ax.set_title('Cross-Environment Benchmark: Mean Normalised Return')
ax.set_ylim(0, 1.15)
ax.grid(axis='y', linestyle='--', alpha=0.45, color='#bdbdbd', linewidth=0.8)
ax.legend(ncols=2, frameon=False, loc='upper left', bbox_to_anchor=(0.01, 0.995), columnspacing=0.9, handlelength=1.0, handletextpad=0.4, borderaxespad=0.2)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUTDIR / f'benchmark_normalized_return_bar.{ext}', dpi=220, bbox_inches='tight')
plt.close(fig)

# 2. Main aggregate grouped bar: fraction of lost return recovered.
fig, ax = plt.subplots(figsize=(7.8, 5.0))
style_axes(ax)
centers = np.arange(len(SEVERITIES)) * 1.7
w = 0.22
for idx, cond in enumerate(METHODS):
    xs = centers + (idx - 1) * w
    ys = [np.nanmean(agg_rec[(sev, cond)]) for sev in SEVERITIES]
    es = [stderr(agg_rec[(sev, cond)]) for sev in SEVERITIES]
    ax.bar(xs, ys, width=w, color=METHOD_COLORS[cond], edgecolor='#666666', linewidth=0.7, label=METHOD_LABELS[cond])
    ax.errorbar(xs, ys, yerr=es, fmt='none', ecolor='#444444', capsize=3, linewidth=1)
ax.axhline(0.0, color='#888888', linewidth=0.8)
ax.axhline(1.0, color='#cccccc', linewidth=0.8, linestyle='--')
ax.set_xticks(centers)
ax.set_xticklabels([s.title() for s in SEVERITIES])
ax.set_ylabel('Fraction of lost return recovered')
ax.set_title('Cross-Environment Benchmark: Recovery Fraction')
ax.set_ylim(-0.35, 1.05)
ax.grid(axis='y', linestyle='--', alpha=0.45, color='#bdbdbd', linewidth=0.8)
ax.legend(frameon=False, ncols=2, loc='upper left', bbox_to_anchor=(0.01, 0.995), columnspacing=0.9, handletextpad=0.4, borderaxespad=0.2)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUTDIR / f'benchmark_recovery_fraction_bar.{ext}', dpi=220, bbox_inches='tight')
plt.close(fig)

# 3. Per-task heatmap of recovery fraction.
heat_cols = [(sev, cond) for sev in SEVERITIES for cond in METHODS]
heat_data = np.full((len(MAIN_TASKS), len(heat_cols)), np.nan)
for i, task in enumerate(MAIN_TASKS):
    for j, key in enumerate(heat_cols):
        heat_data[i, j] = heat_vals.get(task, {}).get(key, math.nan)
fig, ax = plt.subplots(figsize=(10.0, 4.6))
style_axes(ax)
im = ax.imshow(heat_data, aspect='auto', cmap='viridis', vmin=-0.3, vmax=1.0)
ax.set_yticks(np.arange(len(MAIN_TASKS)))
ax.set_yticklabels([t.replace('dmc_', '').replace('_', '\n') for t in MAIN_TASKS])
ax.set_xticks(np.arange(len(heat_cols)))
ax.set_xticklabels([f'{sev[:3].title()}\n{METHOD_LABELS[cond].replace("Rehearsal ", "").replace("Corrupt", "Corr")}' for sev, cond in heat_cols])
ax.set_title('Per-Task Recovery Fraction Heatmap')
for i in range(len(MAIN_TASKS)):
    for j in range(len(heat_cols)):
        val = heat_data[i, j]
        if not math.isnan(val):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if val < 0.45 else 'black', fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label='Fraction recovered')
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUTDIR / f'benchmark_recovery_heatmap.{ext}', dpi=220, bbox_inches='tight')
plt.close(fig)

# 4. Distribution plot across tasks by severity, rehearsal only.
BOX_METHODS = ['rehearsal_sb1', 'rehearsal_sb512']
fig, ax = plt.subplots(figsize=(7.8, 5.2))
style_axes(ax)
positions = []
boxdata = []
colors = []
centers = np.arange(len(SEVERITIES)) * 1.9
w = 0.24
for s_idx, sev in enumerate(SEVERITIES):
    for m_idx, cond in enumerate(BOX_METHODS):
        positions.append(centers[s_idx] + (m_idx - 0.5) * w)
        boxdata.append(agg_rec[(sev, cond)])
        colors.append(METHOD_COLORS[cond])
parts = ax.boxplot(
    boxdata,
    positions=positions,
    widths=0.18,
    patch_artist=True,
    showfliers=True,
    medianprops={'color': '#222222', 'linewidth': 1.2},
)
for patch, color in zip(parts['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.95)
    patch.set_edgecolor('#666666')
    patch.set_linewidth(0.9)
for whisker in parts['whiskers']:
    whisker.set_color('#666666')
for cap in parts['caps']:
    cap.set_color('#666666')
for flier in parts['fliers']:
    flier.set(marker='o', markersize=3, markerfacecolor='#333333', alpha=0.5, markeredgewidth=0)
ax.axhline(0.0, color='#888888', linewidth=0.8)
ax.axhline(1.0, color='#cccccc', linewidth=0.8, linestyle='--')
ax.set_xticks(centers)
ax.set_xticklabels([s.title() for s in SEVERITIES])
ax.set_ylabel('Fraction of lost return recovered')
ax.set_title('Recovery Spread Across Tasks by Severity', pad=6)
ax.set_ylim(-0.35, 1.05)
ax.grid(axis='y', linestyle='--', alpha=0.45, color='#bdbdbd', linewidth=0.8)
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=METHOD_COLORS[c], edgecolor='#666666') for c in BOX_METHODS]
ax.legend(
    handles,
    [METHOD_LABELS[c] for c in BOX_METHODS],
    frameon=False,
    ncols=2,
    loc='upper center',
    bbox_to_anchor=(0.01, 0.995),
    columnspacing=1.0,
    handletextpad=0.5,
    borderaxespad=0.2,
)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUTDIR / f'benchmark_recovery_boxplot.{ext}', dpi=220, bbox_inches='tight')
plt.close(fig)

# 5. Simple pooled side-by-side rehearsal comparison across all tasks and severities.
fig, ax = plt.subplots(figsize=(5.8, 5.0))
style_axes(ax)
pooled_methods = ['rehearsal_sb1', 'rehearsal_sb512']
pooled_data = [[v for sev in SEVERITIES for v in agg_rec[(sev, cond)]] for cond in pooled_methods]
parts = ax.boxplot(
    pooled_data,
    widths=0.45,
    patch_artist=True,
    showfliers=True,
    medianprops={'color': '#222222', 'linewidth': 1.2},
)
for patch, cond in zip(parts['boxes'], pooled_methods):
    patch.set_facecolor(METHOD_COLORS[cond])
    patch.set_alpha(0.95)
    patch.set_edgecolor('#666666')
    patch.set_linewidth(0.9)
for whisker in parts['whiskers']:
    whisker.set_color('#666666')
for cap in parts['caps']:
    cap.set_color('#666666')
for flier in parts['fliers']:
    flier.set(marker='o', markersize=3, markerfacecolor='#333333', alpha=0.5, markeredgewidth=0)
ax.axhline(0.0, color='#888888', linewidth=0.8)
ax.axhline(1.0, color='#cccccc', linewidth=0.8, linestyle='--')
ax.set_xticks([1, 2])
ax.set_xticklabels(['IR', 'IR with MCPB'])
ax.set_ylabel('Fraction of lost return recovered')
ax.set_title('Pooled Recovery Comparison Across Benchmark', pad=10)
ax.set_ylim(-0.35, 1.05)
ax.grid(axis='y', linestyle='--', alpha=0.45, color='#bdbdbd', linewidth=0.8)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUTDIR / f'benchmark_rehearsal_pooled_boxplot.{ext}', dpi=220, bbox_inches='tight')
plt.close(fig)

print('Wrote plots to', OUTDIR)
for name in sorted(p.name for p in OUTDIR.iterdir() if p.suffix in {'.png', '.pdf'}):
    print(name)
