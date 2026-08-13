#!/usr/bin/env python3
"""Generate the figures and compact source tables for the RLScape report.

The numerical arrays below are the audited summaries extracted from the FIFO
and episode-reservoir run artifacts copied from office-gpu on 2026-08-08. The
raw JSONL logs remain the authoritative source; these compact arrays make the
paper package reproducible without bundling roughly 250 MB of logs.
"""

from __future__ import annotations

import csv
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent
FIG = ROOT / "figures"
DATA = ROOT / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

STEPS = np.array([250, 500, 750, 1000, 1250, 1500])
GOALS = ["Kill goblin", "Bury bones", "Chop logs"]
SHORT = ["Kill", "Bury", "Chop"]

# Rows are goals; columns are checkpoints in STEPS (thousands of steps).
FIFO_S = np.array([
    [.91, .96, .75, .06, .98, .73],
    [np.nan, np.nan, .21, 1.00, .08, .12],
    [np.nan, np.nan, np.nan, np.nan, 1.00, 1.00],
])
FIFO_D = np.array([
    [.80, .94, .70, .04, .92, .76],
    [np.nan, np.nan, .30, 1.00, .28, .04],
    [np.nan, np.nan, np.nan, np.nan, 1.00, 1.00],
])
RES_S = np.array([
    [.49, .78, .70, .02, .56, .16],
    [np.nan, np.nan, .51, .04, .49, .08],
    [np.nan, np.nan, np.nan, np.nan, .88, .05],
])
RES_D = np.array([
    [.26, .74, .76, .02, .46, .24],
    [np.nan, np.nan, .48, .00, .34, .04],
    [np.nan, np.nan, np.nan, np.nan, .60, .06],
])
RES_IR_DELTA_S = np.array([
    [-.13, -.05, -.02, .01, .03, .12],
    [np.nan, np.nan, -.22, -.03, -.18, -.02],
    [np.nan, np.nan, np.nan, np.nan, -.36, .07],
])
RES_IR_DELTA_D = np.array([
    [.02, .04, -.10, .02, .10, .16],
    [np.nan, np.nan, -.12, .00, -.18, -.02],
    [np.nan, np.nan, np.nan, np.nan, -.18, .02],
])

# Success rate in successive 50k-step bins during each run.
BIN_STEPS = np.arange(50, 1501, 50)
FIFO_TRAIN = np.array([
    .253, .332, .369, .347, .766, .837, .821, .929, .902, .988,
    .000, .000, .000, .104, .824, .959, .800, .998, .997, 1.000,
    .378, .959, .969, .994, .999, .999, .995, .999, 1.000, .996,
])
RES_TRAIN = np.array([
    .319, .302, .303, .590, .552, .834, .811, .834, .958, .816,
    .936, .660, .091, .727, .713, .705, .946, .391, .762, .962,
    .113, .525, .838, .704, .716, .853, .475, .764, .914, .495,
])

RES_RAM = np.array([14.79, 10.97, 17.30, 16.80, 29.52, 28.53])
FIFO_RAM = np.array([21.65, 21.87, 21.76, 22.42, 22.75, 23.19])
RES_PROC = np.array([19.95, 19.72, 22.07, 22.82, 34.26, 34.28])
RES_ITEMS = np.array([137830, 102171, 161154, 156514, 275046, 265802])

COLORS = {
    "fifo": "#4c78a8",
    "reservoir": "#f58518",
    "ir": "#54a24b",
    "kill": "#4c78a8",
    "bury": "#f58518",
    "chop": "#54a24b",
}


def style() -> None:
  mpl.rcParams.update({
      "font.family": "DejaVu Sans",
      "font.size": 9,
      "axes.titlesize": 10,
      "axes.labelsize": 9,
      "legend.fontsize": 8,
      "figure.dpi": 160,
      "savefig.dpi": 300,
      "savefig.bbox": "tight",
      "axes.spines.top": False,
      "axes.spines.right": False,
      "axes.grid": True,
      "grid.alpha": .18,
  })


def save(fig: mpl.figure.Figure, name: str) -> None:
  fig.savefig(FIG / f"{name}.pdf")
  fig.savefig(FIG / f"{name}.png")
  plt.close(fig)


def shade_phases(ax: mpl.axes.Axes) -> None:
  colors = ["#dbe9f6", "#fce5cc", "#dcefd9"]
  for lo, hi, color in zip([0, 500, 1000], [500, 1000, 1500], colors):
    ax.axvspan(lo, hi, color=color, alpha=.55, zorder=0)
  for x in [500, 1000]:
    ax.axvline(x, color="0.35", ls="--", lw=.8)
  ax.text(250, 1.055, "Kill goblin", ha="center", fontsize=8)
  ax.text(750, 1.055, "Bury bones", ha="center", fontsize=8)
  ax.text(1250, 1.055, "Chop logs", ha="center", fontsize=8)


def training_curve() -> None:
  fig, ax = plt.subplots(figsize=(7.15, 3.15))
  shade_phases(ax)
  ax.plot(BIN_STEPS, FIFO_TRAIN, color=COLORS["fifo"], marker="o",
          ms=2.8, lw=1.6, label="FIFO (200k sequence starts)")
  ax.plot(BIN_STEPS, RES_TRAIN, color=COLORS["reservoir"], marker="o",
          ms=2.8, lw=1.6, label="Goal-stratified episode reservoir")
  ax.set(xlim=(0, 1500), ylim=(-.02, 1.10), xlabel="Environment steps (thousands)",
         ylabel="Training success rate in 50k-step bin")
  ax.legend(loc="lower right", frameon=True, ncol=2)
  ax.set_title("Sequential task acquisition is strong but reservoir training is less stable")
  save(fig, "training_success")


def heat(ax, values, title, cmap="viridis", vmin=0, vmax=1,
         fmt="{:.0%}") -> None:
  masked = np.ma.masked_invalid(values)
  cm = mpl.cm.get_cmap(cmap).copy()
  cm.set_bad("#eeeeee")
  im = ax.imshow(masked, aspect="auto", cmap=cm, vmin=vmin, vmax=vmax)
  ax.set_xticks(np.arange(len(STEPS)), [f"{x/1000:g}M" for x in STEPS])
  ax.set_yticks(np.arange(len(GOALS)), SHORT)
  ax.set_xlabel("Checkpoint")
  ax.set_title(title)
  ax.grid(False)
  for i in range(values.shape[0]):
    for j in range(values.shape[1]):
      v = values[i, j]
      label = "--" if np.isnan(v) else fmt.format(v)
      color = "0.55" if np.isnan(v) else ("white" if abs(v) > .52 else "black")
      ax.text(j, i, label, ha="center", va="center", fontsize=7.5, color=color)
  return im


def frozen_heatmaps() -> None:
  fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True,
                           sharex=True, sharey=True)
  im = heat(axes[0, 0], FIFO_S, "FIFO: sampled policy")
  heat(axes[0, 1], RES_S, "Reservoir: sampled policy")
  heat(axes[1, 0], FIFO_D, "FIFO: deterministic policy")
  heat(axes[1, 1], RES_D, "Reservoir: deterministic policy")
  fig.colorbar(im, ax=axes, shrink=.82, label="Frozen-actor success rate")
  fig.suptitle("Checkpoint evaluation matrix: acquisition and backward retention", fontsize=11)
  save(fig, "frozen_eval_heatmaps")


def ir_heatmaps() -> None:
  fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), constrained_layout=True,
                           sharex=True, sharey=True)
  im = heat(axes[0], RES_IR_DELTA_S, "Sampled policy", cmap="RdYlGn",
            vmin=-.40, vmax=.40, fmt="{:+.0%}")
  heat(axes[1], RES_IR_DELTA_D, "Deterministic policy", cmap="RdYlGn",
       vmin=-.40, vmax=.40, fmt="{:+.0%}")
  fig.colorbar(im, ax=axes, shrink=.82, label="Actor IR minus frozen success rate")
  fig.suptitle("Online actor IR is conditional: harmful at several strong checkpoints,\n"
               "helpful after the final collapse", fontsize=11)
  save(fig, "ir_delta_heatmaps")


def checkpoint_trajectories() -> None:
  fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.65), sharey=True,
                           constrained_layout=True)
  for i, (ax, goal) in enumerate(zip(axes, GOALS)):
    valid = ~np.isnan(FIFO_S[i])
    ax.plot(STEPS[valid], FIFO_S[i, valid], "o-", color=COLORS["fifo"],
            lw=1.7, ms=4, label="FIFO frozen")
    valid = ~np.isnan(RES_S[i])
    ax.plot(STEPS[valid], RES_S[i, valid], "o-", color=COLORS["reservoir"],
            lw=1.7, ms=4, label="Reservoir frozen")
    ir = RES_S[i] + RES_IR_DELTA_S[i]
    ax.plot(STEPS[valid], ir[valid], "s--", color=COLORS["ir"], lw=1.3,
            ms=3.5, label="Reservoir + IR")
    ax.set_title(goal)
    ax.set_xlabel("Checkpoint (k steps)")
    ax.set_ylim(-.03, 1.03)
    ax.set_xticks(STEPS[valid], [str(x) for x in STEPS[valid]], rotation=45)
  axes[0].set_ylabel("Sampled-policy success rate")
  handles, labels = axes[-1].get_legend_handles_labels()
  fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
             bbox_to_anchor=(.5, 1.10))
  save(fig, "checkpoint_trajectories")


def retention_dumbbell() -> None:
  fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True,
                           constrained_layout=True)
  for ax, data, title in zip(axes, [(FIFO_S, RES_S), (FIFO_D, RES_D)],
                             ["Sampled policy", "Deterministic policy"]):
    y_positions, labels = [], []
    y = 0
    for replay_name, matrix, color in [
        ("FIFO", data[0], COLORS["fifo"]),
        ("Reservoir", data[1], COLORS["reservoir"]),
    ]:
      for i, goal in enumerate(SHORT):
        vals = matrix[i, ~np.isnan(matrix[i])]
        peak, final = np.max(vals), vals[-1]
        ax.plot([final, peak], [y, y], color=color, lw=2)
        ax.scatter([peak], [y], color=color, marker="o", s=31,
                   label="Peak" if y == 0 else None)
        ax.scatter([final], [y], color=color, marker="x", s=38,
                   label="Final" if y == 0 else None)
        y_positions.append(y)
        labels.append(f"{replay_name} / {goal}")
        ax.text((peak + final) / 2, y + .13, f"forget {peak-final:.0%}",
                ha="center", fontsize=7)
        y += 1
      y += .55
    ax.set(xlim=(-.03, 1.03), xlabel="Success rate", title=title)
    ax.invert_yaxis()
  axes[0].set_yticks(y_positions, labels)
  axes[1].tick_params(labelleft=False)
  axes[0].legend(loc="lower right", frameon=False)
  fig.suptitle("Maximum observed performance versus final-checkpoint retention", fontsize=11)
  save(fig, "peak_to_final_forgetting")


def aggregate_worst_case() -> None:
  fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
  for ax, matrices, title in zip(
      axes, [(FIFO_S, RES_S), (FIFO_D, RES_D)],
      ["Sampled policy", "Deterministic policy"]):
    for matrix, color, label, marker in [
        (matrices[0], COLORS["fifo"], "FIFO mean", "o"),
        (matrices[1], COLORS["reservoir"], "Reservoir mean", "o"),
    ]:
      means = np.nanmean(matrix, axis=0)
      worst = np.nanmin(matrix, axis=0)
      ax.plot(STEPS, means, marker=marker, color=color, lw=1.7, label=label)
      ax.plot(STEPS, worst, marker="v", color=color, lw=1.1, ls="--",
              alpha=.85, label=label.replace("mean", "worst"))
    ax.set(xlabel="Checkpoint (k steps)", ylabel="Success rate", title=title,
           ylim=(-.03, 1.03))
    ax.set_xticks(STEPS, [str(x) for x in STEPS], rotation=45)
  axes[1].legend(loc="upper right", ncol=2, fontsize=7, frameon=True)
  fig.suptitle("Average performance can hide catastrophic worst-task failure", fontsize=11)
  save(fig, "mean_and_worst_task")


def ir_scatter() -> None:
  xs, ys, labels = [], [], []
  for i, goal in enumerate(SHORT):
    for j, step in enumerate(STEPS):
      if not np.isnan(RES_S[i, j]):
        xs.append(RES_S[i, j])
        ys.append(RES_IR_DELTA_S[i, j])
        labels.append((goal, step))
  xs, ys = np.asarray(xs), np.asarray(ys)
  coef = np.polyfit(xs, ys, 1)
  corr = np.corrcoef(xs, ys)[0, 1]
  fig, ax = plt.subplots(figsize=(5.1, 3.5))
  for goal, color in zip(SHORT, [COLORS["kill"], COLORS["bury"], COLORS["chop"]]):
    idx = np.array([g == goal for g, _ in labels])
    ax.scatter(xs[idx], ys[idx], s=45, color=color, label=goal, alpha=.9,
               edgecolor="white", linewidth=.5)
  line_x = np.linspace(0, 1, 100)
  ax.plot(line_x, np.polyval(coef, line_x), color="0.25", ls="--", lw=1.2,
          label=f"Post-hoc fit, r={corr:.2f}")
  ax.axhline(0, color="0.2", lw=.8)
  ax.axvline(.25, color="0.5", lw=.8, ls=":")
  ax.text(.02, .145, "weak frozen policy", fontsize=8, color="0.35")
  ax.set(xlim=(-.03, 1.03), ylim=(-.42, .20),
         xlabel="Frozen sampled-policy success rate",
         ylabel="IR change in success rate")
  ax.set_title("IR tends to help weak policies and damage strong ones (post-hoc)")
  ax.legend(frameon=True, ncol=2)
  save(fig, "ir_strength_scatter")


def ir_sorted() -> None:
  records = []
  for mode, base, delta in [("S", RES_S, RES_IR_DELTA_S),
                            ("D", RES_D, RES_IR_DELTA_D)]:
    for i, goal in enumerate(SHORT):
      for j, step in enumerate(STEPS):
        if not np.isnan(delta[i, j]):
          records.append((delta[i, j], f"{step/1000:g}M {goal} {mode}"))
  records.sort()
  vals = np.array([x[0] for x in records])
  labels = [x[1] for x in records]
  colors = [COLORS["ir"] if x > 0 else "#e45756" if x < 0 else "0.6" for x in vals]
  fig, ax = plt.subplots(figsize=(6.7, 4.6))
  y = np.arange(len(vals))
  ax.barh(y, vals, color=colors)
  ax.set_yticks(y, labels, fontsize=7)
  ax.axvline(0, color="0.2", lw=.8)
  ax.set(xlabel="Actor IR minus frozen success rate",
         title="All 24 paired aggregate comparisons (S=sampled, D=deterministic)")
  save(fig, "ir_all_conditions")


def resources() -> None:
  fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
  axes[0].plot(STEPS, FIFO_RAM, "o-", color=COLORS["fifo"], label="FIFO replay")
  axes[0].plot(STEPS, RES_RAM, "o-", color=COLORS["reservoir"], label="Reservoir replay")
  axes[0].plot(STEPS, RES_PROC, "s--", color="#b279a2", label="Reservoir process")
  axes[0].set(xlabel="Checkpoint (k steps)", ylabel="Host memory (GiB)",
              title="Host-memory footprint")
  axes[0].legend(frameon=True, fontsize=7)
  axes[1].plot(STEPS, RES_ITEMS / 1000, "o-", color=COLORS["reservoir"])
  axes[1].axhline(3000, color="0.5", ls=":", lw=.8,
                  label="3,000-episode capacity (not transitions)")
  axes[1].set(xlabel="Checkpoint (k steps)", ylabel="Stored transitions (thousands)",
              title="Variable-length reservoir occupancy")
  axes[1].legend(frameon=True, fontsize=7)
  save(fig, "resource_footprint")


def write_tables() -> None:
  with (DATA / "evaluation_matrix.csv").open("w", newline="") as f:
    out = csv.writer(f)
    out.writerow(["replay", "policy_mode", "checkpoint_steps", "goal",
                  "frozen_success", "actor_ir_success", "ir_delta"])
    for replay, mode, base, delta in [
        ("fifo", "sampled", FIFO_S, np.full_like(FIFO_S, np.nan)),
        ("fifo", "deterministic", FIFO_D, np.full_like(FIFO_D, np.nan)),
        ("reservoir", "sampled", RES_S, RES_IR_DELTA_S),
        ("reservoir", "deterministic", RES_D, RES_IR_DELTA_D),
    ]:
      for i, goal in enumerate(GOALS):
        for j, step in enumerate(STEPS):
          if np.isnan(base[i, j]):
            continue
          ir = "" if np.isnan(delta[i, j]) else base[i, j] + delta[i, j]
          d = "" if np.isnan(delta[i, j]) else delta[i, j]
          out.writerow([replay, mode, step * 1000, goal, base[i, j], ir, d])
  with (DATA / "training_50k_bins.csv").open("w", newline="") as f:
    out = csv.writer(f)
    out.writerow(["step", "fifo_success", "reservoir_success"])
    out.writerows(zip(BIN_STEPS * 1000, FIFO_TRAIN, RES_TRAIN))
  with (DATA / "resource_milestones.csv").open("w", newline="") as f:
    out = csv.writer(f)
    out.writerow(["step", "fifo_replay_gib", "reservoir_replay_gib",
                  "reservoir_process_gib", "reservoir_transitions"])
    out.writerows(zip(STEPS * 1000, FIFO_RAM, RES_RAM, RES_PROC, RES_ITEMS))


def main() -> None:
  style()
  training_curve()
  frozen_heatmaps()
  ir_heatmaps()
  checkpoint_trajectories()
  retention_dumbbell()
  aggregate_worst_case()
  ir_scatter()
  ir_sorted()
  resources()
  write_tables()


if __name__ == "__main__":
  main()
