#!/usr/bin/env python3
"""Generate the figures and compact source tables for the RLScape report.

The numerical arrays and compact tables below are audited summaries extracted
from the FIFO, episode-reservoir, actor-recovery, critic-provenance, and
counterfactual-head-repair and conservative-value-dose artifacts copied from
the execution host in August 2026.
The raw JSONL logs remain the
authoritative source; these inputs make the paper package reproducible without
bundling the much larger raw run roots.
"""

from __future__ import annotations

import csv
import json
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

STAGE2 = DATA / "stage2_controlled_actor_recovery"
STAGE2_FIGURES = "stage2_controlled_actor_recovery"
STAGE2_GOALS = ["kill_goblin", "bury_bones", "chop_logs"]
STAGE2_GOAL_LABELS = ["Kill goblin", "Bury bones", "Chop logs"]
STAGE2_FAMILIES = {
    "standard_ir": ("Standard IR", "#4c78a8", "o"),
    "entropy_off_ir": ("Entropy-off IR", "#b279a2", "s"),
    "reward_only_ir": ("Reward-only IR", "#54a24b", "^"),
}

STAGE3B = DATA / "stage3b_natural_forgetting_recovery"
STAGE3B_FIGURES = "stage3b_natural_forgetting_recovery"
STAGE3B_CHECKPOINTS = [1_250_000, 1_500_000]
STAGE3B_CONDITIONS = [
    "frozen", "standard_ir_h6", "reward_only_ir_h6",
    "reward_only_ir_h15", "reward_only_ir_h20",
]

STAGE3A = DATA / "stage3a_critic_provenance_calibration"
STAGE3A_FIGURES = "stage3a_critic_provenance_calibration"
STAGE3A_LABELS = ["strong", "weak"]
STAGE3A_COLORS = {"strong": "#4c78a8", "weak": "#e45756"}

STAGE4A = DATA / "stage4a_counterfactual_head_repair"
STAGE4A_FIGURES = "stage4a_counterfactual_head_repair"
STAGE4A_AUDIT_ARMS = [
    "fifo_original", "reservoir_original", "factual_only",
    "counterfactual", "counterfactual_bounded",
]
STAGE4A_AUDIT_LABELS = [
    "FIFO", "Reservoir", "Factual-only", "Counterfactual", "CF + bounded",
]
STAGE4A_CONDITIONS = [
    "original_frozen", "original_standard_h6", "original_reward_only_h15",
    "factual_reward_only_h15", "counterfactual_reward_only_h15",
    "counterfactual_bounded_reward_only_h15",
    "counterfactual_bounded_h6", "counterfactual_bounded_h15",
]
STAGE4A_CONDITION_LABELS = [
    "Frozen", "Original standard H=6", "Original reward-only H=15",
    "Factual reward-only H=15", "CF reward-only H=15",
    "CF+bdd reward-only H=15", "CF+bdd bootstrap H=6",
    "CF+bdd bootstrap H=15",
]

STAGE4B = DATA / "stage4b_conservative_value"
STAGE4B_FIGURES = "stage4b_conservative_value"
STAGE4B_CONDITIONS = [
    "frozen", "reward_only_h15", "mixed_b005_h15", "mixed_b010_h15",
    "mixed_b025_h15", "mixed_b025_clip050_h15",
    "mixed_b025_ucap15e6_h15", "mixed_b025_h6", "mixed_b100_h15",
]
STAGE4B_LABELS = {
    "frozen": "Frozen", "reward_only_h15": "Reward\nonly",
    "mixed_b005_h15": r"$\beta=.05$",
    "mixed_b010_h15": r"$\beta=.10$",
    "mixed_b025_h15": r"$\beta=.25$",
    "mixed_b025_clip050_h15": "$\\beta=.25$\nclip .50",
    "mixed_b025_ucap15e6_h15": "$\\beta=.25$\nupdate cap",
    "mixed_b025_h6": "$\\beta=.25$\nH=6",
    "mixed_b100_h15": r"$\beta=1$",
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
  path = FIG / name
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path.with_suffix(".pdf"))
  fig.savefig(path.with_suffix(".png"))
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
  # ``matplotlib.colormaps.get_cmap`` is unavailable on the older Matplotlib
  # release used by one of the report-building hosts.
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
    out = csv.writer(f, lineterminator="\n")
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
    out = csv.writer(f, lineterminator="\n")
    out.writerow(["step", "fifo_success", "reservoir_success"])
    out.writerows(zip(BIN_STEPS * 1000, FIFO_TRAIN, RES_TRAIN))
  with (DATA / "resource_milestones.csv").open("w", newline="") as f:
    out = csv.writer(f, lineterminator="\n")
    out.writerow(["step", "fifo_replay_gib", "reservoir_replay_gib",
                  "reservoir_process_gib", "reservoir_transitions"])
    out.writerows(zip(STEPS * 1000, FIFO_RAM, RES_RAM, RES_PROC, RES_ITEMS))


def read_stage2() -> list[dict[str, str]]:
  with (STAGE2 / "main_results.csv").open() as f:
    return list(csv.DictReader(f))


def stage2_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], float]:
  return {
      (row["condition"], row["goal"], row["policy_mode"]):
          float(row["success_rate"])
      for row in rows
  }


def stage2_corruption_calibration() -> None:
  with (STAGE2 / "corruption_selection.json").open() as f:
    payload = json.load(f)
  def strength(candidate: dict) -> float:
    if "manifest" in candidate:
      return float(candidate["manifest"]["strength"])
    token = candidate["name"].split("s0p", 1)[1].split("_", 1)[0]
    return float(token) / (10 ** len(token))
  candidates = sorted(payload["candidates"],
                      key=strength)
  strengths = np.array([strength(x) for x in candidates])
  fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), constrained_layout=True)
  for goal, label, color in zip(STAGE2_GOALS, STAGE2_GOAL_LABELS,
                                [COLORS["kill"], COLORS["bury"], COLORS["chop"]]):
    axes[0].plot(strengths, [x["goal_success"][goal] for x in candidates],
                 "o-", color=color, label=label)
    axes[1].plot(strengths, [x["goal_retention"][goal] for x in candidates],
                 "o-", color=color, label=label)
  aggregate = [x["success_rate"] for x in candidates]
  retention = [x["retention"] for x in candidates]
  axes[0].plot(strengths, aggregate, "D--", color="0.15", label="Aggregate")
  axes[1].plot(strengths, retention, "D--", color="0.15", label="Aggregate")
  selected = payload["selected"]["manifest"]["strength"]
  for ax in axes:
    ax.axvline(selected, color="#e45756", ls=":", lw=1.4,
               label="Selected 40% mask")
    ax.set(xlabel="Independent actor-weight zero-mask probability",
           xlim=(.08, .62), ylim=(-.03, 1.45))
  axes[0].set(ylabel="Calibration success rate",
              title="Behavior after actor-only corruption")
  axes[1].axhline(payload["target_retention"], color="0.45", ls="--", lw=1,
                  label="35% retention target")
  axes[1].set(ylabel="Fraction of clean success retained",
              title="Corruption qualification")
  handles, labels = axes[1].get_legend_handles_labels()
  fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
             bbox_to_anchor=(.5, 1.11))
  save(fig, f"{STAGE2_FIGURES}/corruption_calibration")


def stage2_condition_matrix(rows: list[dict[str, str]]) -> None:
  lookup = stage2_lookup(rows)
  conditions = [
      "clean_no_adapt", "corrupt_no_adapt",
      "standard_ir_h3", "standard_ir_h6", "standard_ir_h15", "standard_ir_h20",
      "entropy_off_ir_h3", "entropy_off_ir_h6", "entropy_off_ir_h15", "entropy_off_ir_h20",
      "reward_only_ir_h3", "reward_only_ir_h6", "reward_only_ir_h15", "reward_only_ir_h20",
      "reference_distill",
  ]
  labels = [
      "Clean actor", "Corrupted actor",
      "Standard H=3", "Standard H=6", "Standard H=15", "Standard H=20",
      "Entropy-off H=3", "Entropy-off H=6", "Entropy-off H=15", "Entropy-off H=20",
      "Reward-only H=3", "Reward-only H=6", "Reward-only H=15", "Reward-only H=20",
      "Clean-policy distillation",
  ]
  fig, axes = plt.subplots(1, 2, figsize=(7.25, 6.2), constrained_layout=True,
                           sharex=True, sharey=True)
  for ax, mode, title in zip(axes, ["sampled", "deterministic"],
                             ["Sampled policy", "Deterministic policy"]):
    matrix = np.array([[lookup[(condition, goal, mode)]
                        for goal in STAGE2_GOALS] for condition in conditions])
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(3), STAGE2_GOAL_LABELS, rotation=22, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title)
    ax.grid(False)
    for i in range(matrix.shape[0]):
      for j in range(matrix.shape[1]):
        value = matrix[i, j]
        ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=7,
                color="white" if value < .22 or value > .67 else "black")
    for boundary in [1.5, 5.5, 9.5, 13.5]:
      ax.axhline(boundary, color="white", lw=1.5)
  fig.colorbar(im, ax=axes, shrink=.72, label="Real-environment success rate")
  fig.suptitle("Stage 2: controlled actor recovery across every objective and horizon",
               fontsize=11)
  save(fig, f"{STAGE2_FIGURES}/condition_matrix")


def stage2_horizon_aggregate(rows: list[dict[str, str]]) -> None:
  lookup = stage2_lookup(rows)
  horizons = [3, 6, 15, 20]
  fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True,
                           sharey=True)
  for ax, mode, title in zip(axes, ["sampled", "deterministic"],
                             ["Sampled policy", "Deterministic policy"]):
    clean = np.mean([lookup[("clean_no_adapt", goal, mode)] for goal in STAGE2_GOALS])
    corrupt = np.mean([lookup[("corrupt_no_adapt", goal, mode)] for goal in STAGE2_GOALS])
    ax.axhline(clean, color="0.25", ls="--", lw=1.2,
               label=f"Clean actor ({clean:.1%})")
    ax.axhline(corrupt, color="#e45756", ls=":", lw=1.4,
               label=f"Corrupted actor ({corrupt:.1%})")
    for family, (label, color, marker) in STAGE2_FAMILIES.items():
      values = []
      for horizon in horizons:
        condition = f"{family}_h{horizon}"
        values.append(np.mean([lookup[(condition, goal, mode)]
                               for goal in STAGE2_GOALS]))
      ax.plot(horizons, values, marker=marker, color=color, lw=1.8,
              ms=5, label=label)
    ax.set(xlabel="Imagination horizon H", title=title, ylim=(.1, .9),
           xticks=horizons)
  axes[0].set_ylabel("Mean real-environment success across goals")
  axes[1].legend(frameon=True, fontsize=7, loc="best")
  fig.suptitle("Removing critic bootstrap changes actor IR from harmful to strongly restorative",
               fontsize=11)
  save(fig, f"{STAGE2_FIGURES}/horizon_aggregate")


def stage2_reward_only_by_goal(rows: list[dict[str, str]]) -> None:
  lookup = stage2_lookup(rows)
  horizons = [3, 6, 15, 20]
  fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.8), constrained_layout=True,
                           sharex=True, sharey=True)
  for i, mode in enumerate(["sampled", "deterministic"]):
    for j, (goal, goal_label) in enumerate(zip(STAGE2_GOALS, STAGE2_GOAL_LABELS)):
      ax = axes[i, j]
      clean = lookup[("clean_no_adapt", goal, mode)]
      corrupt = lookup[("corrupt_no_adapt", goal, mode)]
      standard = [lookup[(f"standard_ir_h{h}", goal, mode)] for h in horizons]
      reward = [lookup[(f"reward_only_ir_h{h}", goal, mode)] for h in horizons]
      ax.axhline(clean, color="0.25", ls="--", lw=1.0, label="Clean")
      ax.axhline(corrupt, color="#e45756", ls=":", lw=1.2, label="Corrupted")
      ax.plot(horizons, standard, "o-", color="#4c78a8", lw=1.4,
              label="Standard IR")
      ax.plot(horizons, reward, "^-", color="#54a24b", lw=1.7,
              label="Reward-only IR")
      ax.set(xticks=horizons, ylim=(-.03, 1.03))
      if i == 0:
        ax.set_title(goal_label)
      if i == 1:
        ax.set_xlabel("Horizon H")
      if j == 0:
        ax.set_ylabel(f"{mode.capitalize()}\nsuccess rate")
  handles, labels = axes[0, 2].get_legend_handles_labels()
  fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
             bbox_to_anchor=(.5, 1.04))
  fig.suptitle("Reward-only rehearsal recovers different skills at different horizon scales",
               fontsize=11, y=1.09)
  save(fig, f"{STAGE2_FIGURES}/reward_only_by_goal")


def stage2_paired_effects() -> None:
  with (STAGE2 / "comparison_vs_baselines.csv").open() as f:
    rows = list(csv.DictReader(f))
  selected = [
      "standard_ir_h3", "standard_ir_h6", "standard_ir_h15", "standard_ir_h20",
      "entropy_off_ir_h3", "entropy_off_ir_h6", "entropy_off_ir_h15", "entropy_off_ir_h20",
      "reward_only_ir_h3", "reward_only_ir_h6", "reward_only_ir_h15", "reward_only_ir_h20",
      "reference_distill",
  ]
  labels = [
      "Standard H=3", "Standard H=6", "Standard H=15", "Standard H=20",
      "Entropy-off H=3", "Entropy-off H=6", "Entropy-off H=15", "Entropy-off H=20",
      "Reward-only H=3", "Reward-only H=6", "Reward-only H=15", "Reward-only H=20",
      "Clean-policy distillation",
  ]
  index = {(r["condition"], r["goal"], r["policy_mode"]):
           float(r["paired_delta_vs_corrupt"]) for r in rows}
  fig, axes = plt.subplots(1, 2, figsize=(7.25, 5.55), constrained_layout=True,
                           sharex=True, sharey=True)
  for ax, mode, title in zip(axes, ["sampled", "deterministic"],
                             ["Sampled policy", "Deterministic policy"]):
    matrix = np.array([[index[(condition, goal, mode)]
                        for goal in STAGE2_GOALS] for condition in selected])
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-.8, vmax=.8)
    ax.set_xticks(range(3), STAGE2_GOAL_LABELS, rotation=22, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title)
    ax.grid(False)
    for i in range(matrix.shape[0]):
      for j in range(matrix.shape[1]):
        ax.text(j, i, f"{matrix[i,j]:+.0%}", ha="center", va="center",
                fontsize=7, color="white" if abs(matrix[i, j]) > .5 else "black")
    for boundary in [3.5, 7.5, 11.5]:
      ax.axhline(boundary, color="white", lw=1.5)
  fig.colorbar(im, ax=axes, shrink=.75,
               label="Paired success change versus corrupted actor")
  fig.suptitle("Stage 2 paired treatment effects: critic-free reward optimization is decisive",
               fontsize=11)
  save(fig, f"{STAGE2_FIGURES}/paired_effects")


def write_stage2_aggregate(rows: list[dict[str, str]]) -> None:
  conditions = []
  for row in rows:
    if row["condition"] not in conditions:
      conditions.append(row["condition"])
  with (STAGE2 / "aggregate_success.csv").open("w", newline="") as f:
    out = csv.writer(f, lineterminator="\n")
    out.writerow(["condition", "policy_mode", "successes", "episodes",
                  "aggregate_success"])
    for condition in conditions:
      for mode in ["sampled", "deterministic"]:
        subset = [row for row in rows if row["condition"] == condition
                  and row["policy_mode"] == mode]
        successes = sum(int(row["successes"]) for row in subset)
        episodes = sum(int(row["episodes"]) for row in subset)
        out.writerow([condition, mode, successes, episodes,
                      successes / episodes])


def read_stage3b(name: str) -> list[dict[str, str]]:
  with (STAGE3B / name).open() as f:
    return list(csv.DictReader(f))


def read_stage3a(name: str) -> list[dict[str, str]]:
  with (STAGE3A / name).open() as f:
    return list(csv.DictReader(f))


def stage3a_real_state_calibration() -> None:
  rows = read_stage3a("real_state_calibration.csv")
  x = np.arange(len(STAGE2_GOALS))
  width = .35
  fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.8),
                           constrained_layout=True)
  for index, label in enumerate(STAGE3A_LABELS):
    selected = {int(row["goal"]): row for row in rows
                if row["checkpoint"] == label}
    offset = (index - .5) * width
    for axis, field in zip(
        axes, ["online_bias", "slow_bias", "online_slow_gap_mae"]):
      axis.bar(x + offset, [float(selected[g][field]) for g in range(3)],
               width, label=label, color=STAGE3A_COLORS[label])
  for axis, title, ylabel in zip(
      axes,
      ["Online critic", "Slow critic", "Online--slow disagreement"],
      ["Bias vs realized return", "Bias vs realized return",
       "Mean absolute gap"]):
    axis.set_xticks(x, STAGE2_GOAL_LABELS, rotation=25, ha="right")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
  axes[-1].legend(frameon=False)
  fig.suptitle("Stage 3A: critic calibration on real posterior states",
               fontsize=11)
  save(fig, f"{STAGE3A_FIGURES}/real_state_calibration")


def stage3a_horizon_decomposition() -> None:
  rows = read_stage3a("combined_rollout_summary.csv")
  fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.8),
                           constrained_layout=True)
  for label in STAGE3A_LABELS:
    selected = [row for row in rows
                if row["checkpoint"] == label
                and row["proposal"] == "actor"
                and row["control_scope"] == "all"
                and row["relationship"] == "factual"]
    horizons = sorted({int(float(row["horizon"])) for row in selected})
    for axis, field in zip(
        axes, ["return_mean", "reward_only_mean", "bootstrap_mean"]):
      values = [np.mean([float(row[field]) for row in selected
                         if int(float(row["horizon"])) == horizon])
                for horizon in horizons]
      axis.plot(horizons, values, "o-", label=label,
                color=STAGE3A_COLORS[label])
  for axis, title in zip(
      axes, ["Full IR target", "Reward-only component", "Bootstrap component"]):
    axis.set_title(title)
    axis.set_xlabel("Imagination horizon H")
    axis.set_ylabel("Mean target contribution")
    axis.set_xticks([1, 3, 6, 15, 20])
  axes[-1].legend(frameon=False)
  fig.suptitle("Horizon-dependent provenance of the actor-selected IR target",
               fontsize=11)
  save(fig, f"{STAGE3A_FIGURES}/horizon_target_decomposition")


def stage3a_actor_vs_recorded() -> None:
  rows = read_stage3a("actor_vs_recorded.csv")
  fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.8),
                           constrained_layout=True)
  for label in STAGE3A_LABELS:
    selected = [row for row in rows
                if row["checkpoint"] == label
                and row["relationship"] == "factual"]
    horizons = sorted({int(row["horizon"]) for row in selected})
    for axis, field in zip(
        axes, ["actor_minus_recorded_return",
               "actor_minus_recorded_bootstrap"]):
      values = [np.mean([float(row[field]) for row in selected
                         if int(row["horizon"]) == horizon])
                for horizon in horizons]
      axis.plot(horizons, values, "o-", label=label,
                color=STAGE3A_COLORS[label])
  axes[0].set_title("Full return difference")
  axes[1].set_title("Bootstrap difference")
  for axis in axes:
    axis.axhline(0, color="black", lw=1)
    axis.set_xlabel("Imagination horizon H")
    axis.set_ylabel("Actor minus recorded proposal")
    axis.set_xticks([1, 3, 6, 15, 20])
  axes[-1].legend(frameon=False)
  fig.suptitle("Does the actor select more optimistic imagined trajectories?",
               fontsize=11)
  save(fig, f"{STAGE3A_FIGURES}/actor_vs_recorded_proposal")


def stage3b_aggregate_horizon() -> None:
  rows = read_stage3b("main_results.csv")
  lookup = {
      (int(row["checkpoint_step"]), row["condition"], row["goal"],
       row["policy_mode"]): float(row["success_rate"])
      for row in rows
  }
  horizons = [6, 15, 20]
  fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True,
                           sharex=True, sharey=True)
  for i, step in enumerate(STAGE3B_CHECKPOINTS):
    for j, mode in enumerate(["sampled", "deterministic"]):
      ax = axes[i, j]
      mean = lambda condition: np.mean([
          lookup[(step, condition, goal, mode)] for goal in STAGE2_GOALS])
      frozen = mean("frozen")
      standard = mean("standard_ir_h6")
      reward = [mean(f"reward_only_ir_h{h}") for h in horizons]
      ax.axhline(frozen, color="0.35", ls="--", lw=1.3,
                 label=f"Frozen ({frozen:.1%})")
      ax.scatter([6], [standard], marker="s", s=45, color="#4c78a8",
                 zorder=3, label=f"Standard IR H=6 ({standard:.1%})")
      ax.plot(horizons, reward, "o-", color="#54a24b", lw=2, ms=5,
              label="Reward-only IR")
      ax.set(xticks=horizons, ylim=(0, 1.02),
             title=f"{step / 1e6:.2f}M checkpoint — {mode}")
      if i == 1:
        ax.set_xlabel("Imagination horizon H")
      if j == 0:
        ax.set_ylabel("Mean success across goals")
      ax.legend(frameon=True, fontsize=7, loc="best")
  fig.suptitle("Natural forgetting: critic-free rehearsal improves aggregate behavior",
               fontsize=11)
  save(fig, f"{STAGE3B_FIGURES}/aggregate_horizon_response")


def stage3b_paired_effect_matrix() -> None:
  rows = read_stage3b("comparison_vs_frozen.csv")
  rows = [row for row in rows if row["condition"] != "frozen"]
  lookup = {
      (int(row["checkpoint_step"]), row["condition"], row["goal"],
       row["policy_mode"]): float(row["paired_delta_vs_frozen"])
      for row in rows
  }
  conditions = STAGE3B_CONDITIONS[1:]
  labels = ["Standard H=6", "Reward-only H=6", "Reward-only H=15",
            "Reward-only H=20"]
  columns = [(step, mode, goal)
             for step in STAGE3B_CHECKPOINTS
             for mode in ["sampled", "deterministic"]
             for goal in STAGE2_GOALS]
  matrix = np.array([
      [lookup[(step, condition, goal, mode)]
       for step, mode, goal in columns]
      for condition in conditions
  ])
  fig, ax = plt.subplots(figsize=(7.25, 2.9), constrained_layout=True)
  im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-.8, vmax=.8)
  ax.set_xticks(range(len(columns)), [
      f"{step / 1e6:.2f}M\n{mode[0].upper()} {goal.split('_')[0].title()}"
      for step, mode, goal in columns], rotation=45, ha="right", fontsize=7)
  ax.set_yticks(range(len(labels)), labels)
  ax.grid(False)
  for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
      value = matrix[i, j]
      ax.text(j, i, f"{value:+.0%}", ha="center", va="center", fontsize=7,
              color="white" if abs(value) > .5 else "black")
  for boundary in [2.5, 5.5, 8.5]:
    ax.axvline(boundary, color="white", lw=1.5)
  fig.colorbar(im, ax=ax, shrink=.75,
               label="Paired success change versus frozen actor")
  ax.set_title("Stage 3B paired effects across both checkpoints and policy modes")
  save(fig, f"{STAGE3B_FIGURES}/paired_effect_matrix")


def read_stage4a_results() -> list[dict[str, str]]:
  with (STAGE4A / "main_results.csv").open(newline="") as handle:
    return list(csv.DictReader(handle))


def read_stage4a_audit(arm: str) -> dict:
  path = STAGE4A / "attempts" / "audit" / arm / "01" / "summary.json"
  return json.loads(path.read_text())


def stage4a_head_selectivity() -> None:
  summaries = [read_stage4a_audit(arm) for arm in STAGE4A_AUDIT_ARMS]
  top1 = [entry["completion_selectivity"]["reward_top1_accuracy"]
          for entry in summaries]
  margins = [entry["completion_selectivity"]["reward_margin"]
             for entry in summaries]
  con_match = [entry["completion_selectivity"]["continuation_matching_mean"]
               for entry in summaries]
  con_other = [entry["completion_selectivity"]["continuation_nonmatching_mean"]
               for entry in summaries]
  x = np.arange(len(summaries))
  colors = ["#4c78a8", "#f58518", "#b279a2", "#54a24b", "#2f855a"]
  fig, axes = plt.subplots(1, 3, figsize=(7.55, 3.2),
                           constrained_layout=True)
  axes[0].bar(x, top1, color=colors)
  axes[0].axhline(.2, color="0.25", ls="--", lw=1, label="5-goal chance")
  axes[0].set(ylabel="Completion-goal top-1 accuracy", ylim=(0, 1.05),
              title="Goal identification")
  axes[0].legend(frameon=False, fontsize=7)
  axes[1].bar(x, margins, color=colors)
  axes[1].axhline(0, color="0.25", lw=1)
  axes[1].set(ylabel="Matching − nonmatching reward", ylim=(-.08, .64),
              title="Reward selectivity")
  width = .36
  axes[2].bar(x - width / 2, con_match, width, color="#e45756",
              label="Completed goal")
  axes[2].bar(x + width / 2, con_other, width, color="#72b7b2",
              label="Other goals")
  axes[2].set(ylabel="Predicted continuation", ylim=(0, 1.05),
              title="Goal-specific stopping")
  axes[2].legend(frameon=False, fontsize=7)
  for ax in axes:
    ax.set_xticks(x, STAGE4A_AUDIT_LABELS, rotation=36, ha="right",
                  fontsize=6.5)
    ax.grid(axis="x", visible=False)
  fig.suptitle("Stage 4A: counterfactual labels repair held-out goal semantics",
               fontsize=11)
  save(fig, f"{STAGE4A_FIGURES}/head_selectivity")


def stage4a_bounded_value() -> None:
  summary = read_stage4a_audit("counterfactual_bounded")
  values = summary["value"]
  targets = np.array([x["value_inclusive"]["target_mean"] for x in values])
  original = np.array([x["value_inclusive"]["prediction_mean"] for x in values])
  bounded = np.array([x["bounded_value_inclusive"]["prediction_mean"]
                      for x in values])
  original_auc = np.array([x["initial_value_success_auroc"] for x in values])
  bounded_auc = np.array([x["bounded_initial_success_auroc"] for x in values])
  x = np.arange(3)
  width = .25
  fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0),
                           constrained_layout=True)
  axes[0].bar(x - width, targets, width, label="Realized target", color="0.55")
  axes[0].bar(x, original, width, label="Original critic", color="#e45756")
  axes[0].bar(x + width, bounded, width, label="Bounded head", color="#54a24b")
  axes[0].set(ylabel="Mean value on held-out states", ylim=(0, 1.85),
              title="Scale calibration")
  axes[0].legend(frameon=False, fontsize=7)
  axes[1].bar(x - width / 2, original_auc, width, label="Original critic",
              color="#e45756")
  axes[1].bar(x + width / 2, bounded_auc, width, label="Bounded head",
              color="#54a24b")
  axes[1].axhline(.5, color="0.25", ls="--", lw=1, label="Chance")
  axes[1].set(ylabel="Initial-state success AUROC", ylim=(0, 1),
              title="Success ranking")
  axes[1].legend(frameon=False, fontsize=7)
  for ax in axes:
    ax.set_xticks(x, STAGE2_GOAL_LABELS, rotation=18, ha="right")
    ax.grid(axis="x", visible=False)
  fig.suptitle("A separate bounded success head repairs scale and improves ranking",
               fontsize=11)
  save(fig, f"{STAGE4A_FIGURES}/bounded_value_calibration")


def stage4a_success_matrix() -> None:
  rows = read_stage4a_results()
  lookup = {(row["condition"], row["policy_mode"], row["goal"]):
            float(row["success_rate"]) for row in rows}
  fig, axes = plt.subplots(1, 2, figsize=(7.45, 4.8),
                           constrained_layout=True, sharey=True)
  for ax, mode in zip(axes, ["sampled", "deterministic"]):
    matrix = np.array([[lookup[(condition, mode, goal)]
                        for goal in STAGE2_GOALS]
                       for condition in STAGE4A_CONDITIONS])
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(3), STAGE2_GOAL_LABELS, rotation=20, ha="right")
    ax.set_yticks(range(len(STAGE4A_CONDITION_LABELS)),
                  STAGE4A_CONDITION_LABELS, fontsize=7)
    ax.set_title(f"{mode.title()} policy")
    ax.grid(False)
    for i in range(matrix.shape[0]):
      for j in range(matrix.shape[1]):
        value = matrix[i, j]
        ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=7,
                color="white" if value < .35 or value > .78 else "black")
  fig.colorbar(im, ax=axes, shrink=.78, label="Real-environment success rate")
  fig.suptitle("Stage 4A paired actor-adaptation assay", fontsize=11)
  save(fig, f"{STAGE4A_FIGURES}/success_matrix")


def stage4a_paired_effects() -> None:
  rows = read_stage4a_results()
  adapted = STAGE4A_CONDITIONS[1:]
  lookup = {(row["condition"], row["policy_mode"], row["goal"]):
            float(row["paired_delta_vs_original_frozen"]) for row in rows}
  columns = [(mode, goal) for mode in ["sampled", "deterministic"]
             for goal in STAGE2_GOALS]
  matrix = np.array([[lookup[(condition, mode, goal)]
                      for mode, goal in columns] for condition in adapted])
  fig, ax = plt.subplots(figsize=(7.35, 3.8), constrained_layout=True)
  im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-.65, vmax=.65)
  ax.set_xticks(range(6), [
      f"{mode[0].upper()}\n{goal.split('_')[0].title()}"
      for mode, goal in columns])
  ax.set_yticks(range(len(adapted)), STAGE4A_CONDITION_LABELS[1:], fontsize=7)
  ax.axvline(2.5, color="white", lw=2)
  ax.grid(False)
  for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
      value = matrix[i, j]
      ax.text(j, i, f"{value:+.0%}", ha="center", va="center", fontsize=7,
              color="white" if abs(value) > .40 else "black")
  fig.colorbar(im, ax=ax, shrink=.8,
               label="Paired success change versus frozen actor")
  ax.set_title("Reward-only objectives improve broadly; original standard IR is unsafe")
  save(fig, f"{STAGE4A_FIGURES}/paired_effects")


def stage4a_mean_worst() -> None:
  rows = read_stage4a_results()
  lookup = {(row["condition"], row["policy_mode"], row["goal"]):
            float(row["success_rate"]) for row in rows}
  fig, axes = plt.subplots(1, 2, figsize=(7.35, 4.3), sharey=True)
  colors = mpl.cm.tab10(np.linspace(0, .9, len(STAGE4A_CONDITIONS)))
  for ax, mode in zip(axes, ["sampled", "deterministic"]):
    for condition, label, color in zip(
        STAGE4A_CONDITIONS, STAGE4A_CONDITION_LABELS, colors):
      vals = np.array([lookup[(condition, mode, goal)] for goal in STAGE2_GOALS])
      ax.scatter(vals.mean(), vals.min(), s=43, color=color, label=label,
                 edgecolor="white", linewidth=.4)
    ax.plot([0, 1], [0, 1], color="0.75", lw=1)
    ax.set(xlim=(.28, .93), ylim=(0, .93), xlabel="Mean success across goals",
           title=mode.title())
  axes[0].set_ylabel("Worst-goal success")
  handles, labels = axes[1].get_legend_handles_labels()
  fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
             fontsize=7, bbox_to_anchor=(.5, .01))
  fig.suptitle("Aggregate performance and weakest-task safety", fontsize=11)
  fig.subplots_adjust(left=.09, right=.99, top=.84, bottom=.37, wspace=.10)
  save(fig, f"{STAGE4A_FIGURES}/mean_worst_task")


def read_stage4b(name: str) -> list[dict[str, str]]:
  with (STAGE4B / name).open(newline="") as handle:
    return list(csv.DictReader(handle))


def stage4b_value_dose() -> None:
  rows = read_stage4b("goal_summary.csv")
  doses = [
      "reward_only_h15", "mixed_b005_h15", "mixed_b010_h15",
      "mixed_b025_h15", "mixed_b100_h15",
  ]
  colors = [COLORS["kill"], COLORS["bury"], COLORS["chop"]]
  for mode in ["sampled", "deterministic"]:
    lookup = {(row["condition"], row["goal"]): row for row in rows
              if row["policy_mode"] == mode}
    fig, ax = plt.subplots(figsize=(7.15, 3.8), constrained_layout=True)
    for goal, label, color in zip(STAGE2_GOALS, STAGE2_GOAL_LABELS, colors):
      xs = [float(lookup[(condition, goal)]["mix_beta"])
            for condition in doses]
      ys = [float(lookup[(condition, goal)]["success_rate"])
            for condition in doses]
      ax.plot(xs, ys, marker="o", color=color, label=label)
    ax.set_xscale("symlog", linthresh=.05)
    ax.set_xticks([0, .05, .1, .25, 1], ["0", ".05", ".10", ".25", "1"])
    ax.set(xlabel=r"Bounded-value dose $\beta$ ($H=15$, unclipped)",
           ylabel="Success rate", ylim=(0, 1),
           title=f"Study VII value-dose response --- {mode}")
    ax.legend(frameon=False)
    save(fig, f"{STAGE4B_FIGURES}/value_dose_{mode}")


def stage4b_mean_worst() -> None:
  rows = read_stage4b("condition_summary.csv")
  fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.7), sharey=True,
                           constrained_layout=True)
  x = np.arange(len(STAGE4B_CONDITIONS))
  for ax, mode in zip(axes, ["sampled", "deterministic"]):
    lookup = {row["condition"]: row for row in rows
              if row["policy_mode"] == mode}
    means = [float(lookup[name]["mean_goal_success"])
             for name in STAGE4B_CONDITIONS]
    worst = [float(lookup[name]["worst_goal_success"])
             for name in STAGE4B_CONDITIONS]
    ax.bar(x - .18, means, width=.36, label="Mean goal", color="#4c78a8")
    ax.bar(x + .18, worst, width=.36, label="Worst goal", color="#e45756")
    ax.set_xticks(x, [STAGE4B_LABELS[name] for name in STAGE4B_CONDITIONS],
                  rotation=34, ha="right", fontsize=6.5)
    ax.set(ylim=(0, 1), title=mode.title())
    ax.grid(axis="x", visible=False)
  axes[0].set_ylabel("Success rate")
  axes[0].legend(frameon=False, fontsize=7)
  fig.suptitle("Study VII mean and weakest-goal safety", fontsize=11)
  save(fig, f"{STAGE4B_FIGURES}/mean_worst_goal")


def stage4b_paired_delta() -> None:
  rows = read_stage4b("goal_summary.csv")
  fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8), sharey=True,
                           constrained_layout=True)
  for ax, mode in zip(axes, ["sampled", "deterministic"]):
    lookup = {(row["condition"], row["goal"]): row for row in rows
              if row["policy_mode"] == mode}
    matrix = np.array([
        [float(lookup[(condition, goal)]["paired_delta_vs_reward_only"])
         for condition in STAGE4B_CONDITIONS]
        for goal in STAGE2_GOALS])
    im = ax.imshow(matrix, vmin=-.5, vmax=.5, cmap="RdBu", aspect="auto")
    ax.set_xticks(range(len(STAGE4B_CONDITIONS)),
                  [STAGE4B_LABELS[name] for name in STAGE4B_CONDITIONS],
                  rotation=36, ha="right", fontsize=6.5)
    ax.set_yticks(range(3), STAGE2_GOAL_LABELS)
    ax.set_title(mode.title())
    ax.grid(False)
    for y in range(3):
      for x in range(len(STAGE4B_CONDITIONS)):
        ax.text(x, y, f"{matrix[y, x]:+.2f}", ha="center", va="center",
                fontsize=6, color="white" if abs(matrix[y, x]) > .34
                else "black")
  fig.colorbar(im, ax=axes, shrink=.78,
               label="Paired success change versus reward-only IR")
  fig.suptitle("Study VII benefit and harm by goal", fontsize=11)
  save(fig, f"{STAGE4B_FIGURES}/paired_delta_heatmap")


def stage4b_replicate_stability() -> None:
  rows = read_stage4b("main_results.csv")
  fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.7), sharey=True,
                           constrained_layout=True)
  for ax, mode in zip(axes, ["sampled", "deterministic"]):
    for replicate in range(3):
      values = []
      for condition in STAGE4B_CONDITIONS:
        selected = [float(row["success_rate"]) for row in rows
                    if row["policy_mode"] == mode and
                    row["condition"] == condition and
                    int(row["replicate"]) == replicate]
        values.append(float(np.mean(selected)))
      ax.plot(range(len(values)), values, marker="o", ms=3,
              label=f"Replicate {replicate}", alpha=.8)
    ax.set_xticks(range(len(STAGE4B_CONDITIONS)),
                  [STAGE4B_LABELS[name] for name in STAGE4B_CONDITIONS],
                  rotation=34, ha="right", fontsize=6.5)
    ax.set(ylim=(0, 1), title=mode.title())
  axes[0].set_ylabel("Macro success across goals")
  axes[0].legend(frameon=False, fontsize=7)
  fig.suptitle("Study VII adaptation-seed stability", fontsize=11)
  save(fig, f"{STAGE4B_FIGURES}/replicate_stability")


def stage4b_mode_tradeoff() -> None:
  rows = read_stage4b("condition_summary.csv")
  lookup = {(row["condition"], row["policy_mode"]): row for row in rows}
  conditions = STAGE4B_CONDITIONS[2:]
  sampled = np.array([
      float(lookup[(name, "sampled")]["mean_goal_delta_vs_reward_only"])
      for name in conditions])
  deterministic = np.array([
      float(lookup[(name, "deterministic")]["mean_goal_delta_vs_reward_only"])
      for name in conditions])
  fig, ax = plt.subplots(figsize=(7.15, 3.8), constrained_layout=True)
  ax.axhline(0, color="0.3", lw=1)
  ax.axvline(0, color="0.3", lw=1)
  ax.axhspan(-.22, 0, xmin=.5, color="#fce5cc", alpha=.45)
  for index, name in enumerate(conditions):
    ax.scatter(sampled[index], deterministic[index], s=45,
               color=mpl.cm.tab10(index / max(1, len(conditions) - 1)),
               edgecolor="white", linewidth=.5)
    ax.annotate(STAGE4B_LABELS[name].replace("\n", " "),
                (sampled[index], deterministic[index]), xytext=(4, 3),
                textcoords="offset points", fontsize=6.5)
  ax.set(xlabel="Sampled macro change versus reward-only",
         ylabel="Deterministic macro change versus reward-only",
         xlim=(-.04, .16), ylim=(-.19, .03),
         title="Small value doses improve sampling without improving the mode")
  save(fig, f"{STAGE4B_FIGURES}/sampled_deterministic_tradeoff")


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
  stage2_rows = read_stage2()
  stage2_corruption_calibration()
  stage2_condition_matrix(stage2_rows)
  stage2_horizon_aggregate(stage2_rows)
  stage2_reward_only_by_goal(stage2_rows)
  stage2_paired_effects()
  write_stage2_aggregate(stage2_rows)
  stage3b_aggregate_horizon()
  stage3b_paired_effect_matrix()
  stage3a_real_state_calibration()
  stage3a_horizon_decomposition()
  stage3a_actor_vs_recorded()
  stage4a_head_selectivity()
  stage4a_bounded_value()
  stage4a_success_matrix()
  stage4a_paired_effects()
  stage4a_mean_worst()
  stage4b_value_dose()
  stage4b_mean_worst()
  stage4b_paired_delta()
  stage4b_replicate_stability()
  stage4b_mode_tradeoff()


if __name__ == "__main__":
  main()
