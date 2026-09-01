#!/usr/bin/env python3
"""Build the standalone MiniGrid baseline report as a polished vector PDF."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORT_PDF = ROOT / "report.pdf"
SEQ_DATA = DATA / "sequential_baseline"
SEQ_ANALYSIS = SEQ_DATA / "analysis"

BLUE = colors.HexColor("#4C78A8")
ORANGE = colors.HexColor("#F58518")
GREEN = colors.HexColor("#54A24B")
RED = colors.HexColor("#E45756")
PURPLE = colors.HexColor("#B279A2")
TEAL = colors.HexColor("#72B7B2")
DARK = colors.HexColor("#222B35")
MID = colors.HexColor("#66717D")
GRID = colors.HexColor("#D8DEE5")
LIGHTBLUE = colors.HexColor("#E8F1FA")
LIGHTORANGE = colors.HexColor("#FFF0DD")
LIGHTGREEN = colors.HexColor("#E9F5E7")
LIGHTRED = colors.HexColor("#FBE8E8")

TASKS = [
    "go_to_door",
    "fetch",
    "go_to_object",
    "put_near",
    "unlock_pickup",
    "pickup_dist",
]
LABELS = {
    "go_to_door": "GoToDoor",
    "fetch": "Fetch",
    "go_to_object": "GoToObject",
    "put_near": "PutNear",
    "unlock_pickup": "UnlockPickup",
    "pickup_dist": "PickupDist",
}
TASK_COLORS = dict(zip(TASKS, (BLUE, ORANGE, GREEN, PURPLE, RED, TEAL)))

SEQ_TASKS = ["go_to_door", "fetch", "go_to_object", "put_near"]
ARMS = ["fifo_100k", "uniform_reservoir", "current_old_50_50"]
ARM_LABELS = {
    "fifo_100k": "FIFO 100k",
    "uniform_reservoir": "Uniform reservoir",
    "current_old_50_50": "50:50 current/old",
}
ARM_COLORS = {
    "fifo_100k": BLUE,
    "uniform_reservoir": ORANGE,
    "current_old_50_50": GREEN,
}


def csv_rows(name: str) -> list[dict[str, str]]:
  with (DATA / name).open(newline="") as stream:
    return list(csv.DictReader(stream))


EVAL_ROWS = csv_rows("evaluation_summary.csv")
RUN_ROWS = csv_rows("run_summary.csv")
TASK_ROWS = csv_rows("task_summary.csv")
SPEC = json.loads((DATA / "experiment_spec.json").read_text())
STATUS = json.loads((DATA / "supervisor_status.json").read_text())

SEQ_EVAL_ROWS = []
SEQ_TRANSFER_ROWS = []
SEQ_REPLAY_ROWS = []
SEQ_COMPONENT_ROWS = []
SEQ_ACTOR_DRIFT_ROWS = []
SEQ_DIAGNOSTIC_ROWS = []
SEQ_SPEC = {}
SEQ_STATUS = {}
if SEQ_DATA.is_dir():
  def seq_csv(name: str) -> list[dict[str, str]]:
    with (SEQ_ANALYSIS / name).open(newline="") as stream:
      return list(csv.DictReader(stream))

  SEQ_EVAL_ROWS = seq_csv("evaluation_curves.csv")
  SEQ_TRANSFER_ROWS = seq_csv("transfer_and_forgetting.csv")
  SEQ_REPLAY_ROWS = seq_csv("replay_allocation.csv")
  SEQ_COMPONENT_ROWS = seq_csv("component_losses_by_goal.csv")
  SEQ_ACTOR_DRIFT_ROWS = seq_csv("actor_same_state_drift.csv")
  SEQ_DIAGNOSTIC_ROWS = seq_csv("diagnostic_inventory.csv")
  SEQ_SPEC = json.loads((SEQ_DATA / "experiment_spec.json").read_text())
  SEQ_STATUS = json.loads((SEQ_DATA / "supervisor_status.json").read_text())


def mean_eval(task: str, checkpoint: int, mode: str) -> tuple[float, float]:
  values = [
      float(row["success_rate"])
      for row in EVAL_ROWS
      if row["task"] == task
      and int(row["checkpoint"]) == checkpoint
      and row["policy_mode"] == mode
  ]
  return fmean(values), pstdev(values)


def lerp_color(left, right, amount: float):
  amount = max(0.0, min(1.0, amount))
  return colors.Color(
      left.red + (right.red - left.red) * amount,
      left.green + (right.green - left.green) * amount,
      left.blue + (right.blue - left.blue) * amount,
  )


def seq_mean_eval(arm: str, task: str, step: int,
                  mode: str = "sampled") -> tuple[float, float]:
  values = [
      float(row["success_rate"])
      for row in SEQ_EVAL_ROWS
      if row["arm"] == arm and row["task"] == task
      and int(row["step"]) == step and row["policy_mode"] == mode
  ]
  return (fmean(values), pstdev(values)) if values else (math.nan, math.nan)


def seq_transfer_mean(arm: str, task: str, field: str,
                      mode: str = "sampled") -> float:
  values = [
      float(row[field])
      for row in SEQ_TRANSFER_ROWS
      if row["arm"] == arm and row["task"] == task
      and row["policy_mode"] == mode
  ]
  return fmean(values)


def replay_shares(arm: str, phase: int) -> list[float]:
  selected = [
      row for row in SEQ_COMPONENT_ROWS
      if row["arm"] == arm
      and phase * 200_000 < int(row["step"]) <= (phase + 1) * 200_000
  ]
  return [
      fmean(float(row[f"train/batch_goal/{goal}"]) for row in selected)
      for goal in range(4)
  ]


def load_head_summaries() -> list[dict]:
  output = []
  base = SEQ_DATA / "diagnostics" / "head"
  if not base.is_dir():
    return output
  for path in base.glob("*/seed_*/step_*/summary.json"):
    arm, seed_name, step_name, _ = path.relative_to(base).parts
    data = json.loads(path.read_text())
    value_rows = data["value"]
    total = sum(row["value_inclusive"]["count"] for row in value_rows)
    correlated = [
        row for row in value_rows
        if row["value_inclusive"]["pearson"] is not None
    ]
    correlated_count = sum(
        row["value_inclusive"]["count"] for row in correlated)
    output.append({
        "arm": arm,
        "seed": int(seed_name.rsplit("_", 1)[1]),
        "step": int(step_name.rsplit("_", 1)[1]),
        "reward_auroc": data["reward_factual"]["auroc"],
        "reward_ap": data["reward_factual"]["average_precision"],
        "reward_margin": data["completion_selectivity"]["reward_margin"],
        "reward_top1": data["completion_selectivity"]["reward_top1_accuracy"],
        "completion_count": data["completion_selectivity"]["count"],
        "value_prediction": sum(
            row["value_inclusive"]["prediction_mean"]
            * row["value_inclusive"]["count"] for row in value_rows) / total,
        "value_target": sum(
            row["value_inclusive"]["target_mean"]
            * row["value_inclusive"]["count"] for row in value_rows) / total,
        "value_bias": sum(
            row["value_inclusive"]["bias"]
            * row["value_inclusive"]["count"] for row in value_rows) / total,
        "value_pearson": (
            sum(row["value_inclusive"]["pearson"]
                * row["value_inclusive"]["count"] for row in correlated)
            / correlated_count if correlated_count else math.nan),
        "slow_gap": sum(
            row["value_slow_gap_mae"] * row["value_inclusive"]["count"]
            for row in value_rows) / total,
    })
  return output


def load_critic_aggregates() -> list[dict]:
  output = []
  base = SEQ_DATA / "diagnostics" / "critic_provenance"
  if not base.is_dir():
    return output
  for path in base.glob("*/seed_*/step_*/summary.json"):
    arm, seed_name, step_name, _ = path.relative_to(base).parts
    data = json.loads(path.read_text())
    for row in data["aggregates"]:
      if row["relationship"] == "factual":
        output.append({
            **row,
            "arm": arm,
            "seed": int(seed_name.rsplit("_", 1)[1]),
            "step": int(step_name.rsplit("_", 1)[1]),
        })
  return output


HEAD_SUMMARIES = load_head_summaries()
CRITIC_AGGREGATES = load_critic_aggregates()


def head_mean(arm: str, step: int, field: str) -> float:
  values = [
      float(row[field]) for row in HEAD_SUMMARIES
      if row["arm"] == arm and row["step"] == step
      and row[field] is not None and math.isfinite(float(row[field]))
  ]
  return fmean(values) if values else math.nan


def critic_weighted_mean(arm: str, horizon: int, proposal: str, field: str,
                         trained_only: bool = True) -> float:
  selected = [
      row for row in CRITIC_AGGREGATES
      if row["arm"] == arm and row["horizon"] == horizon
      and row["proposal"] == proposal
      and (not trained_only or row["step"] > 0)
      and row[field] is not None and math.isfinite(float(row[field]))
  ]
  total = sum(int(row["rows"]) for row in selected)
  return sum(float(row[field]) * int(row["rows"]) for row in selected) / total


class ReportDocTemplate(BaseDocTemplate):

  def __init__(self, filename: str, **kwargs):
    super().__init__(filename, **kwargs)
    frame = Frame(
        self.leftMargin,
        self.bottomMargin,
        self.width,
        self.height,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="body",
    )
    self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPageEnd=self._page))

  def _page(self, canvas, doc):
    canvas.saveState()
    if doc.page > 2:
      canvas.setStrokeColor(GRID)
      canvas.setLineWidth(0.45)
      canvas.line(self.leftMargin, A4[1] - 13.5 * mm,
                  A4[0] - self.rightMargin, A4[1] - 13.5 * mm)
      canvas.setFont("Times-Roman", 7.5)
      canvas.setFillColor(MID)
      canvas.drawString(self.leftMargin, A4[1] - 11.0 * mm,
                        "MiniGrid Continual Learning Experiments")
      canvas.drawRightString(A4[0] - self.rightMargin, A4[1] - 11.0 * mm,
                             "Continual Imagined Rehearsal")
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(MID)
    canvas.drawCentredString(A4[0] / 2, 10.5 * mm, str(doc.page))
    canvas.restoreState()

  def afterFlowable(self, flowable):
    if isinstance(flowable, Paragraph):
      style = flowable.style.name
      if style in ("H1", "H2"):
        level = 0 if style == "H1" else 1
        text = flowable.getPlainText()
        key = f"heading-{level}-{self.seq.nextf('heading')}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)
        self.notify("TOCEntry", (level, text, self.page, key))


class LearningCurves(Flowable):

  def __init__(self, width=515, height=326):
    super().__init__()
    self.width = width
    self.height = height

  def draw(self):
    c = self.canv
    c.saveState()
    panel_w, panel_h = 158, 126
    gap_x, gap_y = 17, 26
    origin_x, origin_y = 5, 10
    checkpoints = [25_000, 50_000, 75_000, 100_000]
    c.setFont("Times-Roman", 7.5)
    c.setStrokeColor(BLUE)
    c.setFillColor(BLUE)
    c.line(183, self.height - 8, 203, self.height - 8)
    c.drawString(207, self.height - 11, "Sampled")
    c.setStrokeColor(ORANGE)
    c.setFillColor(ORANGE)
    c.setDash(2, 1)
    c.line(274, self.height - 8, 294, self.height - 8)
    c.setDash()
    c.drawString(298, self.height - 11, "Deterministic")
    for index, task in enumerate(TASKS):
      row, col = divmod(index, 3)
      x0 = origin_x + col * (panel_w + gap_x)
      y0 = self.height - 26 - (row + 1) * panel_h - row * gap_y
      plot_x, plot_y = x0 + 28, y0 + 23
      plot_w, plot_h = panel_w - 35, panel_h - 47
      c.setFillColor(DARK)
      c.setFont("Times-Bold", 8.3)
      c.drawCentredString(x0 + panel_w / 2, y0 + panel_h - 13, LABELS[task])
      c.setStrokeColor(GRID)
      c.setLineWidth(.4)
      for ytick in (0, .25, .5, .75, 1):
        yy = plot_y + ytick * plot_h
        c.line(plot_x, yy, plot_x + plot_w, yy)
        if col == 0:
          c.setFont("Times-Roman", 6.5)
          c.setFillColor(MID)
          c.drawRightString(plot_x - 4, yy - 2, f"{ytick:.2g}")
      threshold_y = plot_y + .7 * plot_h
      c.setStrokeColor(colors.HexColor("#777777"))
      c.setDash(2, 2)
      c.line(plot_x, threshold_y, plot_x + plot_w, threshold_y)
      c.setDash()
      c.setStrokeColor(DARK)
      c.line(plot_x, plot_y, plot_x, plot_y + plot_h)
      c.line(plot_x, plot_y, plot_x + plot_w, plot_y)
      for checkpoint in checkpoints:
        xx = plot_x + (checkpoint - 25_000) / 75_000 * plot_w
        c.setFont("Times-Roman", 6.5)
        c.setFillColor(MID)
        c.drawCentredString(xx, plot_y - 10, f"{checkpoint // 1000}")
      for mode, color in (("sampled", BLUE), ("deterministic", ORANGE)):
        means, stds = zip(*(mean_eval(task, checkpoint, mode)
                            for checkpoint in checkpoints))
        upper, lower = [], []
        for checkpoint, mean, sd in zip(checkpoints, means, stds):
          xx = plot_x + (checkpoint - 25_000) / 75_000 * plot_w
          upper.append((xx, plot_y + min(1, mean + sd) * plot_h))
          lower.append((xx, plot_y + max(0, mean - sd) * plot_h))
        c.setFillColor(lerp_color(colors.white, color, .17))
        path = c.beginPath()
        path.moveTo(*upper[0])
        for point in upper[1:]:
          path.lineTo(*point)
        for point in reversed(lower):
          path.lineTo(*point)
        path.close()
        c.drawPath(path, fill=1, stroke=0)
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(1.35)
        if mode == "deterministic":
          c.setDash(2, 1)
        points = []
        for checkpoint, mean in zip(checkpoints, means):
          xx = plot_x + (checkpoint - 25_000) / 75_000 * plot_w
          yy = plot_y + mean * plot_h
          points.append((xx, yy))
        for left, right in zip(points, points[1:]):
          c.line(*left, *right)
        c.setDash()
        for xx, yy in points:
          if mode == "sampled":
            c.circle(xx, yy, 1.8, stroke=0, fill=1)
          else:
            c.rect(xx - 1.55, yy - 1.55, 3.1, 3.1, stroke=0, fill=1)
      if col == 0:
        c.saveState()
        c.translate(x0 + 6, plot_y + plot_h / 2)
        c.rotate(90)
        c.setFont("Times-Roman", 7)
        c.setFillColor(DARK)
        c.drawCentredString(0, 0, "Success rate")
        c.restoreState()
      if row == 1:
        c.setFont("Times-Roman", 7)
        c.setFillColor(DARK)
        c.drawCentredString(plot_x + plot_w / 2, y0 + 1, "Training steps (thousands)")
    c.restoreState()


class SeedMatrix(Flowable):

  def __init__(self, width=515, height=242):
    super().__init__()
    self.width = width
    self.height = height

  def draw(self):
    c = self.canv
    c.saveState()
    values = {}
    for row in RUN_ROWS:
      values[(row["task"], int(row["seed"]), "sampled")] = float(row["final_sampled_success"])
      values[(row["task"], int(row["seed"]), "deterministic")] = float(row["final_deterministic_success"])
    label_w, cell_w, cell_h = 78, 47, 25
    panel_w, gap = 3 * cell_w, 35
    base_x, base_y = 88, 32
    for panel, mode in enumerate(("sampled", "deterministic")):
      x0 = base_x + panel * (panel_w + gap)
      c.setFont("Times-Bold", 9)
      c.setFillColor(DARK)
      c.drawCentredString(x0 + panel_w / 2, self.height - 15, mode.capitalize() + " policy")
      for seed in range(3):
        c.setFont("Times-Roman", 7.5)
        c.drawCentredString(x0 + seed * cell_w + cell_w / 2, self.height - 31, f"Seed {seed}")
      for rindex, task in enumerate(TASKS):
        yy = base_y + (len(TASKS) - 1 - rindex) * cell_h
        if panel == 0:
          c.setFont("Times-Roman", 8)
          c.setFillColor(DARK)
          c.drawRightString(x0 - 7, yy + 8, LABELS[task])
        for seed in range(3):
          value = values[(task, seed, mode)]
          xx = x0 + seed * cell_w
          fill = lerp_color(colors.white, colors.HexColor("#1F5F99"), value)
          c.setFillColor(fill)
          c.setStrokeColor(colors.white)
          c.rect(xx, yy, cell_w, cell_h, fill=1, stroke=1)
          c.setFillColor(colors.white if value >= .58 else DARK)
          c.setFont("Times-Bold", 8)
          c.drawCentredString(xx + cell_w / 2, yy + 8, f"{100 * value:.0f}%")
    c.setFont("Times-Roman", 7)
    c.setFillColor(MID)
    c.drawCentredString(self.width / 2, 8, "Each cell contains 50 evaluation episodes at the 100k checkpoint")
    c.restoreState()


class PolicyAgreement(Flowable):

  def __init__(self, width=515, height=330):
    super().__init__()
    self.width = width
    self.height = height

  def draw(self):
    paired = defaultdict(dict)
    for row in EVAL_ROWS:
      key = (row["task"], int(row["training_seed"]), int(row["checkpoint"]))
      paired[key][row["policy_mode"]] = float(row["success_rate"])
    points = [(key, value) for key, value in paired.items()
              if "sampled" in value and "deterministic" in value]
    deltas = [v["sampled"] - v["deterministic"] for _, v in points]
    c = self.canv
    c.saveState()
    x0, y0, size = 74, 37, 260
    c.setStrokeColor(GRID)
    c.setLineWidth(.45)
    for tick in (0, .25, .5, .75, 1):
      xx, yy = x0 + tick * size, y0 + tick * size
      c.line(x0, yy, x0 + size, yy)
      c.line(xx, y0, xx, y0 + size)
      c.setFont("Times-Roman", 7)
      c.setFillColor(MID)
      c.drawRightString(x0 - 5, yy - 2, f"{tick:.2g}")
      c.drawCentredString(xx, y0 - 11, f"{tick:.2g}")
    c.setStrokeColor(DARK)
    c.line(x0, y0, x0, y0 + size)
    c.line(x0, y0, x0 + size, y0)
    c.setDash(3, 2)
    c.line(x0, y0, x0 + size, y0 + size)
    c.setDash()
    for (task, _seed, _checkpoint), value in points:
      xx = x0 + value["deterministic"] * size
      yy = y0 + value["sampled"] * size
      c.setFillColor(TASK_COLORS[task])
      c.setStrokeColor(colors.white)
      c.circle(xx, yy, 2.7, fill=1, stroke=1)
    c.setFont("Times-Roman", 8.5)
    c.setFillColor(DARK)
    c.drawCentredString(x0 + size / 2, 10, "Deterministic success")
    c.saveState()
    c.translate(16, y0 + size / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Sampled success")
    c.restoreState()
    legend_x, legend_y = 365, 275
    c.setFont("Times-Bold", 8.5)
    c.drawString(legend_x, legend_y + 19, "Task")
    for index, task in enumerate(TASKS):
      yy = legend_y - index * 18
      c.setFillColor(TASK_COLORS[task])
      c.circle(legend_x + 4, yy + 2, 3, fill=1, stroke=0)
      c.setFillColor(DARK)
      c.setFont("Times-Roman", 7.8)
      c.drawString(legend_x + 13, yy - 1, LABELS[task])
    box_y = 66
    c.setFillColor(LIGHTBLUE)
    c.setStrokeColor(BLUE)
    c.roundRect(354, box_y, 151, 65, 4, fill=1, stroke=1)
    c.setFillColor(DARK)
    c.setFont("Times-Bold", 8)
    c.drawString(364, box_y + 48, "72 paired conditions")
    c.setFont("Times-Roman", 7.5)
    c.drawString(364, box_y + 33,
                 f"Mean signed difference: {100 * fmean(deltas):+.2f} pp")
    c.drawString(364, box_y + 18,
                 f"Mean absolute difference: {100 * fmean(abs(x) for x in deltas):.2f} pp")
    c.restoreState()


class DifficultySummary(Flowable):

  def __init__(self, width=515, height=292):
    super().__init__()
    self.width = width
    self.height = height

  def draw(self):
    by_task = {row["task"]: row for row in TASK_ROWS}
    c = self.canv
    c.saveState()
    base_y, row_h = 48, 32
    left_x, left_w = 95, 143
    right_x, right_w = 327, 168
    c.setFont("Times-Bold", 9)
    c.setFillColor(DARK)
    c.drawCentredString(left_x + left_w / 2, self.height - 14,
                        "Normalized training success AUC")
    c.drawCentredString(right_x + right_w / 2, self.height - 14,
                        "Steps to 70% rolling success")
    for tick in (0, .1, .2, .3):
      xx = left_x + tick / .32 * left_w
      c.setStrokeColor(GRID)
      c.line(xx, base_y, xx, base_y + row_h * 6)
      c.setFont("Times-Roman", 7)
      c.setFillColor(MID)
      c.drawCentredString(xx, base_y - 12, f"{tick:.1f}")
    for tick in (50, 75, 100):
      xx = right_x + (tick - 50) / 60 * right_w
      c.setStrokeColor(GRID)
      c.line(xx, base_y, xx, base_y + row_h * 6)
      c.setFont("Times-Roman", 7)
      c.setFillColor(MID)
      c.drawCentredString(xx, base_y - 12, f"{tick}k")
    for index, task in enumerate(TASKS):
      yy = base_y + (5 - index) * row_h + row_h / 2
      row = by_task[task]
      auc = float(row["mean_training_auc"])
      auc_sd = float(row["std_training_auc"])
      c.setFillColor(DARK)
      c.setFont("Times-Roman", 8)
      c.drawRightString(left_x - 7, yy - 3, LABELS[task])
      c.setFillColor(TASK_COLORS[task])
      c.rect(left_x, yy - 6, auc / .32 * left_w, 12, fill=1, stroke=0)
      c.setStrokeColor(DARK)
      c.setLineWidth(.65)
      lo = left_x + max(0, auc - auc_sd) / .32 * left_w
      hi = left_x + min(.32, auc + auc_sd) / .32 * left_w
      c.line(lo, yy, hi, yy)
      c.line(lo, yy - 3, lo, yy + 3)
      c.line(hi, yy - 3, hi, yy + 3)
      solved = int(row["solved_seeds"])
      threshold = row["median_training_steps_to_threshold"]
      if threshold:
        step = float(threshold) / 1000
        xx = right_x + (step - 50) / 60 * right_w
        c.setFillColor(TASK_COLORS[task])
        c.circle(xx, yy, 3.4, fill=1, stroke=0)
        c.setFillColor(DARK)
        c.setFont("Times-Roman", 7.5)
        c.drawString(min(xx + 6, right_x + right_w - 43), yy - 3,
                     f"{step:.0f}k ({solved}/3)")
      else:
        c.setFillColor(MID)
        c.setFont("Times-Italic", 7.5)
        c.drawRightString(right_x + right_w, yy - 3,
                          f"not reached ({solved}/3)")
    c.setFont("Times-Roman", 7.2)
    c.setFillColor(MID)
    c.drawCentredString(left_x + left_w / 2, 12, "Error bars: one population SD")
    c.drawCentredString(right_x + right_w / 2, 12, "Median only among seeds reaching threshold")
    c.restoreState()


class SequentialCurves(Flowable):

  def __init__(self, width=515, height=338):
    super().__init__()
    self.width = width
    self.height = height

  def draw(self):
    c = self.canv
    c.saveState()
    panel_w, panel_h = 244, 135
    gap_x, gap_y = 18, 32
    origin_x, origin_y = 4, 8
    c.setFont("Times-Roman", 7.3)
    legend_x = 104
    for index, arm in enumerate(ARMS):
      x = legend_x + index * 128
      c.setStrokeColor(ARM_COLORS[arm])
      c.setFillColor(ARM_COLORS[arm])
      c.setLineWidth(1.5)
      c.line(x, self.height - 8, x + 18, self.height - 8)
      c.drawString(x + 23, self.height - 11, ARM_LABELS[arm])
    for index, task in enumerate(SEQ_TASKS):
      row, col = divmod(index, 2)
      x0 = origin_x + col * (panel_w + gap_x)
      y0 = self.height - 24 - (row + 1) * panel_h - row * gap_y
      px, py = x0 + 31, y0 + 23
      pw, ph = panel_w - 39, panel_h - 46
      for phase, fill in enumerate((LIGHTBLUE, LIGHTORANGE, LIGHTGREEN, colors.HexColor("#F4EBF3"))):
        left = px + phase * pw / 4
        c.setFillColor(lerp_color(colors.white, fill, .42))
        c.rect(left, py, pw / 4, ph, fill=1, stroke=0)
      c.setFont("Times-Bold", 8.5)
      c.setFillColor(DARK)
      c.drawCentredString(x0 + panel_w / 2, y0 + panel_h - 13, LABELS[task])
      c.setLineWidth(.4)
      for ytick in (0, .25, .5, .75, 1):
        yy = py + ytick * ph
        c.setStrokeColor(GRID)
        c.line(px, yy, px + pw, yy)
        c.setFont("Times-Roman", 6.4)
        c.setFillColor(MID)
        c.drawRightString(px - 4, yy - 2, f"{ytick:.2g}")
      for xtick in (0, 200, 400, 600, 800):
        xx = px + xtick / 800 * pw
        c.setStrokeColor(MID if xtick in (200, 400, 600) else GRID)
        c.setLineWidth(.55 if xtick in (200, 400, 600) else .35)
        c.line(xx, py, xx, py + ph)
        c.setFont("Times-Roman", 6.4)
        c.setFillColor(MID)
        c.drawCentredString(xx, py - 10, str(xtick))
      c.setStrokeColor(DARK)
      c.line(px, py, px, py + ph)
      c.line(px, py, px + pw, py)
      for arm in ARMS:
        steps = sorted({
            int(item["step"]) for item in SEQ_EVAL_ROWS
            if item["arm"] == arm and item["task"] == task
            and item["policy_mode"] == "sampled"
        })
        points = []
        for step in steps:
          value, _ = seq_mean_eval(arm, task, step)
          points.append((px + step / 800_000 * pw, py + value * ph))
        c.setStrokeColor(ARM_COLORS[arm])
        c.setFillColor(ARM_COLORS[arm])
        c.setLineWidth(1.35)
        for left, right in zip(points, points[1:]):
          c.line(*left, *right)
        for xx, yy in points:
          c.circle(xx, yy, 1.25, fill=1, stroke=0)
      if col == 0:
        c.saveState()
        c.translate(x0 + 7, py + ph / 2)
        c.rotate(90)
        c.setFillColor(DARK)
        c.setFont("Times-Roman", 7)
        c.drawCentredString(0, 0, "Success rate")
        c.restoreState()
      if row == 1:
        c.setFillColor(DARK)
        c.setFont("Times-Roman", 7)
        c.drawCentredString(px + pw / 2, y0 + 1, "Cumulative training steps (thousands)")
    c.restoreState()


class AcquisitionRetentionMatrix(Flowable):

  def __init__(self, width=515, height=202):
    super().__init__()
    self.width = width
    self.height = height

  def draw(self):
    c = self.canv
    c.saveState()
    label_w, cell_w, cell_h = 87, 45, 31
    panel_w, gap = 4 * cell_w, 43
    base_x, base_y = 87, 29
    for panel, (field, title) in enumerate((
        ("post_acquisition_success", "Immediately after task phase"),
        ("final_success", "At final 800k checkpoint"))):
      x0 = base_x + panel * (panel_w + gap)
      c.setFont("Times-Bold", 8.6)
      c.setFillColor(DARK)
      c.drawCentredString(x0 + panel_w / 2, self.height - 14, title)
      for task_index, task in enumerate(SEQ_TASKS):
        c.setFont("Times-Roman", 6.6)
        c.drawCentredString(
            x0 + task_index * cell_w + cell_w / 2,
            self.height - 31, LABELS[task])
      for arm_index, arm in enumerate(ARMS):
        yy = base_y + (len(ARMS) - 1 - arm_index) * cell_h
        if panel == 0:
          c.setFont("Times-Roman", 7.4)
          c.setFillColor(DARK)
          c.drawRightString(x0 - 6, yy + 11, ARM_LABELS[arm])
        for task_index, task in enumerate(SEQ_TASKS):
          value = seq_transfer_mean(arm, task, field)
          xx = x0 + task_index * cell_w
          c.setFillColor(lerp_color(colors.white, colors.HexColor("#1F5F99"), value))
          c.setStrokeColor(colors.white)
          c.rect(xx, yy, cell_w, cell_h, fill=1, stroke=1)
          c.setFillColor(colors.white if value >= .58 else DARK)
          c.setFont("Times-Bold", 7.5)
          c.drawCentredString(xx + cell_w / 2, yy + 11, f"{100 * value:.0f}%")
    c.setFillColor(MID)
    c.setFont("Times-Roman", 7)
    c.drawCentredString(self.width / 2, 8, "Sampled-policy mean across three seeds; each seed contributes 50 episodes per cell")
    c.restoreState()


class ReplayAllocation(Flowable):

  def __init__(self, width=515, height=188):
    super().__init__()
    self.width = width
    self.height = height

  def draw(self):
    c = self.canv
    c.saveState()
    label_w, bar_w, bar_h, gap = 100, 93, 20, 12
    x0, y0 = 101, 49
    for phase, task in enumerate(SEQ_TASKS):
      xx = x0 + phase * (bar_w + gap)
      c.setFillColor(DARK)
      c.setFont("Times-Bold", 7.5)
      c.drawCentredString(xx + bar_w / 2, self.height - 14, f"Phase {chr(65 + phase)}")
      c.setFont("Times-Roman", 6.8)
      c.drawCentredString(xx + bar_w / 2, self.height - 27, LABELS[task])
    for arm_index, arm in enumerate(ARMS):
      yy = y0 + (len(ARMS) - 1 - arm_index) * 34
      c.setFillColor(DARK)
      c.setFont("Times-Roman", 7.5)
      c.drawRightString(x0 - 7, yy + 7, ARM_LABELS[arm])
      for phase in range(4):
        xx = x0 + phase * (bar_w + gap)
        shares = replay_shares(arm, phase)
        offset = 0.0
        for task_index, value in enumerate(shares):
          if value <= 0:
            continue
          width = value * bar_w
          c.setFillColor(TASK_COLORS[SEQ_TASKS[task_index]])
          c.rect(xx + offset, yy, width, bar_h, fill=1, stroke=0)
          if width >= 21:
            c.setFillColor(colors.white if task_index != 1 else DARK)
            c.setFont("Times-Bold", 6.3)
            c.drawCentredString(xx + offset + width / 2, yy + 7,
                                f"{100 * value:.0f}%")
          offset += width
        c.setStrokeColor(colors.white)
        c.rect(xx, yy, bar_w, bar_h, fill=0, stroke=1)
    legend_x = 120
    for index, task in enumerate(SEQ_TASKS):
      xx = legend_x + index * 96
      c.setFillColor(TASK_COLORS[task])
      c.rect(xx, 12, 8, 8, fill=1, stroke=0)
      c.setFillColor(DARK)
      c.setFont("Times-Roman", 6.8)
      c.drawString(xx + 12, 12, LABELS[task])
    c.restoreState()


class RewardSelectivity(Flowable):

  def __init__(self, width=515, height=218):
    super().__init__()
    self.width = width
    self.height = height

  def draw(self):
    c = self.canv
    c.saveState()
    panel_w, panel_h, gap = 158, 161, 17
    x_origin, y0 = 4, 28
    steps = (0, 200_000, 400_000, 600_000, 800_000)
    for index, arm in enumerate(ARMS):
      x0 = x_origin + index * (panel_w + gap)
      px, py, pw, ph = x0 + 28, y0 + 20, panel_w - 35, panel_h - 44
      c.setFillColor(DARK)
      c.setFont("Times-Bold", 8.2)
      c.drawCentredString(x0 + panel_w / 2, y0 + panel_h - 12, ARM_LABELS[arm])
      for tick in (0, .25, .5, .75, 1):
        yy = py + tick * ph
        c.setStrokeColor(GRID)
        c.line(px, yy, px + pw, yy)
        if index == 0:
          c.setFillColor(MID)
          c.setFont("Times-Roman", 6.2)
          c.drawRightString(px - 4, yy - 2, f"{tick:.2g}")
      chance_y = py + (1 / 6) * ph
      c.setStrokeColor(MID)
      c.setDash(2, 2)
      c.line(px, chance_y, px + pw, chance_y)
      c.setDash()
      for field, color, dashed in (
          ("reward_auroc", BLUE, False), ("reward_top1", RED, True)):
        points = []
        for step in steps:
          xx = px + step / 800_000 * pw
          yy = py + head_mean(arm, step, field) * ph
          points.append((xx, yy))
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(1.45)
        if dashed:
          c.setDash(3, 1.5)
        for left, right in zip(points, points[1:]):
          c.line(*left, *right)
        c.setDash()
        for xx, yy in points:
          c.circle(xx, yy, 1.5, fill=1, stroke=0)
      c.setStrokeColor(DARK)
      c.line(px, py, px, py + ph)
      c.line(px, py, px + pw, py)
      for step in steps:
        xx = px + step / 800_000 * pw
        c.setFillColor(MID)
        c.setFont("Times-Roman", 5.9)
        c.drawCentredString(xx, py - 10, str(step // 1000))
    c.setStrokeColor(BLUE)
    c.setFillColor(BLUE)
    c.line(133, 12, 151, 12)
    c.setFont("Times-Roman", 7)
    c.drawString(156, 9, "Factual reward AUROC")
    c.setStrokeColor(RED)
    c.setDash(3, 1.5)
    c.line(292, 12, 310, 12)
    c.setDash()
    c.setFillColor(RED)
    c.drawString(315, 9, "Completion goal top-1")
    c.restoreState()


class CriticCalibration(Flowable):

  def __init__(self, width=515, height=246):
    super().__init__()
    self.width = width
    self.height = height

  def draw(self):
    c = self.canv
    c.saveState()
    x0, y0, pw, ph = 58, 38, 426, 170
    steps = (0, 200_000, 400_000, 600_000, 800_000)
    ymax = 4.0
    for tick in (0, 1, 2, 3, 4):
      yy = y0 + tick / ymax * ph
      c.setStrokeColor(GRID)
      c.line(x0, yy, x0 + pw, yy)
      c.setFillColor(MID)
      c.setFont("Times-Roman", 6.8)
      c.drawRightString(x0 - 6, yy - 2, str(tick))
    for step in steps:
      xx = x0 + step / 800_000 * pw
      c.setStrokeColor(GRID)
      c.line(xx, y0, xx, y0 + ph)
      c.setFillColor(MID)
      c.setFont("Times-Roman", 6.8)
      c.drawCentredString(xx, y0 - 12, str(step // 1000))
    target = fmean(head_mean(arm, 800_000, "value_target") for arm in ARMS)
    c.setStrokeColor(MID)
    c.setDash(3, 2)
    c.line(x0, y0 + target / ymax * ph, x0 + pw, y0 + target / ymax * ph)
    c.setDash()
    for arm in ARMS:
      points = [
          (x0 + step / 800_000 * pw,
           y0 + head_mean(arm, step, "value_prediction") / ymax * ph)
          for step in steps
      ]
      c.setStrokeColor(ARM_COLORS[arm])
      c.setFillColor(ARM_COLORS[arm])
      c.setLineWidth(1.7)
      for left, right in zip(points, points[1:]):
        c.line(*left, *right)
      for xx, yy in points:
        c.circle(xx, yy, 2, fill=1, stroke=0)
    c.setStrokeColor(DARK)
    c.line(x0, y0, x0, y0 + ph)
    c.line(x0, y0, x0 + pw, y0)
    c.setFillColor(DARK)
    c.setFont("Times-Roman", 7.5)
    c.drawCentredString(x0 + pw / 2, 10, "Cumulative training steps (thousands)")
    c.saveState()
    c.translate(15, y0 + ph / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Predicted / realized discounted return")
    c.restoreState()
    legend_x = 92
    for index, arm in enumerate(ARMS):
      xx = legend_x + index * 126
      c.setStrokeColor(ARM_COLORS[arm])
      c.line(xx, self.height - 15, xx + 17, self.height - 15)
      c.setFillColor(DARK)
      c.setFont("Times-Roman", 6.8)
      c.drawString(xx + 22, self.height - 18, ARM_LABELS[arm])
    c.setStrokeColor(MID)
    c.setDash(3, 2)
    c.line(438, self.height - 15, 455, self.height - 15)
    c.setDash()
    c.setFillColor(DARK)
    c.drawString(460, self.height - 18, "Realized")
    c.restoreState()


class ImaginedReturnProvenance(Flowable):

  def __init__(self, width=515, height=225):
    super().__init__()
    self.width = width
    self.height = height

  def draw(self):
    c = self.canv
    c.saveState()
    panel_w, gap = 158, 17
    x_origin, y0, ph = 4, 34, 137
    horizons = (1, 3, 6, 15)
    ymax = 2.5
    for index, arm in enumerate(ARMS):
      x0 = x_origin + index * (panel_w + gap)
      px, pw = x0 + 28, panel_w - 35
      c.setFillColor(DARK)
      c.setFont("Times-Bold", 8.2)
      c.drawCentredString(x0 + panel_w / 2, self.height - 14, ARM_LABELS[arm])
      for tick in (0, .5, 1, 1.5, 2, 2.5):
        yy = y0 + tick / ymax * ph
        c.setStrokeColor(GRID)
        c.line(px, yy, px + pw, yy)
        if index == 0:
          c.setFillColor(MID)
          c.setFont("Times-Roman", 5.9)
          c.drawRightString(px - 4, yy - 2, f"{tick:g}")
      for proposal, color, dashed in (
          ("actor", PURPLE, False), ("recorded", TEAL, True)):
        points = []
        for hindex, horizon in enumerate(horizons):
          xx = px + hindex / 3 * pw
          value = critic_weighted_mean(arm, horizon, proposal, "return_mean")
          points.append((xx, y0 + value / ymax * ph))
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(1.55)
        if dashed:
          c.setDash(3, 1.5)
        for left, right in zip(points, points[1:]):
          c.line(*left, *right)
        c.setDash()
        for xx, yy in points:
          c.circle(xx, yy, 1.7, fill=1, stroke=0)
      for hindex, horizon in enumerate(horizons):
        xx = px + hindex / 3 * pw
        c.setFillColor(MID)
        c.setFont("Times-Roman", 6.2)
        c.drawCentredString(xx, y0 - 10, str(horizon))
      c.setStrokeColor(DARK)
      c.line(px, y0, px, y0 + ph)
      c.line(px, y0, px + pw, y0)
      bootstrap = critic_weighted_mean(
          arm, 15, "actor", "bootstrap_fraction")
      c.setFillColor(MID)
      c.setFont("Times-Roman", 6.4)
      c.drawCentredString(x0 + panel_w / 2, 10,
                          f"H=15 actor bootstrap: {100 * bootstrap:.1f}%")
    c.setStrokeColor(PURPLE)
    c.setFillColor(PURPLE)
    c.line(172, self.height - 32, 190, self.height - 32)
    c.setFont("Times-Roman", 6.8)
    c.drawString(195, self.height - 35, "Actor-selected")
    c.setStrokeColor(TEAL)
    c.setDash(3, 1.5)
    c.line(286, self.height - 32, 304, self.height - 32)
    c.setDash()
    c.setFillColor(TEAL)
    c.drawString(309, self.height - 35, "Recorded-action")
    c.restoreState()


def styles():
  base = getSampleStyleSheet()
  return {
      "Title": ParagraphStyle(
          "Title", parent=base["Title"], fontName="Times-Bold", fontSize=22,
          leading=25, alignment=TA_CENTER, textColor=DARK, spaceAfter=8),
      "Subtitle": ParagraphStyle(
          "Subtitle", parent=base["Normal"], fontName="Times-Roman",
          fontSize=12.5, leading=16, alignment=TA_CENTER, textColor=MID,
          spaceAfter=18),
      "Meta": ParagraphStyle(
          "Meta", parent=base["Normal"], fontName="Times-Roman", fontSize=10,
          leading=14, alignment=TA_CENTER, textColor=DARK, spaceAfter=5),
      "AbstractTitle": ParagraphStyle(
          "AbstractTitle", parent=base["Heading2"], fontName="Times-Bold",
          fontSize=10.5, leading=13, alignment=TA_CENTER, spaceBefore=9,
          spaceAfter=5, textColor=DARK),
      "Body": ParagraphStyle(
          "Body", parent=base["BodyText"], fontName="Times-Roman", fontSize=9.25,
          leading=12.1, alignment=TA_LEFT, textColor=DARK, spaceAfter=5.5),
      "Small": ParagraphStyle(
          "Small", parent=base["BodyText"], fontName="Times-Roman", fontSize=8,
          leading=10.1, textColor=DARK, spaceAfter=3),
      "H1": ParagraphStyle(
          "H1", parent=base["Heading1"], fontName="Times-Bold", fontSize=15,
          leading=18, textColor=DARK, spaceBefore=13, spaceAfter=7,
          keepWithNext=True),
      "H2": ParagraphStyle(
          "H2", parent=base["Heading2"], fontName="Times-Bold", fontSize=11.5,
          leading=14, textColor=BLUE, spaceBefore=9, spaceAfter=5,
          keepWithNext=True),
      "Caption": ParagraphStyle(
          "Caption", parent=base["BodyText"], fontName="Times-Roman",
          fontSize=7.8, leading=9.5, alignment=TA_LEFT, leftIndent=8,
          rightIndent=8, textColor=DARK, spaceBefore=3, spaceAfter=8),
      "TOCTitle": ParagraphStyle(
          "TOCTitle", parent=base["Heading1"], fontName="Times-Bold",
          fontSize=17, leading=20, textColor=DARK, spaceAfter=10),
      "Reference": ParagraphStyle(
          "Reference", parent=base["BodyText"], fontName="Times-Roman",
          fontSize=8, leading=10, leftIndent=10, firstLineIndent=-10,
          textColor=DARK, spaceAfter=5),
  }


S = styles()


def P(text: str, style="Body"):
  return Paragraph(text, S[style])


def heading(text: str, level=1):
  return Paragraph(text, S["H1" if level == 1 else "H2"])


def bullets(items, numbered=False):
  if not numbered:
    bullet_style = ParagraphStyle(
        "BulletMark", parent=S["Body"], fontName="Times-Roman",
        fontSize=9, leading=12.1, alignment=TA_LEFT, spaceAfter=0)
    body_style = ParagraphStyle(
        "BulletBody", parent=S["Body"], leftIndent=0, firstLineIndent=0,
        spaceAfter=0)
    table = Table(
        [[Paragraph("-", bullet_style), Paragraph(item, body_style)] for item in items],
        colWidths=[14, 480], splitByRow=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return table
  entries = []
  for item in items:
    entries.append(ListItem(P(item), leftIndent=9))
  return ListFlowable(
      entries,
      bulletType="1",
      start="1",
      leftIndent=18,
      bulletFontName="Times-Roman",
      bulletFontSize=8,
      spaceAfter=4,
  )


def report_table(data, widths, font_size=7.5, header_rows=1, alignments=None):
  converted = []
  for row_index, row in enumerate(data):
    converted.append([
        value if isinstance(value, Flowable) else Paragraph(
            str(value), ParagraphStyle(
                f"cell-{row_index}-{col_index}", parent=S["Small"],
                fontName="Times-Bold" if row_index < header_rows else "Times-Roman",
                fontSize=font_size, leading=font_size + 2,
                alignment=(alignments[col_index] if alignments else TA_LEFT),
                textColor=colors.white if row_index < header_rows else DARK,
                spaceAfter=0,
            ))
        for col_index, value in enumerate(row)
    ])
  table = Table(converted, colWidths=widths, repeatRows=header_rows, hAlign="LEFT")
  table.setStyle(TableStyle([
      ("BACKGROUND", (0, 0), (-1, header_rows - 1), BLUE),
      ("VALIGN", (0, 0), (-1, -1), "TOP"),
      ("LEFTPADDING", (0, 0), (-1, -1), 5),
      ("RIGHTPADDING", (0, 0), (-1, -1), 5),
      ("TOPPADDING", (0, 0), (-1, -1), 4),
      ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
      ("LINEBELOW", (0, 0), (-1, 0), 0.6, BLUE),
      ("LINEBELOW", (0, -1), (-1, -1), 0.5, MID),
      ("ROWBACKGROUNDS", (0, header_rows), (-1, -1),
       [colors.white, colors.HexColor("#F5F7F9")]),
  ]))
  return table


def caption(label: str, text: str):
  return P(f"<b>{label}.</b> {text}", "Caption")


def callout(title: str, text: str, background=LIGHTBLUE, border=BLUE):
  content = P(f"<b>{title}</b> {text}")
  table = Table([[content]], colWidths=[490])
  table.setStyle(TableStyle([
      ("BACKGROUND", (0, 0), (-1, -1), background),
      ("BOX", (0, 0), (-1, -1), 0.8, border),
      ("LEFTPADDING", (0, 0), (-1, -1), 10),
      ("RIGHTPADDING", (0, 0), (-1, -1), 10),
      ("TOPPADDING", (0, 0), (-1, -1), 8),
      ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
  ]))
  return table


def build_story():
  story = []
  story += [Spacer(1, 15 * mm), P("MiniGrid Experiments for<br/>Continual Imagined Rehearsal", "Title")]
  story += [P("Independent task audition and visibility-first sequential baseline:<br/>acquisition, forgetting, replay allocation, reward semantics, and critic provenance", "Subtitle")]
  story += [Spacer(1, 4 * mm), P("Steven Davenport", "Meta"),
            P("Continual Reinforcement Learning Research Notes", "Meta"),
            Spacer(1, 3 * mm),
            P("Studies completed 30 August and 1 September 2026; report updated 1 September 2026", "Meta")]
  story += [Spacer(1, 5 * mm), P("Abstract", "AbstractTitle")]
  abstract = (
      "This report records two MiniGrid studies in the Continual Imagined Rehearsal (CIR) programme. Study I trained "
      "six fixed-mission tasks independently for 100,000 environment steps over three seeds. It established that the "
      "reduced 12M DreamerV3 agent was neither uniformly saturated nor uniformly incapable, but rejected the assumption "
      "that the candidate tasks had equal difficulty: PutNear was reliable, GoToDoor, Fetch, and GoToObject were "
      "seed-sensitive, PickupDist was nearly unsolved, and UnlockPickup was never learned. That 1.8M-step audition "
      "completed in 5.14 hours.<br/><br/>"
      "Study II then trained one goal-conditioned agent through GoToDoor, Fetch, GoToObject, and PutNear for 200,000 "
      "steps each. Three seeds were crossed with a 100k FIFO control, a uniform per-goal episode reservoir, and a "
      "reservoir sampled 50:50 between the current and all old goals. The 7.2M-step run evaluated all tasks every 50k "
      "and the active task every 25k under both sampled and deterministic policies, yielding 1,512 conditions and "
      "75,600 episodes. Ninety read-only audits probed identical replay states at five phase boundaries. The complete "
      "workflow took 30.17 hours on one RTX 4090 with no failed scientific unit.<br/><br/>"
      "The central result is a joint acquisition-retention failure. FIFO learned GoToDoor and PutNear but erased "
      "GoToDoor; 50:50 replay acquired three tasks but retained only 28.7% GoToDoor success; uniform replay preserved "
      "teaching exposure but progressively starved the current task and ended at zero success on every task. Forward "
      "transfer was absent. Mechanistically, reward-event AUROC often approached one while same-state completion-goal "
      "top-1 accuracy remained weak. More seriously, the critic became grossly overoptimistic: uniform-reservoir value "
      "predictions reached 3.63 against a fixed realized-return mean of 0.077, the slow critic tracked the same error, "
      "and imagined targets were dominated by bootstrap value. Actor-selected imagination amplified the error. These "
      "findings make unregularized critic-bootstrapped IR unsafe as the next intervention and motivate counterfactual "
      "reward supervision plus bounded or reward-only rehearsal before gating is tested."
  )
  story += [P(abstract, "Small"), PageBreak()]

  story += [P("Contents", "TOCTitle")]
  toc = TableOfContents()
  toc.levelStyles = [
      ParagraphStyle("TOC1", fontName="Times-Bold", fontSize=9.2, leading=15,
                     leftIndent=0, firstLineIndent=0, textColor=DARK),
      ParagraphStyle("TOC2", fontName="Times-Roman", fontSize=8.4, leading=13,
                     leftIndent=16, firstLineIndent=0, textColor=BLUE),
  ]
  story += [toc, PageBreak()]

  story += [heading("1. Research purpose and place in the programme")]
  story += [P(
      "The long-term goal is a safe continual reinforcement-learning agent that can adapt through imagined experience "
      "while minimizing unsafe or wasteful interaction with the real environment. Original Imagined Rehearsal was "
      "motivated by a basic problem in embodied reinforcement learning: a robot normally has to execute its current, "
      "possibly sub-optimal policy in the physical world to obtain feedback. Rehearsing within a learned world model "
      "offers a route to policy improvement before the next real action. Continual Imagined Rehearsal extends that "
      "premise across a changing task stream while explicitly addressing catastrophic forgetting, forward transfer, "
      "and adaptation safety [1].")]
  story += [P(
      "RLScape remains central to that programme, but its 504-location click interface and long training cycles make "
      "it expensive to distinguish environment difficulty from agent failure. MiniGrid [2] provides a complementary "
      "diagnostic tier: it preserves partial visual observation, sparse goal completion, discrete control, randomized "
      "layouts, and multi-step behavior while reducing the action space to seven shared actions and shortening a full "
      "multi-seed matrix from days to hours.")]
  story += [P("This first MiniGrid study asks four questions:")]
  story += [bullets([
      "Can a reduced DreamerV3 agent learn each candidate task within 100,000 environment steps without trivial early saturation?",
      "Which tasks have comparable acquisition difficulty under an identical model, replay, and step budget?",
      "Are conclusions sensitive to sampled versus deterministic action selection?",
      "Which tasks are defensible candidates for the first continual sequence, and which should be deferred or redesigned?",
  ], numbered=True)]
  story += [Spacer(1, 4), callout(
      "Claim boundary.",
      "This is an independent-task difficulty audition. Each task and seed starts from a fresh agent. There are no task "
      "switches, no return to an earlier task, no imagined rehearsal intervention, and therefore no measurement of "
      "catastrophic forgetting, backward transfer, or forward transfer. The study selects the substrate on which those "
      "questions can later be tested."), Spacer(1, 6)]

  story += [heading("2. Study I: fixed-mission MiniGrid baseline")]
  story += [heading("2.1 Controlled task suite", 2)]
  story += [P(
      "The stock task families were converted to fixed-mission variants so language parsing and episode-to-episode "
      "mission changes would not confound the first audition. Layouts, object positions, agent starts, and evaluation "
      "reset seeds still vary. All tasks expose the same seven-action MiniGrid interface: turn left, turn right, move "
      "forward, pick up, drop, toggle, and done. Observations are 64x64 partial RGB frames rendered from the agent's "
      "point of view.")]
  task_table = [
      ["Task", "Family", "Fixed mission", "Success and terminal behavior"],
      ["GoToDoor", "navigation", "go to the red door", "Use done beside the unique red door; toggle or premature done terminates."],
      ["Fetch", "pickup", "fetch the blue ball", "Pick up the blue ball among two distractors; any pickup terminates."],
      ["GoToObject", "navigation", "go to the green key", "Use done beside the green key; pickup, toggle, or premature done terminates."],
      ["PutNear", "pickup and place", "put the yellow ball near the blue box", "Carry and drop the ball in a neighboring cell; wrong pickup or any drop terminates."],
      ["UnlockPickup", "unlock and pickup", "pick up the only box behind the locked door", "Find the key, unlock the door, enter the second room, and pick up the box."],
      ["PickupDist", "pickup", "pick up the grey box", "Pick up the grey box among four distractors; any pickup terminates."],
  ]
  story += [report_table(task_table, [57, 75, 121, 260], font_size=7.0),
            caption("Table 1", "Candidate tasks and their controlled success conditions.")]
  story += [P(
      "GoToDoor, Fetch, GoToObject, and PickupDist use 8x8 rooms; PutNear uses a 6x6 room; UnlockPickup uses the stock "
      "two-room geometry. Every episode is capped at 100 actions. Reward is sparse and positive only on success, with "
      "the standard MiniGrid time discount, so binary success is the primary comparison statistic and mean return also "
      "incorporates solution efficiency.")]

  story += [heading("2.2 Agent and training configuration", 2)]
  story += [P(
      "The agent is DreamerV3 [3] using the repository's reduced size12m preset. The observed optimizer namespace "
      "contained 10,499,273 parameters. The recurrent state-space model uses a deterministic state of width 2,048 and "
      "a 16x16 categorical stochastic state. Network width is 256. Training uses batches of eight sequences of length "
      "32, imagination horizon 15, FIFO replay capacity 100,000, and a replay ratio of 32.")]
  story += [P(
      "The visual dynamics remain goal-agnostic. A six-way one-hot goal_id is concatenated into the reward, "
      "continuation, actor, and value heads. Although only one goal occurs within each independent run, this is the same "
      "interface intended for the later multi-task continual agent.")]
  config_table = [
      ["Component", "Setting", "Component", "Setting"],
      ["Observation", "64x64 partial RGB", "Action space", "Discrete(7)"],
      ["Model preset", "size12m", "Observed parameters", "10,499,273"],
      ["RSSM", "deter 2048; hidden 256; classes 16", "Network width", "256"],
      ["Batch", "8 sequences x 32", "Replay", "FIFO, 100,000"],
      ["Train ratio", "32", "Environment instances", "1"],
      ["Episode limit", "100 actions", "Training budget", "100,000 steps/run"],
  ]
  story += [report_table(config_table, [82, 174, 78, 179], font_size=7.2),
            caption("Table 2", "Pinned Dreamer and environment configuration.")]

  story += [heading("2.3 Experiment matrix and checkpoint evaluation", 2)]
  story += [P(
      "The design is a fully crossed 6-task by 3-seed matrix. Each of the 18 agents trains for 100,000 environment "
      "steps with one environment instance and one training process active at a time. Immutable checkpoints are "
      "published at 25k, 50k, 75k, and 100k steps.")]
  story += [P(
      "Each checkpoint is evaluated under both sampled and deterministic action selection. Every policy mode receives "
      "50 episodes: 6 tasks x 3 seeds x 4 checkpoints x 2 modes x 50 episodes = <b>7,200 evaluation episodes</b>. "
      "The two policy modes for a task, seed, and checkpoint receive the same sequence of environment reset seeds. "
      "Adaptation is disabled, and namespace hashes verify that evaluation leaves the checkpoint unchanged.")]
  protocol = [
      ["Fresh agent", "Training", "Milestones", "Paired evaluation"],
      ["task t, seed s", "100k steps; one env", "25k / 50k / 75k / 100k", "50 sampled + 50 deterministic"],
  ]
  story += [report_table(protocol, [95, 125, 130, 163], font_size=8,
                               alignments=[TA_CENTER] * 4),
            caption("Figure 1", "Independent training and paired checkpoint evaluation protocol; repeated for six tasks and three seeds.")]

  story += [heading("2.4 Predeclared difficulty rule", 2)]
  story += [P(
      "Training difficulty uses binary episode success. For each episode, the analysis computes rolling success over "
      "the last W=20 episodes. Threshold time is the first environment step at which a full 20-episode window reaches "
      "70% success. Saturation time uses 90%, and early saturation means reaching that level within the first 25,000 "
      "steps. Normalized training AUC integrates the rolling success curve over the 100k budget.")]
  story += [P(
      "Two tasks are declared matched when both their mean normalized AUC and their median threshold step agree within "
      "a relative tolerance of 25%. For two nonzero values a and b, max(|a|,|b|) / min(|a|,|b|) must be at most 1.25. "
      "The rule was fixed before results were analyzed. Its reliability limitation becomes important in Section 4.4.")]

  story += [heading("2.5 Execution integrity and provenance", 2)]
  story += [P(
      "The experiment is pinned to Git commit ae3cb917f661d6c3fe76422e476fc6179fbc694e, MiniGrid 3.1.0, "
      "Gymnasium 1.3.0, and JAX 0.4.33. The supervisor records the immutable specification digest, child commands, "
      "timing, return codes, timeouts, retries, and condition summaries. All 18 training attempts and all 144 evaluation "
      "attempts completed on their first attempt with return code zero. Every evaluation recorded exactly 50 episodes "
      "and 50 resets; all checkpoint hashes were equal before and after evaluation.")]

  story += [PageBreak(), heading("3. Operational result: a complete matrix in 5.14 hours")]
  story += [P(
      "The serial run started at 19:07:11 on 29 August 2026 and completed at 00:15:18 on 30 August 2026. Total wall "
      "time was 18,487.8 seconds, or 5.14 hours. Training children occupied 9,721.5 seconds and evaluation children "
      "8,644.3 seconds, leaving roughly 122 seconds of orchestration overhead. Each 100k training run took about 540 "
      "seconds; each 50-episode evaluation condition took about 60 seconds.")]
  execution = [
      ["Quantity", "Value", "Quantity", "Value"],
      ["Training units", "18", "Training steps", "1,800,000"],
      ["Training episodes", "27,892", "Mean collection rate", "185.2 steps/s"],
      ["Checkpoints", "72", "Evaluation conditions", "144"],
      ["Evaluation episodes", "7,200", "Policy modes", "2"],
      ["Environment instances at once", "1", "Concurrent training units", "1"],
      ["Retries or failed units", "0", "Final artifact footprint", "24 GB"],
      ["Training child time", "2.70 h", "Evaluation child time", "2.40 h"],
      ["Total wall time", "5.14 h", "Orchestration overhead", "2.0 min"],
  ]
  story += [report_table(execution, [150, 70, 150, 70], font_size=7.7),
            caption("Table 3", "Experiment scale and operational outcome.")]
  story += [callout(
      "Operational conclusion.",
      "The speed came from the environment, not parallelism. The matrix used one environment, one training process, "
      "and serial task-seed units. MiniGrid therefore has further headroom for vectorized collection or concurrent seeds, "
      "but neither is required for fast baseline iteration.", LIGHTGREEN, GREEN)]

  story += [heading("4. Learning results")]
  story += [heading("4.1 Checkpoint trajectories", 2)]
  story += [P(
      "No task reached 90% rolling training success in the first quarter of the budget, so the reduced agent was not "
      "trivially overpowered. The opposite problem occurred for UnlockPickup and PickupDist: the former never produced "
      "a success, while the latter remained close to zero.")]
  story += [LearningCurves(), caption(
      "Figure 2",
      "Checkpoint success by task and policy mode. Lines show the mean across three seeds; shaded regions show one "
      "population standard deviation. The dashed line marks 70% success. Each point aggregates 150 evaluation episodes. "
      "PutNear's zero-success 75k checkpoint followed by strong 100k performance is a genuine non-monotonic result.")]
  story += [bullets([
      "<b>PutNear</b> was the only consistently strong task, ending at 84.7% sampled and 88.0% deterministic success. Its three seeds reached the rolling threshold, but the 75k checkpoint collapse shows instability.",
      "<b>GoToDoor</b> learned late and inconsistently. It remained at zero through 50k, then reached 46.0% final success under both policy modes.",
      "<b>GoToObject</b> and <b>Fetch</b> showed gradual late learning but did not become reliable across seeds by 100k.",
      "<b>UnlockPickup</b> produced zero training AUC and zero evaluation success at every checkpoint and seed.",
      "<b>PickupDist</b> showed isolated successes but no sustained acquisition.",
  ])]

  story += [heading("4.2 Final outcomes and seed heterogeneity", 2)]
  final_table = [
      ["Task", "Sampled success", "Deterministic success", "Training AUC", "Threshold seeds", "Median step"],
      ["GoToDoor", "46.0 +/- 37.6%", "46.0 +/- 36.8%", "0.154 +/- 0.139", "2/3", "82.4k"],
      ["Fetch", "22.7 +/- 26.4%", "21.3 +/- 24.6%", "0.059 +/- 0.026", "1/3", "98.0k"],
      ["GoToObject", "36.7 +/- 28.7%", "32.7 +/- 23.8%", "0.066 +/- 0.043", "0/3", "--"],
      ["PutNear", "84.7 +/- 16.4%", "88.0 +/- 10.7%", "0.147 +/- 0.015", "3/3", "68.4k"],
      ["UnlockPickup", "0.0 +/- 0.0%", "0.0 +/- 0.0%", "0.000 +/- 0.000", "0/3", "--"],
      ["PickupDist", "1.3 +/- 1.9%", "2.0 +/- 1.6%", "0.025 +/- 0.014", "0/3", "--"],
  ]
  story += [report_table(final_table, [72, 92, 102, 90, 78, 72], font_size=6.8,
                               alignments=[TA_LEFT, TA_CENTER, TA_CENTER, TA_CENTER, TA_CENTER, TA_CENTER]),
            caption("Table 4", "Final 100k-checkpoint outcomes. Mean +/- population SD over three seeds. Threshold seeds count runs reaching 70% rolling training success.")]
  story += [P(
      "The means conceal strong between-seed structure. GoToDoor ranged from 0% to 92% sampled success; Fetch from "
      "4% to 60%; and GoToObject from 0% to 70%. PutNear was the only task above 60% in every seed and mode. This "
      "variance is central to experiment design: it determines whether later continual-learning differences can be "
      "distinguished from ordinary training luck.")]
  story += [SeedMatrix(), caption(
      "Figure 3",
      "Final success for every task, seed, and policy mode. Each cell contains 50 episodes. The matrix exposes one "
      "successful Fetch seed, one failed GoToDoor seed, and the comparatively robust PutNear result.")]

  story += [PageBreak(), heading("4.3 Sampled and deterministic policies agree", 2)]
  story += [P(
      "Across all 72 paired task-seed-checkpoint comparisons, sampled success exceeded deterministic success by only "
      "0.42 percentage points on average. The mean absolute difference was 1.19 points, although the largest individual "
      "difference was 14 points. Most conditions lie on or near the identity line.")]
  story += [PolicyAgreement(), caption(
      "Figure 4",
      "Agreement between sampled and deterministic success. Each point is one task, training seed, and checkpoint; "
      "paired modes use the same sequence of environment reset seeds.")]
  story += [P(
      "Retaining both modes was useful confirmation rather than redundant bookkeeping. It verifies that the task ranking "
      "is not an artifact of one action convention. Small differences should not be over-interpreted with 50 episodes "
      "per condition. Both modes should remain in later CIR evaluations: deterministic behavior detects modal collapse, "
      "while sampled behavior represents the stochastic policy used by imagined rehearsal.")]

  story += [heading("4.4 One formal pairwise match, but no qualified chain", 2)]
  story += [P(
      "GoToDoor and PutNear have similar mean training AUC (0.154 versus 0.147), and their conditional median threshold "
      "times differ by less than 25% (82.4k versus 68.4k). They are therefore the sole formal match among 15 task pairs. "
      "Neither the four-task alternating chain GoToDoor to Fetch to GoToObject to PutNear nor the progressive chain "
      "Fetch to PutNear to UnlockPickup passes all-pairs matching.")]
  story += [DifficultySummary(), caption(
      "Figure 5",
      "Predeclared task-difficulty statistics. AUC error bars show one population SD. Threshold labels report the "
      "median only among seeds reaching 70%, followed by the number reaching it. This conditional definition allows "
      "GoToDoor and PutNear to pass despite different robustness.")]
  pair_table = [
      ["Pair or chain", "AUC matched", "Threshold matched", "Overall"],
      ["GoToDoor - PutNear", "yes", "yes", "matched"],
      ["GoToDoor - Fetch", "no", "yes", "not matched"],
      ["All other 13 pairs", "mixed or no", "no", "not matched"],
      ["Alternating four-task chain", "--", "--", "not qualified"],
      ["Progressive three-task chain", "--", "--", "not qualified"],
  ]
  story += [report_table(pair_table, [180, 105, 115, 100], font_size=7.4,
                               alignments=[TA_LEFT, TA_CENTER, TA_CENTER, TA_CENTER]),
            caption("Table 5", "Pairwise and chain decision summary.")]
  story += [callout(
      "Metric caveat.",
      "Threshold time is calculated only among seeds that cross the threshold. PutNear crosses in 3/3 seeds; "
      "GoToDoor crosses in 2/3, and its AUC standard deviation is more than nine times PutNear's. The reproducible "
      "conclusion is one provisional match under the original rule, not two equally dependable tasks.", LIGHTORANGE, ORANGE)]

  story += [PageBreak(), heading("5. Interpretation for continual imagined rehearsal")]
  story += [heading("5.1 Task-by-task interpretation", 2)]
  story += [P("<b>PutNear is the best present anchor.</b> It requires navigation, pickup, transport, and spatial placement, succeeds across all seeds, and does not saturate in the first quarter. Its 75k collapse nevertheless means checkpoint stability must be monitored; final performance alone would miss this behavior.")]
  story += [P("<b>GoToDoor is useful but underpowered as a sole navigation control.</b> Two seeds learn and one fails completely. This creates headroom for rehearsal but also a high false-positive risk: an apparent continual-learning effect could simply reflect initialization sensitivity.")]
  story += [P("<b>GoToObject and Fetch are promising intermediate candidates.</b> Both show genuine late acquisition in at least one seed and have neither a complete floor nor a reliable ceiling. More budget can reveal whether they converge. Their distinction also preserves task diversity: navigation-to-object versus target pickup among distractors.")]
  story += [P("<b>UnlockPickup is beyond the current learnability envelope.</b> A task that is never learned cannot measure forgetting. It may become valuable later as a forward-transfer or curriculum test after prerequisite behaviors are established.")]
  story += [P("<b>PickupDist is not interchangeable with Fetch.</b> Adding distractors reduces final success to approximately zero despite the identical action interface. It should be simplified, granted more training, or deferred.")]

  story += [heading("5.2 The audition succeeds by rejecting a bad assumption", 2)]
  story += [P(
      "The study did not find three or four equally difficult tasks, but this is not a failed experiment. It prevented "
      "the project from embedding a severe difficulty confound into the first CIR sequence. If UnlockPickup followed "
      "PutNear, later zero performance could be mislabelled catastrophic forgetting or unsafe rehearsal when the "
      "independent baseline already predicts zero. Conversely, combining PutNear and PickupDist would make aggregate "
      "retention dominated by unequal base learnability.")]
  story += [P(
      "The speed advantage is decisive. RLScape remains necessary for validating the large discrete action interface "
      "and project-specific semantics, but MiniGrid can act as the mechanism-development tier. Complete independent "
      "baselines, ablations, and multi-seed safety checks can run before spending days on RLScape or moving toward "
      "simulated and physical robotic arms.")]

  story += [heading("5.3 Recommended Stage II calibration", 2)]
  story += [P(
      "No continual chain should be frozen from this study alone. The next calibration should retain the 12m agent and "
      "shared 100-action interface so task calibration is not confounded with a capacity change.")]
  story += [bullets([
      "Extend GoToDoor, Fetch, GoToObject, and PutNear to 200k steps while retaining 25k checkpoints, testing whether weak tasks converge and whether PutNear remains stable.",
      "Use at least five training seeds for the final candidate panel. Three seeds reveal instability but cannot characterize it precisely.",
      "Replace the pairwise rule with a robustness-aware qualification: at least 4/5 seeds reach 70%, no early saturation, final mean success in a target band such as 40-90%, and acceptable between-seed dispersion.",
      "Treat success, training AUC, threshold time, and checkpoint stability as separate axes instead of forcing one scalar ranking.",
      "Defer UnlockPickup and PickupDist unless their mechanics are simplified or independent learning is demonstrated.",
  ], numbered=True)]
  story += [P(
      "For a short iteration before 200k runs, the current evidence supports a focused four-task audition panel: "
      "PutNear as reliable anchor, GoToDoor as variable navigation, GoToObject as intermediate navigation, and Fetch "
      "as intermediate pickup. This preserves navigation-pickup-navigation-placement diversity, but it is not yet an "
      "approved continual sequence.")]

  story += [heading("5.4 Requirements for the first continual experiment", 2)]
  story += [P(
      "Once tasks qualify independently, one shared goal-conditioned agent should train through a declared sequence and "
      "evaluate every encountered task at each phase boundary. The first continual study should include:")]
  story += [bullets([
      "a no-IR continual Dreamer baseline;",
      "always-on reward-model-only imagined rehearsal as the primary positive comparator;",
      "the same continual replay mechanism and real-environment step budget in every condition;",
      "sampled and deterministic paired evaluations;",
      "absolute success alongside average and maximum forgetting;",
      "forward-transfer measurements against independent from-scratch learning curves;",
      "a return phase that distinguishes retained competence from rapid reacquisition; and",
      "hard integrity checks identifying which actor, critic, and world-model parameters may change during rehearsal.",
  ])]
  story += [callout(
      "Programme logic.",
      "First establish that each component and task behaves as claimed. Then test whether rehearsal should run. Only "
      "after that should gating be judged as a safety and compute-control mechanism.", LIGHTGREEN, GREEN)]

  story += [heading("6. Study II: visibility-first sequential baseline")]
  story += [heading("6.1 Objective and claim boundary", 2)]
  story += [P(
      "Study II is the first true continual MiniGrid baseline. One shared goal-conditioned Dreamer trains through "
      "GoToDoor, Fetch, GoToObject, and PutNear, receiving 200,000 new environment steps per task. The sequence is "
      "navigation to pickup to navigation to pickup-and-place. It does not return to task A; the preserved 800k "
      "checkpoint permits that test later without changing this baseline.")]
  story += [P(
      "The primary objective is visibility. In addition to acquisition, forgetting, backward transfer, and forward "
      "transfer, the run records replay composition, goal-specific training losses, actor drift on identical states, "
      "same-state reward and value calibration, and critic provenance under actor-selected versus recorded actions. "
      "This is deliberately a <b>no-IR baseline</b>: it contains no imagined rehearsal, counterfactual head training, "
      "critic regularization, or gate. Its role is to reveal what a future CIR intervention must repair and beat.")]
  story += [callout(
      "Interpretation boundary.",
      "The replay arms are ordinary continual-learning controls, not CIR variants. A low forgetting score is not "
      "evidence of preservation when the corresponding task was never acquired. Counterfactual value queries are "
      "diagnostics, not supervised targets.", LIGHTORANGE, ORANGE)]

  story += [heading("6.2 Replay arms isolate memory from current-task exposure", 2)]
  arm_design = [
      ["Arm", "Storage", "Sampling in phases A / B / C / D", "Question"],
      ["FIFO 100k", "100k transitions", "current dominated after eviction", "Forgetting control with short memory"],
      ["Uniform reservoir", "about 1,000 episodes per goal", "100% / 50% / 33% / 25% current", "Primary future-CIR baseline; preserve every old teaching signal"],
      ["50:50 current/old", "same per-goal reservoir", "100% / 50% / 50% / 50% current", "Can ordinary current-biased replay solve the acquisition-retention trade-off?"],
  ]
  story += [report_table(arm_design, [91, 101, 137, 184], font_size=6.8),
            caption("Table 6", "Replay arms. The old half in the 50:50 arm is divided uniformly among represented old goals.")]
  story += [P(
      "All arms use the same 12M model, one environment, train ratio 32, 100-action limit, task order, seeds, and "
      "7-action interface. Agent and optimizer snapshots are saved every 25k. Full replay is archived at each 200k "
      "phase boundary, making both exact resumption and later intervention studies possible.")]

  story += [heading("6.3 Dense paired evaluation and read-only diagnostics", 2)]
  story += [P(
      "The active task is evaluated every 25k; all four tasks are evaluated every 50k. Each condition uses 50 sampled "
      "and 50 deterministic episodes with paired reset seeds. The complete matrix is 3 arms x 3 seeds x 168 conditions "
      "per run = <b>1,512 evaluation conditions and 75,600 episodes</b>. Every evaluation verifies that the checkpoint "
      "parameter digest is unchanged.")]
  story += [P(
      "After training, each seed's final uniform-reservoir replay fixes one state bank with up to 16 successful and 16 "
      "failed episodes per trained goal. Every arm's 0, 200k, 400k, 600k, and 800k snapshots is audited on the identical "
      "seed-specific bank. Forty-five head audits sweep all six goal queries over real posterior states; forty-five "
      "critic-provenance audits use 32 posterior samples, two anchors per episode, horizons 1, 3, 6, and 15, and both "
      "actor-selected and recorded-action proposals. All 90 audits preserve exact parameter hashes.")]

  story += [heading("6.4 Operational result: 30.17 hours without a failed unit", 2)]
  seq_execution = [
      ["Quantity", "Value", "Quantity", "Value"],
      ["Training runs", "9", "Training phases", "36"],
      ["Environment steps", "7,200,000", "Snapshots", "288"],
      ["Evaluation conditions", "1,512", "Evaluation episodes", "75,600"],
      ["Head audits", "45", "Critic audits", "45"],
      ["Training child time", "10.53 h", "Evaluation child time", "19.07 h"],
      ["Diagnostic child time", "0.51 h", "Total wall time", "30.17 h"],
      ["Recovered eval launches", "6", "Failed scientific units", "0"],
      ["GPU concurrency", "1 process", "Final artifact footprint", "80 GB"],
  ]
  story += [report_table(seq_execution, [140, 75, 145, 75], font_size=7.3),
            caption("Table 7", "Sequential experiment scale and completion record. The run used one RTX 4090 from 31 August 00:06 to 1 September 06:16 BST.")]
  story += [P(
      "All 36 training phases completed in one attempt, with a mean of 17.54 minutes per 200k phase. Six evaluation "
      "processes received an immediate KeyboardInterrupt before scientific work and were automatically relaunched; "
      "their replacement conditions contain exactly 50 episodes and pass integrity validation. Diagnostics required "
      "no retry. The restart-safe supervisor reached its terminal marker and generated all aggregate tables.")]

  story += [heading("7. Study II behavioral results")]
  story += [heading("7.1 Learning curves show both forgetting and acquisition interference", 2)]
  story += [SequentialCurves(), caption(
      "Figure 6",
      "Sampled-policy success throughout the task stream. Each point is the mean over three seeds and 150 episodes. "
      "Vertical boundaries mark task switches at 200k, 400k, and 600k; colored backgrounds denote phases A-D. "
      "The FIFO arm learns the current task most readily but forgets old tasks. Reservoir replay reduces current-task "
      "exposure and often prevents acquisition before retention can be assessed.")]
  story += [bullets([
      "<b>GoToDoor is a clean forgetting case.</b> FIFO reaches 95.3% immediately after phase A and finishes at 0.7%. Uniform replay reaches 78.0% and finishes at zero. The 50:50 arm reaches 92.0% and retains 28.7%.",
      "<b>Fetch separates the replay arms.</b> FIFO acquires only 18.7%; uniform replay only 1.3%; the 50:50 arm reaches 55.3%, then falls to zero by 800k.",
      "<b>GoToObject remains difficult.</b> Only the 50:50 arm shows meaningful immediate acquisition, 30.0%, and it too finishes at zero. Its weakness is not solely a forgetting effect.",
      "<b>PutNear exposes current-task starvation.</b> FIFO, which is effectively current-only late in phase D, reaches and retains 67.3%. Neither reservoir arm acquires it at all.",
  ])]

  story += [heading("7.2 Acquisition and retention must be reported separately", 2)]
  story += [AcquisitionRetentionMatrix(), caption(
      "Figure 7",
      "Immediate post-acquisition and final success. A low numerical forgetting score can arise because a task was "
      "never learned: uniform replay has lower mean forgetting than FIFO but ends with zero success everywhere.")]
  arm_results = [
      ["Arm", "Post-acquisition", "Final success", "Peak forgetting", "Backward transfer", "Forward transfer"],
      ["FIFO 100k", "45.3%", "17.5%", "36.5%", "-27.8 pp", "-5.0 pp"],
      ["Uniform reservoir", "19.8%", "0.0%", "22.0%", "-19.8 pp", "-4.8 pp"],
      ["50:50 current/old", "44.3%", "7.2%", "40.7%", "-37.2 pp", "-5.8 pp"],
  ]
  story += [report_table(arm_results, [105, 87, 75, 82, 88, 76], font_size=6.9,
                               alignments=[TA_LEFT] + [TA_CENTER] * 5),
            caption("Table 8", "Sampled-policy means across four tasks and three seeds. Forgetting is peak post-acquisition minus final success; backward transfer is final minus immediate post-acquisition.")]
  story += [P(
      "There is no evidence of positive forward transfer. All three arms are slightly negative relative to each task's "
      "initial checkpoint, and pretraining success for later tasks is at or near zero. The main comparison is therefore "
      "not which arm transfers best, but where each arm lies on the acquisition-retention frontier. FIFO supplies the "
      "strongest phase-D acquisition; uniform replay supplies memory exposure but insufficient learning pressure; 50:50 "
      "recovers earlier acquisition without preventing later forgetting.")]

  story += [heading("7.3 The realized replay mixtures match the intended intervention", 2)]
  story += [ReplayAllocation(), caption(
      "Figure 8",
      "Observed training-batch goal fractions, averaged across seeds and within each phase. Uniform replay follows "
      "100/50/33/25% current exposure. The 50:50 arm holds the current task at 50% after phase A. FIFO transitions "
      "gradually at each switch because its 100k capacity occupies half of a 200k phase.")]
  story += [P(
      "The reservoir result is therefore not an implementation accident. Storage retained approximately 1,000 episodes "
      "per represented goal and the batch fractions match the declared policy. The failure is the scientific trade-off "
      "being tested: uniform preservation can make the current task only one quarter of phase-D updates. Even 50% current "
      "exposure is insufficient for PutNear under multi-goal training, despite independent and FIFO evidence that the "
      "task is learnable.")]

  story += [heading("7.4 Sampled and deterministic evaluations tell the same story", 2)]
  story += [P(
      "The aggregate results remain stable under deterministic action selection. FIFO final success is 17.8% rather than "
      "17.5%; uniform replay remains exactly zero; and 50:50 remains 7.2%. Deterministic backward transfer differs from "
      "sampled by at most 2.2 percentage points at arm level. Maintaining both modes is still valuable: the agreement "
      "rules out action sampling as the source of collapse, while future IR may change policy entropy.")]

  story += [heading("8. Study II component diagnostics")]
  story += [heading("8.1 Factual reward detection does not imply goal-conditioned teaching", 2)]
  story += [RewardSelectivity(), caption(
      "Figure 9",
      "Reward-event recognition versus counterfactual goal selectivity on identical posterior states. Solid blue is "
      "factual reward AUROC; dashed red is top-1 identification of the completed goal when all six goal slots are "
      "queried. The horizontal dotted line is six-way chance. Means are over three seed-specific fixed state banks.")]
  head_table = [
      ["Arm at 800k", "Factual AUROC", "Average precision", "Goal top-1", "Reward margin", "Value prediction", "Value target"],
      ["FIFO 100k", "0.530", "0.005", "3.5%", "+0.000", "0.714", "0.077"],
      ["Uniform reservoir", "1.000", "0.996", "23.6%", "+0.071", "2.592", "0.077"],
      ["50:50 current/old", "0.945", "0.511", "16.8%", "+0.031", "1.132", "0.077"],
  ]
  story += [report_table(head_table, [105, 72, 79, 62, 66, 78, 67], font_size=6.4,
                               alignments=[TA_LEFT] + [TA_CENTER] * 6),
            caption("Table 9", "Final same-state head audit. Reward margin is matching minus nonmatching prediction at observed completion events; six-way top-1 chance is 16.7%.")]
  story += [P(
      "This is the same conceptual failure seen in RLScape, now under a much cleaner action interface. The reservoir reward "
      "heads learn that a rare reward event occurred, but the requested goal changes that prediction only weakly. Uniform "
      "replay achieves nearly perfect factual classification while identifying the completed goal only 23.6% of the time. "
      "The 50:50 top-1 rate is essentially chance. A factual reward head can therefore look excellent by AUROC and still "
      "be an unreliable counterfactual teacher for imagined rehearsal.")]

  story += [heading("8.2 The critic becomes severely overoptimistic on real posterior states", 2)]
  story += [CriticCalibration(), caption(
      "Figure 10",
      "Online critic predictions on fixed real posterior states versus their realized inclusive discounted returns. "
      "Each line is a three-seed mean; the dashed realized target remains 0.077 because every arm and checkpoint is "
      "queried on the same seed-specific state bank. The critic error is present before imagination.")]
  story += [P(
      "At 200k, critics already predict 0.65-0.71 against a realized mean of 0.077. The FIFO value later falls near the "
      "target at 600k but rebounds to 0.714 at 800k. Reservoir critics diverge much further: uniform replay reaches "
      "3.626 at 600k, a +3.550 bias, while 50:50 reaches 2.134, a +2.058 bias. These values are structurally impossible "
      "as calibrated success returns in a task with at most one sparse reward.")]
  story += [P(
      "The error is not a harmless intercept. At final checkpoints the value-return Pearson correlation is +0.110 for "
      "FIFO, -0.291 for uniform replay, and -0.133 for 50:50. Ranking is therefore weak or inverted. The slow critic "
      "does not provide an independent safeguard: its mean absolute difference from the online critic remains only "
      "0.016, 0.026, and 0.010 respectively at 800k, tiny relative to the calibration error.")]

  story += [heading("8.3 Standard imagined returns inherit and amplify the critic error", 2)]
  story += [ImaginedReturnProvenance(), caption(
      "Figure 11",
      "Factual imagined return under actor-selected and recorded-action proposals, pooled across trained checkpoints "
      "and three seeds. Every point uses the same posterior anchors. Actor selection amplifies predicted return most "
      "strongly in the reservoir arms; annotations report the actor target's bootstrap fraction at H=15.")]
  provenance_table = [
      ["Arm", "Actor H15", "Recorded H15", "Actor amplification", "Reward-only H15", "Bootstrap fraction"],
      ["FIFO 100k", "0.459", "0.425", "+0.034", "0.154", "81.0%"],
      ["Uniform reservoir", "2.194", "1.812", "+0.381", "0.103", "88.4%"],
      ["50:50 current/old", "1.219", "0.978", "+0.240", "0.148", "82.2%"],
  ]
  story += [KeepTogether([
      report_table(provenance_table, [110, 72, 80, 93, 83, 82], font_size=6.7,
                   alignments=[TA_LEFT] + [TA_CENTER] * 5),
      caption("Table 10", "Pooled factual H=15 imagined-target decomposition over 200k-800k checkpoints. Full target equals predicted reward contribution plus critic bootstrap under Dreamer's continuation-weighted return."),
  ])]
  story += [P(
      "Recorded actions already produce overoptimistic targets, so policy exploitation is not the origin of the error. "
      "Actor-selected actions then raise the target by 0.381 in uniform replay and 0.240 in 50:50 replay. Predicted "
      "reward remains small relative to the full target; the critic bootstrap contributes 81-88% even at H=15 and "
      "approximately 97-99% at H=1. Standard critic-bootstrapped IR would consequently train the actor toward a signal "
      "known to be miscalibrated on both real and imagined states.")]

  story += [heading("8.4 Actor drift confirms that preserved components do not preserve behavior", 2)]
  story += [P(
      "The same-state actor audit compares factual-goal action distributions at adjacent phase boundaries. Mean "
      "Jensen-Shannon divergence per switch is 0.371 for FIFO, 0.373 for uniform replay, and 0.306 for 50:50. Modal "
      "action agreement is only 27.3%, 35.0%, and 47.6% respectively. The 50:50 actor changes less, but its retained "
      "behavior is still poor. These data reinforce a crucial distinction: replay can keep old examples and stabilize "
      "parameters without retaining a competent goal-conditioned policy.")]

  story += [heading("9. Implications for continual imagined rehearsal")]
  story += [heading("9.1 What the baseline establishes", 2)]
  story += [bullets([
      "<b>Catastrophic forgetting is measurable in MiniGrid.</b> GoToDoor supplies a strong, replicated acquisition-to-collapse signal, so Dreamer is not too strong for the selected sequential regime.",
      "<b>Memory availability is not enough.</b> Both reservoir arms retain old goal episodes, yet old behavior degrades and late tasks become harder to acquire.",
      "<b>Uniform replay is a scientifically useful but poor performance baseline.</b> Its low current-task share reveals why CIR was intended to keep the actor focused on the current posterior while components rehearse old goals.",
      "<b>The reward model is only partly serviceable.</b> Factual event recognition can remain strong while counterfactual goal semantics remain weak.",
      "<b>The critic is presently unsafe for unbounded IR.</b> Its real-state calibration is poor, its slow copy repeats the error, and actor-selected imagination increases the target.",
      "<b>Forward transfer is not yet present.</b> The next intervention should first solve acquisition and retention; positive transfer remains a later criterion.",
  ])]

  story += [heading("9.2 Recommended next experimental sequence", 2)]
  story += [P(
      "The next run should not add a gate around the current teacher. Gating decides <i>when</i> to use a signal; it does "
      "not make a semantically confused reward head or overoptimistic critic correct. The evidence supports a staged "
      "mechanistic programme:")]
  story += [bullets([
      "Train the reward and continuation heads with counterfactual goal labels on real replay states, including explicit nonmatching goals. Require high completion-goal top-1 accuracy and a material matching-minus-nonmatching margin on held-out state banks.",
      "Regularize or bound the critic during ordinary training. Candidate controls include Monte Carlo calibration on replay, bounded success-probability values, conservative targets, and explicit penalties for values outside the task's feasible return range.",
      "Before using any critic tail in IR, compare reward-only rehearsal with bounded-bootstrap rehearsal. The unmodified critic-bootstrapped objective should remain a negative control, not the presumed default.",
      "Retain both reservoir allocations. Uniform replay remains the primary component-preservation baseline; 50:50 remains the ordinary-replay comparator that CIR must beat on current acquisition.",
      "Only after the repaired teacher passes frozen-state and imagined-path audits should a safety gate be calibrated. The gate should veto uncertain or out-of-distribution rehearsal, not compensate for known target bias.",
      "Repeat the four-task stream over more seeds after the intervention is fixed, then carry the validated mechanism through Fetch-style simulated manipulation, redesigned RLScape, and finally the CoBot320Pi arm.",
  ], numbered=True)]
  story += [callout(
      "Primary next hypothesis.",
      "Counterfactually supervised reward semantics plus reward-only or bounded-value imagined rehearsal should retain "
      "old-task competence without reducing current-task exposure below the 50:50 replay control. Success requires both "
      "behavioral retention and a calibrated, goal-selective teaching signal.", LIGHTGREEN, GREEN)]

  story += [heading("9.3 Relationship to the RLScape evidence", 2)]
  story += [P(
      "The MiniGrid baseline does not replace RLScape; it removes one major ambiguity from it. RLScape's large click "
      "action space and Berry Bones task could have masqueraded as agent failure. MiniGrid uses seven shared actions and "
      "still reproduces the two most important mechanistic findings: weak counterfactual reward semantics and severe "
      "critic optimism. The critic problem is therefore unlikely to be only an RLScape action-resolution artifact. At "
      "the same time, MiniGrid shows that task difficulty remains consequential: GoToObject and late-task reservoir "
      "acquisition are weak enough that behavior alone cannot identify the failing component.")]

  story += [heading("10. Limitations and claim boundaries")]
  story += [bullets([
      "<b>Three training seeds.</b> Repetition is sufficient to reveal consistent qualitative failure but not to estimate small arm differences precisely.",
      "<b>Fifty episodes per policy mode and checkpoint.</b> Paired resets control comparisons; each individual proportion retains binomial uncertainty.",
      "<b>Task difficulty remains unequal.</b> GoToObject is weak across arms, and PutNear is strongly sensitive to current-task exposure. Aggregate forgetting cannot replace task-level reporting.",
      "<b>No return-to-A phase.</b> The study measures retention at later checkpoints, not rapid reacquisition. The immutable 800k checkpoints allow a later continuation.",
      "<b>No IR intervention.</b> The study diagnoses requirements for CIR but does not show that reward-only, bounded, counterfactual, or gated rehearsal succeeds.",
      "<b>Counterfactual outcomes lack oracle returns.</b> Goal-query differences are valid same-state comparisons, but only factual recorded trajectories receive realized-return calibration labels.",
      "<b>Fixed state banks come from final uniform replay.</b> This enforces identical evidence across arms but conditions diagnostics on states retained by that replay design.",
      "<b>One reduced Dreamer configuration.</b> Results are conditional on the 12M preset, train ratio 32, 100-action horizon, fixed missions, and four-task order.",
      "<b>One serial hardware run.</b> Runtime demonstrates feasibility on the RTX 4090, not cross-platform performance.",
  ], numbered=True)]

  story += [heading("11. Conclusion")]
  story += [P(
      "Together, the two MiniGrid studies establish a fast and revealing mechanism-development tier for CIR. The "
      "independent audition rejected an overconfident task-matching assumption. The sequential baseline then completed "
      "7.2 million training steps, 75,600 paired evaluation episodes, and 90 read-only component audits in 30.17 hours "
      "without a failed scientific unit. Dreamer is not too strong for this regime: catastrophic forgetting is large, "
      "late-task acquisition is replay-sensitive, and forward transfer is absent.")]
  story += [P(
      "The most important outcome is mechanistic. Uniform reservoir replay preserves data but starves current learning; "
      "50:50 replay improves acquisition but does not preserve competence. Reward heads recognize factual events without "
      "reliably encoding which goal those events satisfy. Critics become grossly optimistic on real states, slow critics "
      "repeat the bias, and actor-selected imagination amplifies bootstrap-dominated targets. The mental fallback of "
      "reward-model-only imagined rehearsal is therefore supported, but only after counterfactual goal semantics are "
      "repaired. Critic regularization remains valuable for eventual bounded IR, not a prerequisite for testing a "
      "reward-only positive control. Gating comes after these component failures are corrected and measured.")]

  story += [PageBreak(), heading("Appendix A. Seed-level final results")]
  seed_table = [
      ["Task", "Seed 0 Samp.", "Seed 0 Det.", "Seed 1 Samp.", "Seed 1 Det.", "Seed 2 Samp.", "Seed 2 Det."],
      ["GoToDoor", "92%", "90%", "0%", "0%", "46%", "48%"],
      ["Fetch", "4%", "2%", "60%", "56%", "4%", "6%"],
      ["GoToObject", "0%", "0%", "70%", "56%", "40%", "42%"],
      ["PutNear", "92%", "90%", "100%", "100%", "62%", "74%"],
      ["UnlockPickup", "0%", "0%", "0%", "0%", "0%", "0%"],
      ["PickupDist", "4%", "4%", "0%", "0%", "0%", "2%"],
  ]
  story += [report_table(seed_table, [78] + [72] * 6, font_size=6.8,
                               alignments=[TA_LEFT] + [TA_CENTER] * 6),
            caption("Table A1", "Final sampled and deterministic success for every run. Each cell contains 50 episodes.")]

  story += [heading("Appendix B. Checkpoint means")]
  checkpoint_table = [
      ["Task", "25k S", "25k D", "50k S", "50k D", "75k S", "75k D", "100k S", "100k D"],
      ["GoToDoor", "0.0%", "0.0%", "0.0%", "0.0%", "31.3%", "28.7%", "46.0%", "46.0%"],
      ["Fetch", "0.0%", "0.0%", "0.7%", "0.7%", "8.7%", "8.7%", "22.7%", "21.3%"],
      ["GoToObject", "0.0%", "0.0%", "4.7%", "5.3%", "14.7%", "14.7%", "36.7%", "32.7%"],
      ["PutNear", "2.0%", "0.0%", "20.0%", "15.3%", "0.0%", "0.0%", "84.7%", "88.0%"],
      ["UnlockPickup", "0.0%", "0.0%", "0.0%", "0.0%", "0.0%", "0.0%", "0.0%", "0.0%"],
      ["PickupDist", "0.0%", "0.0%", "0.7%", "0.0%", "3.3%", "4.0%", "1.3%", "2.0%"],
  ]
  story += [report_table(checkpoint_table, [78] + [54] * 8, font_size=6.6,
                               alignments=[TA_LEFT] + [TA_CENTER] * 8),
            caption("Table B1", "Mean success across three seeds. S = sampled; D = deterministic.")]

  story += [heading("Appendix C. Reproducibility package")]
  story += [P(
      "The report directory contains compact evidence packages for both studies: immutable specifications, complete "
      "aggregate evaluation tables, supervisor provenance, replay-allocation summaries, and every per-checkpoint head "
      "and critic summary used here. Raw replay archives, checkpoints, JSONL logs, prediction tensors, and rollout "
      "shards remain authoritative on the RTX 4090 under /home/localadmin/experiment_logs/minigrid/.")]
  repro = [
      ["Artifact", "Role"],
      ["data/experiment_spec.json", "Pinned task, model, evaluation, pairing, and dependency specification"],
      ["data/evaluation_summary.csv", "All 144 checkpoint-policy conditions"],
      ["data/run_summary.csv", "Per-task and per-seed training/final metrics"],
      ["data/task_summary.csv", "Cross-seed task aggregation"],
      ["data/pairwise_difficulty.csv", "Predeclared pairwise decisions"],
      ["data/supervisor_status.json", "Commands, timestamps, attempts, return codes, and runtime versions"],
      ["data/sequential_baseline/analysis/", "All Study II behavior, replay, drift, and component aggregate tables"],
      ["data/sequential_baseline/diagnostics/", "Compact summaries and specifications for all 90 read-only audits"],
      ["data/sequential_baseline/supervisor_status.json", "Complete Study II commands, attempts, timings, and terminal state"],
      ["build_report.py", "Reproducible vector-PDF builder"],
  ]
  story += [report_table(repro, [165, 348], font_size=7.4),
            caption("Table C1", "Portable report package. Raw run artifacts remain authoritative.")]
  story += [P(
      "Regenerate with:<br/><font name='Courier' size='7'>"
      "/home/zythax/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12 "
      "reports/minigrid_baseline_report/build_report.py</font>")]

  story += [PageBreak(), heading("Appendix D. Study II task-level outcomes")]
  seq_task_table = [
      ["Arm", "Task", "Sampled acq.", "Sampled final", "Sampled BWT", "Det. acq.", "Det. final", "Det. BWT"],
      ["FIFO", "GoToDoor", "95.3%", "0.7%", "-94.7 pp", "95.3%", "0.0%", "-95.3 pp"],
      ["FIFO", "Fetch", "18.7%", "0.0%", "-18.7 pp", "12.0%", "0.0%", "-12.0 pp"],
      ["FIFO", "GoToObject", "0.0%", "2.0%", "+2.0 pp", "0.0%", "4.7%", "+4.7 pp"],
      ["FIFO", "PutNear", "67.3%", "67.3%", "+0.0 pp", "66.7%", "66.7%", "+0.0 pp"],
      ["Uniform", "GoToDoor", "78.0%", "0.0%", "-78.0 pp", "74.0%", "0.0%", "-74.0 pp"],
      ["Uniform", "Fetch", "1.3%", "0.0%", "-1.3 pp", "0.0%", "0.0%", "+0.0 pp"],
      ["Uniform", "GoToObject", "0.0%", "0.0%", "+0.0 pp", "0.0%", "0.0%", "+0.0 pp"],
      ["Uniform", "PutNear", "0.0%", "0.0%", "+0.0 pp", "0.0%", "0.0%", "+0.0 pp"],
      ["50:50", "GoToDoor", "92.0%", "28.7%", "-63.3 pp", "94.7%", "28.7%", "-66.0 pp"],
      ["50:50", "Fetch", "55.3%", "0.0%", "-55.3 pp", "58.7%", "0.0%", "-58.7 pp"],
      ["50:50", "GoToObject", "30.0%", "0.0%", "-30.0 pp", "28.0%", "0.0%", "-28.0 pp"],
      ["50:50", "PutNear", "0.0%", "0.0%", "+0.0 pp", "0.0%", "0.0%", "+0.0 pp"],
  ]
  story += [report_table(seq_task_table, [54, 68, 67, 65, 66, 62, 59, 67], font_size=6.0,
                               alignments=[TA_LEFT, TA_LEFT] + [TA_CENTER] * 6),
            caption("Table D1", "Mean immediate post-acquisition success, final success, and backward transfer over three seeds. BWT = final minus post-acquisition.")]
  story += [P(
      "The underlying transfer_and_forgetting.csv retains all 72 task-seed-mode rows, including initial, pretraining, "
      "peak post-acquisition, final, forward-transfer, and forgetting values. The report uses means only for legibility; "
      "seed-level results remain available for uncertainty analysis and later paired intervention comparisons.")]

  story += [heading("Appendix E. Study II audit inventory and integrity")]
  audit_table = [
      ["Evidence", "Scope", "Integrity / role"],
      ["Evaluation matrix", "1,512 conditions; 75,600 episodes", "Exact episode/reset counts; paired modes; unchanged checkpoint digests"],
      ["Replay allocation", "288 logged snapshots", "Observed goal fractions and reservoir occupancy"],
      ["Goal-specific losses", "288 logged snapshots", "Reward, continuation, value, policy, representation, and dynamics loss by replay goal"],
      ["Head audits", "45 jobs; five boundaries x nine runs", "Identical state bank per seed; all six goal queries; unchanged parameters"],
      ["Critic provenance", "45 jobs; H=1/3/6/15", "32 posterior samples; actor and recorded actions; unchanged parameters"],
      ["Actor drift", "144 adjacent-boundary comparisons", "Same states and factual goal query before/after each phase"],
  ]
  story += [report_table(audit_table, [103, 148, 262], font_size=6.8),
            caption("Table E1", "Study II visibility and integrity evidence.")]
  story += [P(
      "The compact report package is approximately 19 MB. The excluded raw remote evidence includes about 700 MB of "
      "diagnostic predictions and per-episode critic shards plus the full 80 GB experiment tree. Compact summaries "
      "record source selections, specification digests, row counts, and before/after parameter hashes so every reported "
      "aggregate can be traced back to its immutable job.")]

  story += [heading("References")]
  refs = [
      "[1] Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural networks. <i>Proceedings of the National Academy of Sciences</i>, 114(13), 3521-3526.",
      "[2] Chevalier-Boisvert, M. et al. (2023). MiniGrid and MiniWorld: Modular and customizable reinforcement learning environments for goal-oriented tasks. <i>Advances in Neural Information Processing Systems</i>, 36.",
      "[3] Hafner, D., Pasukonis, J., Ba, J., and Lillicrap, T. (2023). Mastering diverse domains through world models. <i>arXiv:2301.04104</i>.",
  ]
  story += [P(ref, "Reference") for ref in refs]
  return story


def main() -> int:
  REPORT_PDF.parent.mkdir(parents=True, exist_ok=True)
  doc = ReportDocTemplate(
      str(REPORT_PDF),
      pagesize=A4,
      leftMargin=20 * mm,
      rightMargin=20 * mm,
      topMargin=18 * mm,
      bottomMargin=17 * mm,
      title="MiniGrid Experiments for Continual Imagined Rehearsal",
      author="Steven Davenport",
      subject="Independent task audition and visibility-first sequential baseline",
  )
  doc.multiBuild(build_story())
  print(REPORT_PDF)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
