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


def csv_rows(name: str) -> list[dict[str, str]]:
  with (DATA / name).open(newline="") as stream:
    return list(csv.DictReader(stream))


EVAL_ROWS = csv_rows("evaluation_summary.csv")
RUN_ROWS = csv_rows("run_summary.csv")
TASK_ROWS = csv_rows("task_summary.csv")
SPEC = json.loads((DATA / "experiment_spec.json").read_text())
STATUS = json.loads((DATA / "supervisor_status.json").read_text())


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
                        "MiniGrid Difficulty Audition")
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
  story += [Spacer(1, 21 * mm), P("MiniGrid Difficulty Audition for<br/>Continual Imagined Rehearsal", "Title")]
  story += [P("Initial baseline study of task learnability, seed variance,<br/>policy-mode sensitivity, and curriculum selection", "Subtitle")]
  story += [Spacer(1, 4 * mm), P("Steven Davenport", "Meta"),
            P("Continual Reinforcement Learning Research Notes", "Meta"),
            Spacer(1, 3 * mm),
            P("Experiment completed 30 August 2026; report generated 30 August 2026", "Meta")]
  story += [Spacer(1, 7 * mm), P("Abstract", "AbstractTitle")]
  abstract = (
      "This report establishes the first MiniGrid baseline for the Continual Imagined Rehearsal (CIR) programme. "
      "The purpose was not yet to test forgetting or transfer. It was to identify a fast, controlled set of tasks "
      "that a deliberately reduced DreamerV3 agent could learn without either immediately saturating or failing "
      "completely, before committing compute to sequential A-to-B-to-C-to-A experiments. Six fixed-mission tasks "
      "spanning navigation, pickup, pickup-and-place, and locked-door interaction were trained independently for "
      "100,000 environment steps over three seeds. The resulting matrix contains 1.8 million training steps, 27,892 "
      "training episodes, four immutable checkpoints per run, and 7,200 evaluation episodes split equally between "
      "sampled and deterministic action selection.<br/><br/>"
      "The run completed serially on one RTX 4090 in 5.14 hours with no failed units or retries. Task difficulty was "
      "not balanced. At the final checkpoint, deterministic success was 88.0% for PutNear, 46.0% for GoToDoor, "
      "32.7% for GoToObject, 21.3% for Fetch, 2.0% for PickupDist, and zero for UnlockPickup. PutNear succeeded in "
      "all seeds, whereas GoToDoor, Fetch, and GoToObject were highly seed-sensitive. Sampled and deterministic "
      "evaluations closely agreed across the 72 paired checkpoint conditions: the mean absolute success difference "
      "was only 1.19 percentage points. The predeclared difficulty rule identified GoToDoor and PutNear as the sole "
      "matched pair, but that result is not robust enough to approve a curriculum because the rule ignores failed-seed "
      "prevalence when computing median threshold time. Neither proposed candidate chain passed.<br/><br/>"
      "The experiment therefore succeeds as an audition and rejects the original assumption that these six tasks are "
      "comparably difficult under a common 100k-step budget. MiniGrid is fast enough for broad iteration, but a second "
      "calibration stage is required before the first continual sequence."
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

  story += [heading("6. Limitations and claim boundaries")]
  story += [bullets([
      "<b>Only three training seeds.</b> Means and standard deviations reveal large variation but do not provide stable population estimates.",
      "<b>Only 50 evaluation episodes per condition.</b> Pairing controls reset seeds across modes, but individual success differences retain binomial uncertainty.",
      "<b>One model and optimizer configuration.</b> Ranking is conditional on the 12m Dreamer, FIFO replay, train ratio 32, and 100k budget.",
      "<b>Independent rather than continual training.</b> No result demonstrates retention, forgetting, transfer, or rehearsal benefit.",
      "<b>Fixed missions remove language variation.</b> This strengthens causal control but narrows generalization relative to stock mission sampling.",
      "<b>Task geometry is not standardized.</b> PutNear uses a smaller room and UnlockPickup uses two rooms; difficulty combines behavior and geometry.",
      "<b>The match rule is reliability-blind.</b> Conditional median threshold time excludes failed seeds, so its sole positive match is provisional.",
      "<b>One hardware run.</b> Timing establishes feasibility on the RTX 4090 machine, not cross-platform performance.",
  ], numbered=True)]

  story += [heading("7. Conclusion")]
  story += [P(
      "The first MiniGrid baseline provides a fast, audited foundation for the CIR programme. A single environment and "
      "serial supervisor produced 1.8 million Dreamer training steps plus 7,200 paired evaluation episodes in 5.14 "
      "hours, with no retries, failures, missing episodes, or evaluation-time parameter changes. This confirms "
      "MiniGrid's value as the rapid iteration tier between mechanism tests and expensive RLScape or robotics experiments.")]
  story += [P(
      "The scientific outcome is selective rather than confirmatory. PutNear is reliably learnable; GoToDoor, "
      "GoToObject, and Fetch contain useful but seed-sensitive signal; PickupDist is almost entirely unsuccessful; and "
      "UnlockPickup is beyond the current budget. The original candidate chains are not difficulty-balanced, and the "
      "formal GoToDoor-PutNear match is weakened by unequal seed robustness. The next move is a focused, "
      "robustness-aware calibration, followed by the first true sequential MiniGrid baseline. Only then should "
      "continual imagined rehearsal be judged on forgetting, forward transfer, recurrence, and safety.")]

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
      "The report directory contains the immutable experiment specification, complete evaluation summary, per-run and "
      "per-task summaries, pairwise decision table, supervisor status, builder, and report source. The original raw "
      "replay archives, checkpoints, JSONL logs, and audit streams remain on the RTX 4090 machine at "
      "/home/localadmin/experiment_logs/minigrid/audition_12m_v1.")]
  repro = [
      ["Artifact", "Role"],
      ["data/experiment_spec.json", "Pinned task, model, evaluation, pairing, and dependency specification"],
      ["data/evaluation_summary.csv", "All 144 checkpoint-policy conditions"],
      ["data/run_summary.csv", "Per-task and per-seed training/final metrics"],
      ["data/task_summary.csv", "Cross-seed task aggregation"],
      ["data/pairwise_difficulty.csv", "Predeclared pairwise decisions"],
      ["data/supervisor_status.json", "Commands, timestamps, attempts, return codes, and runtime versions"],
      ["build_report.py", "Reproducible vector-PDF builder"],
  ]
  story += [report_table(repro, [165, 348], font_size=7.4),
            caption("Table C1", "Portable report package. Raw run artifacts remain authoritative.")]
  story += [P(
      "Regenerate with:<br/><font name='Courier' size='7'>"
      "/home/zythax/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12 "
      "reports/minigrid_baseline_report/build_report.py</font>")]

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
      title="MiniGrid Difficulty Audition for Continual Imagined Rehearsal",
      author="Steven Davenport",
      subject="Initial MiniGrid baseline experiment",
  )
  doc.multiBuild(build_story())
  print(REPORT_PDF)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
