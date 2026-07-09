#!/usr/bin/env python3
"""Run Walker intra-episode rehearsal trace sweeps with per-step logging enabled."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from checkpoint_utils import resolve_checkpoint_path


def parse_int_list(text: str) -> list[int]:
  return [int(x.strip()) for x in text.split(',') if x.strip()]


def parse_text_list(text: str) -> list[str]:
  return [x.strip() for x in text.split(',') if x.strip()]


def safe_mean(values: list[float]) -> float:
  vals = [x for x in values if not math.isnan(x)]
  return sum(vals) / len(vals) if vals else float('nan')


def safe_std(values: list[float]) -> float:
  vals = [x for x in values if not math.isnan(x)]
  if len(vals) < 2:
    return 0.0 if len(vals) == 1 else float('nan')
  return statistics.stdev(vals)


def ci95(values: list[float]) -> float:
  vals = [x for x in values if not math.isnan(x)]
  if len(vals) < 2:
    return 0.0 if len(vals) == 1 else float('nan')
  return 1.96 * (statistics.stdev(vals) / math.sqrt(len(vals)))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
  if not path.exists():
    return []
  rows = []
  with path.open() as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        rows.append(json.loads(line))
      except json.JSONDecodeError:
        continue
  return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  if not rows:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  keys = sorted({key for row in rows for key in row.keys()})
  with path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def summarize_run(run_dir: Path, severity: str, config_name: str, seed: int,
                  status: str, returncode: int, wall_seconds: float) -> dict[str, Any]:
  scores_rows = read_jsonl(run_dir / 'scores.jsonl')
  metrics_rows = read_jsonl(run_dir / 'metrics.jsonl')
  trace_rows = read_jsonl(run_dir / 'trace.jsonl')

  scores = [float(row['episode/score']) for row in scores_rows if 'episode/score' in row]
  lengths = [float(row['episode/length']) for row in metrics_rows if 'episode/length' in row]
  fps = [float(row['fps/policy']) for row in metrics_rows if 'fps/policy' in row]
  triggers = [float(row.get('adapt_trigger', 0.0)) for row in trace_rows]

  return {
      'severity': severity,
      'config': config_name,
      'seed': seed,
      'status': status,
      'returncode': returncode,
      'wall_seconds': wall_seconds,
      'episodes': len(scores),
      'trace_rows': len(trace_rows),
      'trace_triggers': sum(1 for x in triggers if x > 0),
      'score_mean': safe_mean(scores),
      'score_std': safe_std(scores),
      'length_mean': safe_mean(lengths),
      'fps_policy_mean': safe_mean(fps),
  }


def summarize_by_config(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
  for row in rows:
    grouped.setdefault((row['severity'], row['config']), []).append(row)

  summary = []
  for (severity, config), items in sorted(grouped.items()):
    ok = [x for x in items if x['status'] == 'ok' and x['episodes'] > 0]
    summary.append({
        'severity': severity,
        'config': config,
        'runs_total': len(items),
        'runs_ok': len(ok),
        'episodes_total': sum(int(x['episodes']) for x in ok),
        'trace_rows_total': sum(int(x['trace_rows']) for x in ok),
        'trace_triggers_total': sum(int(x['trace_triggers']) for x in ok),
        'seeds': ','.join(str(x['seed']) for x in sorted(ok, key=lambda y: y['seed'])),
        'score_mean_mean': safe_mean([float(x['score_mean']) for x in ok]),
        'score_mean_std': safe_std([float(x['score_mean']) for x in ok]),
        'score_mean_ci95': ci95([float(x['score_mean']) for x in ok]),
        'length_mean_mean': safe_mean([float(x['length_mean']) for x in ok]),
        'fps_policy_mean_mean': safe_mean([float(x['fps_policy_mean']) for x in ok]),
        'wall_seconds_mean': safe_mean([float(x['wall_seconds']) for x in ok]),
    })
  return summary


def build_command(args: argparse.Namespace, checkpoint: Path, run_logdir: Path, seed: int,
                  start_batch: int) -> list[str]:
  return [
      args.python_bin,
      'dreamerv3/main.py',
      '--script', 'eval_only',
      '--logdir', str(run_logdir),
      '--configs', args.configs,
      '--task', args.task,
      '--run.from_checkpoint', str(checkpoint),
      '--run.steps', str(int(args.run_steps)),
      '--run.envs', '1',
      '--run.log_every', str(int(args.run_log_every)),
      '--logger.filter', args.logger_filter,
      '--jax.platform', args.jax_platform,
      '--seed', str(seed),
      '--eval_adapt.enabled', 'True',
      '--eval_adapt.steps', str(int(args.adapt_steps)),
      '--eval_adapt.imag_length', str(int(args.adapt_imag_length)),
      '--eval_adapt.lr', str(float(args.adapt_lr)),
      '--eval_adapt.every_k', str(int(args.adapt_every_k)),
      '--eval_adapt.actent', str(float(args.adapt_actent)),
      '--eval_adapt.start_batch', str(int(start_batch)),
      '--eval_adapt.trace.enabled', 'True',
      '--eval_adapt.trace.filename', args.trace_filename,
  ]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--repo_root', default='.')
  parser.add_argument('--outdir', default='')
  parser.add_argument('--python_bin', default=sys.executable)
  parser.add_argument('--checkpoints_root',
                      default='checkpoints/paper_corruptions/dmc_walker_walk/zero_mask')
  parser.add_argument('--severities', default='mild,moderate,severe')
  parser.add_argument('--start_batches', default='1,512')
  parser.add_argument('--configs', default='dmc_vision')
  parser.add_argument('--task', default='dmc_walker_walk')
  parser.add_argument('--seeds', default='0,1,2,3,4')
  parser.add_argument('--run_steps', type=int, default=5000)
  parser.add_argument('--run_log_every', type=int, default=50)
  parser.add_argument('--logger_filter', default='score|length|fps|eval_adapt|epstats')
  parser.add_argument('--jax_platform', default='cuda')
  parser.add_argument('--mujoco_gl', default='egl')
  parser.add_argument('--adapt_steps', type=int, default=1)
  parser.add_argument('--adapt_imag_length', type=int, default=6)
  parser.add_argument('--adapt_lr', type=float, default=1e-4)
  parser.add_argument('--adapt_every_k', type=int, default=1)
  parser.add_argument('--adapt_actent', type=float, default=3e-4)
  parser.add_argument('--trace_filename', default='trace.jsonl')
  parser.add_argument('--skip_existing', action='store_true')
  parser.add_argument('--continue_on_error', action='store_true')
  parser.add_argument('--dry_run', action='store_true')
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
  outdir = Path(args.outdir) if args.outdir else Path(
      'eval_suite/paper/intra_episode_recovery') / f'walker_trace_{timestamp}'
  outdir.mkdir(parents=True, exist_ok=True)

  severities = parse_text_list(args.severities)
  start_batches = parse_int_list(args.start_batches)
  seeds = parse_int_list(args.seeds)
  checkpoints_root = Path(args.checkpoints_root)

  checkpoint_map = {
      severity: resolve_checkpoint_path(checkpoints_root / severity)
      for severity in severities
  }
  experiments = [{
      'severity': severity,
      'checkpoint': str(checkpoint_map[severity]),
      'configs': [{
          'name': f'rehearsal_sb{sb}',
          'start_batch': int(sb),
      } for sb in start_batches],
  } for severity in severities]
  (outdir / 'suite_plan.json').write_text(json.dumps({
      'timestamp': timestamp,
      'args': vars(args),
      'experiments': experiments,
  }, indent=2))

  rows: list[dict[str, Any]] = []
  total = len(severities) * len(start_batches) * len(seeds)
  done = 0
  for severity in severities:
    checkpoint = checkpoint_map[severity]
    severity_dir = outdir / severity
    severity_dir.mkdir(parents=True, exist_ok=True)
    for start_batch in start_batches:
      config_name = f'rehearsal_sb{start_batch}'
      for seed in seeds:
        done += 1
        run_logdir = severity_dir / 'runs' / config_name / f'seed{seed}'
        run_logdir.mkdir(parents=True, exist_ok=True)
        cmd = build_command(args, checkpoint, run_logdir, seed, start_batch)
        (run_logdir / 'command.sh').write_text(' '.join(shlex.quote(x) for x in cmd) + '\n')
        print(f'[{done}/{total}] {severity} {config_name} seed={seed}')

        start = time.time()
        returncode = 0
        status = 'ok'
        if args.skip_existing and (
            (run_logdir / 'scores.jsonl').exists() and
            (run_logdir / 'metrics.jsonl').exists() and
            (run_logdir / args.trace_filename).exists()):
          pass
        elif args.dry_run:
          status = 'dry_run'
        else:
          with (run_logdir / 'stdout.log').open('w') as stdout_f:
            with (run_logdir / 'stderr.log').open('w') as stderr_f:
              proc = subprocess.run(
                  cmd,
                  cwd=args.repo_root,
                  env={**os.environ, 'MUJOCO_GL': args.mujoco_gl},
                  stdout=stdout_f,
                  stderr=stderr_f,
                  check=False,
              )
              returncode = int(proc.returncode)
          if returncode != 0:
            status = 'failed'
            print(f'  FAILED (code={returncode}) -> {run_logdir / "stderr.log"}')
            if not args.continue_on_error:
              rows.append(summarize_run(
                  run_logdir, severity, config_name, seed, status, returncode, time.time() - start))
              write_csv(outdir / 'runs.csv', rows)
              write_csv(outdir / 'summary_by_config.csv', summarize_by_config(rows))
              return 1

        rows.append(summarize_run(
            run_logdir, severity, config_name, seed, status, returncode, time.time() - start))
        write_csv(outdir / 'runs.csv', rows)
        write_csv(outdir / 'summary_by_config.csv', summarize_by_config(rows))

  print(f'Wrote: {outdir / "runs.csv"}')
  print(f'Wrote: {outdir / "summary_by_config.csv"}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
