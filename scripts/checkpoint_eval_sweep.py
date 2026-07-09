#!/usr/bin/env python3
"""Run plain eval sweeps over a list of checkpoints and summarize them."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from checkpoint_utils import expand_checkpoint_paths


def parse_text_list(text: str) -> list[str]:
  return [x.strip() for x in text.split(',') if x.strip()]


def sanitize(text: str) -> str:
  out = []
  for char in text:
    if char.isalnum() or char in ('-', '_'):
      out.append(char)
    else:
      out.append('_')
  return ''.join(out).strip('_') or 'x'


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
  return 1.96 * statistics.stdev(vals) / math.sqrt(len(vals))


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


def load_manifest(checkpoint: Path) -> dict[str, Any]:
  path = checkpoint / 'manifest.json'
  if not path.exists():
    return {}
  try:
    return json.loads(path.read_text())
  except json.JSONDecodeError:
    return {}


def build_command(args: argparse.Namespace, logdir: Path, checkpoint: Path, seed: int) -> list[str]:
  return [
      args.python_bin,
      'dreamerv3/main.py',
      '--script', 'eval_only',
      '--logdir', str(logdir),
      '--configs', args.configs,
      '--task', args.task,
      '--run.from_checkpoint', str(checkpoint),
      '--run.steps', str(int(args.run_steps)),
      '--run.envs', str(int(args.run_envs)),
      '--run.log_every', str(int(args.run_log_every)),
      '--logger.filter', args.logger_filter,
      '--jax.platform', args.jax_platform,
      '--jax.prealloc', str(bool(args.jax_prealloc)),
      '--seed', str(seed),
      '--eval_adapt.enabled', 'False',
  ]


def summarize_run(run_dir: Path, checkpoint_name: str, seed: int, status: str, returncode: int, wall_seconds: float) -> dict[str, Any]:
  scores_rows = read_jsonl(run_dir / 'scores.jsonl')
  metrics_rows = read_jsonl(run_dir / 'metrics.jsonl')

  scores = [float(row['episode/score']) for row in scores_rows if 'episode/score' in row]
  lengths = [float(row['episode/length']) for row in metrics_rows if 'episode/length' in row]
  fps = [float(row['fps/policy']) for row in metrics_rows if 'fps/policy' in row]

  return {
      'checkpoint_name': checkpoint_name,
      'seed': seed,
      'status': status,
      'returncode': returncode,
      'wall_seconds': wall_seconds,
      'episodes': len(scores),
      'score_mean': safe_mean(scores),
      'score_std': safe_std(scores),
      'score_total': sum(scores) if scores else 0.0,
      'success_rate': (
          sum(1 for x in scores if x > float(0.0)) / len(scores)
          if scores else float('nan')),
      'length_mean': safe_mean(lengths),
      'fps_policy_mean': safe_mean(fps),
      'fps_policy_last': fps[-1] if fps else float('nan'),
  }


def summarize_by_checkpoint(rows: list[dict[str, Any]], manifests: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
  grouped: dict[str, list[dict[str, Any]]] = {}
  for row in rows:
    grouped.setdefault(row['checkpoint_name'], []).append(row)

  summary = []
  for checkpoint_name, items in sorted(grouped.items()):
    ok = [x for x in items if x['status'] == 'ok' and x['episodes'] > 0]
    manifest = manifests.get(checkpoint_name, {})
    metrics = {}
    for key in ('score_mean', 'score_total', 'success_rate', 'length_mean', 'fps_policy_mean', 'wall_seconds'):
      vals = [float(x[key]) for x in ok]
      metrics[f'{key}_mean'] = safe_mean(vals)
      metrics[f'{key}_std'] = safe_std(vals)
      metrics[f'{key}_ci95'] = ci95(vals)
    total_eps = sum(int(x['episodes']) for x in ok)
    weighted_score_mean = (
        sum(float(x['score_mean']) * int(x['episodes']) for x in ok) / total_eps
        if total_eps else float('nan'))
    summary.append({
        'checkpoint_name': checkpoint_name,
        'checkpoint': manifest.get('checkpoint', ''),
        'runs_total': len(items),
        'runs_ok': len(ok),
        'episodes_total': total_eps,
        'seeds': ','.join(str(x['seed']) for x in sorted(ok, key=lambda y: y['seed'])),
        'corruption_method': manifest.get('method', ''),
        'corruption_scope': manifest.get('scope', ''),
        'corruption_strength': manifest.get('strength', ''),
        'corruption_relative_l2': manifest.get('relative_l2', ''),
        'score_mean_weighted_by_episodes': weighted_score_mean,
        **metrics,
    })
  return summary


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--repo_root', default='.')
  parser.add_argument('--outdir', default='')
  parser.add_argument('--python_bin', default=sys.executable)
  parser.add_argument('--checkpoints', required=True)
  parser.add_argument('--configs', default='dmc_vision')
  parser.add_argument('--task', default='dmc_walker_walk')
  parser.add_argument('--seeds', default='0')
  parser.add_argument('--run_steps', type=int, default=10000)
  parser.add_argument('--run_envs', type=int, default=1)
  parser.add_argument('--run_log_every', type=int, default=50)
  parser.add_argument('--logger_filter', default='score|length|fps')
  parser.add_argument('--jax_platform', default='cpu')
  parser.add_argument('--jax_prealloc', action='store_true')
  parser.add_argument('--dry_run', action='store_true')
  parser.add_argument('--skip_existing', action='store_true')
  parser.add_argument('--continue_on_error', action='store_true')
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
  outdir = Path(args.outdir) if args.outdir else Path('eval_suite') / f'checkpoint_eval_{timestamp}'
  outdir.mkdir(parents=True, exist_ok=True)
  runs_dir = outdir / 'runs'
  runs_dir.mkdir(exist_ok=True)

  checkpoints = expand_checkpoint_paths(parse_text_list(args.checkpoints))
  seeds = [int(x) for x in parse_text_list(args.seeds)]
  if not checkpoints:
    raise ValueError('No checkpoints parsed from --checkpoints')
  if not seeds:
    raise ValueError('No seeds parsed from --seeds')

  plan = {
      'timestamp': timestamp,
      'args': vars(args),
      'checkpoints': [str(x) for x in checkpoints],
      'seeds': seeds,
  }
  (outdir / 'suite_plan.json').write_text(json.dumps(plan, indent=2))

  manifests = {}
  rows: list[dict[str, Any]] = []
  total = len(checkpoints) * len(seeds)
  done = 0
  for checkpoint in checkpoints:
    manifest = load_manifest(checkpoint)
    manifest['checkpoint'] = str(checkpoint)
    manifests[checkpoint.name] = manifest
    for seed in seeds:
      done += 1
      run_logdir = runs_dir / sanitize(checkpoint.name) / f'seed{seed}'
      run_logdir.mkdir(parents=True, exist_ok=True)
      cmd = build_command(args, run_logdir, checkpoint, seed)
      (run_logdir / 'command.sh').write_text(' '.join(shlex.quote(x) for x in cmd) + '\n')
      print(f'[{done}/{total}] {checkpoint.name} seed={seed}')

      start = time.time()
      status = 'ok'
      returncode = 0
      skipped_existing = 0
      if args.skip_existing and (run_logdir / 'scores.jsonl').exists() and (run_logdir / 'metrics.jsonl').exists():
        skipped_existing = 1
      elif args.dry_run:
        status = 'dry_run'
      else:
        with (run_logdir / 'stdout.log').open('w') as stdout_f:
          with (run_logdir / 'stderr.log').open('w') as stderr_f:
            proc = subprocess.run(
                cmd,
                cwd=args.repo_root,
                stdout=stdout_f,
                stderr=stderr_f,
                check=False,
            )
            returncode = int(proc.returncode)
        if returncode != 0:
          status = 'failed'
          print(f'  FAILED (code={returncode}) -> {run_logdir / "stderr.log"}')
          if not args.continue_on_error:
            row = summarize_run(run_logdir, checkpoint.name, seed, status, returncode, time.time() - start)
            row['skipped_existing'] = skipped_existing
            rows.append(row)
            write_csv(outdir / 'runs.csv', rows)
            print('Stopped early due to failure. Use --continue_on_error to keep going.')
            return 1

      row = summarize_run(run_logdir, checkpoint.name, seed, status, returncode, time.time() - start)
      row['skipped_existing'] = skipped_existing
      rows.append(row)
      write_csv(outdir / 'runs.csv', rows)

  summary = summarize_by_checkpoint(rows, manifests)
  write_csv(outdir / 'summary_by_checkpoint.csv', summary)
  print(f'Wrote: {outdir / "runs.csv"}')
  print(f'Wrote: {outdir / "summary_by_checkpoint.csv"}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
