#!/usr/bin/env python3
"""Run and summarize multi-seed evaluation sweeps for rehearsal vs baseline."""

from __future__ import annotations

import argparse
import csv
import itertools
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


def parse_int_list(text: str) -> list[int]:
  return [int(x.strip()) for x in text.split(',') if x.strip()]


def parse_float_list(text: str) -> list[float]:
  return [float(x.strip()) for x in text.split(',') if x.strip()]


def safe_mean(values: list[float]) -> float:
  vals = [x for x in values if not math.isnan(x)]
  return sum(vals) / len(vals) if vals else float('nan')


def safe_std(values: list[float]) -> float:
  vals = [x for x in values if not math.isnan(x)]
  if len(vals) < 2:
    return 0.0 if len(vals) == 1 else float('nan')
  return statistics.stdev(vals)


def safe_quantile(values: list[float], q: float) -> float:
  vals = sorted(x for x in values if not math.isnan(x))
  if not vals:
    return float('nan')
  if len(vals) == 1:
    return vals[0]
  idx = (len(vals) - 1) * q
  lo = int(math.floor(idx))
  hi = int(math.ceil(idx))
  if lo == hi:
    return vals[lo]
  w = idx - lo
  return vals[lo] * (1 - w) + vals[hi] * w


def ci95(values: list[float]) -> float:
  vals = [x for x in values if not math.isnan(x)]
  if len(vals) < 2:
    return 0.0 if len(vals) == 1 else float('nan')
  return 1.96 * (statistics.stdev(vals) / math.sqrt(len(vals)))


def sanitize_token(text: str) -> str:
  out = []
  for char in text:
    if char.isalnum():
      out.append(char)
    elif char in ('.', '-', '+'):
      out.append('_')
  return ''.join(out).strip('_') or 'x'


def build_experiments(args: argparse.Namespace) -> list[dict[str, Any]]:
  if args.suite_json:
    raw = json.loads(Path(args.suite_json).read_text())
    experiments = []
    defaults = {
        'enabled': False,
        'steps': args.default_adapt_steps,
        'imag_length': args.default_adapt_imag_length,
        'lr': args.default_adapt_lr,
        'every_k': 0,
        'actent': args.default_adapt_actent,
        'start_batch': args.default_adapt_start_batch,
    }
    if not isinstance(raw, list):
      raise ValueError('--suite_json must contain a JSON list.')
    for exp in raw:
      if 'name' not in exp or 'eval_adapt' not in exp:
        raise ValueError('Each suite entry needs name + eval_adapt.')
      eval_adapt = {**defaults, **exp['eval_adapt']}
      experiments.append({
          'name': exp['name'],
          'eval_adapt': {
              'enabled': bool(eval_adapt['enabled']),
              'steps': int(eval_adapt['steps']),
              'imag_length': int(eval_adapt['imag_length']),
              'lr': float(eval_adapt['lr']),
              'every_k': int(eval_adapt['every_k']),
              'actent': float(eval_adapt['actent']),
              'start_batch': int(eval_adapt['start_batch']),
          },
      })
    return experiments

  exps: list[dict[str, Any]] = []
  if args.include_baseline:
    exps.append({
        'name': args.baseline_name,
        'eval_adapt': {
            'enabled': False,
            'steps': args.default_adapt_steps,
            'imag_length': args.default_adapt_imag_length,
            'lr': args.default_adapt_lr,
            'every_k': 0,
            'actent': args.default_adapt_actent,
            'start_batch': args.default_adapt_start_batch,
        },
    })

  for steps, imag_length, lr, every_k, actent, start_batch in itertools.product(
      parse_int_list(args.adapt_steps_grid),
      parse_int_list(args.adapt_imag_length_grid),
      parse_float_list(args.adapt_lr_grid),
      parse_int_list(args.adapt_every_k_grid),
      parse_float_list(args.adapt_actent_grid),
      parse_int_list(args.adapt_start_batch_grid),
  ):
    name = (
        f"reh_s{steps}_h{imag_length}_"
        f"lr{sanitize_token(f'{lr:.2e}')}_"
        f"k{every_k}_ae{sanitize_token(f'{actent:.2e}')}_sb{start_batch}"
    )
    exps.append({
        'name': name,
        'eval_adapt': {
            'enabled': True,
            'steps': int(steps),
            'imag_length': int(imag_length),
            'lr': float(lr),
            'every_k': int(every_k),
            'actent': float(actent),
            'start_batch': int(start_batch),
        },
    })
  return exps


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


def summarize_run(
    run_dir: Path,
    config_name: str,
    seed: int,
    status: str,
    returncode: int,
    wall_seconds: float,
    success_threshold: float,
) -> dict[str, Any]:
  scores_rows = read_jsonl(run_dir / 'scores.jsonl')
  metrics_rows = read_jsonl(run_dir / 'metrics.jsonl')

  scores = [
      float(row['episode/score'])
      for row in scores_rows
      if 'episode/score' in row
  ]
  lengths = [
      float(row['episode/length'])
      for row in metrics_rows
      if 'episode/length' in row
  ]
  fps = [
      float(row['fps/policy'])
      for row in metrics_rows
      if 'fps/policy' in row
  ]

  # Optional rehearsal diagnostics from aggregated epstats windows.
  diag = {}
  diag_keys = (
      'epstats/log/eval_adapt/trigger/sum',
      'epstats/log/eval_adapt/adapt_opt/updates/avg',
      'epstats/log/eval_adapt/adapt_opt/loss/avg',
      'epstats/log/eval_adapt/adapt_opt/grad_norm/avg',
      'epstats/log/eval_adapt/adapt_opt/update_rms/avg',
  )
  for key in diag_keys:
    values = [float(row[key]) for row in metrics_rows if key in row]
    diag[f'diag/{key}'] = safe_mean(values)

  result = {
      'config': config_name,
      'seed': seed,
      'status': status,
      'returncode': returncode,
      'wall_seconds': wall_seconds,
      'episodes': len(scores),
      'score_mean': safe_mean(scores),
      'score_std': safe_std(scores),
      'score_median': safe_quantile(scores, 0.5),
      'score_q10': safe_quantile(scores, 0.1),
      'score_q90': safe_quantile(scores, 0.9),
      'score_min': min(scores) if scores else float('nan'),
      'score_max': max(scores) if scores else float('nan'),
      'success_rate': (
          sum(1 for x in scores if x > success_threshold) / len(scores)
          if scores else float('nan')),
      'length_mean': safe_mean(lengths),
      'length_median': safe_quantile(lengths, 0.5),
      'fps_policy_mean': safe_mean(fps),
      'fps_policy_last': fps[-1] if fps else float('nan'),
  }
  result.update(diag)
  return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  keys = sorted({key for row in rows for key in row.keys()})
  with path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def summarize_by_config(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  grouped: dict[str, list[dict[str, Any]]] = {}
  for row in rows:
    grouped.setdefault(row['config'], []).append(row)

  summary = []
  for config, items in sorted(grouped.items()):
    ok = [x for x in items if x['status'] == 'ok' and x['episodes'] > 0]
    metrics = {}
    metric_keys = (
        'score_mean',
        'success_rate',
        'length_mean',
        'fps_policy_mean',
        'wall_seconds',
    )
    for key in metric_keys:
      vals = [float(x[key]) for x in ok]
      metrics[f'{key}_mean'] = safe_mean(vals)
      metrics[f'{key}_std'] = safe_std(vals)
      metrics[f'{key}_ci95'] = ci95(vals)
    total_eps = sum(int(x['episodes']) for x in ok)
    weighted_score_mean = (
        sum(float(x['score_mean']) * int(x['episodes']) for x in ok) / total_eps
        if total_eps else float('nan')
    )
    summary.append({
        'config': config,
        'runs_total': len(items),
        'runs_ok': len(ok),
        'episodes_total': total_eps,
        'seeds': ','.join(str(x['seed']) for x in sorted(ok, key=lambda y: y['seed'])),
        'score_mean_weighted_by_episodes': weighted_score_mean,
        **metrics,
    })
  return summary


def compare_vs_baseline(
    rows: list[dict[str, Any]],
    baseline_name: str,
) -> list[dict[str, Any]]:
  ok = [x for x in rows if x['status'] == 'ok' and x['episodes'] > 0]
  base_rows = [x for x in ok if x['config'] == baseline_name]
  if not base_rows:
    return []
  base_by_seed = {int(x['seed']): x for x in base_rows}
  by_config: dict[str, list[dict[str, Any]]] = {}
  for row in ok:
    if row['config'] == baseline_name:
      continue
    seed = int(row['seed'])
    if seed not in base_by_seed:
      continue
    base = base_by_seed[seed]
    by_config.setdefault(row['config'], []).append({
        'seed': seed,
        'delta_score_mean': float(row['score_mean']) - float(base['score_mean']),
        'delta_success_rate': float(row['success_rate']) - float(base['success_rate']),
        'delta_fps_policy_mean': (
            float(row['fps_policy_mean']) - float(base['fps_policy_mean'])),
        'delta_length_mean': float(row['length_mean']) - float(base['length_mean']),
    })

  out = []
  for config, items in sorted(by_config.items()):
    out.append({
        'config': config,
        'paired_seeds': len(items),
        'seeds': ','.join(str(x['seed']) for x in sorted(items, key=lambda y: y['seed'])),
        'delta_score_mean_mean': safe_mean([x['delta_score_mean'] for x in items]),
        'delta_score_mean_ci95': ci95([x['delta_score_mean'] for x in items]),
        'delta_success_rate_mean': safe_mean([x['delta_success_rate'] for x in items]),
        'delta_success_rate_ci95': ci95([x['delta_success_rate'] for x in items]),
        'delta_fps_policy_mean_mean': safe_mean([x['delta_fps_policy_mean'] for x in items]),
        'delta_fps_policy_mean_ci95': ci95([x['delta_fps_policy_mean'] for x in items]),
        'delta_length_mean_mean': safe_mean([x['delta_length_mean'] for x in items]),
        'delta_length_mean_ci95': ci95([x['delta_length_mean'] for x in items]),
    })
  return out


def build_command(
    args: argparse.Namespace,
    run_logdir: Path,
    seed: int,
    eval_adapt: dict[str, Any],
) -> list[str]:
  cmd = [
      args.python_bin,
      'dreamerv3/main.py',
      '--script', 'eval_only',
      '--logdir', str(run_logdir),
      '--configs', args.configs,
      '--run.from_checkpoint', args.checkpoint,
      '--run.from_checkpoint_regex', args.from_checkpoint_regex,
      '--run.steps', str(args.run_steps),
      '--run.envs', str(args.run_envs),
      '--run.log_every', str(args.run_log_every),
      '--logger.filter', args.logger_filter,
      '--jax.platform', args.jax_platform,
      '--seed', str(seed),
      '--eval_adapt.enabled', str(bool(eval_adapt['enabled'])),
      '--eval_adapt.steps', str(int(eval_adapt['steps'])),
      '--eval_adapt.imag_length', str(int(eval_adapt['imag_length'])),
      '--eval_adapt.lr', str(float(eval_adapt['lr'])),
      '--eval_adapt.every_k', str(int(eval_adapt['every_k'])),
      '--eval_adapt.actent', str(float(eval_adapt['actent'])),
      '--eval_adapt.start_batch', str(int(eval_adapt['start_batch'])),
  ]
  if args.task:
    cmd.extend(['--task', args.task])
  return cmd


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--repo_root', default='.')
  parser.add_argument('--outdir', default='')
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--from_checkpoint_regex', default='')
  parser.add_argument('--python_bin', default=sys.executable)
  parser.add_argument('--configs', default='procgen,size1m')
  parser.add_argument('--task', default='')
  parser.add_argument('--seeds', default='0,1,2,3,4')
  parser.add_argument('--run_steps', type=int, default=50000)
  parser.add_argument('--run_envs', type=int, default=1)
  parser.add_argument('--run_log_every', type=int, default=10)
  parser.add_argument(
      '--logger_filter',
      default='score|length|fps|eval_adapt|epstats',
  )
  parser.add_argument('--jax_platform', default='cpu')
  parser.add_argument('--success_threshold', type=float, default=0.0)
  parser.add_argument('--include_baseline', dest='include_baseline', action='store_true')
  parser.add_argument('--no_baseline', dest='include_baseline', action='store_false')
  parser.set_defaults(include_baseline=True)
  parser.add_argument('--baseline_name', default='baseline')
  parser.add_argument('--suite_json', default='')
  parser.add_argument('--adapt_steps_grid', default='1,3,5')
  parser.add_argument('--adapt_imag_length_grid', default='10')
  parser.add_argument('--adapt_lr_grid', default='1e-4')
  parser.add_argument('--adapt_every_k_grid', default='0,25')
  parser.add_argument('--adapt_actent_grid', default='3e-4')
  parser.add_argument('--adapt_start_batch_grid', default='1')
  parser.add_argument('--default_adapt_steps', type=int, default=3)
  parser.add_argument('--default_adapt_imag_length', type=int, default=10)
  parser.add_argument('--default_adapt_lr', type=float, default=1e-4)
  parser.add_argument('--default_adapt_actent', type=float, default=3e-4)
  parser.add_argument('--default_adapt_start_batch', type=int, default=1)
  parser.add_argument('--dry_run', action='store_true')
  parser.add_argument('--skip_existing', action='store_true')
  parser.add_argument('--continue_on_error', action='store_true')
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
  outdir = Path(args.outdir) if args.outdir else Path('eval_suite') / timestamp
  outdir.mkdir(parents=True, exist_ok=True)
  runs_dir = outdir / 'runs'
  runs_dir.mkdir(exist_ok=True)

  experiments = build_experiments(args)
  seeds = parse_int_list(args.seeds)
  if not seeds:
    raise ValueError('No seeds parsed from --seeds')

  suite_plan = {
      'timestamp': timestamp,
      'args': vars(args),
      'experiments': experiments,
      'seeds': seeds,
  }
  (outdir / 'suite_plan.json').write_text(json.dumps(suite_plan, indent=2))

  rows: list[dict[str, Any]] = []
  total = len(experiments) * len(seeds)
  done = 0
  for exp in experiments:
    exp_name = exp['name']
    for seed in seeds:
      done += 1
      run_logdir = runs_dir / exp_name / f'seed{seed}'
      run_logdir.mkdir(parents=True, exist_ok=True)
      cmd = build_command(args, run_logdir, seed, exp['eval_adapt'])
      cmd_str = ' '.join(shlex.quote(x) for x in cmd)
      (run_logdir / 'command.sh').write_text(cmd_str + '\n')
      print(f'[{done}/{total}] {exp_name} seed={seed}')

      start = time.time()
      returncode = 0
      status = 'ok'
      skipped_existing = 0
      if args.skip_existing and (
          (run_logdir / 'scores.jsonl').exists() and
          (run_logdir / 'metrics.jsonl').exists()):
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
            row = summarize_run(
                run_logdir,
                exp_name,
                seed,
                status,
                returncode,
                time.time() - start,
                args.success_threshold,
            )
            rows.append(row)
            write_csv(outdir / 'runs.csv', rows)
            print('Stopped early due to failure. Use --continue_on_error to keep going.')
            return 1

      row = summarize_run(
          run_logdir,
          exp_name,
          seed,
          status,
          returncode,
          time.time() - start,
          args.success_threshold,
      )
      row['skipped_existing'] = skipped_existing
      rows.append(row)
      write_csv(outdir / 'runs.csv', rows)

  summary = summarize_by_config(rows)
  delta = compare_vs_baseline(rows, args.baseline_name)
  write_csv(outdir / 'summary_by_config.csv', summary)
  write_csv(outdir / 'comparison_vs_baseline.csv', delta)
  print(f'Wrote: {outdir / "runs.csv"}')
  print(f'Wrote: {outdir / "summary_by_config.csv"}')
  if delta:
    print(f'Wrote: {outdir / "comparison_vs_baseline.csv"}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
