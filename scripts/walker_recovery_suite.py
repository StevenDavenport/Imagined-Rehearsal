#!/usr/bin/env python3
"""Run Walker recovery sweeps across corrupted checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
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


def read_csv(path: Path) -> list[dict[str, str]]:
  if not path.exists():
    return []
  with path.open() as f:
    return list(csv.DictReader(f))


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


def build_command(args: argparse.Namespace, outdir: Path, checkpoint: Path) -> list[str]:
  cmd = [
      args.python_bin,
      'scripts/eval_suite.py',
      '--repo_root', args.repo_root,
      '--outdir', str(outdir),
      '--checkpoint', str(checkpoint),
      '--configs', args.configs,
      '--task', args.task,
      '--seeds', args.seeds,
      '--run_steps', str(int(args.run_steps)),
      '--run_envs', str(int(args.run_envs)),
      '--run_log_every', str(int(args.run_log_every)),
      '--logger_filter', args.logger_filter,
      '--jax_platform', args.jax_platform,
      '--success_threshold', str(float(args.success_threshold)),
  ]
  if args.suite_json:
    cmd.extend(['--suite_json', args.suite_json])
  else:
    cmd.extend([
        '--include_baseline',
        '--baseline_name', args.baseline_name,
        '--adapt_steps_grid', str(int(args.adapt_steps)),
        '--adapt_imag_length_grid', str(int(args.adapt_imag_length)),
        '--adapt_lr_grid', str(float(args.adapt_lr)),
        '--adapt_every_k_grid', str(int(args.adapt_every_k)),
        '--adapt_actent_grid', str(float(args.adapt_actent)),
        '--adapt_start_batch_grid', args.adapt_start_batch_grid,
        '--default_adapt_steps', str(int(args.adapt_steps)),
        '--default_adapt_imag_length', str(int(args.adapt_imag_length)),
        '--default_adapt_lr', str(float(args.adapt_lr)),
        '--default_adapt_actent', str(float(args.adapt_actent)),
        '--default_adapt_start_batch', str(int(
            parse_text_list(args.adapt_start_batch_grid)[0])),
    ])
  if args.skip_existing:
    cmd.append('--skip_existing')
  if args.continue_on_error:
    cmd.append('--continue_on_error')
  if args.dry_run:
    cmd.append('--dry_run')
  return cmd


def annotate_rows(
    rows: list[dict[str, str]],
    checkpoint: Path,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
  out_rows = []
  for row in rows:
    item = dict(row)
    item['checkpoint'] = str(checkpoint)
    item['checkpoint_name'] = checkpoint.name
    item['corruption_method'] = manifest.get('method', '')
    item['corruption_target'] = manifest.get('target', 'actor')
    item['corruption_scope'] = manifest.get('scope', '')
    item['corruption_strength'] = manifest.get('strength', '')
    item['corruption_relative_l2'] = manifest.get('relative_l2', '')
    item['actor_relative_l2'] = manifest.get('actor_relative_l2', '')
    item['critic_relative_l2'] = manifest.get('critic_relative_l2', '')
    out_rows.append(item)
  return out_rows


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--repo_root', default='.')
  parser.add_argument('--outdir', default='')
  parser.add_argument('--python_bin', default=sys.executable)
  parser.add_argument('--checkpoints', required=True)
  parser.add_argument('--configs', default='dmc_vision')
  parser.add_argument('--task', default='dmc_walker_walk')
  parser.add_argument('--seeds', default='0,1,2,3,4')
  parser.add_argument('--run_steps', type=int, default=10000)
  parser.add_argument('--run_envs', type=int, default=1)
  parser.add_argument('--run_log_every', type=int, default=50)
  parser.add_argument('--logger_filter', default='score|length|fps|eval_adapt|epstats')
  parser.add_argument('--jax_platform', default='cuda')
  parser.add_argument('--success_threshold', type=float, default=0.0)
  parser.add_argument('--baseline_name', default='baseline')
  parser.add_argument('--suite_json', default='')
  parser.add_argument('--adapt_steps', type=int, default=1)
  parser.add_argument('--adapt_imag_length', type=int, default=5)
  parser.add_argument('--adapt_lr', type=float, default=1e-4)
  parser.add_argument('--adapt_every_k', type=int, default=1)
  parser.add_argument('--adapt_actent', type=float, default=0.0)
  parser.add_argument('--adapt_start_batch_grid', default='1,512')
  parser.add_argument('--dry_run', action='store_true')
  parser.add_argument('--skip_existing', action='store_true')
  parser.add_argument('--continue_on_error', action='store_true')
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
  outdir = Path(args.outdir) if args.outdir else Path('eval_suite') / f'walker_recovery_{timestamp}'
  outdir.mkdir(parents=True, exist_ok=True)

  checkpoints = expand_checkpoint_paths(parse_text_list(args.checkpoints))
  if not checkpoints:
    raise ValueError('No checkpoints parsed from --checkpoints')

  plan = {
      'timestamp': timestamp,
      'args': vars(args),
      'checkpoints': [str(x) for x in checkpoints],
  }
  (outdir / 'suite_plan.json').write_text(json.dumps(plan, indent=2))

  suite_rows = []
  all_runs = []
  all_summary = []
  all_delta = []

  total = len(checkpoints)
  for index, checkpoint in enumerate(checkpoints, 1):
    ckpt_name = checkpoint.name
    ckpt_outdir = outdir / sanitize(ckpt_name)
    cmd = build_command(args, ckpt_outdir, checkpoint)
    (ckpt_outdir / 'command.sh').parent.mkdir(parents=True, exist_ok=True)
    (ckpt_outdir / 'command.sh').write_text(' '.join(shlex.quote(x) for x in cmd) + '\n')

    print(f'[{index}/{total}] {ckpt_name}')
    start = time.time()
    status = 'ok'
    returncode = 0
    if args.dry_run:
      status = 'dry_run'
    else:
      proc = subprocess.run(cmd, cwd=args.repo_root, check=False)
      returncode = int(proc.returncode)
      if returncode != 0:
        status = 'failed'

    manifest = load_manifest(checkpoint)
    suite_rows.append({
        'checkpoint': str(checkpoint),
        'checkpoint_name': ckpt_name,
        'status': status,
        'returncode': returncode,
        'wall_seconds': time.time() - start,
        'corruption_method': manifest.get('method', ''),
        'corruption_target': manifest.get('target', 'actor'),
        'corruption_scope': manifest.get('scope', ''),
        'corruption_strength': manifest.get('strength', ''),
        'corruption_relative_l2': manifest.get('relative_l2', ''),
        'actor_relative_l2': manifest.get('actor_relative_l2', ''),
        'critic_relative_l2': manifest.get('critic_relative_l2', ''),
    })
    write_csv(outdir / 'suite_runs.csv', suite_rows)

    if status == 'failed' and not args.continue_on_error:
      print(f'Stopped early due to failure in {ckpt_name}.')
      return 1
    if status != 'ok':
      continue

    all_runs.extend(annotate_rows(read_csv(ckpt_outdir / 'runs.csv'), checkpoint, manifest))
    all_summary.extend(annotate_rows(
        read_csv(ckpt_outdir / 'summary_by_config.csv'),
        checkpoint,
        manifest,
    ))
    all_delta.extend(annotate_rows(
        read_csv(ckpt_outdir / 'comparison_vs_baseline.csv'),
        checkpoint,
        manifest,
    ))
    write_csv(outdir / 'all_runs.csv', all_runs)
    write_csv(outdir / 'all_summary_by_config.csv', all_summary)
    write_csv(outdir / 'all_comparison_vs_baseline.csv', all_delta)

  print(f'Wrote: {outdir / "suite_runs.csv"}')
  if all_runs:
    print(f'Wrote: {outdir / "all_runs.csv"}')
  if all_summary:
    print(f'Wrote: {outdir / "all_summary_by_config.csv"}')
  if all_delta:
    print(f'Wrote: {outdir / "all_comparison_vs_baseline.csv"}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
