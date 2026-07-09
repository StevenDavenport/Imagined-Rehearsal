#!/usr/bin/env python3
"""Run a generic clean-baseline and corrupted recovery suite for rehearsal."""

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

from checkpoint_utils import expand_checkpoint_paths, resolve_checkpoint_path


def parse_text_list(text: str) -> list[str]:
  return [x.strip() for x in text.split(',') if x.strip()]


def resolve_labeled_checkpoints(items: list[str]) -> list[tuple[str, Path]]:
  labeled = []
  for item in items:
    raw = Path(item).expanduser()
    label = raw.name
    labeled.append((label, resolve_checkpoint_path(raw)))
  return labeled


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


def annotate_rows(
    rows: list[dict[str, str]],
    suite_label: str,
    checkpoint: Path,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
  annotated = []
  for row in rows:
    item = dict(row)
    item['suite_label'] = suite_label
    item['checkpoint'] = str(checkpoint)
    item['checkpoint_name'] = checkpoint.name
    item['corruption_method'] = manifest.get('method', '')
    item['corruption_scope'] = manifest.get('scope', '')
    item['corruption_strength'] = manifest.get('strength', '')
    item['corruption_severity'] = manifest.get('severity', '')
    item['corruption_relative_l2'] = manifest.get('relative_l2', '')
    annotated.append(item)
  return annotated


def build_clean_command(
    args: argparse.Namespace,
    outdir: Path,
) -> list[str]:
  return [
      args.python_bin,
      'scripts/eval_suite.py',
      '--repo_root', args.repo_root,
      '--outdir', str(outdir),
      '--checkpoint', str(resolve_checkpoint_path(args.source_checkpoint)),
      '--configs', args.configs,
      '--task', args.task,
      '--seeds', args.seeds,
      '--run_steps', str(int(args.run_steps)),
      '--run_envs', str(int(args.run_envs)),
      '--run_log_every', str(int(args.run_log_every)),
      '--logger_filter', args.logger_filter,
      '--jax_platform', args.jax_platform,
      '--success_threshold', str(float(args.success_threshold)),
      '--include_baseline',
      '--baseline_name', args.clean_baseline_name,
      '--suite_json',
      str(_write_clean_suite_json(args, outdir)),
  ]


def _write_clean_suite_json(args: argparse.Namespace, outdir: Path) -> Path:
  outdir.mkdir(parents=True, exist_ok=True)
  path = outdir / 'clean_suite.json'
  path.write_text(json.dumps([
      {
          'name': args.clean_baseline_name,
          'eval_adapt': {
              'enabled': False,
              'steps': 1,
              'imag_length': 1,
              'lr': 1e-4,
              'every_k': 0,
              'actent': 0.0,
              'start_batch': 1,
          },
      },
      {
          'name': args.clean_rehearsal_name_sb1,
          'eval_adapt': {
              'enabled': True,
              'steps': int(args.adapt_steps),
              'imag_length': int(args.adapt_imag_length),
              'lr': float(args.adapt_lr),
              'every_k': int(args.adapt_every_k),
              'actent': float(args.adapt_actent),
              'start_batch': 1,
          },
      },
      {
          'name': args.clean_rehearsal_name_sb512,
          'eval_adapt': {
              'enabled': True,
              'steps': int(args.adapt_steps),
              'imag_length': int(args.adapt_imag_length),
              'lr': float(args.adapt_lr),
              'every_k': int(args.adapt_every_k),
              'actent': float(args.adapt_actent),
              'start_batch': 512,
          },
      },
  ], indent=2))
  return path


def build_corrupt_command(
    args: argparse.Namespace,
    outdir: Path,
    checkpoint: Path,
) -> list[str]:
  outdir.mkdir(parents=True, exist_ok=True)
  suite_path = outdir / 'suite.json'
  suite_path.write_text(json.dumps([
      {
          'name': args.corrupt_baseline_name,
          'eval_adapt': {
              'enabled': False,
              'steps': int(args.adapt_steps),
              'imag_length': int(args.adapt_imag_length),
              'lr': float(args.adapt_lr),
              'every_k': 0,
              'actent': float(args.adapt_actent),
              'start_batch': 1,
          },
      },
      {
          'name': args.rehearsal_name_sb1,
          'eval_adapt': {
              'enabled': True,
              'steps': int(args.adapt_steps),
              'imag_length': int(args.adapt_imag_length),
              'lr': float(args.adapt_lr),
              'every_k': int(args.adapt_every_k),
              'actent': float(args.adapt_actent),
              'start_batch': 1,
          },
      },
      {
          'name': args.rehearsal_name_sb512,
          'eval_adapt': {
              'enabled': True,
              'steps': int(args.adapt_steps),
              'imag_length': int(args.adapt_imag_length),
              'lr': float(args.adapt_lr),
              'every_k': int(args.adapt_every_k),
              'actent': float(args.adapt_actent),
              'start_batch': 512,
          },
      },
  ], indent=2))
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
      '--suite_json', str(suite_path),
  ]
  if args.skip_existing:
    cmd.append('--skip_existing')
  if args.continue_on_error:
    cmd.append('--continue_on_error')
  if args.dry_run:
    cmd.append('--dry_run')
  return cmd


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--repo_root', default='.')
  parser.add_argument('--outdir', default='')
  parser.add_argument('--python_bin', default=sys.executable)
  parser.add_argument('--source_checkpoint', required=True)
  parser.add_argument('--corrupted_checkpoints', required=True)
  parser.add_argument('--configs', required=True)
  parser.add_argument('--task', required=True)
  parser.add_argument('--seeds', default='0,1,2,3,4')
  parser.add_argument('--run_steps', type=int, default=10000)
  parser.add_argument('--run_envs', type=int, default=1)
  parser.add_argument('--run_log_every', type=int, default=50)
  parser.add_argument('--logger_filter', default='score|length|fps|eval_adapt|epstats')
  parser.add_argument('--jax_platform', default='cuda')
  parser.add_argument('--success_threshold', type=float, default=0.0)
  parser.add_argument('--clean_baseline_name', default='clean_baseline')
  parser.add_argument('--clean_rehearsal_name_sb1', default='clean_rehearsal_sb1')
  parser.add_argument('--clean_rehearsal_name_sb512', default='clean_rehearsal_sb512')
  parser.add_argument('--corrupt_baseline_name', default='corrupt_baseline')
  parser.add_argument('--rehearsal_name_sb1', default='rehearsal_sb1')
  parser.add_argument('--rehearsal_name_sb512', default='rehearsal_sb512')
  parser.add_argument('--adapt_steps', type=int, default=1)
  parser.add_argument('--adapt_imag_length', type=int, default=6)
  parser.add_argument('--adapt_lr', type=float, default=1e-4)
  parser.add_argument('--adapt_every_k', type=int, default=1)
  parser.add_argument('--adapt_actent', type=float, default=3e-4)
  parser.add_argument('--dry_run', action='store_true')
  parser.add_argument('--skip_existing', action='store_true')
  parser.add_argument('--continue_on_error', action='store_true')
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
  outdir = Path(args.outdir) if args.outdir else Path('eval_suite') / f'rehearsal_benchmark_{timestamp}'
  outdir.mkdir(parents=True, exist_ok=True)

  source_checkpoint = resolve_checkpoint_path(args.source_checkpoint)
  corrupted_inputs = parse_text_list(args.corrupted_checkpoints)
  corrupted = resolve_labeled_checkpoints(corrupted_inputs)
  if not corrupted:
    raise ValueError('No corrupted checkpoints parsed from --corrupted_checkpoints')

  plan = {
      'timestamp': timestamp,
      'args': vars(args),
      'source_checkpoint': str(source_checkpoint),
      'corrupted_checkpoints': [
          {'label': label, 'checkpoint': str(path)} for label, path in corrupted],
  }
  (outdir / 'suite_plan.json').write_text(json.dumps(plan, indent=2))

  suite_rows = []
  all_runs = []
  all_summary = []
  all_delta = []

  clean_outdir = outdir / 'clean'
  clean_cmd = build_clean_command(args, clean_outdir)
  (clean_outdir / 'command.sh').parent.mkdir(parents=True, exist_ok=True)
  (clean_outdir / 'command.sh').write_text(' '.join(shlex.quote(x) for x in clean_cmd) + '\n')
  start = time.time()
  status = 'ok'
  returncode = 0
  if args.dry_run:
    status = 'dry_run'
  else:
    proc = subprocess.run(clean_cmd, cwd=args.repo_root, check=False)
    returncode = int(proc.returncode)
    if returncode != 0:
      status = 'failed'
  suite_rows.append({
      'suite_label': 'clean',
      'checkpoint': str(source_checkpoint),
      'checkpoint_name': source_checkpoint.name,
      'status': status,
      'returncode': returncode,
      'wall_seconds': time.time() - start,
      'corruption_method': '',
      'corruption_scope': '',
      'corruption_strength': '',
      'corruption_severity': '',
      'corruption_relative_l2': '',
  })
  write_csv(outdir / 'suite_runs.csv', suite_rows)
  if status == 'failed' and not args.continue_on_error:
    return 1
  if status == 'ok':
    clean_manifest = {}
    all_runs.extend(annotate_rows(read_csv(clean_outdir / 'runs.csv'), 'clean', source_checkpoint, clean_manifest))
    all_summary.extend(annotate_rows(read_csv(clean_outdir / 'summary_by_config.csv'), 'clean', source_checkpoint, clean_manifest))
    all_delta.extend(annotate_rows(read_csv(clean_outdir / 'comparison_vs_baseline.csv'), 'clean', source_checkpoint, clean_manifest))
    write_csv(outdir / 'all_runs.csv', all_runs)
    write_csv(outdir / 'all_summary_by_config.csv', all_summary)
    write_csv(outdir / 'all_comparison_vs_baseline.csv', all_delta)

  total = len(corrupted)
  for index, (checkpoint_label, checkpoint) in enumerate(corrupted, 1):
    manifest = load_manifest(checkpoint)
    ckpt_outdir = outdir / sanitize(checkpoint.name)
    cmd = build_corrupt_command(args, ckpt_outdir, checkpoint)
    (ckpt_outdir / 'command.sh').parent.mkdir(parents=True, exist_ok=True)
    (ckpt_outdir / 'command.sh').write_text(' '.join(shlex.quote(x) for x in cmd) + '\n')
    print(f'[{index}/{total}] {checkpoint.name}')
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
    suite_rows.append({
        'suite_label': (manifest.get('severity') or checkpoint_label or checkpoint.name),
        'checkpoint': str(checkpoint),
        'checkpoint_name': checkpoint.name,
        'status': status,
        'returncode': returncode,
        'wall_seconds': time.time() - start,
        'corruption_method': manifest.get('method', ''),
        'corruption_scope': manifest.get('scope', ''),
        'corruption_strength': manifest.get('strength', ''),
        'corruption_severity': manifest.get('severity', ''),
        'corruption_relative_l2': manifest.get('relative_l2', ''),
    })
    write_csv(outdir / 'suite_runs.csv', suite_rows)
    if status == 'failed' and not args.continue_on_error:
      return 1
    if status != 'ok':
      continue
    suite_label = manifest.get('severity') or checkpoint_label or checkpoint.name
    all_runs.extend(annotate_rows(read_csv(ckpt_outdir / 'runs.csv'), suite_label, checkpoint, manifest))
    all_summary.extend(annotate_rows(read_csv(ckpt_outdir / 'summary_by_config.csv'), suite_label, checkpoint, manifest))
    all_delta.extend(annotate_rows(read_csv(ckpt_outdir / 'comparison_vs_baseline.csv'), suite_label, checkpoint, manifest))
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
