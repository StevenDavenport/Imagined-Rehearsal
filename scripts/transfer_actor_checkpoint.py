#!/usr/bin/env python3
"""Create a target checkpoint with actor parameters transplanted from a source checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from checkpoint_utils import resolve_checkpoint_path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--source_actor', type=Path, required=True)
  parser.add_argument('--target_checkpoint', type=Path, required=True)
  parser.add_argument('--outdir', type=Path, required=True)
  parser.add_argument('--scope', default='all', choices=('all', 'head'))
  parser.add_argument('--force', action='store_true')
  return parser.parse_args()


def load_agent_checkpoint(source: Path) -> dict[str, Any]:
  agent_path = source / 'agent.pkl'
  if not agent_path.exists():
    raise FileNotFoundError(f'Missing agent checkpoint: {agent_path}')
  with agent_path.open('rb') as f:
    data = pickle.load(f)
  if not isinstance(data, dict) or 'params' not in data or 'counters' not in data:
    raise ValueError(f'Unexpected checkpoint payload in {agent_path}')
  return data


def clone_agent_checkpoint(data: dict[str, Any]) -> dict[str, Any]:
  params = {}
  for key, value in data['params'].items():
    params[key] = value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
  counters = copy.deepcopy(data['counters'])
  return {'params': params, 'counters': counters}


def actor_keys(params: dict[str, Any], scope: str) -> list[str]:
  prefix = 'pol/' if scope == 'all' else 'pol/head/action/'
  keys = sorted(k for k in params if k.startswith(prefix))
  if not keys:
    raise ValueError(f'No actor keys found for scope={scope!r}')
  return keys


def rms(value: np.ndarray) -> float:
  arr = np.asarray(value, np.float64)
  if not arr.size:
    return 0.0
  return float(np.sqrt(np.mean(np.square(arr))))


def relative_l2(before: list[np.ndarray], after: list[np.ndarray]) -> float:
  num = 0.0
  den = 0.0
  for lhs, rhs in zip(before, after):
    lhs64 = np.asarray(lhs, np.float64)
    rhs64 = np.asarray(rhs, np.float64)
    num += float(np.square(rhs64 - lhs64).sum())
    den += float(np.square(lhs64).sum())
  return float(np.sqrt(num / max(den, 1e-12)))


def clear_outdir(outdir: Path) -> None:
  if not outdir.exists():
    return
  for path in sorted(outdir.glob('**/*'), reverse=True):
    if path.is_file() or path.is_symlink():
      path.unlink()
  for path in sorted(outdir.glob('**/*'), reverse=True):
    if path.is_dir():
      path.rmdir()
  outdir.rmdir()


def main() -> int:
  args = parse_args()
  source = resolve_checkpoint_path(args.source_actor)
  target = resolve_checkpoint_path(args.target_checkpoint)
  outdir = args.outdir.expanduser().resolve()

  if outdir.exists() and not args.force:
    raise FileExistsError(f'Output exists, use --force to overwrite: {outdir}')
  if outdir.exists() and args.force:
    clear_outdir(outdir)

  source_data = load_agent_checkpoint(source)
  target_data = clone_agent_checkpoint(load_agent_checkpoint(target))

  source_keys = actor_keys(source_data['params'], args.scope)
  target_keys = actor_keys(target_data['params'], args.scope)
  if source_keys != target_keys:
    missing = sorted(set(source_keys) ^ set(target_keys))
    raise ValueError(f'Actor key mismatch between source and target: {missing[:20]}')

  before = []
  after = []
  for key in source_keys:
    src = source_data['params'][key]
    dst = target_data['params'][key]
    if getattr(src, 'shape', None) != getattr(dst, 'shape', None):
      raise ValueError(
          f'Shape mismatch for {key}: source={getattr(src, "shape", None)} '
          f'target={getattr(dst, "shape", None)}')
    before.append(dst.copy())
    target_data['params'][key] = src.copy() if isinstance(src, np.ndarray) else copy.deepcopy(src)
    after.append(target_data['params'][key])

  outdir.mkdir(parents=True, exist_ok=True)
  with (outdir / 'agent.pkl').open('wb') as f:
    pickle.dump(target_data, f, protocol=pickle.HIGHEST_PROTOCOL)
  (outdir / 'done').write_text('')

  manifest = {
      'source_actor_checkpoint': str(source),
      'target_checkpoint': str(target),
      'transfer_type': 'actor_only',
      'scope': args.scope,
      'actor_key_count': len(source_keys),
      'actor_param_count': int(sum(int(np.prod(target_data['params'][key].shape)) for key in source_keys)),
      'relative_l2_vs_target_actor': relative_l2(before, after),
      'source_actor_rms_mean': float(np.mean([rms(source_data['params'][key]) for key in source_keys])),
      'target_actor_rms_before_mean': float(np.mean([rms(x) for x in before])),
      'target_actor_rms_after_mean': float(np.mean([rms(x) for x in after])),
      'keys': source_keys,
  }
  (outdir / 'manifest.json').write_text(json.dumps(manifest, indent=2))

  print(f'Created transplanted checkpoint: {outdir}')
  print(f'  source actor: {source}')
  print(f'  target checkpoint: {target}')
  print(f'  actor keys: {len(source_keys)}')
  print(f'  relative_l2_vs_target_actor: {manifest["relative_l2_vs_target_actor"]:.4f}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
