#!/usr/bin/env python3
"""Create actor-only corrupted checkpoint copies for any Dreamer checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from checkpoint_utils import resolve_checkpoint_path


def fmt_strength(value: float) -> str:
  text = f'{value:.3f}'.rstrip('0').rstrip('.')
  return text.replace('.', 'p')


PRESETS = {
    'gaussian_ladder': [
        {
            'name': f'gauss_all_s{str(sigma).replace(".", "p")}_seed0',
            'method': 'gaussian',
            'scope': 'all',
            'strength': sigma,
        }
        for sigma in (0.02, 0.05, 0.10, 0.20)
    ],
    'zero_mask_ladder': [
        {
            'name': f'zero_mask_all_s{str(level).replace(".", "p")}_seed0',
            'method': 'zero_mask',
            'scope': 'all',
            'strength': level,
        }
        for level in (0.20, 0.50, 0.80)
    ],
    'zero_mask_calibration_grid': [
        {
            'name': f'zero_mask_all_s{fmt_strength(level)}_seed0',
            'method': 'zero_mask',
            'scope': 'all',
            'strength': level,
        }
        for level in (
            0.05, 0.10, 0.15, 0.20, 0.25,
            0.30, 0.35, 0.40, 0.45, 0.50,
            0.55, 0.60, 0.65, 0.70, 0.75,
            0.80, 0.85, 0.90, 0.95,
        )
    ],
    'zero_mask_severity_triplet': [
        {
            'name': f'zero_mask_{label}_all_s{fmt_strength(level)}_seed0',
            'method': 'zero_mask',
            'scope': 'all',
            'strength': level,
            'severity': label,
        }
        for label, level in (
            ('mild', 0.20),
            ('moderate', 0.50),
            ('severe', 0.80),
        )
    ],
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--source', type=Path, required=True)
  parser.add_argument('--out_root', type=Path, required=True)
  parser.add_argument('--preset', default='gaussian_ladder', choices=sorted(PRESETS))
  parser.add_argument(
      '--method',
      default='',
      choices=('', 'fresh_random', 'gaussian', 'mix_random', 'resample', 'zero_mask'))
  parser.add_argument('--scope', default='all', choices=('all', 'head'))
  parser.add_argument('--strength', type=float, default=0.0)
  parser.add_argument('--name', default='')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--force', action='store_true')
  parser.add_argument('--list_presets', action='store_true')
  return parser.parse_args()


def print_presets() -> None:
  for name, specs in PRESETS.items():
    print(name)
    for spec in specs:
      print(f"  - {spec['name']}: {spec['method']} {spec['scope']} {spec['strength']}")


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
  if scope == 'all':
    prefix = 'pol/'
  elif scope == 'head':
    prefix = 'pol/head/action/'
  else:
    raise NotImplementedError(scope)
  keys = sorted(k for k in params if k.startswith(prefix))
  if not keys:
    raise ValueError(f'No actor keys found for scope={scope!r}')
  return keys


def rms(value: np.ndarray) -> float:
  arr = np.asarray(value, np.float64)
  if not arr.size:
    return 0.0
  return float(np.sqrt(np.mean(np.square(arr))))


def gaussian(arr: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
  scale = max(rms(arr), 1e-8)
  noise = rng.standard_normal(arr.shape).astype(np.float32)
  out = arr.astype(np.float32) + float(strength) * scale * noise
  return out.astype(arr.dtype, copy=False)


def fresh_random(arr: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
  del strength
  scale = max(rms(arr), 1e-8)
  out = (scale * rng.standard_normal(arr.shape)).astype(np.float32)
  return out.astype(arr.dtype, copy=False)


def mix_random(arr: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
  scale = max(rms(arr), 1e-8)
  rand = (scale * rng.standard_normal(arr.shape)).astype(np.float32)
  alpha = float(strength)
  out = (1.0 - alpha) * arr.astype(np.float32) + alpha * rand
  return out.astype(arr.dtype, copy=False)


def resample(arr: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
  scale = max(rms(arr), 1e-8)
  rand = (scale * rng.standard_normal(arr.shape)).astype(np.float32)
  alpha = float(strength)
  out = (1.0 - alpha) * arr.astype(np.float32) + alpha * rand
  return out.astype(arr.dtype, copy=False)


def zero_mask(arr: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
  keep = rng.random(arr.shape) >= float(strength)
  out = arr.astype(np.float32) * keep.astype(np.float32)
  return out.astype(arr.dtype, copy=False)


METHODS = {
    'fresh_random': fresh_random,
    'gaussian': gaussian,
    'mix_random': mix_random,
    'resample': resample,
    'zero_mask': zero_mask,
}


def build_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
  if args.method:
    if not args.name:
      level = 's' if args.method != 'mix_random' else 'a'
      name = f'{args.method}_{args.scope}_{level}{fmt_strength(args.strength)}_seed{args.seed}'
    else:
      name = args.name
    return [{
        'name': name,
        'method': args.method,
        'scope': args.scope,
        'strength': float(args.strength),
    }]

  specs = []
  for idx, spec in enumerate(PRESETS[args.preset]):
    item = dict(spec)
    if item['name'].endswith('_seed0'):
      item['name'] = item['name'][:-6] + f'_seed{args.seed + idx}'
    specs.append(item)
  return specs


def relative_l2(before: list[np.ndarray], after: list[np.ndarray]) -> float:
  num = 0.0
  den = 0.0
  for lhs, rhs in zip(before, after):
    lhs64 = np.asarray(lhs, np.float64)
    rhs64 = np.asarray(rhs, np.float64)
    num += float(np.square(rhs64 - lhs64).sum())
    den += float(np.square(lhs64).sum())
  return float(np.sqrt(num / max(den, 1e-12)))


def save_variant(
    source: Path,
    outdir: Path,
    data: dict[str, Any],
    spec: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
  params = data['params']
  keys = actor_keys(params, spec['scope'])
  before = [params[key].copy() for key in keys]

  rng = np.random.default_rng(seed)
  fn = METHODS[spec['method']]
  for key in keys:
    params[key] = fn(params[key], spec['strength'], rng)
  after = [params[key] for key in keys]

  outdir.mkdir(parents=True, exist_ok=True)
  with (outdir / 'agent.pkl').open('wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
  (outdir / 'done').write_text('')

  manifest = {
      'source_checkpoint': str(source),
      'corruption_name': spec['name'],
      'method': spec['method'],
      'scope': spec['scope'],
      'strength': float(spec['strength']),
      'severity': spec.get('severity', ''),
      'seed': int(seed),
      'actor_key_count': len(keys),
      'actor_param_count': int(sum(int(np.prod(params[key].shape)) for key in keys)),
      'relative_l2': relative_l2(before, after),
      'actor_rms_before_mean': float(np.mean([rms(x) for x in before])),
      'actor_rms_after_mean': float(np.mean([rms(x) for x in after])),
      'keys': keys,
  }
  (outdir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
  return manifest


def main() -> int:
  args = parse_args()
  if args.list_presets:
    print_presets()
    return 0

  source = resolve_checkpoint_path(args.source)
  out_root = args.out_root
  specs = build_specs(args)
  base = load_agent_checkpoint(source)

  out_root.mkdir(parents=True, exist_ok=True)
  written = []
  for idx, spec in enumerate(specs):
    outdir = out_root / spec['name']
    if outdir.exists() and not args.force:
      raise FileExistsError(f'Output exists, use --force to overwrite: {outdir}')
    if outdir.exists() and args.force:
      for path in sorted(outdir.glob('**/*'), reverse=True):
        if path.is_file():
          path.unlink()
      for path in sorted(outdir.glob('**/*'), reverse=True):
        if path.is_dir():
          path.rmdir()
      outdir.rmdir()

    data = clone_agent_checkpoint(base)
    manifest = save_variant(source, outdir, data, spec, args.seed + idx)
    written.append((outdir, manifest))

  print('Created corrupted checkpoints:')
  for outdir, manifest in written:
    print(
        f"  {outdir} | {manifest['method']} {manifest['scope']} "
        f"strength={manifest['strength']} rel_l2={manifest['relative_l2']:.4f}")
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
