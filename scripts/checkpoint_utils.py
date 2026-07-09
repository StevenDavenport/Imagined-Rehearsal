#!/usr/bin/env python3
"""Helpers for resolving Dreamer checkpoint paths."""

from __future__ import annotations

from pathlib import Path


def _resolve_latest(root: Path) -> Path | None:
  latest = root / 'latest'
  if not latest.exists():
    return None
  target = root / latest.read_text().strip()
  if (target / 'done').exists():
    return target.resolve()
  return None


def resolve_checkpoint_path(pathlike: str | Path) -> Path:
  path = Path(pathlike).expanduser().resolve()

  if (path / 'done').exists():
    return path

  resolved = _resolve_latest(path)
  if resolved is not None:
    return resolved

  ckpt_root = path / 'ckpt'
  resolved = _resolve_latest(ckpt_root)
  if resolved is not None:
    return resolved

  raise FileNotFoundError(
      'Could not resolve a checkpoint path. Expected either an exact '
      f'checkpoint folder with done, a checkpoint root with latest, or '
      f'a training logdir containing ckpt/latest. Got: {path}')


def expand_checkpoint_paths(items: list[str | Path]) -> list[Path]:
  expanded: list[Path] = []
  for item in items:
    path = Path(item).expanduser().resolve()
    try:
      expanded.append(resolve_checkpoint_path(path))
      continue
    except FileNotFoundError:
      pass

    if path.is_dir():
      children = sorted(x for x in path.iterdir() if x.is_dir() and (x / 'done').exists())
      if children:
        expanded.extend(x.resolve() for x in children)
        continue

    raise FileNotFoundError(
        f'Could not resolve checkpoint input or expand checkpoint directory: {path}')
  return expanded
