"""Shared, dependency-light helpers for MiniGrid audition experiments."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import pathlib
import statistics
import subprocess
from typing import Any

from embodied.envs.minigrid_registry import TASK_NAMES
from scripts.checkpoint_utils import resolve_checkpoint_path


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
  if not path.is_file():
    return []
  rows = []
  with path.open() as handle:
    for line in handle:
      if line.strip():
        rows.append(json.loads(line))
  return rows


def write_json(path: pathlib.Path, value: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
  temporary.replace(path)


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
  if not rows:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  keys = sorted({key for row in rows for key in row})
  temporary = path.with_suffix(path.suffix + '.tmp')
  with temporary.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def spec_digest(spec: dict[str, Any]) -> str:
  payload = json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()
  return hashlib.sha256(payload).hexdigest()


def git_revision(repo_root: pathlib.Path) -> str:
  result = subprocess.run(
      ['git', 'rev-parse', 'HEAD'], cwd=repo_root,
      capture_output=True, text=True, check=False)
  return result.stdout.strip() if result.returncode == 0 else 'unknown'


def package_versions(names=('minigrid', 'gymnasium', 'jax')) -> dict[str, str]:
  versions = {}
  for name in names:
    try:
      versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      versions[name] = 'missing'
  return versions


def checkpoint_step(logdir: pathlib.Path) -> int:
  try:
    checkpoint = resolve_checkpoint_path(logdir)
  except FileNotFoundError:
    return 0
  import pickle
  with (checkpoint / 'step.pkl').open('rb') as handle:
    return int(pickle.load(handle))


def evaluation_summary(
    logdir: pathlib.Path,
    *,
    task: str,
    training_seed: int,
    eval_seed: int,
    checkpoint: int,
    policy_mode: str,
    requested_episodes: int,
    paired_rng_seed: int | None = None,
    paired_rng_stride: int | None = None,
    returncode: int | None = None,
) -> dict[str, Any]:
  rows = read_jsonl(logdir / 'episodes.jsonl')
  returns = [float(row.get('episode/score', 0.0)) for row in rows]
  lengths = [
      max(0.0, float(row.get('episode/length', 0.0)) - 1)
      for row in rows]
  integrity_path = logdir / 'eval_integrity.json'
  integrity = (
      json.loads(integrity_path.read_text())
      if integrity_path.is_file() else {})
  resets = reset_seeds(logdir)
  integrity_verified = bool(
      integrity.get('equal', False) and
      integrity.get('policy_mode') == policy_mode and
      integrity.get('paired_rng') is True and
      (paired_rng_seed is None or
       integrity.get('paired_rng_seed') == int(paired_rng_seed)) and
      (paired_rng_stride is None or
       integrity.get('paired_rng_stride') == int(paired_rng_stride)))
  summary = {
      'task': task,
      'training_seed': int(training_seed),
      'eval_seed': int(eval_seed),
      'checkpoint': int(checkpoint),
      'policy_mode': policy_mode,
      'returncode': returncode,
      'requested_episodes': int(requested_episodes),
      'episodes': len(rows),
      'episode_count_exact': len(rows) == int(requested_episodes),
      'reset_count': len(resets),
      'reset_count_exact': len(resets) == int(requested_episodes),
      'mean_return': statistics.fmean(returns) if returns else 0.0,
      'median_return': statistics.median(returns) if returns else 0.0,
      'return_std': statistics.pstdev(returns) if len(returns) > 1 else 0.0,
      'successes': sum(value > 0 for value in returns),
      'success_rate': (
          sum(value > 0 for value in returns) / len(returns)
          if returns else 0.0),
      'mean_action_length': statistics.fmean(lengths) if lengths else 0.0,
      'median_action_length': statistics.median(lengths) if lengths else 0.0,
      'integrity_equal': integrity.get('equal', False),
      'integrity_verified': integrity_verified,
      'complete': bool(
          returncode in (None, 0) and len(rows) == int(requested_episodes) and
          len(resets) == int(requested_episodes) and integrity_verified),
      'logdir': str(logdir),
  }
  return summary


def reset_seeds(logdir: pathlib.Path) -> list[int]:
  return [
      int(row['reset_seed'])
      for row in read_jsonl(logdir / 'audit_env0.jsonl')
      if row.get('kind') == 'reset' and 'reset_seed' in row
  ]


def validate_tasks(tasks) -> tuple[str, ...]:
  tasks = tuple(str(task) for task in tasks)
  unknown = [task for task in tasks if task not in TASK_NAMES]
  if not tasks or unknown or len(set(tasks)) != len(tasks):
    raise ValueError(
        f'Tasks must be a nonempty unique subset of {TASK_NAMES}; got '
        f'{tasks}, unknown={unknown}')
  return tasks
