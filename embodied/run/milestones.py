"""Immutable, self-contained training milestone archives."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import pickle
import shutil
import time
import uuid


class MilestoneError(RuntimeError):
  pass


def archive_name(step: int) -> str:
  return f'step_{int(step):012d}'


def archive_path(root: str | pathlib.Path, step: int) -> pathlib.Path:
  return pathlib.Path(root).expanduser().resolve() / archive_name(step)


def checkpoint_step(checkpoint: str | pathlib.Path) -> int:
  path = pathlib.Path(checkpoint)
  with (path / 'step.pkl').open('rb') as handle:
    return int(pickle.load(handle))


def file_digest(path: pathlib.Path) -> dict:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(block)
  return {
      'path': path.name,
      'bytes': path.stat().st_size,
      'sha256': digest.hexdigest(),
  }


def write_json(path: pathlib.Path, value) -> None:
  path = pathlib.Path(str(path))
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(
      json.dumps(value, indent=2, sort_keys=True) + '\n')
  temporary.replace(path)


def create(
    checkpoint,
    replay,
    root: str | pathlib.Path,
    step: int,
    *,
    metadata: dict | None = None,
) -> pathlib.Path:
  """Create one exact checkpoint+replay archive and publish it atomically."""
  step = int(step)
  root = pathlib.Path(root).expanduser().resolve()
  root.mkdir(parents=True, exist_ok=True)
  final = archive_path(root, step)
  if final.exists():
    validate(final, expected_step=step, full=False)
    return final

  temporary = root / (
      f'.{archive_name(step)}.tmp-{os.getpid()}-{uuid.uuid4().hex}')
  if temporary.exists():
    shutil.rmtree(temporary)
  temporary.mkdir(parents=True)
  try:
    checkpoint_dir = temporary / 'checkpoint'
    replay_dir = temporary / 'replay'
    print(f'Creating immutable milestone: {final}', flush=True)
    checkpoint.save(path=str(checkpoint_dir))
    saved_step = checkpoint_step(checkpoint_dir)
    if saved_step != step:
      raise MilestoneError(
          f'Milestone checkpoint step mismatch: {saved_step} != {step}')
    replay_files = replay.export(replay_dir, link=True)
    if not replay_files:
      raise MilestoneError('Milestone replay export is empty')
    checkpoint_files = [
        file_digest(path) for path in sorted(checkpoint_dir.iterdir())
        if path.is_file()
    ]
    manifest = {
        'format': 1,
        'created_unix': time.time(),
        'step': step,
        'checkpoint': 'checkpoint',
        'replay': 'replay',
        'checkpoint_files': checkpoint_files,
        'replay_files': replay_files,
        'metadata': dict(metadata or {}),
    }
    write_json(temporary / 'manifest.json', manifest)
    (temporary / 'archive_complete').write_bytes(b'')
    temporary.replace(final)
    validate(final, expected_step=step, full=False)
    print(f'Published immutable milestone: {final}', flush=True)
    return final
  except BaseException:
    if temporary.exists():
      shutil.rmtree(temporary)
    raise


def create_checkpoint(
    checkpoint,
    root: str | pathlib.Path,
    step: int,
    *,
    metadata: dict | None = None,
) -> pathlib.Path:
  """Create an immutable agent/optimizer checkpoint without replay data."""
  step = int(step)
  root = pathlib.Path(root).expanduser().resolve()
  root.mkdir(parents=True, exist_ok=True)
  final = archive_path(root, step)
  if final.exists():
    validate_checkpoint(final, expected_step=step, full=False)
    return final

  temporary = root / (
      f'.{archive_name(step)}.tmp-{os.getpid()}-{uuid.uuid4().hex}')
  if temporary.exists():
    shutil.rmtree(temporary)
  temporary.mkdir(parents=True)
  try:
    checkpoint_dir = temporary / 'checkpoint'
    print(f'Creating immutable checkpoint snapshot: {final}', flush=True)
    checkpoint.save(path=str(checkpoint_dir))
    saved_step = checkpoint_step(checkpoint_dir)
    if saved_step != step:
      raise MilestoneError(
          f'Snapshot checkpoint step mismatch: {saved_step} != {step}')
    files = [
        file_digest(path) for path in sorted(checkpoint_dir.iterdir())
        if path.is_file()]
    manifest = {
        'format': 1,
        'kind': 'checkpoint_snapshot',
        'created_unix': time.time(),
        'step': step,
        'checkpoint': 'checkpoint',
        'checkpoint_files': files,
        'metadata': dict(metadata or {}),
    }
    write_json(temporary / 'manifest.json', manifest)
    (temporary / 'archive_complete').write_bytes(b'')
    temporary.replace(final)
    validate_checkpoint(final, expected_step=step, full=False)
    print(f'Published immutable checkpoint snapshot: {final}', flush=True)
    return final
  except BaseException:
    if temporary.exists():
      shutil.rmtree(temporary)
    raise


def validate_checkpoint(
    path: str | pathlib.Path,
    *,
    expected_step: int | None = None,
    full: bool = False,
) -> dict:
  """Validate an agent/optimizer-only checkpoint snapshot."""
  path = pathlib.Path(path).expanduser().resolve()
  marker = path / 'archive_complete'
  manifest_path = path / 'manifest.json'
  checkpoint_dir = path / 'checkpoint'
  if not marker.is_file() or not manifest_path.is_file():
    raise MilestoneError(f'Incomplete checkpoint snapshot: {path}')
  manifest = json.loads(manifest_path.read_text())
  if manifest.get('kind') != 'checkpoint_snapshot':
    raise MilestoneError(f'Not a checkpoint snapshot: {path}')
  step = checkpoint_step(checkpoint_dir)
  if int(manifest.get('step', -1)) != step:
    raise MilestoneError(
        f'Snapshot manifest/checkpoint mismatch at {path}: '
        f'{manifest.get("step")} != {step}')
  if expected_step is not None and step != int(expected_step):
    raise MilestoneError(
        f'Unexpected snapshot step at {path}: {step} != {expected_step}')
  records = manifest.get('checkpoint_files', ())
  if not records:
    raise MilestoneError(f'Snapshot checkpoint inventory is empty: {path}')
  for record in records:
    filename = checkpoint_dir / record['path']
    if not filename.is_file():
      raise MilestoneError(f'Snapshot file is missing: {filename}')
    if filename.stat().st_size != int(record['bytes']):
      raise MilestoneError(f'Snapshot file size changed: {filename}')
    if full and file_digest(filename)['sha256'] != record['sha256']:
      raise MilestoneError(f'Snapshot digest changed: {filename}')
  return manifest


def validate(
    path: str | pathlib.Path,
    *,
    expected_step: int | None = None,
    full: bool = False,
) -> dict:
  path = pathlib.Path(path).expanduser().resolve()
  marker = path / 'archive_complete'
  manifest_path = path / 'manifest.json'
  checkpoint_dir = path / 'checkpoint'
  replay_dir = path / 'replay'
  if not marker.is_file() or not manifest_path.is_file():
    raise MilestoneError(f'Incomplete milestone archive: {path}')
  manifest = json.loads(manifest_path.read_text())
  step = checkpoint_step(checkpoint_dir)
  if int(manifest.get('step', -1)) != step:
    raise MilestoneError(
        f'Milestone manifest/checkpoint mismatch at {path}: '
        f'{manifest.get("step")} != {step}')
  if expected_step is not None and step != int(expected_step):
    raise MilestoneError(
        f'Unexpected milestone step at {path}: {step} != {expected_step}')

  for folder, key in (
      (checkpoint_dir, 'checkpoint_files'),
      (replay_dir, 'replay_files'),
  ):
    records = manifest.get(key, ())
    if not records:
      raise MilestoneError(f'Milestone {key} inventory is empty: {path}')
    for record in records:
      filename = folder / record['path' if key == 'checkpoint_files' else 'name']
      if not filename.is_file():
        raise MilestoneError(f'Milestone file is missing: {filename}')
      if filename.stat().st_size != int(record['bytes']):
        raise MilestoneError(f'Milestone file size changed: {filename}')
      if full:
        current = file_digest(filename)['sha256']
        if current != record['sha256']:
          raise MilestoneError(f'Milestone digest changed: {filename}')
  return manifest


def seed_replay(
    archive: str | pathlib.Path,
    destination: str | pathlib.Path,
) -> dict:
  """Materialize an immutable milestone replay into a fresh live directory.

  Existing matching files are accepted so initialization is restart-safe. Any
  unexpected replay chunk or size mismatch is rejected rather than mixed into
  the resumed run. Hard links are preferred and preserve source immutability;
  copying is the cross-filesystem fallback.
  """
  archive = pathlib.Path(archive).expanduser().resolve()
  destination = pathlib.Path(destination).expanduser().resolve()
  manifest = validate(archive, full=False)
  records = tuple(manifest['replay_files'])
  expected = {record['name']: record for record in records}
  destination.mkdir(parents=True, exist_ok=True)
  unexpected = sorted(
      path.name for path in destination.glob('*.npz')
      if path.name not in expected)
  if unexpected:
    raise MilestoneError(
        f'Unexpected replay chunks in resume destination {destination}: '
        f'{unexpected[:5]}')
  source_root = archive / 'replay'
  for record in records:
    source = source_root / record['name']
    target = destination / record['name']
    expected_bytes = int(record['bytes'])
    if target.is_file():
      if target.stat().st_size != expected_bytes:
        raise MilestoneError(f'Resume replay file size changed: {target}')
      continue
    temporary = destination / f'.{target.name}.resume-{os.getpid()}'
    if temporary.exists():
      temporary.unlink()
    try:
      try:
        os.link(source, temporary)
      except OSError:
        shutil.copy2(source, temporary)
      if temporary.stat().st_size != expected_bytes:
        raise MilestoneError(
            f'Resume replay copy size mismatch: {temporary}')
      temporary.replace(target)
    finally:
      if temporary.exists():
        temporary.unlink()
  return manifest
