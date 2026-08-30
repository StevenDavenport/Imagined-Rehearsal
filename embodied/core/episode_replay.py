"""Complete-episode FIFO and reservoir replay with exact persistence."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import threading
import time

import numpy as np

from . import limiters


class EpisodeReplay:
  """Goal-stratified episode reservoir for continual learning.

  Episodes are admitted independently within each goal using Algorithm R.
  Sampling is either uniform over goals represented in memory, or uses a
  fixed current-versus-old allocation. Within a selected goal, episodes and
  valid fragment starts are sampled uniformly. Short episodes are never
  discarded: independently sampled fragments are concatenated with explicit
  ``is_first`` boundaries until the requested sequence length is reached.
  """

  FORMAT = 1

  def __init__(
      self, length, capacity, groups, directory, group_key='goal_id',
      retention='reservoir', online=False, save_wait=False, name='unnamed',
      seed=0, sampling='uniform_groups', current_group=0,
      current_fraction=0.5, **unused):
    del save_wait, unused
    self.length = int(length)
    self.capacity = int(capacity)
    self.groups = int(groups)
    self.group_key = str(group_key)
    self.retention = str(retention)
    self.sampling = str(sampling)
    self.current_group = int(current_group)
    self.current_fraction = float(current_fraction)
    self.name = str(name)
    if self.length < 1 or self.capacity < 1 or self.groups < 1:
      raise ValueError((self.length, self.capacity, self.groups))
    if self.capacity % self.groups:
      raise ValueError(
          f'episode capacity {self.capacity} must divide across '
          f'{self.groups} goals')
    if self.retention not in ('fifo', 'reservoir'):
      raise ValueError(f'Unknown episode retention: {self.retention!r}')
    if self.sampling not in ('uniform_groups', 'current_old'):
      raise ValueError(f'Unknown episode sampling: {self.sampling!r}')
    if not 0 <= self.current_group < self.groups:
      raise ValueError(
          f'current_group={self.current_group} outside [0, {self.groups})')
    if not 0 <= self.current_fraction <= 1:
      raise ValueError(
          f'current_fraction must be in [0, 1], got {self.current_fraction}')
    if online:
      raise ValueError('EpisodeReplay does not support online sampling')
    if not directory:
      raise ValueError('EpisodeReplay requires a persistent directory')

    self.group_capacity = self.capacity // self.groups
    self.directory = pathlib.Path(str(directory)).expanduser().resolve()
    self.directory.mkdir(parents=True, exist_ok=True)
    admission_seed, sampling_seed = np.random.SeedSequence(seed).spawn(2)
    self.admission_rng = np.random.default_rng(admission_seed)
    self.sampling_rng = np.random.default_rng(sampling_seed)
    self.lock = threading.RLock()
    self.slots = {group: [] for group in range(self.groups)}
    self.episodes_seen = np.zeros(self.groups, np.int64)
    self.partials = {}
    self.next_episode_id = 0
    self.transitions_seen = 0
    self.metrics = dict(
        inserts=0, samples=0, admitted=0, replaced=0, rejected=0,
        abandoned=0)
    self.sampled_goals = np.zeros(self.groups, np.int64)
    self.sampled_goals_total = np.zeros(self.groups, np.int64)

  def __len__(self):
    with self.lock:
      return sum(record['length'] for records in self.slots.values()
                 for record in records)

  def _scalar(self, value, name):
    value = np.asarray(value)
    if value.size != 1:
      raise ValueError(f'{name} must be scalar, got {value.shape}')
    return value.reshape(()).item()

  def add(self, step, worker=0):
    with self.lock:
      self._add(step, worker)

  def _add(self, step, worker):
    step = {key: np.asarray(value) for key, value in step.items()
            if not key.startswith('log/')}
    is_first = bool(self._scalar(step['is_first'], 'is_first'))
    is_last = bool(self._scalar(step['is_last'], 'is_last'))
    group = int(self._scalar(step[self.group_key], self.group_key))
    if not 0 <= group < self.groups:
      raise ValueError(
          f'{self.group_key}={group} outside [0, {self.groups})')

    partial = self.partials.get(worker)
    if is_first:
      if partial:
        # A process restart cannot resume the external RLScape episode even
        # though the exact replay partial was checkpointed. Never admit that
        # incomplete trajectory; start cleanly from the new reset instead.
        self.metrics['abandoned'] += 1
      partial = {
          'episode_id': self.next_episode_id,
          'goal_id': group,
          'steps': [],
      }
      self.next_episode_id += 1
      self.partials[worker] = partial
    elif not partial:
      raise ValueError(f'Worker {worker} transition arrived before is_first')
    if partial['goal_id'] != group:
      raise ValueError(
          f'Goal changed within episode: {partial["goal_id"]} -> {group}')

    index = len(partial['steps'])
    identity = int(partial['episode_id']).to_bytes(16, 'big')
    step['stepid'] = np.frombuffer(
        identity + index.to_bytes(4, 'big'), np.uint8).copy()
    partial['steps'].append(step)
    self.transitions_seen += 1
    self.metrics['inserts'] += 1
    if is_last:
      self._close(worker)

  def _stack(self, steps):
    keys = tuple(steps[0])
    if any(tuple(step) != keys for step in steps):
      raise ValueError('Episode transition keys changed within an episode')
    return {key: np.stack([step[key] for step in steps]) for key in keys}

  def _close(self, worker):
    partial = self.partials.pop(worker)
    group = partial['goal_id']
    data = self._stack(partial['steps'])
    goals = np.asarray(data[self.group_key]).reshape(-1)
    if not np.all(goals == group):
      raise ValueError(f'Non-constant {self.group_key} in completed episode')
    record = {
        'episode_id': int(partial['episode_id']),
        'goal_id': int(group),
        'length': len(partial['steps']),
        'filename': f'episode_{int(partial["episode_id"]):016d}.npz',
        'data': data,
        'persisted': False,
    }
    self.episodes_seen[group] += 1
    seen = int(self.episodes_seen[group])
    slots = self.slots[group]
    if len(slots) < self.group_capacity:
      slots.append(record)
      self.metrics['admitted'] += 1
    elif self.retention == 'fifo':
      slots.pop(0)
      slots.append(record)
      self.metrics['admitted'] += 1
      self.metrics['replaced'] += 1
    else:
      candidate = int(self.admission_rng.integers(0, seen))
      if candidate < self.group_capacity:
        slots[candidate] = record
        self.metrics['admitted'] += 1
        self.metrics['replaced'] += 1
      else:
        self.metrics['rejected'] += 1

  def sample(self, batch, mode='train'):
    if mode not in ('train', 'report', 'eval'):
      raise ValueError(mode)
    batch = int(batch)
    if batch < 1:
      raise ValueError(f'batch must be positive, got {batch}')
    available = self._available_groups
    limiters.wait(
        lambda: bool(available()), f'Replay buffer {self.name} is empty')
    with self.lock:
      groups = self._sample_groups(available(), batch)
      sequences = [self._sample_sequence(int(group)) for group in groups]
      if mode == 'train':
        self.metrics['samples'] += batch
        counts = np.bincount(groups, minlength=self.groups)
        self.sampled_goals += counts
        self.sampled_goals_total += counts
      return {
          key: np.stack([sequence[key] for sequence in sequences])
          for key in sequences[0]}

  def _sample_groups(self, available, batch):
    available = np.asarray(available, np.int64)
    if self.sampling == 'uniform_groups':
      return self.sampling_rng.choice(available, size=batch, replace=True)
    old = available[available != self.current_group]
    if self.current_group not in available or not len(old):
      return self.sampling_rng.choice(available, size=batch, replace=True)
    current = self.sampling_rng.random(batch) < self.current_fraction
    groups = self.sampling_rng.choice(old, size=batch, replace=True)
    groups[current] = self.current_group
    return groups

  def _available_groups(self):
    with self.lock:
      return [group for group, records in self.slots.items() if records]

  def _sample_sequence(self, group):
    pieces = []
    remaining = self.length
    while remaining:
      records = self.slots[group]
      record = records[int(self.sampling_rng.integers(0, len(records)))]
      start = int(self.sampling_rng.integers(0, record['length']))
      amount = min(remaining, record['length'] - start)
      piece = {key: value[start:start + amount]
               for key, value in record['data'].items()}
      piece['is_first'] = np.array(piece['is_first'], copy=True)
      piece['is_first'][0] = True
      pieces.append(piece)
      remaining -= amount
    sequence = {
        key: np.concatenate([piece[key] for piece in pieces], 0)
        for key in pieces[0]}
    if 'is_last' in sequence:
      sequence['is_last'] = np.array(sequence['is_last'], copy=True)
      next_first = np.roll(sequence['is_first'], -1)
      next_first[-1] = False
      sequence['is_last'] |= next_first
    return sequence

  def update(self, data):
    # RLScape uses replay_context=0 and therefore emits no replay updates.
    # Reject future silent misuse rather than pretending mutable sequence
    # fields can be mapped back into fragmented episodes.
    remaining = set(data) - {'stepid'}
    if remaining:
      raise NotImplementedError(
          f'EpisodeReplay updates are unsupported: {sorted(remaining)}')

  def stats(self):
    with self.lock:
      return self._stats()

  def _stats(self):
    transitions = [
        sum(record['length'] for record in self.slots[group])
        for group in range(self.groups)]
    episodes = [len(self.slots[group]) for group in range(self.groups)]
    ram = sum(
        value.nbytes
        for records in self.slots.values() for record in records
        for value in record['data'].values())
    inserts = self.metrics['inserts']
    samples = self.metrics['samples']
    result = {
        'items': sum(transitions),
        'episodes': sum(episodes),
        'chunks': 0,
        'streams': len(self.partials),
        'ram_gb': ram / 1024 ** 3,
        'inserts': inserts,
        'samples': samples,
        'updates': 0,
        'replay_ratio': (
            self.length * samples / inserts if inserts else np.nan),
        'episodes_seen': int(self.episodes_seen.sum()),
        'admitted': self.metrics['admitted'],
        'replaced': self.metrics['replaced'],
        'rejected': self.metrics['rejected'],
        'abandoned': self.metrics['abandoned'],
        'active_goals': sum(bool(count) for count in episodes),
        'sampling/current_group': self.current_group,
        'sampling/current_fraction': self.current_fraction,
        'sampling/is_current_old': int(self.sampling == 'current_old'),
    }
    for group in range(self.groups):
      result[f'goal_{group}/episodes'] = episodes[group]
      result[f'goal_{group}/transitions'] = transitions[group]
      result[f'goal_{group}/sampled'] = int(self.sampled_goals[group])
      result[f'goal_{group}/sampled_total'] = int(
          self.sampled_goals_total[group])
    for key in self.metrics:
      self.metrics[key] = 0
    self.sampled_goals.fill(0)
    return result

  def _write_episode(self, record):
    path = self.directory / record['filename']
    if record.get('persisted', False):
      return
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
      np.savez_compressed(handle, **record['data'])
    temporary.replace(path)
    record['persisted'] = True

  def _write_partial(self, worker, partial):
    # The name is stable when save() and milestone export() run back-to-back,
    # yet changes whenever the partial grows. Thus the checkpoint manifest and
    # exported replay refer to the same exact prefix without overwriting older
    # milestone hard links.
    filename = (
        f'partial_{int(partial["episode_id"]):016d}_'
        f'{len(partial["steps"]):08d}_{int(worker):020d}.npz')
    path = self.directory / filename
    data = self._stack(partial['steps'])
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
      np.savez_compressed(handle, **data)
    temporary.replace(path)
    return {
        'worker': int(worker),
        'episode_id': int(partial['episode_id']),
        'goal_id': int(partial['goal_id']),
        'filename': filename,
        'length': len(partial['steps']),
    }

  def save(self, wait=None):
    del wait
    with self.lock:
      return self._save()

  def _save(self):
    generation = f'{time.time_ns()}_{os.getpid()}'
    for records in self.slots.values():
      for record in records:
        self._write_episode(record)
    partials = [
        self._write_partial(worker, partial)
        for worker, partial in self.partials.items()]
    state = {
        'format': self.FORMAT,
        'kind': 'episode_replay',
        'retention': self.retention,
        'length': self.length,
        'capacity': self.capacity,
        'groups': self.groups,
        'group_key': self.group_key,
        # The sampling policy is phase-local training configuration rather
        # than replay contents. Record it for provenance, but deliberately do
        # not require it to match on load: sequential phases must load the
        # exact same reservoir while changing which goal is current.
        'sampling': self.sampling,
        'current_group': self.current_group,
        'current_fraction': self.current_fraction,
        'episodes_seen': self.episodes_seen.tolist(),
        'transitions_seen': int(self.transitions_seen),
        'next_episode_id': int(self.next_episode_id),
        'admission_rng_state': self.admission_rng.bit_generator.state,
        'sampling_rng_state': self.sampling_rng.bit_generator.state,
        'metrics': {key: int(value) for key, value in self.metrics.items()},
        'sampled_goals': self.sampled_goals.tolist(),
        'sampled_goals_total': self.sampled_goals_total.tolist(),
        'slots': {
            str(group): [{key: record[key] for key in (
                'episode_id', 'goal_id', 'length', 'filename')}
                         for record in records]
            for group, records in self.slots.items()},
        'partials': partials,
    }
    temporary = self.directory / f'.manifest_{generation}.tmp'
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
    temporary.replace(self.directory / 'manifest.json')
    return state

  def _load_npz(self, filename):
    path = self.directory / filename
    if not path.is_file():
      raise FileNotFoundError(path)
    with np.load(path) as archive:
      return {key: np.array(archive[key], copy=True) for key in archive.files}

  def load(self, data=None, directory=None, amount=None):
    if directory is not None or amount is not None:
      raise ValueError('EpisodeReplay loads only its exact saved manifest')
    with self.lock:
      return self._load(data)

  def _load(self, data):
    if data is None:
      manifest = self.directory / 'manifest.json'
      if not manifest.is_file():
        return
      data = json.loads(manifest.read_text())
    expected = (
        data.get('format'), data.get('kind'), data.get('retention'),
        data.get('length'), data.get('capacity'), data.get('groups'),
        data.get('group_key'))
    actual = (
        self.FORMAT, 'episode_replay', self.retention, self.length,
        self.capacity, self.groups, self.group_key)
    if expected != actual:
      raise ValueError(f'EpisodeReplay manifest mismatch: {expected} != {actual}')

    self.slots = {group: [] for group in range(self.groups)}
    for group in range(self.groups):
      for metadata in data['slots'][str(group)]:
        record = dict(metadata)
        record['data'] = self._load_npz(record['filename'])
        record['persisted'] = True
        self.slots[group].append(record)
    self.partials = {}
    for metadata in data.get('partials', ()):
      stacked = self._load_npz(metadata['filename'])
      steps = [
          {key: value[index] for key, value in stacked.items()}
          for index in range(int(metadata['length']))]
      self.partials[int(metadata['worker'])] = {
          'episode_id': int(metadata['episode_id']),
          'goal_id': int(metadata['goal_id']),
          'steps': steps,
      }
    self.episodes_seen = np.asarray(data['episodes_seen'], np.int64)
    self.transitions_seen = int(data['transitions_seen'])
    self.next_episode_id = int(data['next_episode_id'])
    self.admission_rng.bit_generator.state = data['admission_rng_state']
    self.sampling_rng.bit_generator.state = data['sampling_rng_state']
    self.metrics = {
        key: int(value) for key, value in data.get('metrics', {}).items()}
    for key in (
        'inserts', 'samples', 'admitted', 'replaced', 'rejected', 'abandoned'):
      self.metrics.setdefault(key, 0)
    self.sampled_goals = np.asarray(
        data.get('sampled_goals', [0] * self.groups), np.int64)
    self.sampled_goals_total = np.asarray(
        data.get('sampled_goals_total', [0] * self.groups), np.int64)

  def export(self, directory, link=True):
    with self.lock:
      destination = pathlib.Path(directory)
      if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f'Replay export destination is not empty: {destination}')
      destination.mkdir(parents=True, exist_ok=True)
      state = self._save()
      names = {
          record['filename']
          for records in state['slots'].values() for record in records}
      names.update(record['filename'] for record in state['partials'])
      manifest = destination / 'manifest.json'
      manifest.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
      records = [self._export_record(manifest, manifest, linked=False)]
      for name in sorted(names):
        source = self.directory / name
        target = destination / name
        linked = False
        if link:
          try:
            os.link(source, target)
            linked = True
          except OSError:
            pass
        if not linked:
          shutil.copy2(source, target)
        records.append(self._export_record(source, target, linked=linked))
      return records

  def _export_record(self, source, target, linked):
    del source
    digest = hashlib.sha256()
    with target.open('rb') as handle:
      for block in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(block)
    return {
        'name': target.name,
        'bytes': target.stat().st_size,
        'sha256': digest.hexdigest(),
        'linked': bool(linked),
    }
