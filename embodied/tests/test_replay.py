import collections
import pathlib
import threading
import time

import elements
import embodied
import numpy as np
import pytest


REPLAYS_UNLIMITED = [
    embodied.replay.Replay,
    # embodied.replay.Reverb,
]

REPLAYS_SAVECHUNKS = [
    embodied.replay.Replay,
]

REPLAYS_UNIFORM = [
    embodied.replay.Replay,
]


def unbatched(dataset):
  for batch in dataset:
    yield {k: v[0] for k, v in batch.items()}


@pytest.mark.filterwarnings('ignore:.*Pillow.*')
@pytest.mark.filterwarnings('ignore:.*the imp module.*')
@pytest.mark.filterwarnings('ignore:.*distutils.*')
class TestReplay:

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  def test_multiple_keys(self, Replay):
    replay = Replay(length=5, capacity=10)
    for step in range(30):
      replay.add({'image': np.zeros((64, 64, 3)), 'action': np.zeros(12)})
    seq = next(unbatched(replay.dataset(1)))
    assert set(seq.keys()) == {'stepid', 'image', 'action'}
    assert seq['stepid'].shape == (5, 20)
    assert seq['image'].shape == (5, 64, 64, 3)
    assert seq['action'].shape == (5, 12)

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  @pytest.mark.parametrize(
      'length,workers,capacity',
      [(1, 1, 1), (2, 1, 2), (5, 1, 10), (1, 2, 2), (5, 3, 15), (2, 7, 20)])
  def test_capacity_exact(self, Replay, length, workers, capacity):
    replay = Replay(length, capacity)
    for step in range(30):
      for worker in range(workers):
        replay.add({'step': step}, worker)
      target = min(workers * max(0, (step + 1) - length + 1), capacity)
      assert len(replay) == target

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  @pytest.mark.parametrize(
      'length,workers,capacity,chunksize',
      [(1, 1, 1, 128), (2, 1, 2, 128), (5, 1, 10, 128), (1, 2, 2, 128),
       (5, 3, 15, 128), (2, 7, 20, 128), (7, 2, 27, 4)])
  def test_sample_sequences(
      self, Replay, length, workers, capacity, chunksize):
    replay = Replay(length, capacity, chunksize=chunksize)
    for step in range(30):
      for worker in range(workers):
        replay.add({'step': step, 'worker': worker}, worker)
    dataset = unbatched(replay.dataset(1))
    for _ in range(10):
      seq = next(dataset)
      assert (seq['step'] - seq['step'][0] == np.arange(length)).all()
      assert (seq['worker'] == seq['worker'][0]).all()

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  @pytest.mark.parametrize(
      'length,capacity', [(1, 1), (2, 2), (5, 10), (1, 2), (5, 15), (2, 20)])
  def test_sample_single(self, Replay, length, capacity):
    replay = Replay(length, capacity)
    for step in range(length):
      replay.add({'step': step})
    dataset = unbatched(replay.dataset(1))
    for _ in range(10):
      seq = next(dataset)
      assert (seq['step'] == np.arange(length)).all()

  @pytest.mark.parametrize('Replay', REPLAYS_UNIFORM)
  def test_sample_uniform(self, Replay):
    replay = Replay(capacity=20, length=5, seed=0)
    for step in range(7):
      replay.add({'step': step})
    assert len(replay) == 3
    histogram = collections.defaultdict(int)
    dataset = unbatched(replay.dataset(1))
    for _ in range(100):
      seq = next(dataset)
      histogram[seq['step'][0]] += 1
    assert len(histogram) == 3, histogram
    histogram = tuple(histogram.values())
    assert histogram[0] > 20
    assert histogram[1] > 20
    assert histogram[2] > 20

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  def test_workers_simple(self, Replay):
    replay = Replay(length=2, capacity=20)
    replay.add({'step': 0}, worker=0)
    replay.add({'step': 1}, worker=1)
    replay.add({'step': 2}, worker=0)
    replay.add({'step': 3}, worker=1)
    dataset = unbatched(replay.dataset(1))
    for _ in range(10):
      seq = next(dataset)
      assert tuple(seq['step']) in ((0, 2), (1, 3))

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  def test_workers_random(self, Replay, length=4, capacity=30):
    rng = np.random.default_rng(seed=0)
    replay = Replay(length, capacity)
    streams = {i: iter(range(10)) for i in range(3)}
    for _ in range(40):
      worker = int(rng.integers(0, 3, ()))
      try:
        step = {'step': next(streams[worker]), 'stream': worker}
        replay.add(step, worker=worker)
      except StopIteration:
        pass
    histogram = collections.defaultdict(int)
    dataset = unbatched(replay.dataset(1))
    for _ in range(10):
      seq = next(dataset)
      assert (seq['step'] - seq['step'][0] == np.arange(length)).all()
      assert (seq['stream'] == seq['stream'][0]).all()
      histogram[int(seq['stream'][0])] += 1
    assert all(count > 0 for count in histogram.values())

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  @pytest.mark.parametrize(
      'length,workers,capacity',
      [(1, 1, 1), (2, 1, 2), (5, 1, 10), (1, 2, 2), (5, 3, 15), (2, 7, 20)])
  def test_worker_delay(self, Replay, length, workers, capacity):
    replay = Replay(length, capacity)
    rng = np.random.default_rng(seed=0)
    streams = [iter(range(10)) for _ in range(workers)]
    while streams:
      try:
        worker = rng.integers(0, len(streams))
        replay.add({'step': next(streams[worker])}, worker)
      except StopIteration:
        del streams[worker]

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  @pytest.mark.parametrize(
      'length,capacity,chunksize',
      [(1, 1, 128), (3, 10, 128), (5, 100, 128), (5, 25, 2)])
  def test_restore_exact(self, tmpdir, Replay, length, capacity, chunksize):
    elements.UUID.reset(debug=True)
    replay = Replay(
        length, capacity, directory=tmpdir, chunksize=chunksize,
        save_wait=True)
    for step in range(30):
      replay.add({'step': step})
    num_items = np.clip(30 - length + 1, 0, capacity)
    assert len(replay) == num_items
    data = replay.save()
    replay = Replay(length, capacity, directory=tmpdir)
    replay.load(data)
    assert len(replay) == num_items
    dataset = unbatched(replay.dataset(1))
    for _ in range(len(replay)):
      assert len(next(dataset)['step']) == length

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  @pytest.mark.parametrize(
      'length,capacity,chunksize',
      [(1, 1, 128), (3, 10, 128), (5, 100, 128), (5, 25, 2)])
  def test_restore_noclear(self, tmpdir, Replay, length, capacity, chunksize):
    elements.UUID.reset(debug=True)
    replay = Replay(
        length, capacity, directory=tmpdir, chunksize=chunksize,
        save_wait=True)
    for _ in range(30):
      replay.add({'foo': 13})
    num_items = np.clip(30 - length + 1, 0, capacity)
    assert len(replay) == num_items
    data = replay.save()
    for _ in range(30):
      replay.add({'foo': 42})
    replay.load(data)
    dataset = unbatched(replay.dataset(1))
    if capacity < num_items:
      for _ in range(len(replay)):
        assert next(dataset)['foo'] == 13

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  @pytest.mark.parametrize('workers', [1, 2, 5])
  @pytest.mark.parametrize('length,capacity', [(1, 1), (3, 10), (5, 100)])
  def test_restore_workers(self, tmpdir, Replay, workers, length, capacity):
    capacity *= workers
    replay = Replay(
        length, capacity, directory=tmpdir, save_wait=True)
    for step in range(50):
      for worker in range(workers):
        replay.add({'step': step}, worker)
    num_items = np.clip((50 - length + 1) * workers, 0, capacity)
    assert len(replay) == num_items
    data = replay.save()
    replay = Replay(length, capacity, directory=tmpdir)
    replay.load(data)
    assert len(replay) == num_items
    dataset = unbatched(replay.dataset(1))
    for _ in range(len(replay)):
      assert len(next(dataset)['step']) == length

  @pytest.mark.parametrize('Replay', REPLAYS_SAVECHUNKS)
  @pytest.mark.parametrize(
      'length,capacity,chunksize', [(1, 1, 1), (3, 10, 5), (5, 100, 12)])
  def test_restore_chunks_exact(
      self, tmpdir, Replay, length, capacity, chunksize):
    elements.UUID.reset(debug=True)
    assert len(list(elements.Path(tmpdir).glob('*.npz'))) == 0
    replay = Replay(
        length, capacity, directory=tmpdir, chunksize=chunksize,
        save_wait=True)
    for step in range(30):
      replay.add({'step': step})
    num_items = np.clip(30 - length + 1, 0, capacity)
    assert len(replay) == num_items
    data = replay.save()
    filenames = list(elements.Path(tmpdir).glob('*.npz'))
    lengths = [int(x.stem.split('-')[3]) for x in filenames]
    stored_steps = min(capacity + length - 1, 30)
    total_chunks = int(np.ceil(30 / chunksize))
    pruned_chunks = int(np.floor((30 - stored_steps) / chunksize))
    assert len(filenames) == total_chunks - pruned_chunks
    last_chunk_empty = total_chunks * chunksize - 30
    saved_steps = (total_chunks - pruned_chunks) * chunksize - last_chunk_empty
    assert sum(lengths) == saved_steps
    assert all(1 <= x <= chunksize for x in lengths)
    replay = Replay(length, capacity, directory=tmpdir, chunksize=chunksize)
    replay.load(data)
    assert sorted(elements.Path(tmpdir).glob('*.npz')) == sorted(filenames)
    assert len(replay) == num_items
    dataset = unbatched(replay.dataset(1))
    for _ in range(len(replay)):
      assert len(next(dataset)['step']) == length

  @pytest.mark.parametrize('Replay', REPLAYS_SAVECHUNKS)
  @pytest.mark.parametrize('workers', [1, 2, 5])
  @pytest.mark.parametrize(
      'length,capacity,chunksize', [(1, 1, 1), (3, 10, 5), (5, 100, 12)])
  def test_restore_chunks_workers(
      self, tmpdir, Replay, workers, length, capacity, chunksize):
    capacity *= workers
    replay = Replay(
        length, capacity, directory=tmpdir, chunksize=chunksize,
        save_wait=True)
    for step in range(50):
      for worker in range(workers):
        replay.add({'step': step}, worker)
    num_items = np.clip((50 - length + 1) * workers, 0, capacity)
    assert len(replay) == num_items
    data = replay.save()
    filenames = list(elements.Path(tmpdir).glob('*.npz'))
    lengths = [int(x.stem.split('-')[3]) for x in filenames]
    stored_steps = min(capacity // workers + length - 1, 50)
    total_chunks = int(np.ceil(50 / chunksize))
    pruned_chunks = int(np.floor((50 - stored_steps) / chunksize))
    assert len(filenames) == (total_chunks - pruned_chunks) * workers
    last_chunk_empty = total_chunks * chunksize - 50
    saved_steps = (total_chunks - pruned_chunks) * chunksize - last_chunk_empty
    assert sum(lengths) == saved_steps * workers
    replay = Replay(length, capacity, directory=tmpdir, chunksize=chunksize)
    replay.load(data)
    assert len(replay) == num_items
    dataset = unbatched(replay.dataset(1))
    for _ in range(len(replay)):
      assert len(next(dataset)['step']) == length

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  @pytest.mark.parametrize(
      'length,capacity,chunksize',
      [(1, 1, 128), (3, 10, 128), (5, 100, 128), (5, 25, 2)])
  def test_restore_insert(self, tmpdir, Replay, length, capacity, chunksize):
    elements.UUID.reset(debug=True)
    replay = Replay(
        length, capacity, directory=tmpdir, chunksize=chunksize,
        save_wait=True)
    inserts = int(1.5 * chunksize)
    for step in range(inserts):
      replay.add({'step': step})
    num_items = np.clip(inserts - length + 1, 0, capacity)
    assert len(replay) == num_items
    data = replay.save()
    replay = Replay(length, capacity, directory=tmpdir)
    replay.load(data)
    assert len(replay) == num_items
    dataset = unbatched(replay.dataset(1))
    for _ in range(len(replay)):
      assert len(next(dataset)['step']) == length
    for step in range(inserts):
      replay.add({'step': step})
    num_items = np.clip(2 * (inserts - length + 1), 0, capacity)
    assert len(replay) == num_items

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  def test_threading(
      self, tmpdir, Replay, length=5, capacity=128, chunksize=32,
      adders=8, samplers=4):
    elements.UUID.reset(debug=True)
    replay = Replay(
        length, capacity, directory=tmpdir, chunksize=chunksize,
        save_wait=True)
    running = [True]

    def adder():
      ident = threading.get_ident()
      step = 0
      while running[0]:
        replay.add({'step': step}, worker=ident)
        step += 1
        time.sleep(0.001)

    def sampler():
      dataset = unbatched(replay.dataset(1))
      while running[0]:
        seq = next(dataset)
        assert (seq['step'] - seq['step'][0] == np.arange(length)).all()
        time.sleep(0.001)

    workers = []
    for _ in range(adders):
      workers.append(threading.Thread(target=adder))
    for _ in range(samplers):
      workers.append(threading.Thread(target=sampler))

    try:
      [worker.start() for worker in workers]
      for _ in range(4):

        time.sleep(0.1)
        stats = replay.stats()
        assert stats['inserts'] > 0
        assert stats['samples'] > 0

        print('SAVING')
        data = replay.save()
        time.sleep(0.1)

        print('LOADING')
        replay.load(data)

    finally:
      running[0] = False
      [worker.join() for worker in workers]

    assert len(replay) == capacity


class TestEpisodeReplay:

  def add_episode(self, replay, goal, value, length=2, worker=0):
    for step in range(length):
      replay.add({
          'goal_id': np.int32(goal),
          'value': np.int32(value),
          'is_first': np.bool_(step == 0),
          'is_last': np.bool_(step == length - 1),
          'is_terminal': np.bool_(False),
      }, worker)

  def test_algorithm_r_goal_balance_and_short_episode_sampling(self, tmpdir):
    replay = embodied.replay.EpisodeReplay(
        length=5, capacity=6, groups=3, directory=tmpdir,
        retention='reservoir', seed=7)
    for goal in range(3):
      for episode in range(50):
        self.add_episode(replay, goal, 100 * goal + episode, length=1)

    assert replay.episodes_seen.tolist() == [50, 50, 50]
    assert [len(replay.slots[x]) for x in range(3)] == [2, 2, 2]
    batch = replay.sample(600)
    counts = np.bincount(batch['goal_id'][:, 0], minlength=3)
    assert np.all((150 < counts) & (counts < 250)), counts
    assert batch['value'].shape == (600, 5)
    assert batch['is_first'].all()  # One-step episodes are joined explicitly.
    assert np.all(batch['goal_id'] == batch['goal_id'][:, :1])

  def test_algorithm_r_empirical_inclusion_probability(self, tmpdir):
    trials = 1_000
    counts = np.zeros(10, np.int64)
    for seed in range(trials):
      replay = embodied.replay.EpisodeReplay(
          length=1, capacity=2, groups=1, directory=tmpdir,
          retention='reservoir', seed=seed)
      for episode in range(10):
        self.add_episode(replay, 0, episode, length=1)
      for record in replay.slots[0]:
        counts[int(record['data']['value'][0])] += 1

    # Every stream episode has inclusion probability M / n = 0.2.
    assert np.all(np.abs(counts / trials - 0.2) < 0.04), counts

  def test_sampling_does_not_change_future_admission(self, tmpdir):
    left = embodied.replay.EpisodeReplay(
        length=1, capacity=2, groups=1, directory=tmpdir / 'left',
        retention='reservoir', seed=5)
    right = embodied.replay.EpisodeReplay(
        length=1, capacity=2, groups=1, directory=tmpdir / 'right',
        retention='reservoir', seed=5)
    for replay in (left, right):
      for episode in range(2):
        self.add_episode(replay, 0, episode, length=1)
    left.sample(1_000)
    for replay in (left, right):
      for episode in range(2, 100):
        self.add_episode(replay, 0, episode, length=1)

    retained = lambda replay: [
        int(record['data']['value'][0]) for record in replay.slots[0]]
    assert retained(left) == retained(right)

  def test_current_old_sampling_keeps_current_half_and_old_goals_uniform(
      self, tmpdir):
    replay = embodied.replay.EpisodeReplay(
        length=1, capacity=6, groups=3, directory=tmpdir,
        retention='reservoir', sampling='current_old', current_group=2,
        current_fraction=0.5, seed=13)
    for goal in range(3):
      self.add_episode(replay, goal, goal, length=1)

    batch = replay.sample(20_000)
    counts = np.bincount(batch['goal_id'][:, 0], minlength=3) / 20_000
    assert np.allclose(counts, [0.25, 0.25, 0.5], atol=0.02), counts
    stats = replay.stats()
    assert stats['sampling/current_group'] == 2
    assert stats['sampling/current_fraction'] == 0.5
    assert stats['sampling/is_current_old'] == 1

  def test_current_old_sampling_falls_back_before_current_is_available(
      self, tmpdir):
    replay = embodied.replay.EpisodeReplay(
        length=1, capacity=6, groups=3, directory=tmpdir,
        retention='reservoir', sampling='current_old', current_group=2,
        current_fraction=0.5, seed=17)
    self.add_episode(replay, 0, 0, length=1)
    self.add_episode(replay, 1, 1, length=1)
    batch = replay.sample(2_000)
    counts = np.bincount(batch['goal_id'][:, 0], minlength=3)
    assert counts[2] == 0
    assert abs(int(counts[0]) - int(counts[1])) < 150

  def test_sampling_policy_can_change_when_exact_reservoir_is_restored(
      self, tmpdir):
    replay = embodied.replay.EpisodeReplay(
        length=1, capacity=6, groups=3, directory=tmpdir,
        retention='reservoir', sampling='uniform_groups', seed=19)
    for goal in range(3):
      self.add_episode(replay, goal, goal, length=1)
    state = replay.save()

    restored = embodied.replay.EpisodeReplay(
        length=1, capacity=6, groups=3, directory=tmpdir,
        retention='reservoir', sampling='current_old', current_group=1,
        current_fraction=0.6, seed=999)
    restored.load(state)
    assert [len(restored.slots[x]) for x in range(3)] == [1, 1, 1]
    counts = np.bincount(
        restored.sample(10_000)['goal_id'][:, 0], minlength=3) / 10_000
    assert np.allclose(counts, [0.2, 0.6, 0.2], atol=0.025), counts

  def test_exact_restore_future_admission_partials_and_export(self, tmpdir):
    source = pathlib.Path(str(tmpdir)) / 'source'
    replay = embodied.replay.EpisodeReplay(
        length=3, capacity=6, groups=3, directory=source,
        retention='reservoir', seed=3)
    for goal in range(3):
      for episode in range(10):
        self.add_episode(replay, goal, episode, length=2)
    replay.add({
        'goal_id': np.int32(0), 'value': np.int32(999),
        'is_first': np.bool_(True), 'is_last': np.bool_(False),
        'is_terminal': np.bool_(False)}, worker=4)
    state = replay.save()

    restored = embodied.replay.EpisodeReplay(
        length=3, capacity=6, groups=3, directory=source,
        retention='reservoir', seed=999)
    restored.load(state)
    assert 4 in restored.partials
    for candidate in (replay, restored):
      candidate.add({
          'goal_id': np.int32(0), 'value': np.int32(999),
          'is_first': np.bool_(False), 'is_last': np.bool_(True),
          'is_terminal': np.bool_(False)}, worker=4)
      for goal in range(3):
        for episode in range(10, 20):
          self.add_episode(candidate, goal, episode, length=2)
    assert [[record['episode_id'] for record in replay.slots[group]]
            for group in range(3)] == [
                [record['episode_id'] for record in restored.slots[group]]
                for group in range(3)]
    expected = replay.sample(20)
    actual = restored.sample(20)
    assert expected.keys() == actual.keys()
    assert all(np.array_equal(expected[key], actual[key]) for key in expected)

    destination = pathlib.Path(str(tmpdir)) / 'export'
    records = restored.export(destination)
    assert records and (destination / 'manifest.json').is_file()
    assert all((destination / record['name']).is_file()
               for record in records)
    exported = embodied.replay.EpisodeReplay(
        length=3, capacity=6, groups=3, directory=destination,
        retention='reservoir', seed=123)
    exported.load(restored.save())
    assert [[record['episode_id'] for record in exported.slots[group]]
            for group in range(3)] == [
                [record['episode_id'] for record in restored.slots[group]]
                for group in range(3)]

  def test_fresh_reset_abandons_restored_partial(self, tmpdir):
    replay = embodied.replay.EpisodeReplay(
        length=2, capacity=2, groups=1, directory=tmpdir,
        retention='reservoir', seed=0)
    replay.add({
        'goal_id': np.int32(0), 'value': np.int32(1),
        'is_first': np.bool_(True), 'is_last': np.bool_(False),
        'is_terminal': np.bool_(False)}, worker=0)
    replay.add({
        'goal_id': np.int32(0), 'value': np.int32(2),
        'is_first': np.bool_(True), 'is_last': np.bool_(True),
        'is_terminal': np.bool_(False)}, worker=0)

    assert replay.episodes_seen.tolist() == [1]
    assert replay.slots[0][0]['data']['value'].tolist() == [2]
    assert replay.stats()['abandoned'] == 1
