import numpy as np

from embodied.run.train import _copy_scalar_logs


def test_episode_log_scalars_are_copied_before_epstats_mutation():
  source = np.array(1.0)
  copied = _copy_scalar_logs({
      'log/task_success/max': source,
      'not_a_log': np.array(2.0),
  })

  source[...] = 7.0

  assert copied.keys() == {'log/task_success/max'}
  assert copied['log/task_success/max'] == 1.0
