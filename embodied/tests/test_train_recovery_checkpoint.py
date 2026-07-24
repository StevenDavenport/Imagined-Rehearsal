from types import SimpleNamespace

import elements
import embodied
import pytest

from embodied.envs.dummy import Dummy
from embodied.run.train import train
from scripts.rlscape_m2_train import checkpoint_step


class FailingEnv(Dummy):

  def __init__(self):
    super().__init__('disc', size=(8, 8), length=100)
    self.calls = 0
    self.closed = False

  def step(self, action):
    self.calls += 1
    if self.calls == 4:
      raise RuntimeError('external bridge failed')
    return super().step(action)

  def close(self):
    self.closed = True


def test_train_saves_exact_recovery_checkpoint_on_env_error(tmp_path):
  env = FailingEnv()
  agent = embodied.RandomAgent(env.obs_space, env.act_space)

  def make_stream(replay, mode):
    del mode
    while True:
      yield replay.sample(1)

  args = SimpleNamespace(
      logdir=str(tmp_path),
      usage={},
      batch_size=1,
      batch_length=2,
      train_ratio=0,
      log_every=1000,
      report_every=1000,
      save_every=1000,
      envs=1,
      debug=True,
      report_batches=1,
      consec_report=1,
      steps=100,
      ckpt_keep=2,
      from_checkpoint='',
      from_checkpoint_regex='',
  )
  logger = elements.Logger(
      elements.Counter(), [elements.logger.TerminalOutput()])

  with pytest.raises(RuntimeError, match='external bridge failed'):
    train(
        lambda: agent,
        lambda: embodied.replay.Replay(length=2, capacity=100),
        lambda index: env,
        make_stream,
        lambda: logger,
        args,
    )

  assert checkpoint_step(tmp_path) == 3
  assert env.closed
