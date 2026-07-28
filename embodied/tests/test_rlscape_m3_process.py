import json
import os
import pathlib
import subprocess
import sys
import time

from scripts import rlscape_m3_process as managed


def process_record(process, command, cwd):
  return {
      'format': 1,
      'pid': process.pid,
      'pgid': process.pid,
      'start_ticks': managed._start_ticks(process.pid),
      'command': command,
      'cwd': str(cwd),
      'started_unix': time.time(),
  }


def test_managed_child_writes_heartbeats_and_clears_ownership(tmp_path):
  activity = tmp_path / 'activity'
  command = [
      sys.executable, '-c',
      (
          'import pathlib,time; '
          f'pathlib.Path({str(activity)!r}).write_text("alive"); '
          'time.sleep(0.1)'
      ),
  ]
  active = tmp_path / 'active.json'
  heartbeats = []

  result = managed.run_managed(
      command,
      cwd=tmp_path,
      active_path=active,
      console_path=tmp_path / 'console.log',
      activity_paths=[activity],
      poll_seconds=0.02,
      startup_timeout=2,
      stall_timeout=2,
      heartbeat=heartbeats.append,
  )

  assert result['returncode'] == 0
  assert not result['adopted']
  assert not result['timed_out']
  assert heartbeats
  assert not active.exists()
  assert activity.read_text() == 'alive'


def test_managed_child_adopts_exact_live_process(tmp_path):
  command = [sys.executable, '-c', 'import time; time.sleep(0.15)']
  process = subprocess.Popen(
      command, cwd=tmp_path, start_new_session=True)
  active = tmp_path / 'active.json'
  active.write_text(json.dumps(process_record(process, command, tmp_path)))
  try:
    result = managed.run_managed(
        command,
        cwd=tmp_path,
        active_path=active,
        console_path=tmp_path / 'console.log',
        poll_seconds=0.02,
        startup_timeout=2,
        stall_timeout=2,
    )
  finally:
    process.wait(timeout=2)

  assert result['adopted']
  assert result['returncode'] is None
  assert not active.exists()


def test_managed_child_times_out_only_its_owned_group(tmp_path):
  command = [sys.executable, '-c', 'import time; time.sleep(10)']

  started = time.time()
  result = managed.run_managed(
      command,
      cwd=tmp_path,
      active_path=tmp_path / 'active.json',
      console_path=tmp_path / 'console.log',
      poll_seconds=0.02,
      startup_timeout=0.1,
      stall_timeout=0.1,
  )

  assert result['timed_out']
  assert time.time() - started < 5
  assert not managed._group_alive(result['pgid'])


def test_live_active_record_rejects_different_command(tmp_path):
  command = [sys.executable, '-c', 'import time; time.sleep(10)']
  process = subprocess.Popen(
      command, cwd=tmp_path, start_new_session=True)
  active = tmp_path / 'active.json'
  active.write_text(json.dumps(process_record(process, command, tmp_path)))
  try:
    try:
      managed.run_managed(
          [sys.executable, '-c', 'print("different")'],
          cwd=tmp_path,
          active_path=active,
          console_path=tmp_path / 'console.log',
          poll_seconds=0.02,
          startup_timeout=1,
          stall_timeout=1,
      )
    except RuntimeError as error:
      assert 'different managed child' in str(error)
    else:
      raise AssertionError('Different command unexpectedly adopted')
  finally:
    os.killpg(process.pid, 15)
    process.wait(timeout=2)
