"""Small restart-safe process supervisor used by RLScape Experiment 1."""

from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import time
from typing import Callable


def _write_json(path: pathlib.Path, value) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
  temporary.replace(path)


def _proc_fields(pid: int) -> list[str] | None:
  try:
    raw = pathlib.Path(f'/proc/{pid}/stat').read_text()
    closing = raw.rfind(')')
    if closing < 0:
      return None
    return raw[closing + 2:].split()
  except (FileNotFoundError, PermissionError):
    return None


def _start_ticks(pid: int) -> int | None:
  try:
    # Fields after the command begin at field 3 (state); starttime is field 22.
    return int(_proc_fields(pid)[19])
  except (IndexError, TypeError, ValueError):
    return None


def _cmdline(pid: int) -> list[str] | None:
  try:
    raw = pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()
  except (FileNotFoundError, PermissionError):
    return None
  return [part.decode(errors='replace') for part in raw.split(b'\0') if part]


def record_is_live(record: dict) -> bool:
  pid = int(record.get('pid', -1))
  return bool(
      pid > 1 and _start_ticks(pid) == int(record.get('start_ticks', -1)) and
      _cmdline(pid) == list(record.get('command', ())))


def active_matches(
    active_path: pathlib.Path,
    command: list[str],
    cwd: pathlib.Path,
) -> bool:
  if not active_path.exists():
    return False
  record = json.loads(active_path.read_text())
  return bool(
      record_is_live(record) and
      record.get('command') == list(command) and
      record.get('cwd') == str(cwd))


def _group_alive(pgid: int) -> bool:
  inspected = False
  try:
    entries = pathlib.Path('/proc').iterdir()
  except (FileNotFoundError, PermissionError):
    entries = ()
  for entry in entries:
    if not entry.name.isdigit():
      continue
    fields = _proc_fields(int(entry.name))
    if not fields:
      continue
    inspected = True
    try:
      state = fields[0]
      process_group = int(fields[2])
    except (IndexError, ValueError):
      continue
    if process_group == pgid and state != 'Z':
      return True
  if inspected:
    return False
  try:
    os.killpg(pgid, 0)
    return True
  except ProcessLookupError:
    return False
  except PermissionError:
    return True


def terminate_group(
    pgid: int,
    *,
    interrupt_grace: float = 300,
    terminate_grace: float = 60,
) -> None:
  if not _group_alive(pgid):
    return
  try:
    os.killpg(pgid, signal.SIGINT)
  except ProcessLookupError:
    return
  deadline = time.time() + interrupt_grace
  while _group_alive(pgid) and time.time() < deadline:
    time.sleep(min(1, max(0.1, deadline - time.time())))
  if not _group_alive(pgid):
    return
  os.killpg(pgid, signal.SIGTERM)
  deadline = time.time() + terminate_grace
  while _group_alive(pgid) and time.time() < deadline:
    time.sleep(min(1, max(0.1, deadline - time.time())))
  if _group_alive(pgid):
    os.killpg(pgid, signal.SIGKILL)


def _latest_mtime(paths) -> float:
  mtimes = []
  for path in paths:
    path = pathlib.Path(path)
    if path.exists():
      mtimes.append(path.stat().st_mtime)
  return max(mtimes, default=0.0)


def run_managed(
    command: list[str],
    *,
    cwd: pathlib.Path,
    active_path: pathlib.Path,
    console_path: pathlib.Path | None,
    activity_paths=(),
    stream_output: bool = False,
    poll_seconds: float = 30,
    startup_timeout: float = 3600,
    stall_timeout: float = 7200,
    heartbeat: Callable[[dict], None] | None = None,
) -> dict:
  """Start or adopt one exact child command and monitor only its process group."""
  cwd = cwd.resolve()
  command = list(command)
  record = None
  if active_path.exists():
    candidate = json.loads(active_path.read_text())
    if record_is_live(candidate):
      if (
          candidate.get('command') != command or
          candidate.get('cwd') != str(cwd)
      ):
        raise RuntimeError(
            'A different managed child is still running: '
            f'{active_path}')
      record = candidate
    else:
      stale = active_path.with_name(
          f'{active_path.stem}.stale-{int(time.time())}.json')
      active_path.replace(stale)

  process = None
  console = None
  adopted = record is not None
  if not adopted:
    if console_path is not None and not stream_output:
      console_path.parent.mkdir(parents=True, exist_ok=True)
      console = console_path.open('a')
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=os.environ.copy(),
        stdout=console,
        stderr=subprocess.STDOUT if console else None,
        start_new_session=True,
    )
    start_ticks = _start_ticks(process.pid)
    if start_ticks is None:
      process.terminate()
      raise RuntimeError('Could not establish child process identity')
    record = {
        'format': 1,
        'pid': process.pid,
        'pgid': process.pid,
        'start_ticks': start_ticks,
        'command': command,
        'cwd': str(cwd),
        'started_unix': time.time(),
    }
    _write_json(active_path, record)

  pid = int(record['pid'])
  pgid = int(record['pgid'])
  started = float(record.get('started_unix', time.time()))
  paths = [
      *(activity_paths or ()),
      *([console_path] if console_path is not None else ()),
  ]
  last_mtime = _latest_mtime(paths)
  last_activity = time.time()
  saw_activity = adopted
  timed_out = False
  try:
    while record_is_live(record):
      current_mtime = _latest_mtime(paths)
      if current_mtime > last_mtime:
        last_mtime = current_mtime
        last_activity = time.time()
        saw_activity = True
      timeout = stall_timeout if saw_activity else startup_timeout
      snapshot = {
          'pid': pid,
          'pgid': pgid,
          'adopted': adopted,
          'started_unix': started,
          'heartbeat_unix': time.time(),
          'last_activity_unix': last_activity,
          'saw_activity': saw_activity,
          'timeout_seconds': timeout,
      }
      if heartbeat:
        heartbeat(snapshot)
      if time.time() - last_activity > timeout:
        timed_out = True
        terminate_group(pgid)
        break
      time.sleep(max(0.1, poll_seconds))
    returncode = process.poll() if process is not None else None
  except BaseException:
    terminate_group(pgid)
    raise
  finally:
    if console:
      console.close()
    # The Python launcher normally closes Java itself. This only removes a
    # straggler that remains inside the exact process group we created.
    if _group_alive(pgid):
      terminate_group(pgid, interrupt_grace=15, terminate_grace=15)
    if active_path.exists():
      current = json.loads(active_path.read_text())
      if (
          int(current.get('pid', -1)) == pid and
          int(current.get('start_ticks', -1)) == int(record['start_ticks'])
      ):
        active_path.unlink()
  return {
      'returncode': returncode,
      'adopted': adopted,
      'timed_out': timed_out,
      'pid': pid,
      'pgid': pgid,
      'started_unix': started,
      'finished_unix': time.time(),
  }
