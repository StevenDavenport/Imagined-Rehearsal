"""Shared contracts for Stage 5B/5C gated IR experiments."""

from __future__ import annotations

import json
import pathlib

from scripts import rlscape_stage2_actor_recovery as stage2
from scripts import rlscape_stage5a_gating as stage5a
from scripts.rlscape_m1_pilot import spec_digest


OBJECTIVES = ('standard', 'reward_only')
GATES = ('js', 'two_stage')


def load_gate_artifact(stage5a_root: pathlib.Path) -> dict:
  root = stage5a_root.expanduser().resolve()
  complete = root / 'stage5a_complete'
  selected_path = root / 'selected_gates.json'
  if not complete.is_file():
    raise FileNotFoundError(
        f'Stage 5A is not complete; missing marker: {complete}')
  if not selected_path.is_file():
    raise FileNotFoundError(selected_path)
  selected = json.loads(selected_path.read_text())
  if selected.get('canonical_stage') != 'stage5a':
    raise RuntimeError(f'Invalid Stage 5A gate artifact: {selected_path}')
  digest = selected.pop('digest', None)
  if not digest or spec_digest(selected) != digest:
    raise RuntimeError(f'Stage 5A gate artifact digest mismatch: {selected_path}')
  selected['digest'] = digest
  selected['_path'] = str(selected_path)
  return selected


def no_ir_condition() -> dict:
  return dict(
      name='no_ir', family='no_ir', checkpoint='source', enabled=False,
      objective='standard', objective_label='no_ir', horizon=15,
      actent=0.0, gate_enabled=False, gate_kind='none',
      entropy_threshold=1.0, js_threshold=1.0,
      min_success_rate=4 / 128, direction_threshold=1.0)


def ir_condition(objective: str, gate: str, selected: dict) -> dict:
  if objective not in OBJECTIVES:
    raise ValueError(objective)
  if gate not in ('always', *GATES):
    raise ValueError(gate)
  thresholds = selected.get(gate, {}) if gate != 'always' else {}
  return dict(
      name=f'{gate}_{objective}', family=gate, checkpoint='source',
      enabled=True, objective=objective, objective_label=objective,
      horizon=15, actent=0.0, gate_enabled=(gate != 'always'),
      gate_kind='none' if gate == 'always' else gate,
      entropy_threshold=float(thresholds.get('entropy_threshold', 1.0)),
      js_threshold=float(thresholds.get('js_threshold', 1.0)),
      min_success_rate=float(thresholds.get('min_success_rate', 4 / 128)),
      direction_threshold=float(thresholds.get('direction_threshold', 1.0)))


def conditions(selected: dict) -> list[dict]:
  result = [no_ir_condition()]
  result += [ir_condition(objective, 'always', selected)
             for objective in OBJECTIVES]
  for gate in GATES:
    result += [ir_condition(objective, gate, selected)
               for objective in OBJECTIVES]
  assert len(result) == 7
  assert len({item['name'] for item in result}) == len(result)
  return result


def virtual_reward_condition(
    name: str, *, dormant_every: int, active_every: int = 1,
    deactivate_after: int = 3, min_improvement: float = 0.0,
    max_baseline_score: float = 1e30) -> dict:
  """Reward-only IR guarded by a paired tentative-update probe."""
  if min(dormant_every, active_every, deactivate_after) < 1:
    raise ValueError('Virtual gate cadence and hysteresis must be positive')
  return dict(
      name=name, family='virtual_update', checkpoint='source', enabled=True,
      objective='reward_only', objective_label='reward_only', horizon=15,
      actent=0.0, gate_enabled=True, gate_kind='virtual_update',
      entropy_threshold=1.0, js_threshold=1.0,
      min_success_rate=4 / 128, direction_threshold=1.0,
      virtual_score='reward_return',
      virtual_min_improvement=float(min_improvement),
      virtual_max_baseline_score=float(max_baseline_score),
      virtual_dormant_every=int(dormant_every),
      virtual_active_every=int(active_every),
      virtual_deactivate_after=int(deactivate_after))


def build_eval_command(
    args, *, logdir: pathlib.Path, checkpoint: pathlib.Path,
    condition: dict, goal: str, policy_mode: str, episodes: int,
    environment_seed: int, agent_seed: int, persistence: str = 'episode',
    output_checkpoint: pathlib.Path | None = None,
    resume_adapt_state: bool = False, warmup_episodes: int = 1,
) -> list[str]:
  command = stage2.build_command(
      python=args.python, repo_root=args.repo_root, logdir=logdir,
      checkpoint=checkpoint, reference_checkpoint=checkpoint,
      condition=condition, goal=goal, policy_mode=policy_mode,
      episodes=episodes, seed=environment_seed,
      episode_length=args.episode_length, grid_columns=args.grid_columns,
      grid_rows=args.grid_rows, adapt_steps=args.adapt_steps,
      adapt_lr=args.adapt_lr, adapt_every_k=args.adapt_every_k,
      start_batch=args.start_batch, mvn_path=args.mvn_path,
      java_home=args.java_home,
      # Stage 5A selected its gates on the Stage 4A counterfactual+bounded
      # checkpoint. Stage 5B/5C must instantiate the same parameter tree even
      # when the current objective does not consume the bounded-value head.
      extra_configs=('rlscape_stage4a_bounded_value',))
  command += [
      '--run.eval_warmup_episodes', str(int(warmup_episodes)),
      '--eval_adapt.persistence', persistence,
      '--eval_adapt.resume_adapt_state', str(bool(resume_adapt_state)),
      '--eval_adapt.paired_rng', 'True',
      '--eval_adapt.paired_rng_seed', str(int(agent_seed)),
      '--eval_adapt.paired_rng_stride', str(args.paired_rng_stride),
      '--eval_adapt.trace.enabled', 'True',
      '--eval_adapt.trace.filename', 'stage5_trace.jsonl',
      '--eval_adapt.gate.enabled', str(bool(condition['gate_enabled'])),
      '--eval_adapt.gate.kind', condition['gate_kind'],
      '--eval_adapt.gate.audit', 'False',
      '--eval_adapt.gate.entropy_threshold',
      str(float(condition['entropy_threshold'])),
      '--eval_adapt.gate.js_threshold',
      str(float(condition['js_threshold'])),
      '--eval_adapt.gate.min_success_rate',
      str(float(condition['min_success_rate'])),
      '--eval_adapt.gate.direction_threshold',
      str(float(condition['direction_threshold'])),
  ]
  if condition['gate_kind'] == 'virtual_update':
    command += [
        '--eval_adapt.gate.virtual_score',
        str(condition['virtual_score']),
        '--eval_adapt.gate.virtual_min_improvement',
        str(float(condition['virtual_min_improvement'])),
        '--eval_adapt.gate.virtual_max_baseline_score',
        str(float(condition['virtual_max_baseline_score'])),
        '--eval_adapt.gate.virtual_dormant_every',
        str(int(condition['virtual_dormant_every'])),
        '--eval_adapt.gate.virtual_active_every',
        str(int(condition['virtual_active_every'])),
        '--eval_adapt.gate.virtual_deactivate_after',
        str(int(condition['virtual_deactivate_after'])),
    ]
  if output_checkpoint is not None:
    command += [
        '--eval_adapt.output_checkpoint', str(output_checkpoint)]
  return command


def summarize_eval(
    logdir: pathlib.Path, *, condition: dict, goal: str,
    policy_mode: str, episodes: int, environment_seed: int,
    returncode: int | None, warmup_episodes: int,
    agent_seed: int | None = None, paired_rng_stride: int | None = None,
) -> dict:
  result = stage2.summarize_condition(
      logdir, condition=condition, goal=goal, policy_mode=policy_mode,
      episodes=episodes, seed=environment_seed, returncode=returncode)
  integrity_path = logdir / 'eval_integrity.json'
  integrity = (
      json.loads(integrity_path.read_text())
      if integrity_path.is_file() else {})
  resets = stage2.reset_identities(logdir)
  result.update(stage5a.trace_summary(
      logdir, filename='stage5_trace.jsonl').copy())
  result.update({
      'objective_label': condition['objective_label'],
      'gate_kind': condition['gate_kind'],
      'gate_enabled': bool(condition['gate_enabled']),
      'warmup_episodes': int(warmup_episodes),
      'reset_count': len(resets),
      'persistence': integrity.get('persistence', ''),
      'paired_rng_verified': bool(
          agent_seed is None or (
              integrity.get('paired_rng') is True and
              integrity.get('paired_rng_seed') == int(agent_seed) and
              integrity.get('paired_rng_stride') == int(paired_rng_stride))),
      'persistent_frozen_component_violations': integrity.get(
          'persistent_frozen_component_violations', {}),
  })
  result['complete'] = bool(
      returncode in (None, 0) and result['episode_count_exact'] and
      result['integrity_equal'] and
      result['paired_rng_verified'] and
      not result['persistent_frozen_component_violations'] and
      len(resets) == int(episodes) + int(warmup_episodes))
  return result
