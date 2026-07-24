#!/usr/bin/env python3
"""Frozen-world-model critic corruption, fidelity, and repair experiment.

The runner is deliberately staged and resumable. It collects a fixed healthy
actor dataset, measures open-loop model fidelity, compares clean and corrupted
critics on identical anchors, screens critic-only rehearsal horizons, confirms
the best qualifying horizons at MCPB-512, and emits an automatic decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import pickle
import sys
from functools import partial as bind
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np

import corrupt_actor_checkpoint as corrupt
from checkpoint_utils import resolve_checkpoint_path


elements = embodied = jax = yaml = dv3_main = None


def load_runtime_dependencies() -> None:
  """Keep pure analysis helpers importable in lightweight test environments."""
  global elements, embodied, jax, yaml, dv3_main
  if elements is not None:
    return
  import elements as elements_module
  import embodied as embodied_module
  import jax as jax_module
  import ruamel.yaml as yaml_module
  from dreamerv3 import main as main_module
  elements = elements_module
  embodied = embodied_module
  jax = jax_module
  yaml = yaml_module
  dv3_main = main_module


ADAPT_NAMESPACES = (
    '^(?!(adapt_opt|adapt_actor_opt|adapt_critic_opt|model_opt)/)')


def parse_ints(text: str) -> list[int]:
  return [int(x.strip()) for x in text.split(',') if x.strip()]


def json_dump(path: pathlib.Path, value: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
  if not rows:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  keys = sorted({key for row in rows for key in row})
  with path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
  if not path.exists():
    return []
  with path.open() as f:
    return list(csv.DictReader(f))


def rankdata(values: np.ndarray) -> np.ndarray:
  order = np.argsort(values, kind='mergesort')
  ranks = np.empty(len(values), np.float64)
  sorted_values = values[order]
  start = 0
  while start < len(values):
    end = start + 1
    while end < len(values) and sorted_values[end] == sorted_values[start]:
      end += 1
    ranks[order[start:end]] = 0.5 * (start + end - 1)
    start = end
  return ranks


def correlation(lhs: np.ndarray, rhs: np.ndarray, ranks: bool = False) -> float:
  lhs = np.asarray(lhs, np.float64).reshape(-1)
  rhs = np.asarray(rhs, np.float64).reshape(-1)
  finite = np.isfinite(lhs) & np.isfinite(rhs)
  lhs, rhs = lhs[finite], rhs[finite]
  if len(lhs) < 2 or lhs.std() == 0 or rhs.std() == 0:
    return float('nan')
  if ranks:
    lhs, rhs = rankdata(lhs), rankdata(rhs)
  return float(np.corrcoef(lhs, rhs)[0, 1])


def error_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
  pred = np.asarray(pred, np.float64).reshape(-1)
  target = np.asarray(target, np.float64).reshape(-1)
  finite = np.isfinite(pred) & np.isfinite(target)
  pred, target = pred[finite], target[finite]
  if not len(pred):
    return {k: float('nan') for k in (
        'mae', 'rmse', 'bias', 'pearson', 'spearman', 'nrmse')}
  diff = pred - target
  scale = max(float(target.std()), 1e-8)
  return {
      'mae': float(np.abs(diff).mean()),
      'rmse': float(np.sqrt(np.square(diff).mean())),
      'bias': float(diff.mean()),
      'pearson': correlation(pred, target),
      'spearman': correlation(pred, target, ranks=True),
      'nrmse': float(np.sqrt(np.square(diff).mean()) / scale),
  }


def quantile_bands(values: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
  """Assign stable low/middle/high labels, including tied-value datasets."""
  values = np.asarray(values, np.float64).reshape(-1)
  if not len(values):
    return np.zeros(0, np.int8), {0: 'low', 1: 'middle', 2: 'high'}
  ranks = rankdata(values)
  denominator = max(len(values) - 1, 1)
  bands = np.minimum((3 * ranks / denominator).astype(np.int8), 2)
  return bands, {0: 'low', 1: 'middle', 2: 'high'}


def bootstrap_median_ci(
    values: list[float], seed: int = 0, draws: int = 5000
) -> tuple[float, float]:
  values = np.asarray(values, np.float64)
  if not len(values):
    return float('nan'), float('nan')
  rng = np.random.default_rng(seed)
  samples = rng.choice(values, (draws, len(values)), replace=True)
  medians = np.median(samples, axis=1)
  lo, hi = np.quantile(medians, (0.025, 0.975))
  return float(lo), float(hi)


def discounted_return(rewards: np.ndarray, gamma: float) -> np.ndarray:
  rewards = np.asarray(rewards, np.float64)
  weights = gamma ** np.arange(rewards.shape[-1], dtype=np.float64)
  return (rewards * weights).sum(-1)


def categorical_kl(post: np.ndarray, prior: np.ndarray) -> np.ndarray:
  def logsoftmax(x):
    x = np.asarray(x, np.float64)
    top = x.max(-1, keepdims=True)
    return x - top - np.log(np.exp(x - top).sum(-1, keepdims=True))
  lp, lq = logsoftmax(post), logsoftmax(prior)
  return (np.exp(lp) * (lp - lq)).sum(-1).mean(-1)


def namespace_digest(params: dict[str, Any], prefixes: tuple[str, ...]) -> str:
  digest = hashlib.sha256()
  for key in sorted(k for k in params if k.startswith(prefixes)):
    arr = np.asarray(jax.device_get(params[key]))
    digest.update(key.encode())
    digest.update(str(arr.dtype).encode())
    digest.update(np.asarray(arr.shape, np.int64).tobytes())
    digest.update(arr.tobytes())
  return digest.hexdigest()


def copy_params(params: dict[str, Any]) -> dict[str, Any]:
  return jax.tree.map(lambda x: x.copy(), params)


def make_config(args, runtime: pathlib.Path):
  raw = yaml.YAML(typ='safe').load(
      elements.Path(ROOT / 'dreamerv3' / 'configs.yaml').read())
  config = elements.Config(raw['defaults'])
  for name in args.configs:
    config = config.update(raw[name])
  config = config.update(
      task=args.task,
      seed=args.seed,
      logdir=str(runtime),
      run=config.run.update(
          envs=1, debug=True, from_checkpoint=str(args.clean_checkpoint)),
      jax=config.jax.update(
          platform=args.jax_platform, prealloc=False, precompile=False),
      eval_adapt=config.eval_adapt.update(
          enabled=True, train_critic=True, critic_lr=args.critic_lr,
          steps=1, imag_length=6, start_batch=args.screen_particles))
  return config


def load_agent(args, outdir: pathlib.Path):
  runtime = outdir / 'runtime'
  runtime.mkdir(parents=True, exist_ok=True)
  config = make_config(args, runtime)
  agent = dv3_main.make_agent(config)
  elements.checkpoint.load(resolve_checkpoint_path(args.clean_checkpoint), dict(
      agent=bind(agent.load, regex=ADAPT_NAMESPACES)))
  clean = agent.clone_params()
  elements.checkpoint.load(resolve_checkpoint_path(args.critic_checkpoint), dict(
      agent=bind(agent.load, regex='^(val|slowval)/')))
  corrupted = agent.clone_params()
  return config, agent, clean, corrupted


def prepare_corruption(
    args, outdir: pathlib.Path, strength: float = 0.20,
    use_supplied: bool = True) -> pathlib.Path:
  if use_supplied and args.critic_checkpoint:
    return resolve_checkpoint_path(args.critic_checkpoint)
  source = resolve_checkpoint_path(args.clean_checkpoint)
  root = outdir / 'checkpoints'
  root.mkdir(parents=True, exist_ok=True)
  name = f'critic_zero_mask_s{str(strength).replace(".", "p")}_seed0'
  target = root / name
  if not target.exists():
    data = corrupt.load_agent_checkpoint(source)
    spec = dict(
        name=name, method='zero_mask', scope='all', strength=strength,
        severity='candidate')
    corrupt.save_variant(
        source, target, corrupt.clone_agent_checkpoint(data), spec, 0,
        'critic')
  return target


def _leaf_list(value, batch: int) -> list[np.ndarray]:
  arr = np.asarray(value)
  assert arr.shape[0] == batch, arr.shape
  # Dreamer enables JAX's transfer guard, so offline latent anchors must use
  # explicit transfers before the jitted carry stack sees them.
  return [jax.device_put(arr[i]) for i in range(batch)]


def carry_from_arrays(agent, deter, stoch, logit):
  deter = np.asarray(deter, np.float32)
  stoch = np.asarray(stoch, np.float32)
  logit = np.asarray(logit, np.float32)
  batch = len(deter)
  carry = list(agent.init_policy(batch))
  dyn = dict(carry[1])
  dyn['deter'] = _leaf_list(deter, batch)
  dyn['stoch'] = _leaf_list(stoch, batch)
  dyn['logit'] = _leaf_list(logit, batch)
  carry[1] = dyn
  return tuple(carry)


def collect(args, outdir: pathlib.Path) -> None:
  dataset = outdir / 'dataset'
  manifest_path = dataset / 'manifest.json'
  if manifest_path.exists() and args.skip_existing:
    print('Dataset exists:', dataset)
    return
  config, agent, clean, _ = load_agent(args, outdir)
  policy_params = agent.extract_policy_params(clean)
  driver = embodied.Driver(
      [bind(dv3_main.make_env, config, 0)], parallel=False)
  episodes: list[dict[str, list[Any]]] = []
  current: dict[str, list[Any]] | None = None

  def policy(carry, obs, **kwargs):
    nonlocal current
    carry, acts, outs = agent.policy(
        carry, obs, mode='eval', params=policy_params)
    if bool(obs['is_first'][0]):
      current = {k: [] for k in (
          'action', 'reward', 'is_first', 'is_last', 'is_terminal',
          'deter', 'stoch', 'logit')}
      episodes.append(current)
    assert current is not None
    current['action'].append(np.asarray(acts['action'][0], np.float32))
    for key in ('reward', 'is_first', 'is_last', 'is_terminal'):
      current[key].append(np.asarray(obs[key][0]))
    dyn = carry[1]
    current['deter'].append(np.asarray(dyn['deter'][0], np.float16))
    current['stoch'].append(np.asarray(dyn['stoch'][0], np.uint8))
    current['logit'].append(np.asarray(dyn['logit'][0], np.float16))
    return carry, acts, outs

  driver.reset(agent.init_policy)
  driver(policy, episodes=args.episodes)
  driver.close()
  dataset.mkdir(parents=True, exist_ok=True)
  rows = []
  for index, episode in enumerate(episodes[:args.episodes]):
    arrays = {k: np.stack(v) for k, v in episode.items()}
    path = dataset / f'episode_{index:03d}.npz'
    np.savez_compressed(path, **arrays)
    rows.append(dict(
        episode=index, length=len(arrays['reward']),
        score=float(arrays['reward'].sum()), file=path.name,
        split='train' if index < args.train_episodes else 'validation'))
  json_dump(manifest_path, dict(
      task=args.task, checkpoint=str(resolve_checkpoint_path(args.clean_checkpoint)),
      episodes=rows, anchor_stride=args.anchor_stride,
      max_fidelity_horizon=max(args.fidelity_horizons)))
  write_csv(dataset / 'episodes.csv', rows)


def load_dataset(outdir: pathlib.Path):
  root = outdir / 'dataset'
  manifest = json.loads((root / 'manifest.json').read_text())
  episodes = []
  for row in manifest['episodes']:
    with np.load(root / row['file']) as data:
      episodes.append({k: data[k] for k in data.files})
  return manifest, episodes


def anchors_from_episodes(args, episodes, split: str):
  lo = 0 if split == 'train' else args.train_episodes
  hi = args.train_episodes if split == 'train' else args.episodes
  max_h = max(args.fidelity_horizons)
  anchors = []
  for epi in range(lo, min(hi, len(episodes))):
    ep = episodes[epi]
    for step in range(0, len(ep['reward']) - max_h - 1, args.anchor_stride):
      anchors.append((epi, step))
  return anchors


def batch_carry(agent, episodes, batch):
  return carry_from_arrays(
      agent,
      np.stack([episodes[e]['deter'][t] for e, t in batch]),
      np.stack([episodes[e]['stoch'][t] for e, t in batch]),
      np.stack([episodes[e]['logit'][t] for e, t in batch]))


def batched(items, size):
  for start in range(0, len(items), size):
    yield items[start:start + size]


def state_values(agent, params, episodes, anchors, batch_size=16):
  values = []
  slowvalues = []
  rewards = []
  entropies = []
  slowentropies = []
  for batch in batched(anchors, batch_size):
    outs = agent.critic_state(batch_carry(agent, episodes, batch), params=params)
    values.extend(np.asarray(outs['value']).reshape(-1).tolist())
    slowvalues.extend(np.asarray(outs['slowvalue']).reshape(-1).tolist())
    rewards.extend(np.asarray(outs['reward']).reshape(-1).tolist())
    entropies.extend(
        np.asarray(outs['value_entropy']).reshape(-1).tolist())
    slowentropies.extend(
        np.asarray(outs['slowvalue_entropy']).reshape(-1).tolist())
  return tuple(np.asarray(x) for x in (
      values, slowvalues, rewards, entropies, slowentropies))


def proposal_coverage(args, outdir, agent, clean, episodes, anchors):
  """Compare where healthy and damaged actors drive the frozen model."""
  candidates = [('healthy_actor', clean)]
  if args.proposal_checkpoint:
    elements.checkpoint.load(
        resolve_checkpoint_path(args.proposal_checkpoint),
        dict(agent=bind(agent.load, regex='^pol/')))
    candidates.append(('damaged_actor', agent.clone_params()))
  rows = []
  max_h = max(args.fidelity_horizons)
  for actor_name, params in candidates:
    for batch in batched(anchors, args.probe_batch):
      outs = agent.critic_rollout(
          batch_carry(agent, episodes, batch), max_h,
          args.coverage_particles, params=params)
      rewards = np.asarray(outs['reward']).reshape(
          len(batch), args.coverage_particles, max_h)
      actions = np.asarray(outs['action']['action']).reshape(
          len(batch), args.coverage_particles, max_h, -1)
      for index, (epi, step) in enumerate(batch):
        for horizon in args.fidelity_horizons:
          returns = discounted_return(
              rewards[index, :, :horizon], args.gamma)
          sample = rewards[index, :, :horizon]
          rows.append(dict(
              actor=actor_name, episode=epi, step=step, horizon=horizon,
              return_mean=float(returns.mean()),
              return_std=float(returns.std()),
              return_max=float(returns.max()),
              reward_mean=float(sample.mean()),
              reward_above_0p5=float((sample >= 0.5).mean()),
              reward_above_0p8=float((sample >= 0.8).mean()),
              action_std=float(actions[index, :, :horizon].std(axis=0).mean())))
  write_csv(outdir / 'probe' / 'proposal_coverage_raw.csv', rows)
  summary = []
  for actor_name, _ in candidates:
    for horizon in args.fidelity_horizons:
      selected = [r for r in rows if
                  r['actor'] == actor_name and r['horizon'] == horizon]
      summary.append(dict(
          actor=actor_name, horizon=horizon, anchors=len(selected),
          return_mean=float(np.mean([r['return_mean'] for r in selected])),
          return_std=float(np.mean([r['return_std'] for r in selected])),
          return_max=float(np.mean([r['return_max'] for r in selected])),
          reward_above_0p5=float(np.mean([
              r['reward_above_0p5'] for r in selected])),
          reward_above_0p8=float(np.mean([
              r['reward_above_0p8'] for r in selected])),
          action_std=float(np.mean([r['action_std'] for r in selected]))))
  write_csv(outdir / 'probe' / 'proposal_coverage_by_horizon.csv', summary)


def probe(args, outdir: pathlib.Path) -> None:
  _, episodes = load_dataset(outdir)
  _, agent, clean, corrupted = load_agent(args, outdir)
  anchors = anchors_from_episodes(args, episodes, 'validation')
  clean_v, clean_slow, posterior_rew, clean_ent, clean_slow_ent = state_values(
      agent, clean, episodes, anchors, args.probe_batch)
  corrupt_v, corrupt_slow, _, corrupt_ent, corrupt_slow_ent = state_values(
      agent, corrupted, episodes, anchors, args.probe_batch)

  current_reward = np.asarray([episodes[e]['reward'][t] for e, t in anchors])
  rows = []
  for label, pred, target in (
      ('posterior_reward', posterior_rew, current_reward),
      ('corrupt_vs_clean', corrupt_v, clean_v),
      ('corrupt_slow_vs_clean', corrupt_slow, clean_v),
      ('clean_slow_vs_clean', clean_slow, clean_v)):
    rows.append({'probe': label, **error_metrics(pred, target)})

  max_h = max(args.fidelity_horizons)
  fidelity_rows = []
  for batch in batched(anchors, args.probe_batch):
    carry = batch_carry(agent, episodes, batch)
    actions = {'action': np.stack([
        episodes[e]['action'][t:t + max_h] for e, t in batch])}
    outs = agent.critic_recorded_rollout(carry, actions, params=clean)
    pred_rew = np.asarray(outs['reward'])
    prior_logit = np.asarray(outs['logit'])
    prior_deter = np.asarray(outs['deter'])
    for i, (epi, step) in enumerate(batch):
      ep = episodes[epi]
      true_rew = np.asarray(ep['reward'][step + 1:step + max_h + 1])
      true_logit = np.asarray(ep['logit'][step + 1:step + max_h + 1])
      true_deter = np.asarray(ep['deter'][step + 1:step + max_h + 1])
      for horizon in args.fidelity_horizons:
        reward_metrics = error_metrics(
            pred_rew[i, :horizon], true_rew[:horizon])
        pred_return = discounted_return(pred_rew[i:i + 1, :horizon], args.gamma)[0]
        true_return = discounted_return(true_rew[None, :horizon], args.gamma)[0]
        deter_l2 = np.linalg.norm(
            prior_deter[i, horizon - 1].astype(np.float64) -
            true_deter[horizon - 1].astype(np.float64))
        deter_den = max(np.linalg.norm(
            true_deter[horizon - 1].astype(np.float64)), 1e-8)
        fidelity_rows.append(dict(
            episode=epi, step=step, horizon=horizon,
            reward_mae=reward_metrics['mae'],
            predicted_return=pred_return, actual_return=true_return,
            prior_posterior_kl=float(categorical_kl(
                true_logit[horizon - 1], prior_logit[i, horizon - 1])),
            deter_relative_l2=float(deter_l2 / deter_den),
            continuation=float(np.asarray(outs['continuation'])[i, :horizon].mean())))

  summary = []
  for horizon in args.fidelity_horizons:
    selected = [r for r in fidelity_rows if r['horizon'] == horizon]
    mets = error_metrics(
        np.asarray([r['predicted_return'] for r in selected]),
        np.asarray([r['actual_return'] for r in selected]))
    summary.append(dict(
        horizon=horizon,
        reward_step_mae=float(np.mean([r['reward_mae'] for r in selected])),
        prior_posterior_kl=float(np.mean([
            r['prior_posterior_kl'] for r in selected])),
        deter_relative_l2=float(np.mean([
            r['deter_relative_l2'] for r in selected])),
        continuation=float(np.mean([r['continuation'] for r in selected])),
        qualified=bool(
            np.isfinite(mets['pearson']) and mets['pearson'] >= 0.5 and
            np.isfinite(mets['nrmse']) and mets['nrmse'] <= 1.0),
        **{f'return_{k}': v for k, v in mets.items()}))

  disagreement = error_metrics(corrupt_v, clean_v)
  normalized_gap = disagreement['mae'] / max(float(clean_v.std()), 1e-8)
  rows.append({'probe': 'corruption_qualification',
               'normalized_gap': normalized_gap,
               'qualified': normalized_gap >= 0.25})
  rows.extend([
      {'probe': 'clean_value_distribution',
       'mean': float(clean_v.mean()), 'std': float(clean_v.std()),
       'entropy': float(clean_ent.mean()),
       'slow_entropy': float(clean_slow_ent.mean())},
      {'probe': 'corrupt_value_distribution',
       'mean': float(corrupt_v.mean()), 'std': float(corrupt_v.std()),
       'entropy': float(corrupt_ent.mean()),
       'slow_entropy': float(corrupt_slow_ent.mean())},
  ])
  band_rows = []
  for horizon in args.fidelity_horizons:
    selected = [r for r in fidelity_rows if r['horizon'] == horizon]
    labels, names = quantile_bands(np.asarray([
        r['actual_return'] for r in selected]))
    for band, name in names.items():
      subset = [row for row, label in zip(selected, labels) if label == band]
      if not subset:
        continue
      mets = error_metrics(
          np.asarray([r['predicted_return'] for r in subset]),
          np.asarray([r['actual_return'] for r in subset]))
      band_rows.append(dict(
          horizon=horizon, reward_band=name, anchors=len(subset),
          actual_return_mean=float(np.mean([
              r['actual_return'] for r in subset])),
          **{f'return_{k}': value for k, value in mets.items()}))
  write_csv(outdir / 'probe' / 'critic_baseline.csv', rows)
  write_csv(outdir / 'probe' / 'fidelity_raw.csv', fidelity_rows)
  write_csv(outdir / 'probe' / 'fidelity_by_horizon.csv', summary)
  write_csv(outdir / 'probe' / 'fidelity_by_reward_band.csv', band_rows)
  proposal_coverage(args, outdir, agent, clean, episodes, anchors)
  json_dump(outdir / 'probe' / 'qualification.json', dict(
      normalized_critic_gap=normalized_gap,
      corruption_qualified=normalized_gap >= 0.25,
      qualified_horizons=[r['horizon'] for r in summary if r['qualified']]))


def evaluate_repair(
    args, agent, params, clean, episodes, anchors, horizon, update, seed,
    initial_mae):
  value, slowvalue, _, entropy, slowentropy = state_values(
      agent, params, episodes, anchors, args.probe_batch)
  teacher, _, _, _, _ = state_values(
      agent, clean, episodes, anchors, args.probe_batch)
  mets = error_metrics(value, teacher)
  slowmets = error_metrics(slowvalue, teacher)
  closure = 1.0 - mets['mae'] / max(initial_mae, 1e-8)
  return dict(
      seed=seed, horizon=horizon, update=update,
      gap_closure=closure,
      value_mean=float(value.mean()), value_std=float(value.std()),
      value_entropy=float(entropy.mean()),
      slowvalue_entropy=float(slowentropy.mean()),
      slow_mae=slowmets['mae'], **{f'teacher_{k}': v for k, v in mets.items()})


def run_repair_condition(
    args, outdir, agent, clean, corrupted, episodes, train_anchors,
    valid_anchors, horizon, particles, max_updates, seed, stage):
  run_dir = outdir / stage / f'h{horizon}' / f'seed{seed}'
  done = run_dir / 'done.json'
  if done.exists() and args.skip_existing:
    return read_csv(run_dir / 'curve.csv')
  rng = np.random.default_rng(seed)
  schedule = rng.integers(0, len(train_anchors), size=max_updates)
  params = copy_params(corrupted)
  with agent.n_adapt.lock:
    agent.n_adapt.value = seed * 1_000_000 + horizon * 10_000
  teacher, _, _, _, _ = state_values(
      agent, clean, episodes, valid_anchors, args.probe_batch)
  initial, _, _, _, _ = state_values(
      agent, params, episodes, valid_anchors, args.probe_batch)
  initial_mae = error_metrics(initial, teacher)['mae']
  checkpoints = sorted(set(
      x for x in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
      if x <= max_updates))
  rows = [evaluate_repair(
      args, agent, params, clean, episodes, valid_anchors,
      horizon, 0, seed, initial_mae)]
  last_metrics = {}
  for update in range(1, max_updates + 1):
    anchor = [train_anchors[int(schedule[update - 1])]]
    carry = batch_carry(agent, episodes, anchor)
    params, _, last_metrics = agent.adapt_critic(
        params, carry, steps=1, horizon=horizon, start_batch=particles)
    if update in checkpoints:
      row = evaluate_repair(
          args, agent, params, clean, episodes, valid_anchors,
          horizon, update, seed, initial_mae)
      for key in (
          'adapt_critic_opt/loss', 'adapt_critic_opt/grad_norm',
          'adapt_critic_opt/update_rms'):
        if key in last_metrics:
          row[key.replace('/', '_')] = float(np.asarray(last_metrics[key]))
      rows.append(row)
      write_csv(run_dir / 'curve.csv', rows)
  unchanged = namespace_digest(params, ('enc/', 'dyn/', 'dec/', 'rew/', 'con/', 'pol/'))
  expected = namespace_digest(corrupted, ('enc/', 'dyn/', 'dec/', 'rew/', 'con/', 'pol/'))
  if unchanged != expected:
    raise AssertionError('Critic-only repair modified actor/world-model parameters')
  run_dir.mkdir(parents=True, exist_ok=True)
  critic_state = {
      key: np.asarray(jax.device_get(value)) for key, value in params.items()
      if key.startswith(('val/', 'slowval/', 'adapt_critic_opt/'))}
  with (run_dir / 'critic_state.pkl').open('wb') as f:
    pickle.dump(critic_state, f, protocol=pickle.HIGHEST_PROTOCOL)
  json_dump(done, dict(
      horizon=horizon, seed=seed, particles=particles,
      updates=max_updates, actor_world_digest=unchanged))
  return rows


def repair_stage(args, outdir: pathlib.Path, stage: str) -> None:
  _, episodes = load_dataset(outdir)
  _, agent, clean, corrupted = load_agent(args, outdir)
  train = anchors_from_episodes(args, episodes, 'train')
  valid = anchors_from_episodes(args, episodes, 'validation')
  if stage == 'screen':
    horizons = args.screen_horizons
    particles, updates = args.screen_particles, args.screen_updates
  else:
    selected_path = outdir / 'screen' / 'selected.json'
    if not selected_path.exists():
      summarize_screen(args, outdir)
    selected = json.loads(selected_path.read_text())['confirmation_horizons']
    horizons = [int(x) for x in selected]
    particles, updates = args.confirm_particles, args.confirm_updates
  all_rows = []
  for horizon in horizons:
    for seed in args.rehearsal_seeds:
      rows = run_repair_condition(
          args, outdir, agent, clean, corrupted, episodes, train, valid,
          horizon, particles, updates, seed, stage)
      all_rows.extend(rows)
      write_csv(outdir / stage / 'repair_curve.csv', all_rows)


def summarize_screen(args, outdir: pathlib.Path) -> None:
  rows = read_csv(outdir / 'screen' / 'repair_curve.csv')
  fidelity = read_csv(outdir / 'probe' / 'fidelity_by_horizon.csv')
  qualified = {
      int(r['horizon']) for r in fidelity if r['qualified'].lower() == 'true'}
  summary = []
  for horizon in args.screen_horizons:
    final = [r for r in rows if int(r['horizon']) == horizon and
             int(r['update']) == args.screen_updates]
    closures = [float(r['gap_closure']) for r in final]
    summary.append(dict(
        horizon=horizon, runs=len(final), fidelity_qualified=horizon in qualified,
        median_gap_closure=float(np.median(closures)) if closures else float('nan'),
        mean_gap_closure=float(np.mean(closures)) if closures else float('nan')))
  eligible = [r for r in summary if r['fidelity_qualified'] and r['runs']]
  eligible.sort(key=lambda r: (-r['median_gap_closure'], r['horizon']))
  chosen = [r['horizon'] for r in eligible[:2]]
  if 6 not in chosen:
    chosen.append(6)
  chosen = sorted(set(chosen))
  write_csv(outdir / 'screen' / 'summary.csv', summary)
  json_dump(outdir / 'screen' / 'selected.json', dict(
      confirmation_horizons=chosen,
      rule='top two fidelity-qualified gap closures plus horizon-6 control'))


def summarize(args, outdir: pathlib.Path) -> None:
  summarize_screen(args, outdir)
  probeq = json.loads((outdir / 'probe' / 'qualification.json').read_text())
  confirmation = read_csv(outdir / 'confirm' / 'repair_curve.csv')
  final = [r for r in confirmation if int(r['update']) == args.confirm_updates]
  confirmed_summary = []
  for horizon in sorted({int(r['horizon']) for r in final}):
    closures = [float(r['gap_closure']) for r in final
                if int(r['horizon']) == horizon]
    lo, hi = bootstrap_median_ci(closures, seed=horizon)
    confirmed_summary.append(dict(
        horizon=horizon, runs=len(closures),
        median_gap_closure=float(np.median(closures)),
        mean_gap_closure=float(np.mean(closures)),
        median_ci_low=lo, median_ci_high=hi))
  write_csv(outdir / 'confirm' / 'summary.csv', confirmed_summary)
  best = max(
      confirmed_summary,
      key=lambda row: row['median_gap_closure'], default=None)
  median_closure = (
      best['median_gap_closure'] if best else float('nan'))
  if not probeq['qualified_horizons']:
    direction = 'world_model_fidelity'
  elif not probeq['corruption_qualified']:
    direction = 'recalibrate_critic_corruption'
  elif np.isfinite(median_closure) and median_closure >= 0.50:
    direction = 'rollout_proposal_coverage'
  elif np.isfinite(median_closure) and median_closure < 0.20:
    direction = 'critic_targets_or_bootstrapping'
  else:
    direction = 'critic_repair_hyperparameters'
  json_dump(outdir / 'decision.json', dict(
      next_direction=direction,
      best_confirmation_horizon=best['horizon'] if best else None,
      best_median_gap_closure=median_closure,
      qualified_horizons=probeq['qualified_horizons'],
      corruption_qualified=probeq['corruption_qualified']))
  plot_results(outdir)


def plot_results(outdir: pathlib.Path) -> None:
  """Create compact plots when matplotlib is installed; CSVs stay canonical."""
  try:
    import matplotlib.pyplot as plt
  except ImportError:
    print('matplotlib unavailable; skipping plots')
    return
  plotdir = outdir / 'plots'
  plotdir.mkdir(parents=True, exist_ok=True)
  fidelity = read_csv(outdir / 'probe' / 'fidelity_by_horizon.csv')
  if fidelity:
    horizons = [int(row['horizon']) for row in fidelity]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(horizons, [float(r['return_pearson']) for r in fidelity], 'o-')
    axes[0].axhline(0.5, color='grey', linestyle='--')
    axes[0].set(xlabel='Imagination horizon', ylabel='Return correlation')
    axes[1].plot(horizons, [float(r['return_nrmse']) for r in fidelity], 'o-')
    axes[1].axhline(1.0, color='grey', linestyle='--')
    axes[1].set(xlabel='Imagination horizon', ylabel='Return NRMSE')
    fig.tight_layout()
    fig.savefig(plotdir / 'world_model_fidelity.png', dpi=160)
    plt.close(fig)
  repair = read_csv(outdir / 'confirm' / 'repair_curve.csv')
  if repair:
    fig, ax = plt.subplots(figsize=(6, 4))
    horizons = sorted({int(r['horizon']) for r in repair})
    for horizon in horizons:
      subset = [r for r in repair if int(r['horizon']) == horizon]
      updates = sorted({int(r['update']) for r in subset})
      medians = [np.median([
          float(r['gap_closure']) for r in subset
          if int(r['update']) == update]) for update in updates]
      ax.plot(updates, medians, marker='o', label=f'H={horizon}')
    ax.axhline(0.5, color='grey', linestyle='--')
    ax.set(xlabel='Critic updates', ylabel='Median clean-critic gap closure')
    ax.legend()
    fig.tight_layout()
    fig.savefig(plotdir / 'critic_repair.png', dpi=160)
    plt.close(fig)


def parse_args(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('command', choices=(
      'prepare', 'collect', 'probe', 'screen', 'confirm', 'summarize', 'all'))
  parser.add_argument('--outdir', type=pathlib.Path, required=True)
  parser.add_argument('--clean_checkpoint', type=pathlib.Path, required=True)
  parser.add_argument('--critic_checkpoint', type=pathlib.Path)
  parser.add_argument(
      '--proposal_checkpoint', type=pathlib.Path,
      help='Optional actor+critic corruption used only for rollout coverage.')
  parser.add_argument('--configs', default='dmc_vision')
  parser.add_argument('--task', default='dmc_cheetah_run')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--jax_platform', default='cpu')
  parser.add_argument('--episodes', type=int, default=12)
  parser.add_argument('--train_episodes', type=int, default=6)
  parser.add_argument('--anchor_stride', type=int, default=10)
  parser.add_argument('--fidelity_horizons', default='1,3,6,12,24,48,96')
  parser.add_argument('--screen_horizons', default='3,6,12,24,48')
  parser.add_argument('--rehearsal_seeds', default='0,1,2')
  parser.add_argument('--screen_particles', type=int, default=128)
  parser.add_argument('--confirm_particles', type=int, default=512)
  parser.add_argument('--screen_updates', type=int, default=128)
  parser.add_argument('--confirm_updates', type=int, default=256)
  parser.add_argument('--critic_lr', type=float, default=1e-4)
  parser.add_argument('--gamma', type=float, default=1 - 1 / 333)
  parser.add_argument('--probe_batch', type=int, default=8)
  parser.add_argument('--coverage_particles', type=int, default=16)
  parser.add_argument('--skip_existing', action='store_true')
  args = parser.parse_args(argv)
  args.configs = [x.strip() for x in args.configs.split(',') if x.strip()]
  args.fidelity_horizons = parse_ints(args.fidelity_horizons)
  args.screen_horizons = parse_ints(args.screen_horizons)
  args.rehearsal_seeds = parse_ints(args.rehearsal_seeds)
  if not 0 < args.train_episodes < args.episodes:
    parser.error('Need 0 < train_episodes < episodes')
  return args


def main(argv=None):
  args = parse_args(argv)
  supplied_critic_checkpoint = args.critic_checkpoint is not None
  outdir = args.outdir.resolve()
  outdir.mkdir(parents=True, exist_ok=True)
  if args.command == 'prepare':
    print(prepare_corruption(args, outdir))
    return 0
  load_runtime_dependencies()
  if not args.critic_checkpoint:
    selected = outdir / 'selected_corruption.json'
    candidate = (pathlib.Path(json.loads(selected.read_text())['checkpoint'])
                 if selected.exists() else
                 outdir / 'checkpoints' / 'critic_zero_mask_s0p2_seed0')
    if not candidate.exists():
      candidate = prepare_corruption(args, outdir)
    args.critic_checkpoint = candidate
  json_dump(outdir / 'experiment_plan.json', {
      key: str(value) if isinstance(value, pathlib.Path) else value
      for key, value in vars(args).items()})
  commands = ('collect', 'probe', 'screen', 'confirm', 'summarize')
  requested = commands if args.command == 'all' else (args.command,)
  for command in requested:
    print(f'=== {command} ===')
    if command == 'collect':
      collect(args, outdir)
    elif command == 'probe':
      probe(args, outdir)
      # A critic perturbation must be large enough to measure recovery without
      # changing the actor or world model. Escalate only for auto-generated
      # checkpoints; an explicitly supplied checkpoint is never replaced.
      qualification = json.loads(
          (outdir / 'probe' / 'qualification.json').read_text())
      if not supplied_critic_checkpoint and not qualification[
          'corruption_qualified']:
        for strength in (0.30, 0.50):
          args.critic_checkpoint = prepare_corruption(
              args, outdir, strength, use_supplied=False)
          print(f'Retrying critic qualification at strength {strength:.2f}')
          probe(args, outdir)
          qualification = json.loads(
              (outdir / 'probe' / 'qualification.json').read_text())
          if qualification['corruption_qualified']:
            break
        json_dump(outdir / 'selected_corruption.json', dict(
            checkpoint=str(args.critic_checkpoint),
            qualification=qualification))
    elif command == 'screen':
      repair_stage(args, outdir, 'screen')
      summarize_screen(args, outdir)
    elif command == 'confirm':
      repair_stage(args, outdir, 'confirm')
    elif command == 'summarize':
      summarize(args, outdir)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
