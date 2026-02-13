from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def validate_rollout_tensor_shape(
    rollout: np.ndarray,
    batch: int,
    samples: int,
    horizon: int,
) -> None:
  assert rollout.shape == (batch, samples, horizon), (
      rollout.shape, (batch, samples, horizon))


def posterior_delta_stats(delta: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
  assert delta.ndim == 2, delta.shape
  p_hat = (delta > epsilon).mean(axis=1)
  var_hat = delta.var(axis=1)
  return p_hat.astype(np.float32), var_hat.astype(np.float32)


def sample_posterior_indices(
    logits: np.ndarray,
    samples: int,
    seed: int,
) -> np.ndarray:
  """Sample categorical indices from logits for determinism tests.

  Args:
    logits: [B, S, C]
    samples: number of samples K

  Returns:
    int32 tensor with shape [B, K, S].
  """
  assert logits.ndim == 3, logits.shape
  bsize, stoch, classes = logits.shape
  probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
  probs = probs / probs.sum(axis=-1, keepdims=True)

  rng = np.random.default_rng(seed)
  draws = rng.random((bsize, samples, stoch, 1), dtype=np.float64)
  cdf = np.cumsum(probs[:, None, :, :], axis=-1)
  idx = (draws > cdf).sum(axis=-1)
  idx = np.clip(idx, 0, classes - 1)
  return idx.astype(np.int32)


def compute_future_cum_returns(ret_1: np.ndarray, horizon: int) -> np.ndarray:
  """Strictly-causal future return sum.

  For each t, this returns sum(ret_1[t+1 : t+horizon+1]).
  Last horizon steps are NaN because future is unavailable.
  """
  assert ret_1.ndim == 2, ret_1.shape
  assert horizon >= 1, horizon
  episodes, length = ret_1.shape
  out = np.full((episodes, length), np.nan, np.float32)
  if length <= horizon:
    return out

  # prefix[:, i] = sum(ret_1[:, :i])
  prefix = np.concatenate([
      np.zeros((episodes, 1), np.float64),
      np.cumsum(ret_1.astype(np.float64), axis=1),
  ], axis=1)

  t = np.arange(0, length - horizon, dtype=np.int64)
  left = t + 1
  right = t + horizon + 1
  out[:, t] = (prefix[:, right] - prefix[:, left]).astype(np.float32)
  return out


def build_market_obs(
    feat_t: np.ndarray,
    ret_t: np.ndarray,
    t: int,
    length: int,
) -> dict[str, np.ndarray]:
  assert ret_t.shape == (1,), ret_t.shape
  return {
      'feat': feat_t.astype(np.float32),
      'ret_1': ret_t.astype(np.float32),
      'reward': np.float32(0.0),
      'is_first': np.bool_(t == 0),
      'is_last': np.bool_(t == length - 1),
      'is_terminal': np.bool_(t == length - 1),
  }


def run_mc_eval_on_payload(
    agent,
    payload: dict[str, np.ndarray],
    horizon: int,
    epsilon: float,
    cost_buffer: float,
    mode: str = 'mc_eval',
    max_episodes: int = 0,
) -> dict[str, np.ndarray]:
  feat = payload['feat'].astype(np.float32)
  ret_1 = payload['ret_1'].astype(np.float32)
  ts = payload['timestamp_ns'].astype(np.int64)

  episodes, length, _ = feat.shape
  if max_episodes > 0:
    episodes = min(episodes, max_episodes)
    feat = feat[:episodes]
    ret_1 = ret_1[:episodes]
    ts = ts[:episodes]

  future = compute_future_cum_returns(ret_1[..., 0], horizon)
  if length > horizon:
    assert np.isnan(future[:, length - horizon:]).all()
  carry = agent.init_policy(1)

  rows = {
      'episode': [],
      't': [],
      'timestamp_ns': [],
      'future_return': [],
      'y': [],
      'p_hat': [],
      'var_hat': [],
  }

  threshold = epsilon + cost_buffer
  for ep in range(episodes):
    for t in range(length):
      obs_t = build_market_obs(feat[ep, t], ret_1[ep, t], t=t, length=length)
      obs_batch = {k: v[None] for k, v in obs_t.items()}
      carry, _, outs = agent.policy(carry, obs_batch, mode=mode)

      target = future[ep, t]
      if not np.isfinite(target):
        continue

      p_hat = float(outs['mc/p_hat'][0])
      var_hat = float(outs['mc/var_hat'][0])
      y = bool(target > threshold)

      rows['episode'].append(ep)
      rows['t'].append(t)
      rows['timestamp_ns'].append(int(ts[ep, t]))
      rows['future_return'].append(float(target))
      rows['y'].append(y)
      rows['p_hat'].append(p_hat)
      rows['var_hat'].append(var_hat)

  return {
      k: np.asarray(v)
      for k, v in rows.items()
  }


def aggregate_ensemble_outputs(
    outputs: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
  if not outputs:
    raise ValueError('Expected at least one model output.')

  base = outputs[0]
  key_cols = ('episode', 't', 'timestamp_ns')
  for out in outputs[1:]:
    for key in key_cols:
      if not np.array_equal(out[key], base[key]):
        raise ValueError(f'Ensemble alignment mismatch for key: {key}')
    if not np.array_equal(out['y'], base['y']):
      raise ValueError('Ensemble alignment mismatch for labels.')

  p_stack = np.stack([out['p_hat'] for out in outputs], axis=1)
  v_stack = np.stack([out['var_hat'] for out in outputs], axis=1)

  return {
      'episode': base['episode'],
      't': base['t'],
      'timestamp_ns': base['timestamp_ns'],
      'future_return': base['future_return'],
      'y': base['y'].astype(bool),
      'p_mean': p_stack.mean(axis=1),
      'disagree': p_stack.std(axis=1),
      'var_mean': v_stack.mean(axis=1),
      'p_hat_models': p_stack,
      'var_hat_models': v_stack,
  }


def apply_green_gate(
    p_mean: np.ndarray,
    disagree: np.ndarray,
    var_mean: np.ndarray,
    p_min: float,
    disagree_max: float,
    var_max: float,
) -> np.ndarray:
  return (
      (p_mean >= p_min) &
      (disagree <= disagree_max) &
      (var_mean <= var_max)
  )


def _safe_ratio(num: float, den: float) -> float:
  return float(num / den) if den else float('nan')


def compute_gate_metrics(y: np.ndarray, green: np.ndarray) -> dict[str, float]:
  y = y.astype(bool)
  green = green.astype(bool)
  tp = int((green & y).sum())
  fp = int((green & ~y).sum())
  fn = int((~green & y).sum())
  tn = int((~green & ~y).sum())

  precision = _safe_ratio(tp, tp + fp)
  recall = _safe_ratio(tp, tp + fn)
  coverage = _safe_ratio(tp + fp, len(y))
  abstain_rate = _safe_ratio(fn + tn, len(y))
  false_green_rate = _safe_ratio(fp, tp + fp)

  return {
      'count': int(len(y)),
      'positives': int(y.sum()),
      'greens': int(green.sum()),
      'tp': tp,
      'fp': fp,
      'fn': fn,
      'tn': tn,
      'precision': precision,
      'recall': recall,
      'coverage': coverage,
      'abstain_rate': abstain_rate,
      'false_green_rate': false_green_rate,
  }


def reliability_table(
    p_mean: np.ndarray,
    y: np.ndarray,
    bins: int = 10,
) -> dict[str, np.ndarray]:
  edges = np.linspace(0.0, 1.0, bins + 1)
  index = np.clip(np.digitize(p_mean, edges, right=True) - 1, 0, bins - 1)

  rows = {
      'bin_lo': [],
      'bin_hi': [],
      'count': [],
      'pred_mean': [],
      'empirical_pos_rate': [],
  }
  y = y.astype(bool)
  for b in range(bins):
    mask = (index == b)
    rows['bin_lo'].append(float(edges[b]))
    rows['bin_hi'].append(float(edges[b + 1]))
    rows['count'].append(int(mask.sum()))
    if mask.any():
      rows['pred_mean'].append(float(p_mean[mask].mean()))
      rows['empirical_pos_rate'].append(float(y[mask].mean()))
    else:
      rows['pred_mean'].append(float('nan'))
      rows['empirical_pos_rate'].append(float('nan'))

  return {k: np.asarray(v) for k, v in rows.items()}


def save_predictions_csv(path: str, rows: dict[str, np.ndarray]) -> None:
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  keys = list(rows.keys())
  length = len(rows[keys[0]])
  with open(path, 'w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(keys)
    for i in range(length):
      writer.writerow([rows[k][i] for k in keys])


def save_reliability_csv(path: str, table: dict[str, np.ndarray]) -> None:
  save_predictions_csv(path, table)


def save_metrics_json(path: str, metrics: dict) -> None:
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  Path(path).write_text(json.dumps(metrics, indent=2))
