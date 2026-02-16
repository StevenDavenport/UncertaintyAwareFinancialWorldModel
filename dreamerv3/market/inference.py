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
      'y_up': [],
      'y_down': [],
      'p_hat': [],
      'p_up': [],
      'p_down': [],
      'var_hat': [],
      'mean_hat': [],
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

      p_up = float(outs.get('mc/p_up', outs['mc/p_hat'])[0])
      p_down_raw = outs.get('mc/p_down')
      p_down = float(p_down_raw[0]) if p_down_raw is not None else float(max(0.0, 1.0 - p_up))
      var_hat = float(outs['mc/var_hat'][0])
      mean_hat_raw = outs.get('mc/mean_hat')
      mean_hat = float(mean_hat_raw[0]) if mean_hat_raw is not None else float('nan')
      y_up = bool(target > threshold)
      y_down = bool(target < -threshold)

      rows['episode'].append(ep)
      rows['t'].append(t)
      rows['timestamp_ns'].append(int(ts[ep, t]))
      rows['future_return'].append(float(target))
      rows['y'].append(y_up)
      rows['y_up'].append(y_up)
      rows['y_down'].append(y_down)
      rows['p_hat'].append(p_up)
      rows['p_up'].append(p_up)
      rows['p_down'].append(p_down)
      rows['var_hat'].append(var_hat)
      rows['mean_hat'].append(mean_hat)

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
    if 'y_up' in base and not np.array_equal(out.get('y_up', out['y']), base['y_up']):
      raise ValueError('Ensemble alignment mismatch for y_up labels.')
    if 'y_down' in base and not np.array_equal(out.get('y_down'), base['y_down']):
      raise ValueError('Ensemble alignment mismatch for y_down labels.')

  p_up_stack = np.stack([out.get('p_up', out['p_hat']) for out in outputs], axis=1)
  p_down_stack = np.stack([out.get('p_down', np.zeros_like(out['p_hat'])) for out in outputs], axis=1)
  v_stack = np.stack([out['var_hat'] for out in outputs], axis=1)
  m_stack = np.stack([out.get('mean_hat', np.full_like(out['var_hat'], np.nan)) for out in outputs], axis=1)
  y_up = base.get('y_up', base['y']).astype(bool)
  y_down = base.get('y_down', np.zeros_like(y_up, dtype=bool)).astype(bool)

  return {
      'episode': base['episode'],
      't': base['t'],
      'timestamp_ns': base['timestamp_ns'],
      'future_return': base['future_return'],
      'y': y_up,
      'y_up': y_up,
      'y_down': y_down,
      'y_any': (y_up | y_down),
      'p_mean': p_up_stack.mean(axis=1),
      'p_up_mean': p_up_stack.mean(axis=1),
      'p_down_mean': p_down_stack.mean(axis=1),
      'disagree': p_up_stack.std(axis=1),
      'disagree_up': p_up_stack.std(axis=1),
      'disagree_down': p_down_stack.std(axis=1),
      'var_mean': v_stack.mean(axis=1),
      'mean_mean': m_stack.mean(axis=1),
      'p_hat_models': p_up_stack,
      'p_up_models': p_up_stack,
      'p_down_models': p_down_stack,
      'var_hat_models': v_stack,
      'mean_hat_models': m_stack,
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


def apply_directional_gate(
    p_up_mean: np.ndarray,
    p_down_mean: np.ndarray,
    disagree_up: np.ndarray,
    disagree_down: np.ndarray,
    var_mean: np.ndarray,
    p_min: float,
    disagree_max: float,
    var_max: float,
) -> dict[str, np.ndarray]:
  long_ok = (
      (p_up_mean >= p_min) &
      (disagree_up <= disagree_max) &
      (var_mean <= var_max)
  )
  short_ok = (
      (p_down_mean >= p_min) &
      (disagree_down <= disagree_max) &
      (var_mean <= var_max)
  )

  both = long_ok & short_ok
  prefer_long = p_up_mean > p_down_mean
  prefer_short = p_down_mean > p_up_mean

  long = (long_ok & ~short_ok) | (both & prefer_long)
  short = (short_ok & ~long_ok) | (both & prefer_short)
  green = long | short
  direction = np.zeros_like(p_up_mean, dtype=np.int8)
  direction[long] = 1
  direction[short] = -1

  return {
      'long': long,
      'short': short,
      'green': green,
      'direction': direction,
  }


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


def compute_directional_gate_metrics(
    y_up: np.ndarray,
    y_down: np.ndarray,
    long_green: np.ndarray,
    short_green: np.ndarray,
) -> dict[str, float]:
  y_up = y_up.astype(bool)
  y_down = y_down.astype(bool)
  long_green = long_green.astype(bool)
  short_green = short_green.astype(bool)
  green = long_green | short_green
  y_any = y_up | y_down

  tp_long = int((long_green & y_up).sum())
  fp_long = int((long_green & ~y_up).sum())
  tp_short = int((short_green & y_down).sum())
  fp_short = int((short_green & ~y_down).sum())
  wrong_way = int(((long_green & y_down) | (short_green & y_up)).sum())

  tp = tp_long + tp_short
  fp = fp_long + fp_short
  fn = int((~green & y_any).sum())
  tn = int((~green & ~y_any).sum())

  precision = _safe_ratio(tp, tp + fp)
  recall = _safe_ratio(tp, tp + fn)
  coverage = _safe_ratio(tp + fp, len(y_any))
  abstain_rate = _safe_ratio(fn + tn, len(y_any))
  false_green_rate = _safe_ratio(fp, tp + fp)

  return {
      'count': int(len(y_any)),
      'positives_any': int(y_any.sum()),
      'positives_up': int(y_up.sum()),
      'positives_down': int(y_down.sum()),
      'greens': int(green.sum()),
      'long_greens': int(long_green.sum()),
      'short_greens': int(short_green.sum()),
      'tp': tp,
      'fp': fp,
      'fn': fn,
      'tn': tn,
      'tp_long': tp_long,
      'fp_long': fp_long,
      'tp_short': tp_short,
      'fp_short': fp_short,
      'wrong_way': wrong_way,
      'precision': precision,
      'recall': recall,
      'coverage': coverage,
      'abstain_rate': abstain_rate,
      'false_green_rate': false_green_rate,
  }


def compute_forward_return_paths(ret_1: np.ndarray, horizon: int) -> np.ndarray:
  assert ret_1.ndim == 2, ret_1.shape
  assert horizon >= 1, horizon
  episodes, length = ret_1.shape
  out = np.full((episodes, length, horizon), np.nan, np.float32)
  usable = length - horizon
  if usable <= 0:
    return out
  for h in range(horizon):
    out[:, :usable, h] = ret_1[:, (1 + h):(1 + h + usable)]
  return out


def simulate_fixed_horizon_policy(
    aggregate: dict[str, np.ndarray],
    payload: dict[str, np.ndarray],
    direction: np.ndarray,
    horizon: int,
    stop_mult: float,
    min_stop: float,
    trade_cost: float,
    vol_feature_index: int = 4,
) -> dict[str, np.ndarray | dict[str, float]]:
  ret_1 = payload['ret_1'][..., 0].astype(np.float32)
  feat = payload['feat'].astype(np.float32)
  paths = compute_forward_return_paths(ret_1, horizon)

  ep = aggregate['episode'].astype(np.int64)
  t = aggregate['t'].astype(np.int64)
  ts = aggregate['timestamp_ns'].astype(np.int64)
  direction = direction.astype(np.int8)

  n = len(direction)
  if vol_feature_index < 0 or vol_feature_index >= feat.shape[-1]:
    raise ValueError(
        f'Invalid vol_feature_index={vol_feature_index} for feat dim {feat.shape[-1]}')

  fwd = paths[ep, t]
  vol = np.maximum(feat[ep, t, vol_feature_index], 0.0)
  stop_dist = np.maximum(float(min_stop), float(stop_mult) * vol).astype(np.float32)

  gross_return = np.zeros(n, np.float32)
  net_return = np.zeros(n, np.float32)
  exit_step = np.zeros(n, np.int32)
  stop_hit = np.zeros(n, bool)
  trade = direction != 0

  for i in np.where(trade)[0]:
    cum = np.cumsum(fwd[i].astype(np.float64))
    stop = float(stop_dist[i])

    if direction[i] > 0:
      if stop > 0:
        hit = np.where(cum <= -stop)[0]
      else:
        hit = np.array([], dtype=np.int64)
      if hit.size:
        gross = -stop
        exit_idx = int(hit[0]) + 1
        stop_hit[i] = True
      else:
        gross = float(cum[-1])
        exit_idx = int(horizon)
    else:
      if stop > 0:
        hit = np.where(cum >= stop)[0]
      else:
        hit = np.array([], dtype=np.int64)
      if hit.size:
        gross = -stop
        exit_idx = int(hit[0]) + 1
        stop_hit[i] = True
      else:
        gross = float(-cum[-1])
        exit_idx = int(horizon)

    gross_return[i] = np.float32(gross)
    net_return[i] = np.float32(gross - trade_cost)
    exit_step[i] = exit_idx

  trade_returns = net_return[trade]
  trade_ts = ts[trade]
  order = np.argsort(trade_ts, kind='stable')
  sorted_returns = trade_returns[order] if order.size else trade_returns
  equity = np.cumsum(sorted_returns.astype(np.float64)) if sorted_returns.size else np.zeros(0, np.float64)
  equity_with_start = np.concatenate([np.array([0.0], np.float64), equity], axis=0)
  peak = np.maximum.accumulate(equity_with_start)
  drawdown = equity_with_start - peak
  max_drawdown = float(-drawdown.min()) if drawdown.size else 0.0

  wins = sorted_returns > 0
  gross_profit = float(sorted_returns[wins].sum()) if sorted_returns.size else 0.0
  gross_loss = float(-sorted_returns[~wins].sum()) if sorted_returns.size else 0.0
  profit_factor = _safe_ratio(gross_profit, gross_loss)

  metrics = {
      'trades': int(trade.sum()),
      'long_trades': int((direction > 0).sum()),
      'short_trades': int((direction < 0).sum()),
      'stop_hits': int(stop_hit[trade].sum()),
      'stop_hit_rate': _safe_ratio(int(stop_hit[trade].sum()), int(trade.sum())),
      'win_rate': _safe_ratio(int((trade_returns > 0).sum()), int(trade.sum())),
      'avg_trade_return': float(trade_returns.mean()) if trade_returns.size else float('nan'),
      'median_trade_return': float(np.median(trade_returns)) if trade_returns.size else float('nan'),
      'total_return': float(trade_returns.sum()),
      'gross_profit': gross_profit,
      'gross_loss': gross_loss,
      'profit_factor': profit_factor,
      'max_drawdown': max_drawdown,
  }

  return {
      'trade': trade,
      'direction': direction,
      'stop_dist': stop_dist,
      'stop_hit': stop_hit,
      'exit_step': exit_step,
      'gross_return': gross_return,
      'net_return': net_return,
      'metrics': metrics,
  }


def _asset_ids_from_payload(payload: dict[str, np.ndarray]) -> np.ndarray:
  episodes = payload['timestamp_ns'].shape[0]
  if 'instrument_id' in payload:
    inst = payload['instrument_id']
    if inst.ndim == 2:
      return inst[:, 0].astype(np.int64)
    if inst.ndim == 1:
      return inst.astype(np.int64)
  return np.arange(episodes, dtype=np.int64)


def run_capital_backtest(
    aggregate: dict[str, np.ndarray],
    trade: dict[str, np.ndarray | dict[str, float]],
    payload: dict[str, np.ndarray],
    initial_capital: float = 1.0,
    risk_fraction: float = 0.01,
    max_positions: int = 20,
    max_gross_leverage: float = 3.0,
    one_position_per_asset: bool = True,
) -> dict[str, dict[str, np.ndarray] | dict[str, float]]:
  if initial_capital <= 0:
    raise ValueError(f'initial_capital must be positive, got {initial_capital}')
  if risk_fraction < 0:
    raise ValueError(f'risk_fraction must be non-negative, got {risk_fraction}')
  if max_positions < 1:
    raise ValueError(f'max_positions must be >= 1, got {max_positions}')
  if max_gross_leverage <= 0:
    raise ValueError(f'max_gross_leverage must be > 0, got {max_gross_leverage}')

  trade_mask = trade['trade'].astype(bool)
  direction = trade['direction'].astype(np.int8)
  stop_dist = trade['stop_dist'].astype(np.float64)
  net_return = trade['net_return'].astype(np.float64)
  exit_step = trade['exit_step'].astype(np.int64)

  ep = aggregate['episode'].astype(np.int64)
  t = aggregate['t'].astype(np.int64)
  entry_ts = aggregate['timestamp_ns'].astype(np.int64)
  p_up = aggregate.get('p_up_mean', aggregate['p_mean']).astype(np.float64)
  p_down = aggregate.get('p_down_mean', np.zeros_like(p_up)).astype(np.float64)
  score = np.maximum(p_up, p_down)

  ts_matrix = payload['timestamp_ns'].astype(np.int64)
  exit_t = t + exit_step
  valid_exit = (
      (exit_step > 0) &
      (exit_t >= 0) &
      (exit_t < ts_matrix.shape[1]) &
      (direction != 0) &
      trade_mask
  )
  if not valid_exit.any():
    eq = {
        'timestamp_ns': np.asarray([0], np.int64),
        'equity': np.asarray([float(initial_capital)], np.float64),
        'open_positions': np.asarray([0], np.int32),
        'open_notional': np.asarray([0.0], np.float64),
    }
    empty = {
        'trade_id': np.asarray([], np.int64),
        'episode': np.asarray([], np.int64),
        'asset_id': np.asarray([], np.int64),
        'direction': np.asarray([], np.int8),
        'entry_t': np.asarray([], np.int64),
        'exit_t': np.asarray([], np.int64),
        'entry_timestamp_ns': np.asarray([], np.int64),
        'exit_timestamp_ns': np.asarray([], np.int64),
        'bars_held': np.asarray([], np.int32),
        'signal_score': np.asarray([], np.float64),
        'p_up_mean': np.asarray([], np.float64),
        'p_down_mean': np.asarray([], np.float64),
        'stop_dist': np.asarray([], np.float64),
        'notional': np.asarray([], np.float64),
        'net_return': np.asarray([], np.float64),
        'pnl': np.asarray([], np.float64),
        'equity_before': np.asarray([], np.float64),
        'equity_after': np.asarray([], np.float64),
    }
    return {
        'metrics': {
            'initial_capital': float(initial_capital),
            'final_equity': float(initial_capital),
            'total_return_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'signals': int(trade_mask.sum()),
            'trades_executed': 0,
            'win_rate': float('nan'),
            'avg_pnl': float('nan'),
            'median_pnl': float('nan'),
            'total_pnl': 0.0,
            'profit_factor': float('nan'),
            'skipped_invalid': int((~valid_exit & trade_mask).sum()),
            'skipped_position_limit': 0,
            'skipped_asset_busy': 0,
            'skipped_leverage': 0,
            'skipped_equity': 0,
        },
        'trades': empty,
        'equity': eq,
    }

  candidate_idx = np.where(valid_exit)[0]
  asset_lookup = _asset_ids_from_payload(payload)
  asset_id = asset_lookup[ep[candidate_idx]]
  entry_ts_c = entry_ts[candidate_idx]
  exit_ts_c = ts_matrix[ep[candidate_idx], exit_t[candidate_idx]]
  direction_c = direction[candidate_idx]
  stop_dist_c = np.maximum(stop_dist[candidate_idx], 1e-8)
  net_ret_c = net_return[candidate_idx]
  score_c = score[candidate_idx]
  p_up_c = p_up[candidate_idx]
  p_down_c = p_down[candidate_idx]
  ep_c = ep[candidate_idx]
  t_c = t[candidate_idx]
  exit_t_c = exit_t[candidate_idx]
  bars_c = exit_step[candidate_idx].astype(np.int32)

  entry_ids = {}
  exit_ids = {}
  for j in range(len(candidate_idx)):
    et = int(entry_ts_c[j])
    xt = int(exit_ts_c[j])
    entry_ids.setdefault(et, []).append(j)
    exit_ids.setdefault(xt, []).append(j)

  timeline = sorted(set(entry_ids.keys()) | set(exit_ids.keys()))
  cash = float(initial_capital)
  open_notional = 0.0
  open_positions: dict[int, dict[str, float | int]] = {}
  open_assets: set[int] = set()

  skipped_invalid = int((~valid_exit & trade_mask).sum())
  skipped_position_limit = 0
  skipped_asset_busy = 0
  skipped_leverage = 0
  skipped_equity = 0

  trades = {
      'trade_id': [],
      'episode': [],
      'asset_id': [],
      'direction': [],
      'entry_t': [],
      'exit_t': [],
      'entry_timestamp_ns': [],
      'exit_timestamp_ns': [],
      'bars_held': [],
      'signal_score': [],
      'p_up_mean': [],
      'p_down_mean': [],
      'stop_dist': [],
      'notional': [],
      'net_return': [],
      'pnl': [],
      'equity_before': [],
      'equity_after': [],
  }
  curve = {
      'timestamp_ns': [timeline[0]],
      'equity': [cash],
      'open_positions': [0],
      'open_notional': [0.0],
  }

  for ts in timeline:
    for j in exit_ids.get(ts, []):
      pos = open_positions.pop(j, None)
      if pos is None:
        continue
      aid = int(pos['asset_id'])
      if aid in open_assets:
        open_assets.remove(aid)
      notional = float(pos['notional'])
      pnl = notional * float(pos['net_return'])
      eq_before = cash
      cash += pnl
      open_notional = max(0.0, open_notional - notional)

      trades['trade_id'].append(int(candidate_idx[j]))
      trades['episode'].append(int(pos['episode']))
      trades['asset_id'].append(aid)
      trades['direction'].append(int(pos['direction']))
      trades['entry_t'].append(int(pos['entry_t']))
      trades['exit_t'].append(int(pos['exit_t']))
      trades['entry_timestamp_ns'].append(int(pos['entry_timestamp_ns']))
      trades['exit_timestamp_ns'].append(int(ts))
      trades['bars_held'].append(int(pos['bars_held']))
      trades['signal_score'].append(float(pos['signal_score']))
      trades['p_up_mean'].append(float(pos['p_up_mean']))
      trades['p_down_mean'].append(float(pos['p_down_mean']))
      trades['stop_dist'].append(float(pos['stop_dist']))
      trades['notional'].append(notional)
      trades['net_return'].append(float(pos['net_return']))
      trades['pnl'].append(float(pnl))
      trades['equity_before'].append(float(eq_before))
      trades['equity_after'].append(float(cash))

    candidates = entry_ids.get(ts, [])
    if candidates:
      candidates = sorted(
          candidates, key=lambda j: (score_c[j], abs(net_ret_c[j])), reverse=True)
    for j in candidates:
      aid = int(asset_id[j])
      if one_position_per_asset and aid in open_assets:
        skipped_asset_busy += 1
        continue
      if len(open_positions) >= max_positions:
        skipped_position_limit += 1
        continue
      if cash <= 0:
        skipped_equity += 1
        continue

      stop = float(stop_dist_c[j])
      risk_budget = cash * float(risk_fraction)
      desired_notional = risk_budget / stop if stop > 0 else 0.0
      allowed_notional = max(0.0, float(max_gross_leverage) * cash - open_notional)
      notional = min(desired_notional, allowed_notional)
      if notional <= 0:
        skipped_leverage += 1
        continue

      open_positions[j] = {
          'episode': int(ep_c[j]),
          'asset_id': aid,
          'direction': int(direction_c[j]),
          'entry_t': int(t_c[j]),
          'exit_t': int(exit_t_c[j]),
          'entry_timestamp_ns': int(entry_ts_c[j]),
          'bars_held': int(bars_c[j]),
          'signal_score': float(score_c[j]),
          'p_up_mean': float(p_up_c[j]),
          'p_down_mean': float(p_down_c[j]),
          'stop_dist': float(stop_dist_c[j]),
          'notional': float(notional),
          'net_return': float(net_ret_c[j]),
      }
      open_assets.add(aid)
      open_notional += float(notional)

    curve['timestamp_ns'].append(int(ts))
    curve['equity'].append(float(cash))
    curve['open_positions'].append(int(len(open_positions)))
    curve['open_notional'].append(float(open_notional))

  # Any positions remaining should be impossible under valid exits, but close defensively.
  for j in list(open_positions.keys()):
    pos = open_positions.pop(j)
    aid = int(pos['asset_id'])
    if aid in open_assets:
      open_assets.remove(aid)
    notional = float(pos['notional'])
    pnl = notional * float(pos['net_return'])
    eq_before = cash
    cash += pnl
    open_notional = max(0.0, open_notional - notional)

    trades['trade_id'].append(int(candidate_idx[j]))
    trades['episode'].append(int(pos['episode']))
    trades['asset_id'].append(aid)
    trades['direction'].append(int(pos['direction']))
    trades['entry_t'].append(int(pos['entry_t']))
    trades['exit_t'].append(int(pos['exit_t']))
    trades['entry_timestamp_ns'].append(int(pos['entry_timestamp_ns']))
    trades['exit_timestamp_ns'].append(int(pos['entry_timestamp_ns']))
    trades['bars_held'].append(int(pos['bars_held']))
    trades['signal_score'].append(float(pos['signal_score']))
    trades['p_up_mean'].append(float(pos['p_up_mean']))
    trades['p_down_mean'].append(float(pos['p_down_mean']))
    trades['stop_dist'].append(float(pos['stop_dist']))
    trades['notional'].append(notional)
    trades['net_return'].append(float(pos['net_return']))
    trades['pnl'].append(float(pnl))
    trades['equity_before'].append(float(eq_before))
    trades['equity_after'].append(float(cash))

  equity = np.asarray(curve['equity'], np.float64)
  peak = np.maximum.accumulate(equity)
  drawdown_pct = np.where(peak > 0, (peak - equity) / peak, 0.0)
  max_drawdown_pct = float(drawdown_pct.max()) if drawdown_pct.size else 0.0

  pnl = np.asarray(trades['pnl'], np.float64)
  wins = pnl > 0
  gross_profit = float(pnl[wins].sum()) if pnl.size else 0.0
  gross_loss = float(-pnl[~wins].sum()) if pnl.size else 0.0
  profit_factor = _safe_ratio(gross_profit, gross_loss)

  metrics = {
      'initial_capital': float(initial_capital),
      'final_equity': float(cash),
      'total_return_pct': float(cash / float(initial_capital) - 1.0),
      'max_drawdown_pct': max_drawdown_pct,
      'signals': int(trade_mask.sum()),
      'trades_executed': int(len(pnl)),
      'win_rate': _safe_ratio(int((pnl > 0).sum()), int(len(pnl))),
      'avg_pnl': float(pnl.mean()) if pnl.size else float('nan'),
      'median_pnl': float(np.median(pnl)) if pnl.size else float('nan'),
      'total_pnl': float(pnl.sum()) if pnl.size else 0.0,
      'gross_profit': gross_profit,
      'gross_loss': gross_loss,
      'profit_factor': profit_factor,
      'skipped_invalid': skipped_invalid,
      'skipped_position_limit': int(skipped_position_limit),
      'skipped_asset_busy': int(skipped_asset_busy),
      'skipped_leverage': int(skipped_leverage),
      'skipped_equity': int(skipped_equity),
  }

  return {
      'metrics': metrics,
      'trades': {
          k: np.asarray(v)
          for k, v in trades.items()
      },
      'equity': {
          k: np.asarray(v)
          for k, v in curve.items()
      },
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
