import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from dreamerv3.market import inference


def test_validate_rollout_tensor_shape():
  rollout = np.zeros((2, 3, 4), np.float32)
  inference.validate_rollout_tensor_shape(rollout, batch=2, samples=3, horizon=4)

  with pytest.raises(AssertionError):
    inference.validate_rollout_tensor_shape(rollout, batch=2, samples=4, horizon=4)


def test_sampling_determinism_with_seed():
  logits = np.array(
      [[[-0.1, 0.2, 1.3], [0.0, 0.5, -0.2]]],
      dtype=np.float32,
  )
  a = inference.sample_posterior_indices(logits, samples=16, seed=7)
  b = inference.sample_posterior_indices(logits, samples=16, seed=7)
  c = inference.sample_posterior_indices(logits, samples=16, seed=11)

  assert np.array_equal(a, b)
  assert not np.array_equal(a, c)


def test_future_returns_are_strictly_causal():
  ret = np.array([[10.0, 1.0, 2.0, 3.0]], np.float32)
  future = inference.compute_future_cum_returns(ret, horizon=2)

  # t=0 should only use ret[1] and ret[2], never ret[0].
  assert np.isclose(future[0, 0], 3.0)
  assert np.isclose(future[0, 1], 5.0)
  assert np.isnan(future[0, 2])
  assert np.isnan(future[0, 3])

  ret_changed = ret.copy()
  ret_changed[0, 0] = -999.0
  future_changed = inference.compute_future_cum_returns(ret_changed, horizon=2)
  assert np.isclose(future[0, 0], future_changed[0, 0])


def test_ensemble_aggregation_and_gate_metrics():
  out1 = {
      'episode': np.array([0, 0]),
      't': np.array([5, 6]),
      'timestamp_ns': np.array([100, 101]),
      'future_return': np.array([0.02, -0.01], np.float32),
      'y': np.array([True, False]),
      'p_hat': np.array([0.9, 0.2], np.float32),
      'var_hat': np.array([0.05, 0.10], np.float32),
  }
  out2 = {
      'episode': np.array([0, 0]),
      't': np.array([5, 6]),
      'timestamp_ns': np.array([100, 101]),
      'future_return': np.array([0.02, -0.01], np.float32),
      'y': np.array([True, False]),
      'p_hat': np.array([0.7, 0.4], np.float32),
      'var_hat': np.array([0.15, 0.20], np.float32),
  }

  agg = inference.aggregate_ensemble_outputs([out1, out2])
  np.testing.assert_allclose(agg['p_mean'], np.array([0.8, 0.3], np.float32))
  np.testing.assert_allclose(agg['disagree'], np.array([0.1, 0.1], np.float32))
  np.testing.assert_allclose(agg['var_mean'], np.array([0.10, 0.15], np.float32))

  green = inference.apply_green_gate(
      agg['p_mean'],
      agg['disagree'],
      agg['var_mean'],
      p_min=0.75,
      disagree_max=0.11,
      var_max=0.12,
  )
  assert green.tolist() == [True, False]

  metrics = inference.compute_gate_metrics(agg['y'], green)
  assert metrics['tp'] == 1
  assert metrics['fp'] == 0
  assert metrics['fn'] == 0
  assert metrics['tn'] == 1
  assert np.isclose(metrics['precision'], 1.0)
  assert np.isclose(metrics['coverage'], 0.5)


def test_directional_gate_and_metrics():
  p_up = np.array([0.80, 0.20, 0.75, 0.40], np.float32)
  p_down = np.array([0.10, 0.85, 0.75, 0.20], np.float32)
  d_up = np.array([0.05, 0.06, 0.02, 0.04], np.float32)
  d_down = np.array([0.04, 0.05, 0.03, 0.04], np.float32)
  var = np.array([0.01, 0.02, 0.01, 0.50], np.float32)

  gate = inference.apply_directional_gate(
      p_up, p_down, d_up, d_down, var,
      p_min=0.70, disagree_max=0.08, var_max=0.10)
  assert gate['direction'].tolist() == [1, -1, 0, 0]
  assert gate['green'].tolist() == [True, True, False, False]

  y_up = np.array([True, False, True, False])
  y_down = np.array([False, True, False, False])
  metrics = inference.compute_directional_gate_metrics(
      y_up, y_down, gate['long'], gate['short'])
  assert metrics['tp'] == 2
  assert metrics['fp'] == 0
  assert metrics['wrong_way'] == 0
  assert np.isclose(metrics['precision'], 1.0)
  assert np.isclose(metrics['coverage'], 0.5)


def test_fixed_horizon_trade_sim_with_stops_and_shorts():
  payload = {
      'ret_1': np.array([[[0.0], [-0.02], [0.0], [0.03], [0.0]]], np.float32),
      'feat': np.array([[
          [0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
      ]], np.float32),
  }
  aggregate = {
      'episode': np.array([0, 0, 0], np.int64),
      't': np.array([0, 1, 2], np.int64),
      'timestamp_ns': np.array([1, 2, 3], np.int64),
  }
  direction = np.array([1, -1, 1], np.int8)
  out = inference.simulate_fixed_horizon_policy(
      aggregate=aggregate,
      payload=payload,
      direction=direction,
      horizon=2,
      stop_mult=1.5,
      min_stop=0.0,
      trade_cost=0.001,
      vol_feature_index=4,
  )

  np.testing.assert_allclose(
      out['net_return'],
      np.array([-0.016, -0.016, 0.029], np.float32),
      atol=1e-6)
  assert out['stop_hit'].tolist() == [True, True, False]
  assert out['exit_step'].tolist() == [1, 2, 2]
  assert np.isclose(out['metrics']['total_return'], -0.003, atol=1e-6)
  assert np.isclose(out['metrics']['max_drawdown'], 0.032, atol=1e-6)


def test_capital_backtest_respects_asset_lock_and_updates_equity():
  payload = {
      'timestamp_ns': np.array([[10, 11, 12, 13, 14, 15]], np.int64),
  }
  aggregate = {
      'episode': np.array([0, 0, 0], np.int64),
      't': np.array([0, 1, 3], np.int64),
      'timestamp_ns': np.array([10, 11, 13], np.int64),
      'p_mean': np.array([0.7, 0.8, 0.75], np.float32),
      'p_up_mean': np.array([0.7, 0.8, 0.75], np.float32),
      'p_down_mean': np.array([0.1, 0.1, 0.1], np.float32),
  }
  trade = {
      'trade': np.array([True, True, True]),
      'direction': np.array([1, 1, 1], np.int8),
      'stop_dist': np.array([0.01, 0.01, 0.01], np.float32),
      'net_return': np.array([0.02, 0.03, -0.01], np.float32),
      'exit_step': np.array([2, 2, 1], np.int32),
  }

  out = inference.run_capital_backtest(
      aggregate=aggregate,
      trade=trade,
      payload=payload,
      initial_capital=1.0,
      risk_fraction=0.1,
      max_positions=1,
      max_gross_leverage=20.0,
      one_position_per_asset=True,
  )

  metrics = out['metrics']
  assert metrics['signals'] == 3
  assert metrics['trades_executed'] == 2
  assert metrics['skipped_asset_busy'] == 1
  assert np.isclose(metrics['final_equity'], 1.08, atol=1e-6)
  assert np.isclose(metrics['total_return_pct'], 0.08, atol=1e-6)

  trades = out['trades']
  assert trades['entry_timestamp_ns'].tolist() == [10, 13]
  assert np.allclose(trades['pnl'], np.array([0.2, -0.12], np.float64), atol=1e-6)
