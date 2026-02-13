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
