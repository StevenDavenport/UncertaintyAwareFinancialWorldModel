# Uncertainty-Aware Financial World Model

This repository is a financial forecasting side project built on top of the DreamerV3 codebase.

The core idea is not to learn a trading policy.  
It is to train a world model and use its uncertainty to gate decisions:

- `green` when confidence is high
- `abstain` otherwise

## Project Status

Implemented:

- Market data pipeline for 5-minute OHLCV bars with stationary features
- Dreamer-compatible market environment with zero actions
- Ensemble training launcher (multi-seed)
- Posterior Monte Carlo rollout inference
- Abstention gate using:
  - ensemble mean probability (`p_mean`)
  - ensemble disagreement (`std p_hat`)
  - rollout variance (`var_mean`)
- Evaluation suite with metrics, CSV/JSON artefacts, and sanity plot
- Threshold sweep utility
- Universe downloader utility (Yahoo Finance, 5m/60d)

## Architecture

DreamerV3 world model is reused nearly unchanged:

- RSSM remains action-conditioned
- actions are forced to zeros (`agent.zero_actions=True`)
- reward loss can be disabled (`agent.use_reward_loss=False`)
- market treated as exogenous sequence modeling

## Data Format

Prepared splits (`train.npz`, `val.npz`, `test.npz`) contain:

- `feat`: `[episodes, length, 6]`
- `ret_1`: `[episodes, length, 1]`
- `timestamp_ns`: `[episodes, length]`
- `close`: `[episodes, length]`

Feature vector:

1. 1-step log return
2. candle body ratio
3. log(high/low)
4. range ratio
5. rolling realized volatility
6. rolling log-volume z-score

## Preliminary Results (as of February 13, 2026)

These are early prototype results and not leak-proof production benchmarks.

Single-ticker run (`QQQ`, tiny model, 3-seed ensemble), test-tuned gate:

- precision: `61.5%`
- coverage: `5.3%`
- recall: `8.1%`

Leakage-corrected protocol (`val` tune -> locked `test`) showed overfit:

- locked test precision dropped near base rate (~`40%`)

Universe run (subset eval for fast sweep on CPU):

- dataset: `101` tickers
- episodes: `train=1010`, `val=404`, `test=202`
- quick eval (3 seeds, `samples=8`, `max_episodes=30`):
  - baseline gate precision: `46.3%`, coverage: `21.0%`
  - best precision-first sweep point: `53.0%` precision at `1.13%` coverage

Takeaway: gating behavior works; coverage/precision tradeoff is controllable; more data depth is still needed.

## Quickstart

### 1) Download liquid universe data (Yahoo 5m, last 60d)

```bash
python -m dreamerv3.market.download_universe \
  --universe sp100 \
  --outdir data \
  --interval 5m \
  --period 60d \
  --max_tickers 120 \
  --skip_existing
```

### 2) Build universe dataset

```bash
python -m dreamerv3.market.prepare_universe \
  --input_dir data \
  --tickers "$(python - <<'PY'
import json, pathlib
r = json.loads(pathlib.Path('data/universe_sp100_download_report.json').read_text())
print(','.join(r['success']))
PY
)" \
  --csv_pattern '{ticker}_5m.csv' \
  --outdir data/market/universe_5m \
  --episode_length 256 \
  --stride 256 \
  --train_end 2026-01-10T00:00:00 \
  --val_end 2026-02-01T00:00:00 \
  --strict
```

### 3) Train ensemble (tiny config, CPU)

```bash
python -m dreamerv3.market.train_ensemble \
  --configs market_tiny \
  --seeds 0,1,2,3,4 \
  --logroot ~/logdir/finwm/universe_tiny \
  --env.market.dataset_dir data/market/universe_5m \
  --jax.platform cpu \
  --jax.prealloc False
```

### 4) Evaluate ensemble

```bash
python -m dreamerv3.market.eval_ensemble \
  --logdirs ~/logdir/finwm/universe_tiny/seed_0 ~/logdir/finwm/universe_tiny/seed_1 ~/logdir/finwm/universe_tiny/seed_2 \
  --dataset_dir data/market/universe_5m \
  --split test \
  --horizon 12 \
  --samples 8 \
  --epsilon 0.0005 \
  --cost_buffer 0.0002 \
  --p_min 0.50 \
  --disagree_max 0.12 \
  --var_max 0.40 \
  --max_episodes 30 \
  --outdir ~/logdir/finwm/universe_tiny/eval_h12_test_quick30
```

### 5) Sweep thresholds

```bash
python -m dreamerv3.market.sweep_thresholds \
  --predictions_csv ~/logdir/finwm/universe_tiny/eval_h12_test_quick30/predictions.csv \
  --outdir ~/logdir/finwm/universe_tiny/eval_h12_test_quick30/sweep_relaxed \
  --p_values 0.35,0.40,0.45,0.50,0.55,0.60,0.65 \
  --disagree_values 0.04,0.06,0.08,0.10,0.12,0.15,0.20 \
  --var_values 0.10,0.15,0.20,0.25,0.30,0.40,0.60 \
  --min_coverage 0.01
```

## Important Notes

- Yahoo Finance 5-minute history is limited to the most recent ~60 days.
- For serious validation and lower overfit risk, migrate to a provider with deeper intraday history.
- Always evaluate with strict protocol:
  - tune thresholds on `val`
  - report once on locked `test`

## Key Project Files

- `dreamerv3/market/data.py`
- `dreamerv3/market/prepare_data.py`
- `dreamerv3/market/prepare_universe.py`
- `dreamerv3/market/download_universe.py`
- `dreamerv3/market/train_ensemble.py`
- `dreamerv3/market/eval_ensemble.py`
- `dreamerv3/market/sweep_thresholds.py`
- `dreamerv3/market/inference.py`
- `embodied/envs/market.py`
