# Market Forecasting + Abstention Gate (DreamerV3)

This directory contains a market forecasting prototype built on top of the existing DreamerV3 world model.

## Goal

Use DreamerV3 world-model uncertainty to produce a conservative green-light signal:
- Monte Carlo rollouts from posterior latent belief states
- Ensemble disagreement across independently trained world models
- Tiered abstention gate for when to trigger or abstain

This is not a trading policy. It is an evaluation framework for confidence-gated forecasts.

## What Was Implemented

- Market dataset preprocessing pipeline (CSV OHLCV -> stationary features -> train/val/test splits -> fixed-length episodes)
- Dreamer-compatible market environment with zero-action dynamics
- Config support for market training (`market_qqq`)
- Agent support for:
  - forced zero actions
  - optional reward loss disable (`use_reward_loss`)
  - posterior MC rollout stats during policy pass (`mc/p_hat`, `mc/var_hat`)
- Ensemble launcher for multi-seed training
- Ensemble evaluation script with abstention gate metrics and saved artifacts
- Lightweight correctness tests (shape checks, determinism, causal label construction, ensemble aggregation)

## Key Files

- `dreamerv3/market/data.py`: dataset ingestion, feature engineering, split/chunk/save/load
- `dreamerv3/market/prepare_data.py`: data preparation CLI
- `embodied/envs/market.py`: market environment
- `dreamerv3/configs.yaml`: market config block and MC options
- `dreamerv3/agent.py`: zero-action mode, optional reward loss disable, posterior MC stats
- `dreamerv3/market/inference.py`: ensemble aggregation, gating, metrics, artifact writing
- `dreamerv3/market/train_ensemble.py`: multi-seed launcher
- `dreamerv3/market/eval_ensemble.py`: evaluation entrypoint
- `dreamerv3/tests/test_market_inference.py`: lightweight tests

## Dataset Format

Prepared split files (`train.npz`, `val.npz`, `test.npz`) contain:
- `feat`: float32 `[episodes, length, feature_dim]`
- `ret_1`: float32 `[episodes, length, 1]`
- `timestamp_ns`: int64 `[episodes, length]`
- `close`: float32 `[episodes, length]`

Current engineered features (`feat`) are:
1. 1-step log return
2. candle body ratio
3. log(high/low)
4. range ratio
5. rolling realized volatility
6. rolling log-volume z-score

## Training Behavior

- Actions are forced to zeros when `agent.zero_actions=True`
- Reward loss can be disabled with `agent.use_reward_loss=False`
- For `market_qqq`, reward loss is disabled and reward is zeroed (`reward_mode: zero`)

## Quickstart

### 1) Prepare data

```bash
python -m dreamerv3.market.prepare_data \
  --input_csv data/qqq_5m.csv \
  --outdir data/market/qqq_5m \
  --instrument QQQ \
  --episode_length 256 \
  --stride 256 \
  --train_end 2026-01-10T00:00:00 \
  --val_end 2026-02-01T00:00:00
```

### 1a) Download a liquid ticker universe (yfinance, 5m/60d)

```bash
python -m dreamerv3.market.download_universe \
  --universe sp100 \
  --outdir data \
  --interval 5m \
  --period 60d \
  --max_tickers 120 \
  --skip_existing
```

Options:
- `--universe sp100|sp500|custom`
- `--tickers` for custom lists (comma-separated)
- `--skip_existing` to avoid re-downloading files
- `--report_json` to control where summary JSON is written

### 1b) Prepare multi-ticker universe data (same feature pipeline)

```bash
python -m dreamerv3.market.prepare_universe \
  --input_dir data \
  --tickers QQQ,SPY,AAPL,MSFT,AMZN,NVDA,META,GOOGL,TSLA,AMD \
  --csv_pattern '{ticker}_5m.csv' \
  --outdir data/market/universe_5m \
  --episode_length 256 \
  --stride 256 \
  --train_end 2026-01-10T00:00:00 \
  --val_end 2026-02-01T00:00:00
```

Notes:
- Each ticker CSV must live at `input_dir/csv_pattern`.
- Default behavior skips invalid/missing tickers; add `--strict` to fail fast.
- Output schema is unchanged (`train.npz`, `val.npz`, `test.npz`), so configs/scripts can point `dataset_dir` to `data/market/universe_5m`.

### 2) Train one world model

```bash
python dreamerv3/main.py \
  --configs market_qqq size1m \
  --logdir ~/logdir/finwm/qqq/seed_0 \
  --seed 0
```

### 3) Train ensemble (multi-seed)

```bash
python -m dreamerv3.market.train_ensemble \
  --configs market_qqq size1m \
  --seeds 0,1,2,3,4 \
  --logroot ~/logdir/finwm/qqq
```

### 4) Evaluate ensemble + abstention gate

```bash
python -m dreamerv3.market.eval_ensemble \
  --logdirs ~/logdir/finwm/qqq/seed_0 ~/logdir/finwm/qqq/seed_1 ~/logdir/finwm/qqq/seed_2 ~/logdir/finwm/qqq/seed_3 ~/logdir/finwm/qqq/seed_4 \
  --dataset_dir data/market/qqq_5m \
  --split test \
  --horizon 12 \
  --samples 64 \
  --epsilon 0.0005 \
  --cost_buffer 0.0002 \
  --p_min 0.70 \
  --disagree_max 0.08 \
  --var_max 0.0004 \
  --outdir ~/logdir/finwm/qqq/eval_h12
```

## Evaluation Outputs

The evaluation script writes:
- `metrics.json`: precision/recall/coverage/abstain/false-green and settings
- `predictions.csv`: per-timestep predictions and gate decisions
- `calibration.csv`: reliability-bin table
- `sanity.png`: reliability + short timeline plot

## Gate Definition

Green signal is true only if all conditions hold:
- `p_mean >= p_min`
- `disagree <= disagree_max`
- `var_mean <= var_max`

Else: abstain.

## Ground Truth Label

For each timestep `t` on the test split:
- compute strictly causal future return over horizon `H`: sum of returns on `(t+1 .. t+H)`
- label `y(t)=1` if future return exceeds `epsilon + cost_buffer`

## Tests

```bash
pytest -q dreamerv3/tests/test_market_inference.py
```

Current tests cover:
- rollout shape assertions
- deterministic sampling with fixed seed
- no lookahead leakage in label construction
- ensemble aggregation/gate metric correctness
