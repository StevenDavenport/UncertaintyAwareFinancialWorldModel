from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EPS = 1e-8


@dataclass(frozen=True)
class ColumnNames:
  timestamp: str = 'timestamp'
  open: str = 'open'
  high: str = 'high'
  low: str = 'low'
  close: str = 'close'
  volume: str = 'volume'


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
  assert window >= 1
  out = np.full(values.shape, np.nan, np.float64)
  if values.size < window:
    return out
  cumsum = np.cumsum(np.insert(values, 0, 0.0))
  out[window - 1:] = (cumsum[window:] - cumsum[:-window]) / window
  return out


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
  assert window >= 1
  out = np.full(values.shape, np.nan, np.float64)
  if values.size < window:
    return out
  csum = np.cumsum(np.insert(values, 0, 0.0))
  csum2 = np.cumsum(np.insert(values * values, 0, 0.0))
  mean = (csum[window:] - csum[:-window]) / window
  mean2 = (csum2[window:] - csum2[:-window]) / window
  var = np.maximum(mean2 - mean * mean, 0.0)
  out[window - 1:] = np.sqrt(var)
  return out


def _to_datetime64_ns(value: str) -> np.int64:
  return np.datetime64(value, 'ns').astype(np.int64)


def load_ohlcv_csv(path: str, columns: ColumnNames = ColumnNames()) -> dict[str, np.ndarray]:
  rows = []
  with open(path, 'r', newline='') as handle:
    reader = csv.DictReader(handle)
    required = {
        columns.timestamp,
        columns.open,
        columns.high,
        columns.low,
        columns.close,
        columns.volume,
    }
    missing = required - set(reader.fieldnames or ())
    if missing:
      raise ValueError(f'Missing required columns: {sorted(missing)}')
    for row in reader:
      rows.append((
          _to_datetime64_ns(row[columns.timestamp]),
          float(row[columns.open]),
          float(row[columns.high]),
          float(row[columns.low]),
          float(row[columns.close]),
          float(row[columns.volume]),
      ))

  if not rows:
    raise ValueError(f'No rows found in CSV: {path}')

  data = np.array(rows, dtype=[
      ('timestamp_ns', np.int64),
      ('open', np.float64),
      ('high', np.float64),
      ('low', np.float64),
      ('close', np.float64),
      ('volume', np.float64),
  ])
  order = np.argsort(data['timestamp_ns'])
  data = data[order]

  return {
      'timestamp_ns': data['timestamp_ns'].astype(np.int64),
      'open': data['open'].astype(np.float64),
      'high': data['high'].astype(np.float64),
      'low': data['low'].astype(np.float64),
      'close': data['close'].astype(np.float64),
      'volume': data['volume'].astype(np.float64),
  }


def build_features(
    ohlcv: dict[str, np.ndarray],
    vol_window: int = 48,
) -> dict[str, np.ndarray]:
  assert vol_window >= 2, vol_window

  timestamp_ns = ohlcv['timestamp_ns']
  open_ = ohlcv['open']
  high = ohlcv['high']
  low = ohlcv['low']
  close = ohlcv['close']
  volume = ohlcv['volume']

  ret_1 = np.zeros_like(close, dtype=np.float64)
  ret_1[1:] = np.log(np.maximum(close[1:], EPS) / np.maximum(close[:-1], EPS))

  body = (close - open_) / np.maximum(open_, EPS)
  hl_log = np.log(np.maximum(high, EPS) / np.maximum(low, EPS))
  range_ratio = (high - low) / np.maximum(open_, EPS)

  log_volume = np.log1p(np.maximum(volume, 0.0))
  vol_mean = _rolling_mean(log_volume, vol_window)
  vol_std = _rolling_std(log_volume, vol_window)
  volume_z = (log_volume - vol_mean) / np.maximum(vol_std, 1e-6)
  volume_z = np.nan_to_num(volume_z, nan=0.0, posinf=0.0, neginf=0.0)

  realized_vol = np.sqrt(_rolling_mean(ret_1 * ret_1, vol_window))
  realized_vol = np.nan_to_num(realized_vol, nan=0.0, posinf=0.0, neginf=0.0)

  feat = np.stack([
      ret_1,
      body,
      hl_log,
      range_ratio,
      realized_vol,
      volume_z,
  ], axis=-1).astype(np.float32)

  return {
      'timestamp_ns': timestamp_ns.astype(np.int64),
      'close': close.astype(np.float32),
      'ret_1': ret_1.astype(np.float32)[:, None],
      'feat': feat,
  }


def split_by_time(
    payload: dict[str, np.ndarray],
    train_end: str,
    val_end: str,
) -> dict[str, dict[str, np.ndarray]]:
  train_cut = np.datetime64(train_end, 'ns').astype(np.int64)
  val_cut = np.datetime64(val_end, 'ns').astype(np.int64)
  if val_cut <= train_cut:
    raise ValueError(f'Expected val_end > train_end, got {train_end} and {val_end}')

  ts = payload['timestamp_ns']
  masks = {
      'train': ts < train_cut,
      'val': (ts >= train_cut) & (ts < val_cut),
      'test': ts >= val_cut,
  }

  splits = {}
  for split, mask in masks.items():
    if not mask.any():
      raise ValueError(f'Split {split} is empty; adjust split dates.')
    splits[split] = {k: v[mask] for k, v in payload.items()}
  return splits


def chunk_episodes(
    split_data: dict[str, np.ndarray],
    episode_length: int,
    stride: int,
) -> dict[str, np.ndarray]:
  if episode_length <= 1:
    raise ValueError(f'episode_length must be > 1, got {episode_length}')
  if stride <= 0:
    raise ValueError(f'stride must be > 0, got {stride}')

  total = len(split_data['timestamp_ns'])
  starts = np.arange(0, total - episode_length + 1, stride, dtype=np.int64)
  if starts.size == 0:
    raise ValueError(
        f'Not enough data for one episode: total={total}, episode_length={episode_length}')

  episodes = {}
  for key, arr in split_data.items():
    chunks = [arr[s: s + episode_length] for s in starts]
    episodes[key] = np.stack(chunks, axis=0)

  return episodes


def save_split_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(path, **payload)


def load_split_npz(dataset_dir: str, split: str) -> dict[str, np.ndarray]:
  filename = Path(dataset_dir) / f'{split}.npz'
  if not filename.exists():
    raise FileNotFoundError(f'Missing split file: {filename}')
  with np.load(filename, allow_pickle=False) as data:
    return {k: data[k] for k in data.files}


def prepare_market_dataset(
    input_csv: str,
    outdir: str,
    instrument: str,
    episode_length: int,
    stride: int,
    train_end: str,
    val_end: str,
    vol_window: int = 48,
    columns: ColumnNames = ColumnNames(),
) -> dict[str, int]:
  raw = load_ohlcv_csv(input_csv, columns)
  feat = build_features(raw, vol_window=vol_window)
  splits = split_by_time(feat, train_end=train_end, val_end=val_end)

  outpath = Path(outdir)
  summary = {
      'instrument': instrument,
      'input_csv': str(input_csv),
      'episode_length': int(episode_length),
      'stride': int(stride),
      'train_end': train_end,
      'val_end': val_end,
      'vol_window': int(vol_window),
      'splits': {},
  }

  for split, arrays in splits.items():
    ep = chunk_episodes(arrays, episode_length=episode_length, stride=stride)
    save_split_npz(outpath / f'{split}.npz', ep)
    summary['splits'][split] = {
        'rows': int(len(arrays['timestamp_ns'])),
        'episodes': int(ep['feat'].shape[0]),
        'episode_length': int(ep['feat'].shape[1]),
        'feature_dim': int(ep['feat'].shape[2]),
    }

  (outpath / 'meta.json').write_text(json.dumps(summary, indent=2))
  return {
      split: values['episodes']
      for split, values in summary['splits'].items()
  }
