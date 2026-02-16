from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description='Build a frontend JSON bundle from one or more eval output directories.')
  parser.add_argument(
      '--eval_dirs',
      nargs='+',
      required=True,
      help='List of eval dirs. Supports label=path format.')
  parser.add_argument(
      '--out_json',
      default='frontend/data/bundle.json',
      help='Output JSON bundle path.')
  parser.add_argument(
      '--max_rows_per_run',
      type=int,
      default=0,
      help='Optional cap per run. 0 keeps all rows.')
  return parser.parse_args()


def parse_eval_spec(spec: str) -> tuple[str, Path]:
  if '=' in spec:
    label, path = spec.split('=', 1)
    label = label.strip()
    path = path.strip()
    if not label:
      raise ValueError(f'Invalid labeled spec (empty label): {spec}')
    return label, Path(path)
  path = Path(spec)
  label = path.name
  return label, path


def load_json(path: Path) -> dict:
  if not path.exists():
    raise FileNotFoundError(path)
  with path.open('r') as f:
    return json.load(f)


def parse_float(value: str, default: float = 0.0) -> float:
  try:
    return float(value)
  except Exception:
    return float(default)


def parse_int(value: str, default: int = 0) -> int:
  try:
    return int(float(value))
  except Exception:
    return int(default)


def build_episode_asset_lookup(
    dataset_dir: str | None,
    split: str | None,
    instrument_lookup: dict[str, str] | None,
) -> list[str]:
  if not dataset_dir or not split:
    return []
  meta_path = Path(dataset_dir).expanduser() / 'meta.json'
  if not meta_path.exists():
    return []
  try:
    meta = json.loads(meta_path.read_text())
  except Exception:
    return []
  per_instrument = meta.get('per_instrument')
  if not isinstance(per_instrument, dict):
    return []

  out: list[str] = []
  for inst_id, (inst_name, info) in enumerate(per_instrument.items()):
    try:
      count = int(info['splits'][split]['episodes'])
    except Exception:
      count = 0
    if count <= 0:
      continue
    if instrument_lookup:
      resolved = instrument_lookup.get(str(inst_id), str(inst_name))
    else:
      resolved = str(inst_name)
    out.extend([resolved] * count)
  return out


def load_predictions_rows(path: Path, max_rows: int) -> list[dict]:
  if not path.exists():
    raise FileNotFoundError(path)
  rows = []
  with path.open('r', newline='') as f:
    reader = csv.DictReader(f)
    fieldnames = set(reader.fieldnames or ())
    required = {'timestamp_ns', 'future_return', 'p_mean', 'disagree', 'var_mean'}
    missing = required - set(reader.fieldnames or ())
    if missing:
      raise ValueError(f'{path}: missing required columns: {sorted(missing)}')
    for row in reader:
      parsed = {
          'timestamp_ns': parse_int(row.get('timestamp_ns', '0')),
          'future_return': parse_float(row.get('future_return', '0')),
          'p_mean': parse_float(row.get('p_mean', '0')),
          'disagree': parse_float(row.get('disagree', '0')),
          'var_mean': parse_float(row.get('var_mean', '0')),
      }
      if 'episode' in fieldnames:
        parsed['episode'] = parse_int(row.get('episode', '-1'), -1)
      if 't' in fieldnames:
        parsed['t'] = parse_int(row.get('t', '-1'), -1)
      if 'direction' in fieldnames:
        parsed['direction'] = parse_int(row.get('direction', '0'), 0)
      if 'green' in fieldnames:
        parsed['green'] = parse_int(row.get('green', '0'), 0)
      if 'p_up_mean' in fieldnames:
        parsed['p_up_mean'] = parse_float(row.get('p_up_mean', '0'))
      if 'p_down_mean' in fieldnames:
        parsed['p_down_mean'] = parse_float(row.get('p_down_mean', '0'))
      rows.append(parsed)
      if max_rows > 0 and len(rows) >= max_rows:
        break
  return rows


def load_trade_rows(path: Path) -> list[dict]:
  if not path.exists():
    return []
  rows = []
  with path.open('r', newline='') as f:
    reader = csv.DictReader(f)
    fieldnames = set(reader.fieldnames or ())
    for row in reader:
      parsed = {
          'trade_id': parse_int(row.get('trade_id', '-1'), -1),
          'asset_name': str(row.get('asset_name', '')).strip(),
          'asset_id': parse_int(row.get('asset_id', '-1'), -1),
          'direction': parse_int(row.get('direction', '0'), 0),
          'entry_timestamp_ns': parse_int(row.get('entry_timestamp_ns', '0'), 0),
          'exit_timestamp_ns': parse_int(row.get('exit_timestamp_ns', '0'), 0),
          'bars_held': parse_int(row.get('bars_held', '0'), 0),
          'notional': parse_float(row.get('notional', '0')),
          'net_return': parse_float(row.get('net_return', '0')),
          'pnl': parse_float(row.get('pnl', '0')),
      }
      if 'stop_dist' in fieldnames:
        parsed['stop_dist'] = parse_float(row.get('stop_dist', '0'))
      if 'signal_score' in fieldnames:
        parsed['signal_score'] = parse_float(row.get('signal_score', '0'))
      rows.append(parsed)
  return rows


def load_equity_rows(path: Path, max_rows: int = 0) -> list[dict]:
  if not path.exists():
    return []
  rows = []
  with path.open('r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
      rows.append({
          'timestamp_ns': parse_int(row.get('timestamp_ns', '0')),
          'equity': parse_float(row.get('equity', '0')),
          'open_positions': parse_int(row.get('open_positions', '0')),
          'open_notional': parse_float(row.get('open_notional', '0')),
      })
      if max_rows > 0 and len(rows) >= max_rows:
        break
  return rows


def build_run(label: str, eval_dir: Path, max_rows: int) -> dict:
  metrics_blob = load_json(eval_dir / 'metrics.json')
  thresholds = metrics_blob.get('thresholds', {})
  metrics = metrics_blob.get('metrics', {})
  trade_metrics = metrics_blob.get('trade_metrics', {})
  portfolio_metrics = metrics_blob.get('portfolio_metrics', {})
  instrument_lookup = metrics_blob.get('instrument_lookup', {})
  if not isinstance(instrument_lookup, dict):
    instrument_lookup = {}
  rows = load_predictions_rows(eval_dir / 'predictions.csv', max_rows=max_rows)
  if not rows:
    raise ValueError(f'{eval_dir}: predictions.csv has no rows.')
  trades = load_trade_rows(eval_dir / 'backtest_trades.csv')
  equity = load_equity_rows(eval_dir / 'backtest_equity.csv')

  dataset_dir = metrics_blob.get('dataset_dir', '')
  split = metrics_blob.get('split', '')
  episode_assets = build_episode_asset_lookup(
      dataset_dir=dataset_dir,
      split=split,
      instrument_lookup=instrument_lookup,
  )
  assets = sorted({r['asset_name'] for r in trades if r.get('asset_name')})

  return {
      'id': label,
      'source_dir': str(eval_dir),
      'split': split,
      'horizon': int(metrics_blob.get('horizon', 0)),
      'samples': int(metrics_blob.get('samples', 0)),
      'epsilon': float(metrics_blob.get('epsilon', 0.0)),
      'cost_buffer': float(metrics_blob.get('cost_buffer', 0.0)),
      'dataset_dir': str(dataset_dir),
      'thresholds': {
          'p_min': float(thresholds.get('p_min', 0.5)),
          'disagree_max': float(thresholds.get('disagree_max', 0.1)),
          'var_max': float(thresholds.get('var_max', 0.4)),
      },
      'baseline_metrics': metrics,
      'trade_metrics': trade_metrics,
      'portfolio_metrics': portfolio_metrics,
      'assets': assets,
      'instrument_lookup': instrument_lookup,
      'episode_assets': episode_assets,
      'rows': rows,
      'trades': trades,
      'equity': equity,
  }


def main() -> None:
  args = parse_args()
  runs = []
  for spec in args.eval_dirs:
    label, eval_dir = parse_eval_spec(spec)
    run = build_run(label, eval_dir.expanduser(), max_rows=args.max_rows_per_run)
    runs.append(run)

  runs.sort(key=lambda r: (r['horizon'], r['id']))
  bundle = {
      'project': 'UncertaintyAwareFinancialWorldModel',
      'generated_at_utc': dt.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z',
      'num_runs': len(runs),
      'runs': runs,
  }

  out = Path(args.out_json).expanduser()
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(bundle, indent=2))
  print(f'Wrote frontend bundle: {out}')
  print(f'Runs: {[r["id"] for r in runs]}')


if __name__ == '__main__':
  main()
