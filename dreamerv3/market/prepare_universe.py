import argparse
from pathlib import Path

from .data import ColumnNames
from .data import prepare_market_universe_dataset


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description='Prepare a multi-ticker market dataset for DreamerV3.')
  parser.add_argument('--input_dir', required=True, help='Directory containing per-ticker CSV files.')
  parser.add_argument(
      '--tickers',
      required=True,
      help='Comma-separated tickers, e.g. QQQ,SPY,AAPL,MSFT')
  parser.add_argument(
      '--csv_pattern',
      default='{ticker}_5m.csv',
      help='Filename pattern resolved under input_dir, default: {ticker}_5m.csv')
  parser.add_argument('--outdir', required=True, help='Output directory for train/val/test NPZ files.')
  parser.add_argument('--episode_length', type=int, default=256)
  parser.add_argument('--stride', type=int, default=256)
  parser.add_argument('--train_end', required=True, help='Train split end timestamp (exclusive).')
  parser.add_argument('--val_end', required=True, help='Validation split end timestamp (exclusive).')
  parser.add_argument('--vol_window', type=int, default=48)
  parser.add_argument('--strict', action='store_true', help='Fail on invalid/missing ticker CSVs.')
  parser.add_argument('--time_col', default='timestamp')
  parser.add_argument('--open_col', default='open')
  parser.add_argument('--high_col', default='high')
  parser.add_argument('--low_col', default='low')
  parser.add_argument('--close_col', default='close')
  parser.add_argument('--volume_col', default='volume')
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
  if not tickers:
    raise ValueError('Expected at least one ticker in --tickers.')

  root = Path(args.input_dir)
  inputs = []
  for ticker in tickers:
    candidates = [
        args.csv_pattern.format(ticker=ticker),
        args.csv_pattern.format(ticker=ticker.lower()),
        args.csv_pattern.format(ticker=ticker.upper()),
    ]
    chosen = None
    for name in candidates:
      path = root / name
      if path.exists():
        chosen = path
        break
    if chosen is None:
      # Keep first candidate path for downstream diagnostics.
      chosen = root / candidates[0]
    inputs.append((ticker, str(chosen)))

  columns = ColumnNames(
      timestamp=args.time_col,
      open=args.open_col,
      high=args.high_col,
      low=args.low_col,
      close=args.close_col,
      volume=args.volume_col,
  )

  counts = prepare_market_universe_dataset(
      inputs=inputs,
      outdir=args.outdir,
      episode_length=args.episode_length,
      stride=args.stride,
      train_end=args.train_end,
      val_end=args.val_end,
      vol_window=args.vol_window,
      columns=columns,
      skip_invalid=(not args.strict),
  )
  if counts['train'] <= 0:
    raise ValueError('No training episodes were prepared. Check ticker CSV paths and split dates.')
  print('Prepared universe dataset episodes:', counts)


if __name__ == '__main__':
  main()
