import argparse

from .data import ColumnNames
from .data import prepare_market_dataset


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Prepare market dataset splits for DreamerV3.')
  parser.add_argument('--input_csv', required=True, help='Input OHLCV CSV path.')
  parser.add_argument('--outdir', required=True, help='Output directory for train/val/test NPZ files.')
  parser.add_argument('--instrument', default='QQQ', help='Instrument identifier for metadata.')
  parser.add_argument('--episode_length', type=int, default=256)
  parser.add_argument('--stride', type=int, default=256)
  parser.add_argument('--train_end', required=True, help='Train split end timestamp (exclusive).')
  parser.add_argument('--val_end', required=True, help='Validation split end timestamp (exclusive).')
  parser.add_argument('--vol_window', type=int, default=48)
  parser.add_argument('--time_col', default='timestamp')
  parser.add_argument('--open_col', default='open')
  parser.add_argument('--high_col', default='high')
  parser.add_argument('--low_col', default='low')
  parser.add_argument('--close_col', default='close')
  parser.add_argument('--volume_col', default='volume')
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  columns = ColumnNames(
      timestamp=args.time_col,
      open=args.open_col,
      high=args.high_col,
      low=args.low_col,
      close=args.close_col,
      volume=args.volume_col,
  )
  counts = prepare_market_dataset(
      input_csv=args.input_csv,
      outdir=args.outdir,
      instrument=args.instrument,
      episode_length=args.episode_length,
      stride=args.stride,
      train_end=args.train_end,
      val_end=args.val_end,
      vol_window=args.vol_window,
      columns=columns,
  )
  print('Prepared dataset episodes:', counts)


if __name__ == '__main__':
  main()
