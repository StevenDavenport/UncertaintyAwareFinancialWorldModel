import argparse
import json
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


def _normalize_symbol(symbol: str) -> str:
  symbol = symbol.strip().upper()
  # Yahoo uses '-' for share class separator (e.g. BRK-B).
  return symbol.replace('.', '-')


SP100_FALLBACK = [
    'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'AMD', 'AMGN', 'AMT', 'AMZN',
    'AVGO', 'AXP', 'BA', 'BAC', 'BK', 'BKNG', 'BLK', 'BMY', 'BRK-B', 'C',
    'CAT', 'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS',
    'CVX', 'DHR', 'DIS', 'DUK', 'EMR', 'F', 'FDX', 'GD', 'GE', 'GILD',
    'GM', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM',
    'KHC', 'KMI', 'KO', 'LIN', 'LLY', 'LMT', 'LOW', 'MA', 'MCD', 'MDLZ',
    'MDT', 'MET', 'META', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'NEE', 'NFLX',
    'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PM', 'PYPL', 'QCOM', 'RTX',
    'SBUX', 'SCHW', 'SO', 'SPG', 'T', 'TGT', 'TMO', 'TMUS', 'TSLA', 'TXN',
    'UNH', 'UNP', 'UPS', 'USB', 'V', 'VZ', 'WBA', 'WFC', 'WMT', 'XOM',
]


def _read_html_tables(url: str) -> list[pd.DataFrame]:
  headers = {
      'User-Agent': (
          'Mozilla/5.0 (X11; Linux x86_64) '
          'AppleWebKit/537.36 (KHTML, like Gecko) '
          'Chrome/120.0.0.0 Safari/537.36'
      )
  }
  resp = requests.get(url, headers=headers, timeout=20)
  resp.raise_for_status()
  return pd.read_html(StringIO(resp.text))


def _load_sp500() -> list[str]:
  # Table source is intentionally simple and public.
  tables = _read_html_tables('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
  if not tables:
    raise ValueError('Failed to load S&P 500 table.')
  df = tables[0]
  if 'Symbol' not in df.columns:
    raise ValueError('Unexpected S&P 500 table format (missing Symbol).')
  return [_normalize_symbol(x) for x in df['Symbol'].astype(str).tolist()]


def _load_sp100() -> list[str]:
  try:
    tables = _read_html_tables('https://en.wikipedia.org/wiki/S%26P_100')
    if not tables:
      raise ValueError('Failed to load S&P 100 table.')
    best = None
    for table in tables:
      cols = {str(c).strip() for c in table.columns}
      if 'Symbol' in cols:
        best = table
        break
    if best is None:
      raise ValueError('Unexpected S&P 100 table format (missing Symbol).')
    return [_normalize_symbol(x) for x in best['Symbol'].astype(str).tolist()]
  except Exception as err:
    print(f'Warning: failed to fetch S&P 100 constituents live ({err}); using built-in fallback list.')
    return [_normalize_symbol(x) for x in SP100_FALLBACK]


def _resolve_tickers(kind: str, custom: str) -> list[str]:
  if kind == 'custom':
    tickers = [_normalize_symbol(x) for x in custom.split(',') if x.strip()]
    if not tickers:
      raise ValueError('Expected non-empty --tickers for custom universe.')
    return tickers
  if kind == 'sp100':
    return _load_sp100()
  if kind == 'sp500':
    return _load_sp500()
  raise ValueError(f'Unknown universe: {kind}')


def _download_one(
    ticker: str,
    interval: str,
    period: str,
    min_rows: int,
    retries: int,
    pause_sec: float,
) -> pd.DataFrame | None:
  last_err = None
  for _ in range(max(1, retries)):
    try:
      df = yf.download(
          ticker,
          period=period,
          interval=interval,
          auto_adjust=False,
          prepost=False,
          progress=False,
          threads=False,
      )
      if df is None or df.empty:
        last_err = f'empty data for {ticker}'
      else:
        if isinstance(df.columns, pd.MultiIndex):
          df.columns = [c[0] for c in df.columns]
        needed = ['Open', 'High', 'Low', 'Close', 'Volume']
        if any(c not in df.columns for c in needed):
          last_err = f'missing OHLCV cols for {ticker}'
        else:
          out = df[needed].copy().dropna()
          if len(out) < min_rows:
            last_err = f'too few rows for {ticker}: {len(out)}'
          else:
            out = out.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
            })
            idx = out.index
            if getattr(idx, 'tz', None) is not None:
              idx = idx.tz_convert('UTC').tz_localize(None)
            out.insert(0, 'timestamp', idx.strftime('%Y-%m-%dT%H:%M:%S'))
            return out
    except Exception as err:
      last_err = str(err)
    time.sleep(max(0.0, pause_sec))
  print(f'{ticker}: failed ({last_err})')
  return None


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Download a liquid ticker universe into per-ticker 5m CSVs.')
  parser.add_argument('--universe', choices=['sp100', 'sp500', 'custom'], default='sp100')
  parser.add_argument('--tickers', default='', help='Comma-separated symbols for --universe custom.')
  parser.add_argument('--outdir', default='data')
  parser.add_argument('--interval', default='5m')
  parser.add_argument('--period', default='60d')
  parser.add_argument('--max_tickers', type=int, default=0, help='Optional cap; 0 means all.')
  parser.add_argument('--min_rows', type=int, default=1000)
  parser.add_argument('--retries', type=int, default=3)
  parser.add_argument('--pause_sec', type=float, default=0.15)
  parser.add_argument('--skip_existing', action='store_true')
  parser.add_argument('--csv_pattern', default='{ticker}_5m.csv')
  parser.add_argument('--report_json', default='', help='Optional path for download report JSON.')
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  outdir = Path(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)

  tickers = _resolve_tickers(args.universe, args.tickers)
  tickers = sorted(set(tickers))
  if args.max_tickers > 0:
    tickers = tickers[: args.max_tickers]

  report = {
      'universe': args.universe,
      'interval': args.interval,
      'period': args.period,
      'requested': len(tickers),
      'success': [],
      'failed': [],
      'skipped_existing': [],
  }

  for i, ticker in enumerate(tickers, 1):
    path = outdir / args.csv_pattern.format(ticker=ticker)
    if args.skip_existing and path.exists():
      report['skipped_existing'].append(ticker)
      print(f'[{i}/{len(tickers)}] {ticker}: skipped existing')
      continue

    df = _download_one(
        ticker=ticker,
        interval=args.interval,
        period=args.period,
        min_rows=args.min_rows,
        retries=args.retries,
        pause_sec=args.pause_sec,
    )
    if df is None:
      report['failed'].append(ticker)
      continue
    df.to_csv(path, index=False)
    report['success'].append(ticker)
    print(f'[{i}/{len(tickers)}] {ticker}: wrote {len(df)} rows -> {path}')

  report['num_success'] = len(report['success'])
  report['num_failed'] = len(report['failed'])
  report['num_skipped_existing'] = len(report['skipped_existing'])
  print('Download summary:', {
      'requested': report['requested'],
      'success': report['num_success'],
      'failed': report['num_failed'],
      'skipped_existing': report['num_skipped_existing'],
  })

  report_path = Path(args.report_json) if args.report_json else (outdir / f'universe_{args.universe}_download_report.json')
  report_path.write_text(json.dumps(report, indent=2))
  print(f'Report saved to {report_path}')


if __name__ == '__main__':
  main()
