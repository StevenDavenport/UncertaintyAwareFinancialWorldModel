from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dreamerv3.market.inference import (
    apply_directional_gate,
    compute_directional_gate_metrics,
    compute_gate_metrics,
)


def parse_list(text: str, cast=float) -> list:
  vals = []
  for chunk in text.split(','):
    chunk = chunk.strip()
    if chunk:
      vals.append(cast(chunk))
  if not vals:
    raise ValueError('Expected at least one value.')
  return vals


def load_predictions(path: str) -> dict[str, np.ndarray]:
  rows = []
  with open(path, 'r', newline='') as f:
    reader = csv.DictReader(f)
    fieldnames = set(reader.fieldnames or ())
    for row in reader:
      rows.append(row)
  if not rows:
    raise ValueError(f'No rows found in {path}')

  def col(name, cast=float):
    return np.asarray([cast(r[name]) for r in rows])

  y = np.asarray([bool(int(r['y'])) for r in rows])
  out = {
      'y': y,
      'p_mean': col('p_mean', float),
      'disagree': col('disagree', float),
      'var_mean': col('var_mean', float),
  }
  if {'y_up', 'y_down', 'p_up_mean', 'p_down_mean', 'disagree_up', 'disagree_down'} <= fieldnames:
    out.update({
        'y_up': np.asarray([bool(int(r['y_up'])) for r in rows]),
        'y_down': np.asarray([bool(int(r['y_down'])) for r in rows]),
        'p_up_mean': col('p_up_mean', float),
        'p_down_mean': col('p_down_mean', float),
        'disagree_up': col('disagree_up', float),
        'disagree_down': col('disagree_down', float),
    })
  return out


def sweep(data, p_vals, d_vals, v_vals, signal_mode: str):
  results = []
  y = data['y']
  v = data['var_mean']
  directional_ready = (
      signal_mode == 'directional' and
      all(k in data for k in ('y_up', 'y_down', 'p_up_mean', 'p_down_mean', 'disagree_up', 'disagree_down'))
  )
  p = data['p_mean']
  d = data['disagree']
  for p_min in p_vals:
    for d_max in d_vals:
      for v_max in v_vals:
        if directional_ready:
          gate = apply_directional_gate(
              data['p_up_mean'],
              data['p_down_mean'],
              data['disagree_up'],
              data['disagree_down'],
              v,
              p_min=float(p_min),
              disagree_max=float(d_max),
              var_max=float(v_max),
          )
          mets = compute_directional_gate_metrics(
              data['y_up'], data['y_down'], gate['long'], gate['short'])
          row = {
              'signal_mode': 'directional',
              'p_min': float(p_min),
              'disagree_max': float(d_max),
              'var_max': float(v_max),
              **mets,
          }
        else:
          green = (p >= p_min) & (d <= d_max) & (v <= v_max)
          mets = compute_gate_metrics(y, green)
          row = {
              'signal_mode': 'long_only',
              'p_min': float(p_min),
              'disagree_max': float(d_max),
              'var_max': float(v_max),
              **mets,
          }
        results.append(row)
  return results


def rank_results(results, min_coverage: float):
  valid = [r for r in results if np.isfinite(r['precision']) and r['coverage'] >= min_coverage]
  valid.sort(key=lambda r: (r['precision'], r['coverage'], -r['false_green_rate']), reverse=True)
  return valid


def save_csv(path: Path, rows: list[dict]):
  path.parent.mkdir(parents=True, exist_ok=True)
  if not rows:
    path.write_text('')
    return
  keys = list(rows[0].keys())
  with open(path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)


def save_plot(path: Path, rows: list[dict], topk: int):
  try:
    import matplotlib.pyplot as plt
  except ImportError:
    return
  if not rows:
    return
  top = rows[:topk]
  cov = np.asarray([r['coverage'] for r in top], np.float32)
  pre = np.asarray([r['precision'] for r in top], np.float32)

  path.parent.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(1, 1, figsize=(6, 4))
  ax.scatter(cov, pre, s=28, color='tab:blue')
  for i, r in enumerate(top[:10]):
    ax.annotate(str(i + 1), (cov[i], pre[i]), fontsize=8, xytext=(3, 3), textcoords='offset points')
  ax.set_xlabel('Coverage')
  ax.set_ylabel('Precision')
  ax.set_title('Threshold Sweep (Top Results)')
  ax.set_xlim(-0.02, 1.02)
  ax.set_ylim(-0.02, 1.02)
  fig.tight_layout()
  fig.savefig(path, dpi=150)
  plt.close(fig)


def parse_args():
  p = argparse.ArgumentParser(description='Sweep gate thresholds over predictions.csv')
  p.add_argument('--predictions_csv', required=True)
  p.add_argument('--outdir', required=True)
  p.add_argument('--p_values', default='0.50,0.55,0.60,0.65,0.70')
  p.add_argument('--disagree_values', default='0.05,0.08,0.10,0.15,0.20,0.30')
  p.add_argument('--var_values', default='0.0004,0.001,0.003,0.01,0.03')
  p.add_argument('--signal_mode', default='auto', choices=['auto', 'directional', 'long_only'])
  p.add_argument('--min_coverage', type=float, default=0.01)
  p.add_argument('--topk', type=int, default=25)
  return p.parse_args()


def main():
  args = parse_args()
  outdir = Path(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)

  data = load_predictions(args.predictions_csv)
  p_vals = parse_list(args.p_values, float)
  d_vals = parse_list(args.disagree_values, float)
  v_vals = parse_list(args.var_values, float)

  if args.signal_mode == 'auto':
    signal_mode = 'directional' if 'p_up_mean' in data else 'long_only'
  else:
    signal_mode = args.signal_mode

  all_rows = sweep(data, p_vals, d_vals, v_vals, signal_mode=signal_mode)
  ranked = rank_results(all_rows, min_coverage=args.min_coverage)

  save_csv(outdir / 'sweep_all.csv', all_rows)
  save_csv(outdir / 'sweep_ranked.csv', ranked)
  save_plot(outdir / 'sweep_frontier.png', ranked, topk=args.topk)

  summary = {
      'num_candidates': len(all_rows),
      'num_ranked': len(ranked),
      'signal_mode': signal_mode,
      'min_coverage': args.min_coverage,
      'top': ranked[: min(args.topk, len(ranked))],
  }
  (outdir / 'sweep_summary.json').write_text(json.dumps(summary, indent=2))

  print(f'Saved sweep outputs to {outdir}')
  if ranked:
    best = ranked[0]
    print('Best thresholds:')
    print({k: best[k] for k in ('p_min', 'disagree_max', 'var_max', 'precision', 'coverage', 'false_green_rate', 'greens')})
  else:
    print('No thresholds met min_coverage with finite precision. Consider expanding grid or lowering min_coverage.')


if __name__ == '__main__':
  main()
