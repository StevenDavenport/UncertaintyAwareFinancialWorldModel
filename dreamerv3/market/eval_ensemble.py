from __future__ import annotations

import argparse
from pathlib import Path

import elements
import numpy as np
import ruamel.yaml as yaml

from dreamerv3 import main as mainlib
from dreamerv3.market import data as market_data
from dreamerv3.market import inference


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Evaluate ensemble MC abstention gate on market test data.')
  parser.add_argument('--logdirs', nargs='+', required=True)
  parser.add_argument('--dataset_dir', default='')
  parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
  parser.add_argument('--horizon', type=int, default=12)
  parser.add_argument('--samples', type=int, default=64)
  parser.add_argument('--epsilon', type=float, default=0.0)
  parser.add_argument('--cost_buffer', type=float, default=0.0)
  parser.add_argument('--target_key', default='ret_1')
  parser.add_argument('--p_min', type=float, default=0.70)
  parser.add_argument('--disagree_max', type=float, default=0.08)
  parser.add_argument('--var_max', type=float, default=0.0004)
  parser.add_argument('--signal_mode', default='directional', choices=['directional', 'long_only'])
  parser.add_argument('--stop_mult', type=float, default=2.0)
  parser.add_argument('--min_stop', type=float, default=0.0005)
  parser.add_argument('--trade_cost', type=float, default=-1.0)
  parser.add_argument('--vol_feature_index', type=int, default=4)
  parser.add_argument('--bt_initial_capital', type=float, default=1.0)
  parser.add_argument('--bt_risk_fraction', type=float, default=0.01)
  parser.add_argument('--bt_max_positions', type=int, default=20)
  parser.add_argument('--bt_max_gross_leverage', type=float, default=3.0)
  parser.add_argument('--bt_one_position_per_asset', type=int, default=1)
  parser.add_argument('--calibration_bins', type=int, default=10)
  parser.add_argument('--max_episodes', type=int, default=0)
  parser.add_argument('--jax_platform', default='cpu')
  parser.add_argument('--outdir', required=True)
  return parser.parse_args()


def _load_saved_config(logdir: Path) -> dict:
  path = logdir / 'config.yaml'
  if not path.exists():
    raise FileNotFoundError(f'Missing config file in run directory: {path}')
  return yaml.YAML(typ='safe').load(path.read_text())


def _resolve_dataset_dir(config_dict: dict, explicit: str) -> str:
  if explicit:
    return explicit
  return config_dict['env']['market']['dataset_dir']


def _build_agent(logdir: Path, args: argparse.Namespace):
  cfg = _load_saved_config(logdir)
  cfg['task'] = f'market_{args.split}'
  cfg['logdir'] = str(logdir)
  cfg.setdefault('jax', {})
  cfg['jax']['platform'] = args.jax_platform
  cfg['jax']['enable_policy'] = True

  cfg.setdefault('agent', {})
  cfg['agent']['zero_actions'] = True
  cfg['agent'].setdefault('mc_eval', {})
  cfg['agent']['mc_eval']['enabled'] = True
  cfg['agent']['mc_eval']['horizon'] = int(args.horizon)
  cfg['agent']['mc_eval']['samples'] = int(args.samples)
  cfg['agent']['mc_eval']['epsilon'] = float(args.epsilon)
  cfg['agent']['mc_eval']['target'] = str(args.target_key)

  if args.dataset_dir:
    cfg.setdefault('env', {})
    cfg.setdefault('env', {}).setdefault('market', {})
    cfg['env']['market']['dataset_dir'] = args.dataset_dir

  config = elements.Config(cfg)
  agent = mainlib.make_agent(config)

  ckpt_root = logdir / 'ckpt'
  latest_file = ckpt_root / 'latest'
  if not latest_file.exists():
    raise FileNotFoundError(f'Missing checkpoint pointer file: {latest_file}')
  ckpt_name = latest_file.read_text().strip()
  ckpt_path = ckpt_root / ckpt_name
  if not ckpt_path.exists():
    raise FileNotFoundError(f'Checkpoint snapshot does not exist: {ckpt_path}')
  cp = elements.Checkpoint()
  cp.agent = agent
  cp.load(str(ckpt_path), keys=['agent'])
  return agent, cfg


def _collect_rows(
    aggregate: dict[str, np.ndarray],
    outputs: list[dict[str, np.ndarray]],
    gate: dict[str, np.ndarray],
    trade: dict[str, np.ndarray | dict[str, float]] | None,
) -> dict[str, np.ndarray]:
  green = gate['green']
  direction = gate['direction']
  long_green = gate['long']
  short_green = gate['short']
  rows = {
      'episode': aggregate['episode'],
      't': aggregate['t'],
      'timestamp_ns': aggregate['timestamp_ns'],
      'future_return': aggregate['future_return'],
      'y': aggregate['y'].astype(np.int32),
      'y_up': aggregate['y_up'].astype(np.int32),
      'y_down': aggregate['y_down'].astype(np.int32),
      'p_mean': aggregate['p_mean'],
      'p_up_mean': aggregate['p_up_mean'],
      'p_down_mean': aggregate['p_down_mean'],
      'disagree': aggregate['disagree'],
      'disagree_up': aggregate['disagree_up'],
      'disagree_down': aggregate['disagree_down'],
      'var_mean': aggregate['var_mean'],
      'mean_mean': aggregate['mean_mean'],
      'green': green.astype(np.int32),
      'green_long': long_green.astype(np.int32),
      'green_short': short_green.astype(np.int32),
      'direction': direction.astype(np.int32),
  }
  for idx, output in enumerate(outputs):
    rows[f'p_hat_m{idx}'] = output.get('p_up', output['p_hat'])
    if 'p_down' in output:
      rows[f'p_down_m{idx}'] = output['p_down']
    rows[f'var_hat_m{idx}'] = output['var_hat']
    if 'mean_hat' in output:
      rows[f'mean_hat_m{idx}'] = output['mean_hat']
  if trade is not None:
    rows.update({
        'trade': trade['trade'].astype(np.int32),
        'stop_dist': trade['stop_dist'],
        'stop_hit': trade['stop_hit'].astype(np.int32),
        'exit_step': trade['exit_step'],
        'gross_return': trade['gross_return'],
        'net_return': trade['net_return'],
    })
  return rows


def _save_plot(
    filename: Path,
    aggregate: dict[str, np.ndarray],
    gate: dict[str, np.ndarray],
    reliability: dict[str, np.ndarray],
) -> None:
  try:
    import matplotlib.pyplot as plt
  except ImportError as exc:
    raise RuntimeError('matplotlib is required to generate evaluation plots.') from exc

  filename.parent.mkdir(parents=True, exist_ok=True)

  fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

  valid = reliability['count'] > 0
  axes[0].plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=1)
  axes[0].plot(
      reliability['pred_mean'][valid],
      reliability['empirical_pos_rate'][valid],
      marker='o',
      linewidth=1.5,
      color='tab:blue',
  )
  axes[0].set_title('Reliability')
  axes[0].set_xlabel('Predicted p_mean')
  axes[0].set_ylabel('Empirical P(y=1)')
  axes[0].set_xlim(0, 1)
  axes[0].set_ylim(0, 1)

  n = min(500, len(aggregate['p_mean']))
  x = np.arange(n)
  green = gate['green']
  long_green = gate['long']
  short_green = gate['short']
  axes[1].plot(x, aggregate['p_up_mean'][:n], label='p_up_mean', color='tab:blue')
  axes[1].plot(x, aggregate['p_down_mean'][:n], label='p_down_mean', color='tab:red', alpha=0.85)
  axes[1].plot(x, aggregate['y_up'][:n].astype(np.float32), label='y_up', color='tab:orange', alpha=0.7)
  axes[1].plot(x, aggregate['y_down'][:n].astype(np.float32), label='y_down', color='tab:brown', alpha=0.7)
  long_idx = np.where(long_green[:n])[0]
  short_idx = np.where(short_green[:n])[0]
  axes[1].scatter(long_idx, aggregate['p_up_mean'][:n][long_green[:n]], s=10, color='tab:green', label='long')
  axes[1].scatter(short_idx, aggregate['p_down_mean'][:n][short_green[:n]], s=10, color='tab:purple', label='short')
  abstain_idx = np.where(~green[:n])[0]
  if abstain_idx.size:
    axes[1].scatter(
        abstain_idx[: min(100, abstain_idx.size)],
        aggregate['p_up_mean'][:n][~green[:n]][: min(100, abstain_idx.size)],
        s=6,
        color='gray',
        alpha=0.35,
        label='abstain',
    )
  axes[1].set_title('Timeline (first 500 points)')
  axes[1].set_xlabel('Index')
  axes[1].set_ylabel('Value')
  axes[1].set_ylim(-0.05, 1.05)
  axes[1].legend(loc='lower right')

  fig.tight_layout()
  fig.savefig(filename, dpi=150)
  plt.close(fig)


def main() -> None:
  args = parse_args()
  outdir = Path(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)

  logdirs = [Path(x) for x in args.logdirs]
  if not logdirs:
    raise ValueError('Expected at least one logdir.')

  first_cfg = _load_saved_config(logdirs[0])
  dataset_dir = _resolve_dataset_dir(first_cfg, args.dataset_dir)
  payload = market_data.load_split_npz(dataset_dir, args.split)
  instrument_lookup = None
  if 'instrument_id' not in payload:
    inferred_ids, inferred_lookup = market_data.infer_split_instrument_ids(
        dataset_dir, args.split, int(payload['feat'].shape[0]))
    if inferred_ids is not None:
      payload['instrument_id'] = inferred_ids
      instrument_lookup = inferred_lookup

  model_outputs = []
  for logdir in logdirs:
    print(f'Loading model from {logdir}')
    agent, _ = _build_agent(logdir, args)
    output = inference.run_mc_eval_on_payload(
        agent,
        payload,
        horizon=args.horizon,
        epsilon=args.epsilon,
        cost_buffer=args.cost_buffer,
        mode='mc_eval',
        max_episodes=args.max_episodes,
    )
    model_outputs.append(output)

  aggregate = inference.aggregate_ensemble_outputs(model_outputs)
  if args.signal_mode == 'directional':
    gate = inference.apply_directional_gate(
        aggregate['p_up_mean'],
        aggregate['p_down_mean'],
        aggregate['disagree_up'],
        aggregate['disagree_down'],
        aggregate['var_mean'],
        p_min=args.p_min,
        disagree_max=args.disagree_max,
        var_max=args.var_max,
    )
    metrics = inference.compute_directional_gate_metrics(
        aggregate['y_up'], aggregate['y_down'], gate['long'], gate['short'])
  else:
    green = inference.apply_green_gate(
        aggregate['p_mean'],
        aggregate['disagree'],
        aggregate['var_mean'],
        p_min=args.p_min,
        disagree_max=args.disagree_max,
        var_max=args.var_max,
    )
    gate = {
      'green': green,
      'long': green,
      'short': np.zeros_like(green, dtype=bool),
      'direction': green.astype(np.int8),
    }
    metrics = inference.compute_gate_metrics(aggregate['y'], green)

  trade_cost = args.trade_cost
  if trade_cost < 0:
    trade_cost = 2.0 * float(args.cost_buffer)
  trade = inference.simulate_fixed_horizon_policy(
      aggregate=aggregate,
      payload=payload,
      direction=gate['direction'],
      horizon=args.horizon,
      stop_mult=args.stop_mult,
      min_stop=args.min_stop,
      trade_cost=trade_cost,
      vol_feature_index=args.vol_feature_index,
  )
  backtest = inference.run_capital_backtest(
      aggregate=aggregate,
      trade=trade,
      payload=payload,
      initial_capital=args.bt_initial_capital,
      risk_fraction=args.bt_risk_fraction,
      max_positions=args.bt_max_positions,
      max_gross_leverage=args.bt_max_gross_leverage,
      one_position_per_asset=bool(args.bt_one_position_per_asset),
  )
  reliability = inference.reliability_table(
      aggregate['p_mean'], aggregate['y_up'], bins=args.calibration_bins)

  metrics_blob = {
      'split': args.split,
      'models': [str(x) for x in logdirs],
      'dataset_dir': dataset_dir,
      'horizon': args.horizon,
      'samples': args.samples,
      'epsilon': args.epsilon,
      'cost_buffer': args.cost_buffer,
      'signal_mode': args.signal_mode,
      'trade_eval': {
          'horizon': args.horizon,
          'stop_mult': args.stop_mult,
          'min_stop': args.min_stop,
          'trade_cost': trade_cost,
          'vol_feature_index': args.vol_feature_index,
      },
      'portfolio_backtest': {
          'initial_capital': args.bt_initial_capital,
          'risk_fraction': args.bt_risk_fraction,
          'max_positions': args.bt_max_positions,
          'max_gross_leverage': args.bt_max_gross_leverage,
          'one_position_per_asset': bool(args.bt_one_position_per_asset),
      },
      'thresholds': {
          'p_min': args.p_min,
          'disagree_max': args.disagree_max,
          'var_max': args.var_max,
      },
      'metrics': metrics,
      'trade_metrics': trade['metrics'],
      'portfolio_metrics': backtest['metrics'],
  }
  if instrument_lookup:
    metrics_blob['instrument_lookup'] = {
        str(k): v for k, v in instrument_lookup.items()
    }
    if 'asset_id' in backtest['trades']:
      names = []
      for aid in backtest['trades']['asset_id']:
        aid_int = int(aid)
        names.append(instrument_lookup.get(aid_int, f'asset_{aid_int}'))
      backtest['trades']['asset_name'] = np.asarray(names, dtype=object)

  rows = _collect_rows(aggregate, model_outputs, gate, trade)
  inference.save_predictions_csv(outdir / 'predictions.csv', rows)
  inference.save_predictions_csv(outdir / 'backtest_trades.csv', backtest['trades'])
  inference.save_predictions_csv(outdir / 'backtest_equity.csv', backtest['equity'])
  inference.save_reliability_csv(outdir / 'calibration.csv', reliability)
  inference.save_metrics_json(outdir / 'metrics.json', metrics_blob)
  _save_plot(outdir / 'sanity.png', aggregate, gate, reliability)

  print('Saved evaluation artefacts to', outdir)
  print('Metrics:', metrics)
  print('Trade metrics:', trade['metrics'])
  print('Portfolio metrics:', backtest['metrics'])


if __name__ == '__main__':
  main()
