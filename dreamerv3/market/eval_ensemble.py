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
    green: np.ndarray,
) -> dict[str, np.ndarray]:
  rows = {
      'episode': aggregate['episode'],
      't': aggregate['t'],
      'timestamp_ns': aggregate['timestamp_ns'],
      'future_return': aggregate['future_return'],
      'y': aggregate['y'].astype(np.int32),
      'p_mean': aggregate['p_mean'],
      'disagree': aggregate['disagree'],
      'var_mean': aggregate['var_mean'],
      'green': green.astype(np.int32),
  }
  for idx, output in enumerate(outputs):
    rows[f'p_hat_m{idx}'] = output['p_hat']
    rows[f'var_hat_m{idx}'] = output['var_hat']
  return rows


def _save_plot(
    filename: Path,
    aggregate: dict[str, np.ndarray],
    green: np.ndarray,
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
  axes[1].plot(x, aggregate['p_mean'][:n], label='p_mean', color='tab:blue')
  axes[1].plot(x, aggregate['y'][:n].astype(np.float32), label='y', color='tab:orange', alpha=0.7)
  green_idx = np.where(green[:n])[0]
  axes[1].scatter(green_idx, aggregate['p_mean'][:n][green[:n]], s=10, color='tab:green', label='green')
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
  green = inference.apply_green_gate(
      aggregate['p_mean'],
      aggregate['disagree'],
      aggregate['var_mean'],
      p_min=args.p_min,
      disagree_max=args.disagree_max,
      var_max=args.var_max,
  )

  metrics = inference.compute_gate_metrics(aggregate['y'], green)
  reliability = inference.reliability_table(
      aggregate['p_mean'], aggregate['y'], bins=args.calibration_bins)

  metrics_blob = {
      'split': args.split,
      'models': [str(x) for x in logdirs],
      'dataset_dir': dataset_dir,
      'horizon': args.horizon,
      'samples': args.samples,
      'epsilon': args.epsilon,
      'cost_buffer': args.cost_buffer,
      'thresholds': {
          'p_min': args.p_min,
          'disagree_max': args.disagree_max,
          'var_max': args.var_max,
      },
      'metrics': metrics,
  }

  rows = _collect_rows(aggregate, model_outputs, green)
  inference.save_predictions_csv(outdir / 'predictions.csv', rows)
  inference.save_reliability_csv(outdir / 'calibration.csv', reliability)
  inference.save_metrics_json(outdir / 'metrics.json', metrics_blob)
  _save_plot(outdir / 'sanity.png', aggregate, green, reliability)

  print('Saved evaluation artefacts to', outdir)
  print('Metrics:', metrics)


if __name__ == '__main__':
  main()
