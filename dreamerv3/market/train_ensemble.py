from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_configs(values: list[str]) -> list[str]:
  configs = []
  for value in values:
    for item in value.split(','):
      item = item.strip()
      if item:
        configs.append(item)
  if not configs:
    raise ValueError('Expected at least one config block.')
  return configs


def _parse_seeds(text: str) -> list[int]:
  seeds = []
  for chunk in text.split(','):
    chunk = chunk.strip()
    if chunk:
      seeds.append(int(chunk))
  if not seeds:
    raise ValueError('Expected at least one seed.')
  return seeds


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Launch multiple Dreamer runs for an ensemble.')
  parser.add_argument('--configs', nargs='+', required=True)
  parser.add_argument('--seeds', required=True, help='Comma-separated seeds, e.g. 0,1,2,3,4')
  parser.add_argument('--logroot', required=True)
  parser.add_argument('--python', default=sys.executable)
  parser.add_argument('--dry_run', action='store_true')
  parser.add_argument('--extra_flags', nargs='*', default=[])
  # Forward any unknown args (for example --jax.platform cpu) to dreamerv3/main.py.
  args, unknown = parser.parse_known_args()
  args.extra_flags = list(args.extra_flags) + list(unknown)
  return args


def main() -> None:
  args = parse_args()
  configs = _parse_configs(args.configs)
  seeds = _parse_seeds(args.seeds)
  logroot = Path(args.logroot)
  logroot.mkdir(parents=True, exist_ok=True)

  for seed in seeds:
    run_logdir = logroot / f'seed_{seed}'
    cmd = [
        args.python,
        'dreamerv3/main.py',
        '--configs',
        *configs,
        '--seed',
        str(seed),
        '--logdir',
        str(run_logdir),
        *args.extra_flags,
    ]
    print('Launching:', ' '.join(cmd))
    if args.dry_run:
      continue
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
  main()
