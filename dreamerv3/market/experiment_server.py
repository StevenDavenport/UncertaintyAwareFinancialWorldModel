from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs
from urllib.parse import urlparse

from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer

import numpy as np


@dataclass
class Job:
  job_id: str
  kind: str
  cmd: list[str]
  cwd: str
  log_path: str
  created_ts: float
  status: str
  returncode: int | None = None
  ended_ts: float | None = None
  process: subprocess.Popen | None = None


class JobStore:

  def __init__(self):
    self._lock = threading.Lock()
    self._jobs: dict[str, Job] = {}

  def create(self, job: Job):
    with self._lock:
      self._jobs[job.job_id] = job

  def get(self, job_id: str) -> Job | None:
    with self._lock:
      return self._jobs.get(job_id)

  def all(self) -> list[Job]:
    with self._lock:
      return list(self._jobs.values())

  def refresh(self):
    with self._lock:
      for job in self._jobs.values():
        if job.process and job.status == 'running':
          ret = job.process.poll()
          if ret is not None:
            job.returncode = int(ret)
            job.ended_ts = time.time()
            job.status = 'succeeded' if ret == 0 else 'failed'


def _float(val: Any, default: float) -> str:
  try:
    return str(float(val))
  except Exception:
    return str(default)


def _int(val: Any, default: int) -> str:
  try:
    return str(int(val))
  except Exception:
    return str(default)


def _str(val: Any, default: str = '') -> str:
  if val is None:
    return default
  return str(val)


def _paths_list(lines: Any) -> list[str]:
  if isinstance(lines, list):
    raw = lines
  else:
    raw = _str(lines).splitlines()
  out = []
  for item in raw:
    item = _str(item).strip()
    if item:
      out.append(str(Path(item).expanduser()))
  return out


def build_eval_cmd(payload: dict[str, Any]) -> list[str]:
  logdirs = _paths_list(payload.get('logdirs', []))
  if not logdirs:
    raise ValueError('eval: expected at least one logdir')
  dataset_dir = _str(payload.get('dataset_dir')).strip()
  if not dataset_dir:
    raise ValueError('eval: dataset_dir is required')
  outdir = _str(payload.get('outdir')).strip()
  if not outdir:
    raise ValueError('eval: outdir is required')

  cmd = [
      'python', '-m', 'dreamerv3.market.eval_ensemble',
      '--logdirs', *logdirs,
      '--dataset_dir', dataset_dir,
      '--split', _str(payload.get('split', 'test')),
      '--horizon', _int(payload.get('horizon', 12), 12),
      '--samples', _int(payload.get('samples', 64), 64),
      '--epsilon', _float(payload.get('epsilon', 0.0005), 0.0005),
      '--cost_buffer', _float(payload.get('cost_buffer', 0.0002), 0.0002),
      '--p_min', _float(payload.get('p_min', 0.50), 0.50),
      '--disagree_max', _float(payload.get('disagree_max', 0.12), 0.12),
      '--var_max', _float(payload.get('var_max', 0.40), 0.40),
      '--signal_mode', _str(payload.get('signal_mode', 'directional')),
      '--stop_mult', _float(payload.get('stop_mult', 2.0), 2.0),
      '--min_stop', _float(payload.get('min_stop', 0.0005), 0.0005),
      '--trade_cost', _float(payload.get('trade_cost', -1.0), -1.0),
      '--bt_initial_capital', _float(payload.get('bt_initial_capital', 1.0), 1.0),
      '--bt_risk_fraction', _float(payload.get('bt_risk_fraction', 0.01), 0.01),
      '--bt_max_positions', _int(payload.get('bt_max_positions', 20), 20),
      '--bt_max_gross_leverage', _float(payload.get('bt_max_gross_leverage', 3.0), 3.0),
      '--bt_one_position_per_asset', _int(payload.get('bt_one_position_per_asset', 1), 1),
      '--outdir', str(Path(outdir).expanduser()),
  ]
  max_eps = _int(payload.get('max_episodes', 0), 0)
  if int(max_eps) > 0:
    cmd += ['--max_episodes', max_eps]
  return cmd


def build_sweep_cmd(payload: dict[str, Any]) -> list[str]:
  predictions_csv = _str(payload.get('predictions_csv')).strip()
  outdir = _str(payload.get('outdir')).strip()
  if not predictions_csv or not outdir:
    raise ValueError('sweep: predictions_csv and outdir are required')
  cmd = [
      'python', '-m', 'dreamerv3.market.sweep_thresholds',
      '--predictions_csv', str(Path(predictions_csv).expanduser()),
      '--outdir', str(Path(outdir).expanduser()),
      '--p_values', _str(payload.get('p_values', '0.35,0.40,0.45,0.50,0.55,0.60,0.65')),
      '--disagree_values', _str(payload.get('disagree_values', '0.04,0.06,0.08,0.10,0.12,0.15,0.20')),
      '--var_values', _str(payload.get('var_values', '0.10,0.15,0.20,0.25,0.30,0.40,0.60')),
      '--min_coverage', _float(payload.get('min_coverage', 0.01), 0.01),
  ]
  topk = _int(payload.get('topk', 40), 40)
  if int(topk) > 0:
    cmd += ['--topk', topk]
  return cmd


def build_bundle_cmd(payload: dict[str, Any]) -> list[str]:
  eval_dirs = _paths_list(payload.get('eval_dirs', []))
  out_json = _str(payload.get('out_json')).strip()
  if not eval_dirs:
    raise ValueError('bundle: expected at least one eval dir')
  if not out_json:
    raise ValueError('bundle: out_json is required')
  cmd = [
      'python', '-m', 'dreamerv3.market.build_frontend_bundle',
      '--eval_dirs', *eval_dirs,
      '--out_json', str(Path(out_json).expanduser()),
  ]
  max_rows = _int(payload.get('max_rows_per_run', 0), 0)
  if int(max_rows) > 0:
    cmd += ['--max_rows_per_run', max_rows]
  return cmd


def build_train_cmd(payload: dict[str, Any]) -> list[str]:
  configs = _str(payload.get('configs', 'market_tiny')).split()
  seeds = _str(payload.get('seeds', '0,1,2'))
  logroot = _str(payload.get('logroot')).strip()
  if not logroot:
    raise ValueError('train: logroot is required')

  cmd = [
      'python', '-m', 'dreamerv3.market.train_ensemble',
      '--configs', *configs,
      '--seeds', seeds,
      '--logroot', str(Path(logroot).expanduser()),
  ]
  dataset_dir = _str(payload.get('dataset_dir')).strip()
  if dataset_dir:
    cmd += ['--env.market.dataset_dir', dataset_dir]
  jax_platform = _str(payload.get('jax_platform')).strip()
  if jax_platform:
    cmd += ['--jax.platform', jax_platform]
  prealloc = _str(payload.get('jax_prealloc')).strip()
  if prealloc:
    cmd += ['--jax.prealloc', prealloc]
  return cmd


class Handler(BaseHTTPRequestHandler):
  server_version = 'FinWMRunner/0.1'

  def log_message(self, format: str, *args):
    return

  @property
  def ctx(self):
    return self.server.context  # type: ignore[attr-defined]

  def _send_json(self, code: int, payload: dict[str, Any]):
    blob = json.dumps(payload).encode('utf-8')
    self.send_response(code)
    self.send_header('Content-Type', 'application/json; charset=utf-8')
    self.send_header('Content-Length', str(len(blob)))
    self.end_headers()
    self.wfile.write(blob)

  def _read_json_body(self) -> dict[str, Any]:
    length = int(self.headers.get('Content-Length', '0') or 0)
    raw = self.rfile.read(length) if length > 0 else b'{}'
    try:
      obj = json.loads(raw.decode('utf-8'))
      if isinstance(obj, dict):
        return obj
    except Exception:
      pass
    raise ValueError('Invalid JSON body')

  def _serve_file(self, path: Path):
    if not path.exists() or not path.is_file():
      self.send_error(404, 'Not found')
      return
    ctype, _ = mimetypes.guess_type(str(path))
    ctype = ctype or 'application/octet-stream'
    data = path.read_bytes()
    self.send_response(200)
    self.send_header('Content-Type', ctype)
    self.send_header('Content-Length', str(len(data)))
    self.end_headers()
    self.wfile.write(data)

  def _job_to_dict(self, job: Job, include_tail: int = 0) -> dict[str, Any]:
    self.ctx.store.refresh()
    payload = {
        'job_id': job.job_id,
        'kind': job.kind,
        'cmd': job.cmd,
        'cwd': job.cwd,
        'status': job.status,
        'returncode': job.returncode,
        'created_ts': job.created_ts,
        'ended_ts': job.ended_ts,
        'log_path': job.log_path,
    }
    if include_tail > 0:
      payload['log_tail'] = self._read_log_tail(Path(job.log_path), include_tail)
    return payload

  def _read_log_tail(self, path: Path, lines: int) -> str:
    if not path.exists():
      return ''
    try:
      with path.open('r', errors='replace') as f:
        data = f.readlines()
      return ''.join(data[-lines:])
    except Exception:
      return ''

  def _launch_job(self, kind: str, cmd: list[str]) -> Job:
    job_id = f'{kind}-{uuid.uuid4().hex[:10]}'
    jobs_dir = self.ctx.jobs_dir
    jobs_dir.mkdir(parents=True, exist_ok=True)
    log_path = jobs_dir / f'{job_id}.log'
    logf = log_path.open('w')
    proc = subprocess.Popen(
        cmd,
        cwd=self.ctx.repo_root,
        stdout=logf,
        stderr=subprocess.STDOUT,
        text=True,
    )
    job = Job(
        job_id=job_id,
        kind=kind,
        cmd=cmd,
        cwd=str(self.ctx.repo_root),
        log_path=str(log_path),
        created_ts=time.time(),
        status='running',
        process=proc,
    )
    self.ctx.store.create(job)
    return job

  def do_GET(self):
    parsed = urlparse(self.path)
    path = parsed.path

    if path == '/api/health':
      self._send_json(200, {'ok': True, 'repo_root': str(self.ctx.repo_root)})
      return

    if path == '/api/runs':
      self.ctx.store.refresh()
      jobs = sorted(self.ctx.store.all(), key=lambda j: j.created_ts, reverse=True)
      self._send_json(200, {'runs': [self._job_to_dict(j) for j in jobs]})
      return

    if path.startswith('/api/runs/'):
      job_id = path.split('/')[3] if len(path.split('/')) >= 4 else ''
      job = self.ctx.store.get(job_id)
      if not job:
        self._send_json(404, {'error': 'Unknown job id'})
        return
      query = parse_qs(parsed.query)
      tail = int(query.get('tail', ['0'])[0] or 0)
      self._send_json(200, self._job_to_dict(job, include_tail=max(0, tail)))
      return

    if path == '/api/market/tickers':
      tickers = self.ctx.list_market_tickers()
      self._send_json(200, {'tickers': tickers})
      return

    if path == '/api/market/candles':
      query = parse_qs(parsed.query)
      ticker = _str(query.get('ticker', [''])[0]).strip()
      if not ticker:
        self._send_json(400, {'error': 'ticker query param is required'})
        return
      def _to_int(raw: str, default: int = 0) -> int:
        raw = str(raw).strip()
        if not raw:
          return default
        try:
          return int(float(raw))
        except Exception:
          return default
      start = _to_int(query.get('start_ns', [''])[0], 0)
      end = _to_int(query.get('end_ns', [''])[0], 0)
      limit = _to_int(query.get('limit', ['0'])[0], 0)

      try:
        candles = self.ctx.load_market_candles(ticker)
      except Exception as err:
        self._send_json(400, {'error': str(err)})
        return

      if start > 0:
        candles = [c for c in candles if int(c['timestamp_ns']) >= start]
      if end > 0:
        candles = [c for c in candles if int(c['timestamp_ns']) <= end]
      if limit > 0 and len(candles) > limit:
        candles = candles[-limit:]

      self._send_json(200, {'ticker': ticker, 'count': len(candles), 'candles': candles})
      return

    # Static frontend serving.
    if path == '/':
      self._serve_file(self.ctx.frontend_dir / 'index.html')
      return
    rel = path.lstrip('/')
    target = (self.ctx.frontend_dir / rel).resolve()
    if not str(target).startswith(str(self.ctx.frontend_dir.resolve())):
      self.send_error(403, 'Forbidden')
      return
    self._serve_file(target)

  def do_POST(self):
    parsed = urlparse(self.path)
    path = parsed.path

    try:
      payload = self._read_json_body()
    except Exception as err:
      self._send_json(400, {'error': str(err)})
      return

    if path == '/api/runs/stop':
      job_id = _str(payload.get('job_id')).strip()
      job = self.ctx.store.get(job_id)
      if not job:
        self._send_json(404, {'error': 'Unknown job id'})
        return
      if job.process and job.status == 'running':
        job.process.terminate()
        time.sleep(0.2)
        if job.process.poll() is None:
          job.process.kill()
      self.ctx.store.refresh()
      self._send_json(200, {'ok': True, 'job': self._job_to_dict(job)})
      return

    builder_map = {
        '/api/run/eval': ('eval', build_eval_cmd),
        '/api/run/sweep': ('sweep', build_sweep_cmd),
        '/api/run/bundle': ('bundle', build_bundle_cmd),
        '/api/run/train': ('train', build_train_cmd),
    }
    if path not in builder_map:
      self._send_json(404, {'error': 'Unknown endpoint'})
      return

    kind, builder = builder_map[path]
    try:
      cmd = builder(payload)
      job = self._launch_job(kind, cmd)
      self._send_json(200, {'ok': True, 'job': self._job_to_dict(job)})
    except Exception as err:
      self._send_json(400, {'error': str(err)})


class Context:
  def __init__(
      self,
      repo_root: Path,
      frontend_dir: Path,
      jobs_dir: Path,
      market_data_dir: Path,
      market_csv_pattern: str,
  ):
    self.repo_root = repo_root
    self.frontend_dir = frontend_dir
    self.jobs_dir = jobs_dir
    self.market_data_dir = market_data_dir
    self.market_csv_pattern = market_csv_pattern
    self.store = JobStore()
    self._candle_cache: dict[str, list[dict[str, float | int]]] = {}
    self._cache_lock = threading.Lock()

  def _ticker_csv_path(self, ticker: str) -> Path:
    name = self.market_csv_pattern.replace('{ticker}', ticker)
    return (self.market_data_dir / name).resolve()

  def list_market_tickers(self) -> list[str]:
    pat = self.market_csv_pattern
    if '{ticker}' not in pat:
      return []
    glob_pat = pat.replace('{ticker}', '*')
    out = []
    for path in self.market_data_dir.glob(glob_pat):
      name = path.name
      prefix, suffix = pat.split('{ticker}', 1)
      if not name.startswith(prefix) or not name.endswith(suffix):
        continue
      ticker = name[len(prefix): len(name) - len(suffix) if suffix else None]
      if ticker:
        out.append(ticker)
    return sorted(set(out))

  def load_market_candles(self, ticker: str) -> list[dict[str, float | int]]:
    key = ticker.strip()
    with self._cache_lock:
      cached = self._candle_cache.get(key)
    if cached is not None:
      return cached

    path = self._ticker_csv_path(key)
    if not path.exists():
      raise FileNotFoundError(f'CSV not found for ticker {key}: {path}')

    with path.open('r', newline='') as f:
      reader = csv.DictReader(f)
      fields = reader.fieldnames or []
      lower = {name.lower(): name for name in fields}

      def col(*names: str) -> str | None:
        for name in names:
          if name in lower:
            return lower[name]
        return None

      ts_col = col('timestamp', 'datetime', 'date', 'time')
      o_col = col('open')
      h_col = col('high')
      l_col = col('low')
      c_col = col('close')
      v_col = col('volume')
      if not (ts_col and o_col and h_col and l_col and c_col):
        raise ValueError(
            f'{path}: expected timestamp/open/high/low/close columns, found {fields}')

      rows: list[dict[str, float | int]] = []
      for row in reader:
        try:
          ts_ns = int(np.datetime64(str(row[ts_col]), 'ns').astype(np.int64))
          o = float(row[o_col])
          h = float(row[h_col])
          l = float(row[l_col])
          c = float(row[c_col])
          v = float(row[v_col]) if v_col else 0.0
        except Exception:
          continue
        rows.append({
            'timestamp_ns': ts_ns,
            'open': o,
            'high': h,
            'low': l,
            'close': c,
            'volume': v,
        })
    rows.sort(key=lambda r: int(r['timestamp_ns']))
    with self._cache_lock:
      self._candle_cache[key] = rows
    return rows


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Serve FinWM frontend and run experiments via local API.')
  parser.add_argument('--host', default='127.0.0.1')
  parser.add_argument('--port', type=int, default=8000)
  parser.add_argument('--repo_root', default='.')
  parser.add_argument('--frontend_dir', default='frontend')
  parser.add_argument('--jobs_dir', default='frontend/runs')
  parser.add_argument('--market_data_dir', default='data')
  parser.add_argument('--market_csv_pattern', default='{ticker}_5m.csv')
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  repo_root = Path(args.repo_root).expanduser().resolve()
  frontend_dir = (repo_root / args.frontend_dir).resolve()
  jobs_dir = (repo_root / args.jobs_dir).resolve()
  market_data_dir = (repo_root / args.market_data_dir).resolve()
  if not frontend_dir.exists():
    raise FileNotFoundError(f'Frontend directory not found: {frontend_dir}')

  ctx = Context(
      repo_root=repo_root,
      frontend_dir=frontend_dir,
      jobs_dir=jobs_dir,
      market_data_dir=market_data_dir,
      market_csv_pattern=args.market_csv_pattern,
  )
  server = ThreadingHTTPServer((args.host, args.port), Handler)
  server.context = ctx  # type: ignore[attr-defined]
  print(f'FinWM server running at http://{args.host}:{args.port}')
  print(f'Frontend: {frontend_dir}')
  print(f'Jobs logs: {jobs_dir}')
  print(f'Market data dir: {market_data_dir}')
  server.serve_forever()


if __name__ == '__main__':
  main()
