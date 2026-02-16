'use strict';

const state = {
  bundle: null,
  runs: [],
  activeRun: null,
  activeAsset: '',
  plotlyResizeBound: false,
  candlesByTicker: new Map(),
  renderToken: 0,
};

const el = {
  tabDashboardBtn: document.getElementById('tabDashboardBtn'),
  tabRunnerBtn: document.getElementById('tabRunnerBtn'),
  tabDashboard: document.getElementById('tabDashboard'),
  tabRunner: document.getElementById('tabRunner'),
  loadDefaultBtn: document.getElementById('loadDefaultBtn'),
  fileInput: document.getElementById('fileInput'),
  runSelect: document.getElementById('runSelect'),
  assetSelect: document.getElementById('assetSelect'),
  candleLimit: document.getElementById('candleLimit'),
  reloadCandlesBtn: document.getElementById('reloadCandlesBtn'),
  showSignalsChk: document.getElementById('showSignalsChk'),
  showTradesChk: document.getElementById('showTradesChk'),
  runMeta: document.getElementById('runMeta'),
  chartStatus: document.getElementById('chartStatus'),
  pMin: document.getElementById('pMin'),
  disagreeMax: document.getElementById('disagreeMax'),
  varMax: document.getElementById('varMax'),
  epsilon: document.getElementById('epsilon'),
  costBuffer: document.getElementById('costBuffer'),
  bins: document.getElementById('bins'),
  pMinVal: document.getElementById('pMinVal'),
  disagreeVal: document.getElementById('disagreeVal'),
  varVal: document.getElementById('varVal'),
  binsVal: document.getElementById('binsVal'),
  resetBtn: document.getElementById('resetBtn'),
  precision: document.getElementById('precision'),
  coverage: document.getElementById('coverage'),
  recall: document.getElementById('recall'),
  falseGreen: document.getElementById('falseGreen'),
  greens: document.getElementById('greens'),
  count: document.getElementById('count'),
  pfReturn: document.getElementById('pfReturn'),
  pfMdd: document.getElementById('pfMdd'),
  pfTrades: document.getElementById('pfTrades'),
  reliabilityChart: document.getElementById('reliabilityChart'),
  scatterChart: document.getElementById('scatterChart'),
  candlestickChart: document.getElementById('candlestickChart'),
  tradeTableBody: document.getElementById('tradeTableBody'),

  apiStatus: document.getElementById('apiStatus'),
  evalLogdirs: document.getElementById('evalLogdirs'),
  evalDatasetDir: document.getElementById('evalDatasetDir'),
  evalSplit: document.getElementById('evalSplit'),
  evalHorizon: document.getElementById('evalHorizon'),
  evalSamples: document.getElementById('evalSamples'),
  evalEpsilon: document.getElementById('evalEpsilon'),
  evalCostBuffer: document.getElementById('evalCostBuffer'),
  evalPMin: document.getElementById('evalPMin'),
  evalDisagreeMax: document.getElementById('evalDisagreeMax'),
  evalVarMax: document.getElementById('evalVarMax'),
  evalSignalMode: document.getElementById('evalSignalMode'),
  evalStopMult: document.getElementById('evalStopMult'),
  evalMinStop: document.getElementById('evalMinStop'),
  evalTradeCost: document.getElementById('evalTradeCost'),
  evalBtInitialCapital: document.getElementById('evalBtInitialCapital'),
  evalBtRiskFraction: document.getElementById('evalBtRiskFraction'),
  evalBtMaxPositions: document.getElementById('evalBtMaxPositions'),
  evalBtMaxGrossLeverage: document.getElementById('evalBtMaxGrossLeverage'),
  evalBtOnePositionPerAsset: document.getElementById('evalBtOnePositionPerAsset'),
  evalMaxEpisodes: document.getElementById('evalMaxEpisodes'),
  evalOutdir: document.getElementById('evalOutdir'),
  runEvalBtn: document.getElementById('runEvalBtn'),
  sweepPredCsv: document.getElementById('sweepPredCsv'),
  sweepOutdir: document.getElementById('sweepOutdir'),
  sweepPValues: document.getElementById('sweepPValues'),
  sweepDisagreeValues: document.getElementById('sweepDisagreeValues'),
  sweepVarValues: document.getElementById('sweepVarValues'),
  sweepMinCoverage: document.getElementById('sweepMinCoverage'),
  sweepTopk: document.getElementById('sweepTopk'),
  runSweepBtn: document.getElementById('runSweepBtn'),
  bundleEvalDirs: document.getElementById('bundleEvalDirs'),
  bundleOutJson: document.getElementById('bundleOutJson'),
  bundleMaxRows: document.getElementById('bundleMaxRows'),
  runBundleBtn: document.getElementById('runBundleBtn'),
  trainConfigs: document.getElementById('trainConfigs'),
  trainSeeds: document.getElementById('trainSeeds'),
  trainLogroot: document.getElementById('trainLogroot'),
  trainDatasetDir: document.getElementById('trainDatasetDir'),
  trainJaxPlatform: document.getElementById('trainJaxPlatform'),
  trainJaxPrealloc: document.getElementById('trainJaxPrealloc'),
  runTrainBtn: document.getElementById('runTrainBtn'),
  refreshRunsBtn: document.getElementById('refreshRunsBtn'),
  runsSelect: document.getElementById('runsSelect'),
  stopRunBtn: document.getElementById('stopRunBtn'),
  runLogTail: document.getElementById('runLogTail'),
};

function parseNumber(value, fallback = 0) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function textLines(value) {
  return String(value || '')
    .split(/\r?\n/)
    .map((x) => x.trim())
    .filter(Boolean);
}

function nsToSec(ns) {
  return Math.floor(Number(ns) / 1e9);
}

function fmtPct(value) {
  if (!Number.isFinite(value)) return 'nan';
  return `${(value * 100).toFixed(2)}%`;
}

function fmtNum(value, digits = 4) {
  if (!Number.isFinite(value)) return 'nan';
  return Number(value).toFixed(digits);
}

function fmtTimeNs(ns) {
  if (!Number.isFinite(ns)) return '-';
  const date = new Date(Number(ns) / 1e6);
  if (Number.isNaN(date.getTime())) return '-';
  return date.toISOString().slice(0, 16).replace('T', ' ');
}

async function apiFetch(path, method = 'GET', body = null) {
  const init = {method, headers: {}};
  if (body !== null) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }
  const res = await fetch(path, init);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

function setApiStatus(text, ok) {
  if (!el.apiStatus) return;
  el.apiStatus.textContent = `API: ${text}`;
  el.apiStatus.style.color = ok ? '#2f9e44' : '#bf3f34';
}

function setTab(name) {
  const dashboardActive = name === 'dashboard';
  if (el.tabDashboard) el.tabDashboard.classList.toggle('is-active', dashboardActive);
  if (el.tabRunner) el.tabRunner.classList.toggle('is-active', !dashboardActive);
  if (el.tabDashboardBtn) el.tabDashboardBtn.classList.toggle('is-active', dashboardActive);
  if (el.tabRunnerBtn) el.tabRunnerBtn.classList.toggle('is-active', !dashboardActive);
}

function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) {
    throw new Error('CSV appears empty.');
  }
  const headers = lines[0].split(',').map((x) => x.trim());
  const idx = Object.fromEntries(headers.map((h, i) => [h, i]));
  const required = ['timestamp_ns', 'future_return', 'p_mean', 'disagree', 'var_mean'];
  for (const key of required) {
    if (!(key in idx)) {
      throw new Error(`CSV missing required column: ${key}`);
    }
  }
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(',');
    if (parts.length < headers.length) continue;
    rows.push({
      timestamp_ns: parseNumber(parts[idx.timestamp_ns], 0),
      future_return: parseNumber(parts[idx.future_return], 0),
      p_mean: parseNumber(parts[idx.p_mean], 0),
      disagree: parseNumber(parts[idx.disagree], 0),
      var_mean: parseNumber(parts[idx.var_mean], 0),
      episode: ('episode' in idx) ? parseNumber(parts[idx.episode], -1) : -1,
      direction: ('direction' in idx) ? parseNumber(parts[idx.direction], 0) : 0,
    });
  }
  return rows;
}

function defaultRunFromRows(rows) {
  return {
    id: 'uploaded_csv',
    source_dir: 'uploaded',
    split: 'unknown',
    horizon: 0,
    samples: 0,
    epsilon: 0.0005,
    cost_buffer: 0.0002,
    thresholds: {p_min: 0.5, disagree_max: 0.12, var_max: 0.4},
    baseline_metrics: {},
    trade_metrics: {},
    portfolio_metrics: {},
    rows,
    trades: [],
    equity: [],
    assets: [],
    episode_assets: [],
  };
}

function setRunControls(run) {
  el.pMin.value = String(run.thresholds?.p_min ?? 0.5);
  el.disagreeMax.value = String(run.thresholds?.disagree_max ?? 0.12);
  el.varMax.value = String(run.thresholds?.var_max ?? 0.4);
  el.epsilon.value = String(run.epsilon ?? 0.0005);
  el.costBuffer.value = String(run.cost_buffer ?? 0.0002);
  syncControlLabels();
}

function syncControlLabels() {
  el.pMinVal.textContent = Number(el.pMin.value).toFixed(3);
  el.disagreeVal.textContent = Number(el.disagreeMax.value).toFixed(3);
  el.varVal.textContent = Number(el.varMax.value).toFixed(3);
  el.binsVal.textContent = String(el.bins.value);
}

function updateRunSelector() {
  el.runSelect.innerHTML = '';
  state.runs.forEach((run, idx) => {
    const opt = document.createElement('option');
    const h = run.horizon ? `H${run.horizon}` : 'H?';
    opt.value = String(idx);
    opt.textContent = `${run.id} (${h})`;
    el.runSelect.appendChild(opt);
  });
}

function rowAssetName(row, run) {
  if (row.asset_name) return String(row.asset_name);
  const ep = parseNumber(row.episode, -1);
  const episodeAssets = run.episode_assets || [];
  if (ep >= 0 && ep < episodeAssets.length) {
    return String(episodeAssets[ep]);
  }
  return '';
}

function runAssets(run) {
  if (Array.isArray(run.assets) && run.assets.length) {
    return run.assets.slice();
  }
  const fromTrades = new Set((run.trades || []).map((x) => String(x.asset_name || '')).filter(Boolean));
  if (fromTrades.size) {
    return [...fromTrades].sort();
  }
  const fromRows = new Set();
  for (const row of run.rows || []) {
    const name = rowAssetName(row, run);
    if (name) fromRows.add(name);
  }
  return [...fromRows].sort();
}

function updateAssetSelector(run) {
  const assets = runAssets(run);
  el.assetSelect.innerHTML = '';
  if (!assets.length) {
    const opt = document.createElement('option');
    opt.value = '__ALL__';
    opt.textContent = 'All Rows (No Asset Mapping)';
    el.assetSelect.appendChild(opt);
    state.activeAsset = '__ALL__';
    return;
  }
  for (const asset of assets) {
    const opt = document.createElement('option');
    opt.value = asset;
    opt.textContent = asset;
    el.assetSelect.appendChild(opt);
  }
  if (assets.includes(state.activeAsset)) {
    el.assetSelect.value = state.activeAsset;
  } else {
    state.activeAsset = assets[0];
    el.assetSelect.value = state.activeAsset;
  }
}

function selectRun(index) {
  if (!state.runs.length) return;
  const i = Math.max(0, Math.min(index, state.runs.length - 1));
  state.activeRun = state.runs[i];
  state.candlesByTicker.clear();
  el.runSelect.value = String(i);
  setRunControls(state.activeRun);
  updateAssetSelector(state.activeRun);
  render().catch((err) => {
    alert(err.message);
  });
}

function computeMetrics(rows, cfg) {
  const threshold = cfg.epsilon + cfg.costBuffer;
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;
  const ys = [];
  const greens = [];
  const p = [];

  for (const r of rows) {
    const y = Number(r.future_return) > threshold;
    const g = (
      Number(r.p_mean) >= cfg.pMin &&
      Number(r.disagree) <= cfg.disagreeMax &&
      Number(r.var_mean) <= cfg.varMax
    );
    ys.push(y ? 1 : 0);
    greens.push(g ? 1 : 0);
    p.push(Number(r.p_mean));
    if (g && y) tp++;
    else if (g && !y) fp++;
    else if (!g && y) fn++;
    else tn++;
  }

  const count = rows.length;
  const precision = (tp + fp) ? tp / (tp + fp) : NaN;
  const recall = (tp + fn) ? tp / (tp + fn) : NaN;
  const coverage = count ? (tp + fp) / count : NaN;
  const falseGreen = (tp + fp) ? fp / (tp + fp) : NaN;

  return {tp, fp, fn, tn, count, precision, recall, coverage, falseGreen, ys, greens, p};
}

function computeReliability(p, ys, bins) {
  const b = Math.max(2, bins | 0);
  const edges = [];
  for (let i = 0; i <= b; i++) edges.push(i / b);
  const out = [];
  for (let i = 0; i < b; i++) {
    const lo = edges[i];
    const hi = edges[i + 1];
    let c = 0;
    let pSum = 0;
    let ySum = 0;
    for (let j = 0; j < p.length; j++) {
      const x = p[j];
      const take = (i === b - 1) ? (x >= lo && x <= hi) : (x >= lo && x < hi);
      if (!take) continue;
      c++;
      pSum += x;
      ySum += ys[j];
    }
    out.push({
      lo,
      hi,
      count: c,
      predMean: c ? pSum / c : NaN,
      empirical: c ? ySum / c : NaN,
    });
  }
  return out;
}

function clearCanvas(canvas) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#fbfdfd';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  return ctx;
}

function drawAxes(ctx, width, height, margin, xlabel, ylabel) {
  ctx.strokeStyle = '#cad6dd';
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  ctx.moveTo(margin, height - margin);
  ctx.lineTo(width - margin, height - margin);
  ctx.lineTo(width - margin, margin);
  ctx.stroke();

  ctx.fillStyle = '#57656f';
  ctx.font = '15px "IBM Plex Mono"';
  ctx.fillText(xlabel, width / 2 - 34, height - 12);
  ctx.save();
  ctx.translate(18, height / 2 + 22);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(ylabel, 0, 0);
  ctx.restore();
}

function drawReliability(canvas, relRows) {
  const ctx = clearCanvas(canvas);
  const w = canvas.width;
  const h = canvas.height;
  const m = 58;
  drawAxes(ctx, w, h, m, 'Predicted p_mean', 'Empirical P(y=1)');

  const x = (v) => m + v * (w - 2 * m);
  const y = (v) => h - m - v * (h - 2 * m);

  ctx.strokeStyle = '#9db2bc';
  ctx.setLineDash([6, 5]);
  ctx.beginPath();
  ctx.moveTo(x(0), y(0));
  ctx.lineTo(x(1), y(1));
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = '#0d7a8a';
  ctx.fillStyle = '#0d7a8a';
  ctx.lineWidth = 2.4;
  let first = true;
  ctx.beginPath();
  for (const row of relRows) {
    if (!Number.isFinite(row.predMean) || !Number.isFinite(row.empirical)) continue;
    const px = x(row.predMean);
    const py = y(row.empirical);
    if (first) {
      ctx.moveTo(px, py);
      first = false;
    } else {
      ctx.lineTo(px, py);
    }
  }
  ctx.stroke();

  for (const row of relRows) {
    if (!Number.isFinite(row.predMean) || !Number.isFinite(row.empirical)) continue;
    const px = x(row.predMean);
    const py = y(row.empirical);
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawScatter(canvas, rows, greens) {
  const ctx = clearCanvas(canvas);
  const n = Math.min(2200, rows.length);
  const w = canvas.width;
  const h = canvas.height;
  const m = 58;
  drawAxes(ctx, w, h, m, 'p_mean', 'disagree');

  const x = (v) => m + v * (w - 2 * m);
  const y = (v) => h - m - Math.min(1, v / 0.3) * (h - 2 * m);

  ctx.globalAlpha = 0.45;
  for (let i = 0; i < n; i++) {
    const row = rows[i];
    ctx.fillStyle = greens[i] ? '#2f9e44' : '#7b8a93';
    ctx.beginPath();
    ctx.arc(x(Number(row.p_mean)), y(Number(row.disagree)), 2.2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1.0;
}

function ensureChart() {
  if (!el.candlestickChart) return false;
  if (!window.Plotly) {
    el.chartStatus.textContent = 'Chart: Plotly failed to load.';
    return false;
  }
  if (!state.plotlyResizeBound) {
    window.addEventListener('resize', () => {
      if (!el.candlestickChart || !window.Plotly) return;
      window.Plotly.Plots.resize(el.candlestickChart);
    });
    state.plotlyResizeBound = true;
  }
  return true;
}

function computeFocusRangeSec(candles, rows, trades) {
  if (!candles.length) return null;
  const first = candles[0].time;
  const last = candles[candles.length - 1].time;
  const times = [];

  for (const row of rows) {
    const t = nsToSec(row.timestamp_ns);
    if (Number.isFinite(t)) times.push(t);
  }
  for (const tr of trades) {
    const te = nsToSec(tr.entry_timestamp_ns);
    const tx = nsToSec(tr.exit_timestamp_ns);
    if (Number.isFinite(te)) times.push(te);
    if (Number.isFinite(tx)) times.push(tx);
  }
  if (!times.length) return null;
  const lo = Math.min(...times);
  const hi = Math.max(...times);
  const pad = 12 * 300; // 12 bars of padding on both sides.
  const from = Math.max(first, lo - pad);
  const to = Math.min(last, hi + pad);
  if (!Number.isFinite(from) || !Number.isFinite(to) || from >= to) return null;
  return {from, to};
}

function findFirstIndexAtOrAfter(candles, tsSec) {
  let lo = 0;
  let hi = candles.length - 1;
  let ans = candles.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (candles[mid].time >= tsSec) {
      ans = mid;
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }
  return ans;
}

function findLastIndexAtOrBefore(candles, tsSec) {
  let lo = 0;
  let hi = candles.length - 1;
  let ans = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (candles[mid].time <= tsSec) {
      ans = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return ans;
}

function markerSymbol(shape) {
  if (shape === 'arrowUp') return 'triangle-up';
  if (shape === 'arrowDown') return 'triangle-down';
  if (shape === 'square') return 'square';
  return 'circle';
}

function markersToPlotlyTrace(markers, candles) {
  if (!markers.length || !candles.length) return null;
  const byTime = new Map(candles.map((c) => [c.time, c]));
  const hi = Math.max(...candles.map((c) => c.high));
  const lo = Math.min(...candles.map((c) => c.low));
  const offset = Math.max((hi - lo) * 0.01, 0.02);
  const x = [];
  const y = [];
  const text = [];
  const colors = [];
  const symbols = [];
  for (const m of markers) {
    const c = byTime.get(m.time);
    if (!c) continue;
    x.push(new Date(m.time * 1000));
    y.push(m.position === 'aboveBar' ? c.high + offset : c.low - offset);
    text.push(m.text || '');
    colors.push(m.color || '#3c7ed2');
    symbols.push(markerSymbol(m.shape));
  }
  if (!x.length) return null;
  return {
    type: 'scatter',
    mode: 'markers+text',
    x,
    y,
    text,
    textposition: 'top center',
    textfont: {size: 10, color: '#1f2d3d'},
    marker: {
      color: colors,
      symbol: symbols,
      size: 10,
      line: {color: '#1f2d3d', width: 0.5},
    },
    hovertemplate: '%{x}<br>%{y:.3f}<extra></extra>',
    name: 'Markers',
  };
}

async function loadCandles(ticker, limit) {
  const key = `${ticker}::${limit}`;
  if (state.candlesByTicker.has(key)) {
    return state.candlesByTicker.get(key);
  }
  const params = new URLSearchParams({
    ticker,
    limit: String(limit),
  });
  const out = await apiFetch(`/api/market/candles?${params.toString()}`);
  const rows = Array.isArray(out.candles) ? out.candles : [];
  const dedup = new Map();
  for (const row of rows) {
    const t = nsToSec(row.timestamp_ns);
    const o = Number(row.open);
    const h = Number(row.high);
    const l = Number(row.low);
    const c = Number(row.close);
    const valid = (
      Number.isFinite(t) &&
      Number.isFinite(o) &&
      Number.isFinite(h) &&
      Number.isFinite(l) &&
      Number.isFinite(c) &&
      h >= l
    );
    if (!valid) continue;
    dedup.set(t, {
      time: t,
      open: o,
      high: h,
      low: l,
      close: c,
    });
  }
  const candles = [...dedup.values()].sort((a, b) => a.time - b.time);
  state.candlesByTicker.set(key, candles);
  return candles;
}

function rowsForAsset(run, asset) {
  if (!asset || asset === '__ALL__') return run.rows || [];
  return (run.rows || []).filter((row) => rowAssetName(row, run) === asset);
}

function tradesForAsset(run, asset) {
  if (!asset || asset === '__ALL__') return run.trades || [];
  return (run.trades || []).filter((row) => String(row.asset_name || '') === asset);
}

function buildSignalMarkers(rows, cfg) {
  if (!el.showSignalsChk.checked) return [];
  const threshold = cfg.epsilon + cfg.costBuffer;
  const markers = [];
  for (const row of rows) {
    const green = (
      Number(row.p_mean) >= cfg.pMin &&
      Number(row.disagree) <= cfg.disagreeMax &&
      Number(row.var_mean) <= cfg.varMax
    );
    if (!green) continue;
    const time = nsToSec(row.timestamp_ns);
    const direction = parseNumber(row.direction, (Number(row.future_return) > threshold ? 1 : -1));
    markers.push({
      time,
      position: direction < 0 ? 'aboveBar' : 'belowBar',
      color: direction < 0 ? '#d44e42' : '#2f9e44',
      shape: direction < 0 ? 'arrowDown' : 'arrowUp',
      text: 'G',
    });
    if (markers.length >= 2000) break;
  }
  return markers;
}

function buildTradeMarkers(trades) {
  if (!el.showTradesChk.checked) return [];
  const markers = [];
  for (const tr of trades) {
    const d = parseNumber(tr.direction, 0);
    const pnl = parseNumber(tr.pnl, 0);
    const entryTime = nsToSec(tr.entry_timestamp_ns);
    const exitTime = nsToSec(tr.exit_timestamp_ns);
    markers.push({
      time: entryTime,
      position: d < 0 ? 'aboveBar' : 'belowBar',
      color: '#3c7ed2',
      shape: 'circle',
      text: d < 0 ? 'SE' : 'LE',
    });
    markers.push({
      time: exitTime,
      position: d < 0 ? 'belowBar' : 'aboveBar',
      color: pnl >= 0 ? '#2f9e44' : '#d44e42',
      shape: 'square',
      text: pnl >= 0 ? 'X+' : 'X-',
    });
    if (markers.length >= 3000) break;
  }
  return markers;
}

function renderTradeTable(trades) {
  if (!el.tradeTableBody) return;
  const rows = trades
    .slice()
    .sort((a, b) => Number(b.entry_timestamp_ns) - Number(a.entry_timestamp_ns))
    .slice(0, 250);
  el.tradeTableBody.innerHTML = '';
  if (!rows.length) {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td colspan="7">No trades for this ticker/run.</td>';
    el.tradeTableBody.appendChild(tr);
    return;
  }
  for (const row of rows) {
    const tr = document.createElement('tr');
    const dir = parseNumber(row.direction, 0) < 0 ? 'SHORT' : 'LONG';
    const pnl = parseNumber(row.pnl, 0);
    tr.innerHTML = `
      <td>${fmtTimeNs(row.entry_timestamp_ns)}</td>
      <td>${fmtTimeNs(row.exit_timestamp_ns)}</td>
      <td>${dir}</td>
      <td>${parseNumber(row.bars_held, 0)}</td>
      <td>${fmtNum(parseNumber(row.notional, 0), 3)}</td>
      <td class="${parseNumber(row.net_return, 0) >= 0 ? 'trade-pos' : 'trade-neg'}">${fmtNum(parseNumber(row.net_return, 0), 5)}</td>
      <td class="${pnl >= 0 ? 'trade-pos' : 'trade-neg'}">${fmtNum(pnl, 5)}</td>
    `;
    el.tradeTableBody.appendChild(tr);
  }
}

async function renderChart(run, asset, assetRows, assetTrades, cfg, token) {
  if (!ensureChart()) return;
  if (!asset || asset === '__ALL__') {
    if (window.Plotly) {
      window.Plotly.react(el.candlestickChart, [], {
        height: 540,
        margin: {l: 50, r: 30, t: 10, b: 30},
        paper_bgcolor: '#fcfefd',
        plot_bgcolor: '#fcfefd',
      }, {responsive: true, scrollZoom: true, displaylogo: false});
    }
    el.chartStatus.textContent = 'Chart: choose a ticker with available candles.';
    return;
  }
  const limit = Math.max(200, parseNumber(el.candleLimit.value, 4000));
  let candles = [];
  try {
    candles = await loadCandles(asset, limit);
  } catch (err) {
    if (token !== state.renderToken) return;
    el.chartStatus.textContent = `Chart: ${err.message}`;
    if (window.Plotly) {
      window.Plotly.react(el.candlestickChart, [], {
        height: 540,
        margin: {l: 50, r: 30, t: 10, b: 30},
        paper_bgcolor: '#fcfefd',
        plot_bgcolor: '#fcfefd',
      }, {responsive: true, scrollZoom: true, displaylogo: false});
    }
    return;
  }
  if (token !== state.renderToken) return;
  if (!candles.length) {
    if (window.Plotly) {
      window.Plotly.react(el.candlestickChart, [], {
        height: 540,
        margin: {l: 50, r: 30, t: 10, b: 30},
        paper_bgcolor: '#fcfefd',
        plot_bgcolor: '#fcfefd',
      }, {responsive: true, scrollZoom: true, displaylogo: false});
    }
    el.chartStatus.textContent = `Chart: no valid candles loaded for ${asset}.`;
    return;
  }

  const first = candles[0].time;
  const last = candles[candles.length - 1].time;

  const markersRaw = [
    ...buildSignalMarkers(assetRows, cfg),
    ...buildTradeMarkers(assetTrades),
  ];
  const markers = markersRaw
    .filter((m) => Number.isFinite(m.time) && m.time >= first && m.time <= last)
    .sort((a, b) => a.time - b.time);

  const iFrom = 0;
  const iTo = candles.length - 1;
  const x = candles.map((c) => new Date(c.time * 1000));
  const candleTrace = {
    type: 'candlestick',
    x,
    open: candles.map((c) => c.open),
    high: candles.map((c) => c.high),
    low: candles.map((c) => c.low),
    close: candles.map((c) => c.close),
    increasing: {line: {color: '#2f9e44'}, fillcolor: '#2f9e44'},
    decreasing: {line: {color: '#bf3f34'}, fillcolor: '#bf3f34'},
    whiskerwidth: 0.4,
    name: 'OHLC',
  };
  const closeTrace = {
    type: 'scatter',
    mode: 'lines',
    x,
    y: candles.map((c) => c.close),
    line: {color: '#1f2d3d', width: 1.5},
    hoverinfo: 'skip',
    name: 'Close',
  };
  const markerTrace = markersToPlotlyTrace(markers, candles);
  const traces = markerTrace ? [candleTrace, closeTrace, markerTrace] : [candleTrace, closeTrace];
  const layout = {
    height: 540,
    margin: {l: 55, r: 36, t: 8, b: 36},
    paper_bgcolor: '#fcfefd',
    plot_bgcolor: '#fcfefd',
    showlegend: false,
    dragmode: 'pan',
    xaxis: {
      type: 'date',
      rangeslider: {visible: false},
      range: [x[iFrom], x[iTo]],
      showgrid: true,
      gridcolor: '#edf2f4',
    },
    yaxis: {
      fixedrange: false,
      showgrid: true,
      gridcolor: '#edf2f4',
      tickformat: '.2f',
    },
    hovermode: 'x',
  };
  try {
    window.Plotly.react(
      el.candlestickChart,
      traces,
      layout,
      {responsive: true, scrollZoom: true, displaylogo: false},
    );
  } catch (err) {
    el.chartStatus.textContent = `Chart render error for ${asset}: ${err && err.message ? err.message : String(err)}`;
    return;
  }
  const firstClose = candles[0].close;
  const lastClose = candles[candles.length - 1].close;
  el.chartStatus.textContent = `Chart: ${asset} | candles=${candles.length} | visible_bars=${Math.max(0, iTo - iFrom + 1)} | markers=${markers.length}/${markersRaw.length} | close=${fmtNum(firstClose, 3)}->${fmtNum(lastClose, 3)} | range=${fmtTimeNs(first * 1e9)} -> ${fmtTimeNs(last * 1e9)}`;
}

async function render() {
  if (!state.activeRun) return;
  const token = ++state.renderToken;
  syncControlLabels();

  const run = state.activeRun;
  const asset = el.assetSelect.value || '__ALL__';
  state.activeAsset = asset;
  const rows = rowsForAsset(run, asset);
  const trades = tradesForAsset(run, asset);

  const cfg = {
    pMin: parseNumber(el.pMin.value, 0.5),
    disagreeMax: parseNumber(el.disagreeMax.value, 0.12),
    varMax: parseNumber(el.varMax.value, 0.4),
    epsilon: parseNumber(el.epsilon.value, 0.0005),
    costBuffer: parseNumber(el.costBuffer.value, 0.0002),
    bins: parseNumber(el.bins.value, 10),
  };

  const metrics = computeMetrics(rows, cfg);
  const reliability = computeReliability(metrics.p, metrics.ys, cfg.bins);
  if (token !== state.renderToken) return;

  const positives = metrics.ys.reduce((a, b) => a + b, 0);
  const threshold = cfg.epsilon + cfg.costBuffer;
  el.runMeta.textContent = `${run.id} | split=${run.split || 'n/a'} | horizon=${run.horizon || '?'} | rows=${rows.length}/${(run.rows || []).length} | positives=${positives}/${metrics.count} | target>${threshold.toFixed(6)} | asset=${asset}`;

  el.precision.textContent = fmtPct(metrics.precision);
  el.coverage.textContent = fmtPct(metrics.coverage);
  el.recall.textContent = fmtPct(metrics.recall);
  el.falseGreen.textContent = fmtPct(metrics.falseGreen);
  el.greens.textContent = String(metrics.tp + metrics.fp);
  el.count.textContent = String(metrics.count);

  const pf = run.portfolio_metrics || {};
  el.pfReturn.textContent = Number.isFinite(Number(pf.total_return_pct)) ? fmtPct(Number(pf.total_return_pct)) : '-';
  el.pfMdd.textContent = Number.isFinite(Number(pf.max_drawdown_pct)) ? fmtPct(Number(pf.max_drawdown_pct)) : '-';
  el.pfTrades.textContent = Number.isFinite(Number(pf.trades_executed)) ? String(Number(pf.trades_executed)) : '-';

  drawReliability(el.reliabilityChart, reliability);
  drawScatter(el.scatterChart, rows, metrics.greens);
  renderTradeTable(trades);
  await renderChart(run, asset, rows, trades, cfg, token);
}

function getEvalPayload() {
  return {
    logdirs: textLines(el.evalLogdirs.value),
    dataset_dir: el.evalDatasetDir.value.trim(),
    split: el.evalSplit.value,
    horizon: parseNumber(el.evalHorizon.value, 12),
    samples: parseNumber(el.evalSamples.value, 8),
    epsilon: parseNumber(el.evalEpsilon.value, 0.0005),
    cost_buffer: parseNumber(el.evalCostBuffer.value, 0.0002),
    p_min: parseNumber(el.evalPMin.value, 0.5),
    disagree_max: parseNumber(el.evalDisagreeMax.value, 0.12),
    var_max: parseNumber(el.evalVarMax.value, 0.4),
    signal_mode: (el.evalSignalMode && el.evalSignalMode.value) ? el.evalSignalMode.value : 'directional',
    stop_mult: parseNumber(el.evalStopMult.value, 2.0),
    min_stop: parseNumber(el.evalMinStop.value, 0.0005),
    trade_cost: parseNumber(el.evalTradeCost.value, -1.0),
    bt_initial_capital: parseNumber(el.evalBtInitialCapital.value, 1.0),
    bt_risk_fraction: parseNumber(el.evalBtRiskFraction.value, 0.01),
    bt_max_positions: parseNumber(el.evalBtMaxPositions.value, 20),
    bt_max_gross_leverage: parseNumber(el.evalBtMaxGrossLeverage.value, 3.0),
    bt_one_position_per_asset: parseNumber(el.evalBtOnePositionPerAsset.value, 1),
    max_episodes: parseNumber(el.evalMaxEpisodes.value, 0),
    outdir: el.evalOutdir.value.trim(),
  };
}

function getSweepPayload() {
  return {
    predictions_csv: el.sweepPredCsv.value.trim(),
    outdir: el.sweepOutdir.value.trim(),
    p_values: el.sweepPValues.value.trim(),
    disagree_values: el.sweepDisagreeValues.value.trim(),
    var_values: el.sweepVarValues.value.trim(),
    min_coverage: parseNumber(el.sweepMinCoverage.value, 0.01),
    topk: parseNumber(el.sweepTopk.value, 40),
  };
}

function getBundlePayload() {
  return {
    eval_dirs: textLines(el.bundleEvalDirs.value),
    out_json: el.bundleOutJson.value.trim(),
    max_rows_per_run: parseNumber(el.bundleMaxRows.value, 0),
  };
}

function getTrainPayload() {
  return {
    configs: el.trainConfigs.value.trim(),
    seeds: el.trainSeeds.value.trim(),
    logroot: el.trainLogroot.value.trim(),
    dataset_dir: el.trainDatasetDir.value.trim(),
    jax_platform: el.trainJaxPlatform.value.trim(),
    jax_prealloc: el.trainJaxPrealloc.value.trim(),
  };
}

function renderRuns(runs) {
  const prev = el.runsSelect.value;
  el.runsSelect.innerHTML = '';
  for (const run of runs) {
    const opt = document.createElement('option');
    opt.value = run.job_id;
    const rc = (run.returncode === null || run.returncode === undefined) ? '' : ` rc=${run.returncode}`;
    opt.textContent = `${run.job_id} [${run.status}]${rc}`;
    el.runsSelect.appendChild(opt);
  }
  if (!runs.length) {
    el.runLogTail.textContent = '';
    return;
  }
  const found = runs.find((x) => x.job_id === prev);
  el.runsSelect.value = found ? prev : runs[0].job_id;
}

async function refreshRuns() {
  const data = await apiFetch('/api/runs');
  const runs = data.runs || [];
  renderRuns(runs);
  if (el.runsSelect.value) {
    await refreshRunTail(el.runsSelect.value);
  }
}

async function refreshRunTail(jobId) {
  if (!jobId) return;
  const run = await apiFetch(`/api/runs/${encodeURIComponent(jobId)}?tail=160`);
  el.runLogTail.textContent = run.log_tail || '';
}

async function startJob(endpoint, payload) {
  const out = await apiFetch(endpoint, 'POST', payload);
  const job = out.job || {};
  await refreshRuns();
  if (job.job_id) {
    el.runsSelect.value = job.job_id;
    await refreshRunTail(job.job_id);
  }
}

async function checkApi() {
  try {
    await apiFetch('/api/health');
    setApiStatus('connected', true);
    await refreshRuns();
  } catch (_) {
    setApiStatus('not available (use experiment_server)', false);
  }
}

function setBundle(bundle) {
  if (!bundle || !Array.isArray(bundle.runs) || !bundle.runs.length) {
    throw new Error('Bundle missing runs.');
  }
  state.bundle = bundle;
  state.runs = bundle.runs;
  updateRunSelector();
  selectRun(0);
}

async function loadDefaultBundle() {
  const res = await fetch('./data/bundle.json', {cache: 'no-store'});
  if (!res.ok) {
    throw new Error(`Failed to load default bundle: HTTP ${res.status}`);
  }
  const bundle = await res.json();
  setBundle(bundle);
}

function bindEvents() {
  if (el.tabDashboardBtn) {
    el.tabDashboardBtn.addEventListener('click', () => setTab('dashboard'));
  }
  if (el.tabRunnerBtn) {
    el.tabRunnerBtn.addEventListener('click', () => setTab('runner'));
  }

  el.loadDefaultBtn.addEventListener('click', async () => {
    try {
      await loadDefaultBundle();
    } catch (err) {
      alert(err.message);
    }
  });

  el.fileInput.addEventListener('change', async (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    const text = await file.text();
    try {
      if (file.name.toLowerCase().endsWith('.json')) {
        const bundle = JSON.parse(text);
        setBundle(bundle);
      } else if (file.name.toLowerCase().endsWith('.csv')) {
        const rows = parseCsv(text);
        setBundle({runs: [defaultRunFromRows(rows)]});
      } else {
        throw new Error('Unsupported file type. Use .json or .csv');
      }
    } catch (err) {
      alert(err.message);
    }
  });

  el.runSelect.addEventListener('change', () => selectRun(parseInt(el.runSelect.value, 10) || 0));
  if (el.assetSelect) {
    el.assetSelect.addEventListener('change', () => {
      render().catch((err) => alert(err.message));
    });
  }
  [el.pMin, el.disagreeMax, el.varMax, el.epsilon, el.costBuffer, el.bins, el.showSignalsChk, el.showTradesChk].forEach((node) => {
    node.addEventListener('input', () => {
      render().catch((err) => alert(err.message));
    });
  });
  if (el.candleLimit) {
    el.candleLimit.addEventListener('change', () => {
      state.candlesByTicker.clear();
      render().catch((err) => alert(err.message));
    });
  }
  if (el.reloadCandlesBtn) {
    el.reloadCandlesBtn.addEventListener('click', () => {
      state.candlesByTicker.clear();
      render().catch((err) => alert(err.message));
    });
  }

  el.resetBtn.addEventListener('click', () => {
    if (!state.activeRun) return;
    setRunControls(state.activeRun);
    render().catch((err) => alert(err.message));
  });

  if (el.runEvalBtn) {
    el.runEvalBtn.addEventListener('click', async () => {
      try {
        await startJob('/api/run/eval', getEvalPayload());
      } catch (err) {
        alert(err.message);
      }
    });
  }

  if (el.runSweepBtn) {
    el.runSweepBtn.addEventListener('click', async () => {
      try {
        await startJob('/api/run/sweep', getSweepPayload());
      } catch (err) {
        alert(err.message);
      }
    });
  }

  if (el.runBundleBtn) {
    el.runBundleBtn.addEventListener('click', async () => {
      try {
        await startJob('/api/run/bundle', getBundlePayload());
      } catch (err) {
        alert(err.message);
      }
    });
  }

  if (el.runTrainBtn) {
    el.runTrainBtn.addEventListener('click', async () => {
      try {
        await startJob('/api/run/train', getTrainPayload());
      } catch (err) {
        alert(err.message);
      }
    });
  }

  if (el.refreshRunsBtn) {
    el.refreshRunsBtn.addEventListener('click', async () => {
      try {
        await refreshRuns();
      } catch (err) {
        alert(err.message);
      }
    });
  }

  if (el.runsSelect) {
    el.runsSelect.addEventListener('change', async () => {
      try {
        await refreshRunTail(el.runsSelect.value);
      } catch (err) {
        alert(err.message);
      }
    });
  }

  if (el.stopRunBtn) {
    el.stopRunBtn.addEventListener('click', async () => {
      try {
        const jobId = el.runsSelect.value;
        if (!jobId) return;
        await apiFetch('/api/runs/stop', 'POST', {job_id: jobId});
        await refreshRuns();
      } catch (err) {
        alert(err.message);
      }
    });
  }
}

function init() {
  setTab('dashboard');
  bindEvents();
  checkApi();
  setInterval(() => {
    checkApi();
  }, 5000);
  loadDefaultBundle().catch(() => {
    el.runMeta.textContent = 'No default bundle found. Upload bundle JSON or predictions CSV.';
    el.chartStatus.textContent = 'Chart: load a run and select a ticker.';
  });
}

init();
