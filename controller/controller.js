#!/usr/bin/env node
'use strict';

/**
 * Smart Resource Allocation Controller
 * ======================================
 * Connects the ML Prediction API <-> K8s Scaling Executor to form
 * a closed-loop proactive autoscaling system.
 *
 * Architecture:
 *
 *   ┌──────────────┐   48×21 window   ┌──────────────────┐
 *   │  Simulation   │ ───────────────► │   ML Prediction  │
 *   │  Data (CSV)   │                  │   API (FastAPI)  │
 *   └──────────────┘                   └────────┬─────────┘
 *                                               │ prediction
 *                                               ▼
 *                                      ┌──────────────────┐
 *                    ┌────────────────►│  This Controller  │
 *                    │  every 60s      │   (Node.js)       │
 *                    └─────────────────┤                   │
 *                                      └────────┬─────────┘
 *                                               │ scale request
 *                                               ▼
 *                                      ┌──────────────────┐
 *                                      │  K8s Executor    │
 *                                      │  (Node.js)       │
 *                                      │  kubectl scale   │
 *                                      └──────────────────┘
 *
 * Usage:
 *   node controller/controller.js                            # Real-time (60s)
 *   node controller/controller.js --interval 10             # Faster demo
 *   node controller/controller.js --interval 0 --steps 20  # Instant, 20 steps
 *   node controller/controller.js --dry-run                 # Skip executor calls
 *   node controller/controller.js --validate                # Validate + rollback
 *   node controller/controller.js --dry-run --validate --interval 0 --steps 60 --start-offset 770
 */

// ============================================================
// Dependencies  (Node 18+ built-ins only — no npm install needed)
// ============================================================
const https = require('https');
const http  = require('http');

// ============================================================
// Configuration
// ============================================================

let ML_API_URL   = process.env.ML_API_URL
  || 'https://k8s-model-api.whiteglacier-fec535bb.southeastasia.azurecontainerapps.io';

let EXECUTOR_URL = process.env.EXECUTOR_URL
  || 'http://localhost:6000/api/v1/scale-with-metrics';

const LOOKBACK = 48; // Sliding window size (48 rows × 21 features)

// Column order MUST match the ML API expectation exactly
const FEATURE_COLS = [
  'request_rate_rps',         'latency_p95_ms',           'latency_p99_ms',
  'error_rate_percent',       'queue_length',              'pod_cpu_usage_percent_avg',
  'pod_cpu_usage_percent_p95','pod_memory_usage_mb_avg',   'pod_memory_usage_mb_p95',
  'hour_sin',                 'hour_cos',                  'day_sin',
  'day_cos',                  'mesh_inbound_rps',          'mesh_inbound_latency_p95',
  'mesh_inbound_error_rate',  'degree_centrality',         'eigenvector_centrality',
  'betweenness_centrality',   'closeness_centrality',
];

// ============================================================
// ANSI colour helpers
// ============================================================

const RESET = '\x1b[0m';
const COLORS = {
  INFO:     '\x1b[94m',  // Blue
  PREDICT:  '\x1b[96m',  // Cyan
  SCALE:    '\x1b[91m',  // Red
  SKIP:     '\x1b[90m',  // Gray
  OK:       '\x1b[92m',  // Green
  WARN:     '\x1b[93m',  // Yellow
  ERROR:    '\x1b[91m',  // Red
  'DRY-RUN':'\x1b[95m',  // Magenta
  VALIDATE: '\x1b[96m',  // Cyan
  ROLLBACK: '\x1b[95m',  // Magenta
};

function log(level, message) {
  const ts    = new Date().toISOString().slice(11, 19); // HH:MM:SS UTC
  const color = COLORS[level] || '';
  const lbl   = level.padStart(8, ' ').slice(0, 8);
  console.log(`  ${color}[${ts}] [${lbl}]${RESET} ${message}`);
}

// ============================================================
// 1. HTTP helper — Promise-based fetch (no node-fetch needed)
// ============================================================

function request(url, { method = 'GET', body = null, timeoutMs = 30_000 } = {}) {
  return new Promise((resolve, reject) => {
    const parsed  = new URL(url);
    const lib     = parsed.protocol === 'https:' ? https : http;
    const options = {
      hostname: parsed.hostname,
      port:     parsed.port || (parsed.protocol === 'https:' ? 443 : 80),
      path:     parsed.pathname + parsed.search,
      method,
      headers:  { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    };

    const chunks = [];
    const req = lib.request(options, (res) => {
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => {
        const raw  = Buffer.concat(chunks).toString();
        const data = JSON.parse(raw);
        if (res.statusCode >= 400) {
          return reject(new Error(`HTTP ${res.statusCode}: ${raw.slice(0, 200)}`));
        }
        resolve(data);
      });
    });

    req.setTimeout(timeoutMs, () => {
      req.destroy();
      reject(new Error(`Request timed out after ${timeoutMs}ms`));
    });

    req.on('error', reject);

    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}

// ============================================================
// 2. Simulation data fetch
// ============================================================

async function fetchSimulationData() {
  log('INFO', 'Fetching simulation data from ML API...');
  const body = await request(`${ML_API_URL}/simulation-data`, { timeoutMs: 60_000 });
  const data = body.data;
  log('INFO', `Loaded ${data.length} simulation rows (total dataset: ${body.total_rows})`);
  return data;
}

// ============================================================
// 3. Window builder
// ============================================================

/**
 * Convert 48 raw data rows into a 48×21 numeric window.
 * @returns {{ windowData: number[][], windowEndUtc: string }}
 */
function buildWindow(rows) {
  const windowData = rows.map((row) => {
    const values = FEATURE_COLS.map((col) => parseFloat(row[col]));
    values.push(parseFloat(row['current_pod_count']));
    return values;
  });
  const windowEndUtc = rows[rows.length - 1].timestamp;
  return { windowData, windowEndUtc };
}

// ============================================================
// 4. ML API caller
// ============================================================

async function callPredict(windowData, windowEndUtc, serviceId = 'Order') {
  const payload = {
    window_data:    windowData,
    window_end_utc: windowEndUtc,
    service_id:     serviceId,
    input_source:   'controller',
  };
  return request(`${ML_API_URL}/predict`, { method: 'POST', body: payload, timeoutMs: 30_000 });
}

// ============================================================
// 5. Prediction → Executor payload converter
// ============================================================

/**
 * Build the executor POST payload from ML prediction + raw metrics.
 * Returns null if scale_action === 'no_change'.
 */
function buildExecutorPayload(prediction, rawMetrics) {
  const { current_pods, predicted_pods, scale_action } = prediction;

  if (scale_action === 'no_change') return null;

  const requestPods = Math.abs(predicted_pods - current_pods);

  const errorPct   = parseFloat(rawMetrics.error_rate_percent  ?? 0);
  const errorRate  = Math.min(Math.max(errorPct / 100.0, 0), 1);
  const successRate = parseFloat((1.0 - errorRate).toFixed(4));

  const cpuPct = parseFloat(rawMetrics.pod_cpu_usage_percent_avg ?? 0);
  let   memPct = parseFloat(rawMetrics.pod_memory_usage_mb_p95   ?? 0) / 1024 * 100;
  memPct = Math.min(parseFloat(memPct.toFixed(2)), 100);

  const latencyBefore = parseFloat(parseFloat(rawMetrics.latency_p95_ms ?? 0).toFixed(2));

  return {
    services: [{
      deployment:   (prediction.service_id || 'Order').toLowerCase(),
      request_pods: requestPods,
      scale_action,
      metrics: {
        successRate,
        errorRate:       parseFloat(errorRate.toFixed(4)),
        p95LatencyBefore: latencyBefore,
        p95LatencyAfter:  parseFloat((latencyBefore * 0.6).toFixed(2)),
        cpuPercent:       parseFloat(cpuPct.toFixed(2)),
        memPercent:       memPct,
        restartCount:     0,
        trafficRecovery:  0.95,
      },
    }],
  };
}

// ============================================================
// 6. Executor caller
// ============================================================

async function sendToExecutor(payload, dryRun = false) {
  if (dryRun) {
    log('DRY-RUN', `Would send to executor:\n${JSON.stringify(payload, null, 2)}`);
    return { status: 'dry-run', message: 'Skipped (dry-run mode)' };
  }
  try {
    const result = await request(EXECUTOR_URL, { method: 'POST', body: payload });
    return result;
  } catch (err) {
    if (err.code === 'ECONNREFUSED' || err.message.includes('connect')) {
      log('WARN', `Executor not reachable at ${EXECUTOR_URL} — is it running?`);
      return { status: 'error', message: 'Executor service unreachable' };
    }
    log('ERROR', `Executor call failed: ${err.message}`);
    return { status: 'error', message: err.message };
  }
}

// ============================================================
// 7. Validate + Rollback
// ============================================================

/**
 * After a scaling action, re-query ML API with the NEXT window (step+1)
 * to verify the decision was correct.
 *
 * Conflict logic:
 *   scaled_up   but next says scale_down → ROLLBACK
 *   scaled_down but next says scale_up   → ROLLBACK
 *   otherwise                            → VALIDATED ✓
 */
async function validateScaleAction(data, step, originalPrediction, dryRun = false) {
  const nextStart = step + 1;
  const nextEnd   = nextStart + LOOKBACK;

  if (nextEnd > data.length) {
    return { status: 'skipped', message: 'Not enough lookahead data', rollbackPayload: null };
  }

  let nextPred;
  try {
    const nextRows = data.slice(nextStart, nextEnd);
    const { windowData, windowEndUtc } = buildWindow(nextRows);
    nextPred = await callPredict(windowData, windowEndUtc, originalPrediction.service_id || 'Order');
  } catch (err) {
    log('WARN', `Validate: ML API call failed — ${err.message}`);
    return { status: 'skipped', message: `ML API error: ${err.message}`, rollbackPayload: null };
  }

  const origAction   = originalPrediction.scale_action;
  const nextAction   = nextPred.scale_action;
  const origPredPods = originalPrediction.predicted_pods;
  const nextCurrPods = nextPred.current_pods;
  const nextPredPods = nextPred.predicted_pods;

  const conflict = (
    (origAction === 'scale_up'   && nextAction === 'scale_down') ||
    (origAction === 'scale_down' && nextAction === 'scale_up')
  );

  if (conflict) {
    const rollbackAction = origAction === 'scale_up' ? 'scale_down' : 'scale_up';
    const rollbackPods   = Math.max(Math.abs(origPredPods - nextCurrPods), 1);

    const rollbackPayload = {
      services: [{
        deployment:   (originalPrediction.service_id || 'Order').toLowerCase(),
        request_pods: rollbackPods,
        scale_action: rollbackAction,
        metrics: {
          successRate:      0.95,
          errorRate:        0.05,
          p95LatencyBefore: 0,
          p95LatencyAfter:  0,
          cpuPercent:       0,
          memPercent:       0,
          restartCount:     0,
          trafficRecovery:  0.90,
        },
      }],
    };

    const reason = `Original action=${origAction} (→${origPredPods} pods) ` +
                   `but next step says ${nextAction} (→${nextPredPods} pods) — ROLLING BACK`;

    log('ROLLBACK', reason);

    if (dryRun) {
      log('DRY-RUN', `Rollback payload:\n${JSON.stringify(rollbackPayload, null, 2)}`);
    } else {
      const result = await sendToExecutor(rollbackPayload, false);
      log('ROLLBACK', `Rollback executor result: ${JSON.stringify(result)}`);
    }

    return { status: 'rollback', message: reason, rollbackPayload };
  }

  const reason = `Confirmed: ${origAction} still valid — ` +
                 `next window predicts ${nextAction} (${nextPredPods} pods)`;
  log('VALIDATE', `✓ ${reason}`);
  return { status: 'validated', message: reason, rollbackPayload: null };
}

// ============================================================
// 8. Main controller loop
// ============================================================

async function runController({
  intervalSec  = 60,
  maxSteps     = null,
  dryRun       = false,
  executorUrl  = EXECUTOR_URL,
  startOffset  = 0,
  validate     = false,
} = {}) {
  EXECUTOR_URL = executorUrl;

  // ── Banner ──
  console.log();
  console.log('='.repeat(76));
  console.log('  SMART RESOURCE ALLOCATION CONTROLLER');
  console.log('  ML Prediction API  ←→  K8s Scaling Executor');
  console.log('='.repeat(76));
  console.log(`  ML API:     ${ML_API_URL}`);
  console.log(`  Executor:   ${EXECUTOR_URL}`);
  console.log(`  Interval:   ${intervalSec}s${intervalSec === 60 ? ' (real-time)' : ''}`);
  console.log(`  Dry-run:    ${dryRun}`);
  console.log(`  Validate:   ${validate}  (rollback on model conflict)`);
  console.log(`  Max steps:  ${maxSteps ?? 'unlimited'}`);
  console.log('='.repeat(76));

  // ── Fetch all simulation data once ──
  let data = await fetchSimulationData();

  if (startOffset > 0) {
    data = data.slice(startOffset);
    log('INFO', `Skipped first ${startOffset} rows (--start-offset)`);
  }

  let totalSteps = data.length - LOOKBACK;
  if (maxSteps) totalSteps = Math.min(totalSteps, maxSteps);
  log('INFO', `Ready to run ${totalSteps} prediction steps`);
  console.log();

  // ── Tracking ──
  const stats = {
    predictions:    0,
    scale_ups:      0,
    scale_downs:    0,
    no_changes:     0,
    executor_calls: 0,
    executor_errors:0,
    validated:      0,
    rollbacks:      0,
    latencies:      [],
  };

  const hdr = `  ${'Step'.padEnd(6)} ${'Time (sim)'.padEnd(22)} ${'Curr'.padEnd(6)} ${'Pred'.padEnd(6)} ` +
              `${'Action'.padEnd(14)} ${'Exec Result'.padEnd(20)} Latency`;
  console.log(hdr);
  console.log('  ' + '-'.repeat(84));

  // ── Action colours ──
  const actionDisplay = {
    scale_up:   `\x1b[91m▲ SCALE UP\x1b[0m  `,
    scale_down: `\x1b[92m▼ SCALE DOWN\x1b[0m`,
    no_change:  `\x1b[90m— NO CHANGE\x1b[0m `,
  };

  // ── Main loop ──
  for (let step = 0; step < totalSteps; step++) {
    const windowRows = data.slice(step, step + LOOKBACK);
    if (windowRows.length < LOOKBACK) break;

    const { windowData, windowEndUtc } = buildWindow(windowRows);
    const latestRow = windowRows[windowRows.length - 1];

    // Call ML API
    let prediction;
    try {
      prediction = await callPredict(windowData, windowEndUtc);
    } catch (err) {
      log('ERROR', `Step ${step + 1}: ML API failed — ${err.message}`);
      continue;
    }

    const { current_pods: current, predicted_pods: predicted,
            scale_action: action,  latency_ms: latency } = prediction;

    stats.predictions++;
    stats.latencies.push(latency);

    // ── Decide scaling ──
    let execResult = '—';

    if (action === 'no_change') {
      stats.no_changes++;
      execResult = 'skipped';
    } else {
      const payload = buildExecutorPayload(prediction, latestRow);
      if (payload) {
        const result = await sendToExecutor(payload, dryRun);
        stats.executor_calls++;

        if (result.status === 'error') {
          stats.executor_errors++;
          execResult = `ERR: ${(result.message || '?').slice(0, 15)}`;
        } else if (dryRun) {
          execResult = 'dry-run OK';
        } else {
          execResult = 'executed';
        }

        if (action === 'scale_up')   stats.scale_ups++;
        else                         stats.scale_downs++;

        // ── Validate + Rollback ──
        if (validate) {
          const vresult = await validateScaleAction(data, step, prediction, dryRun);
          if (vresult.status === 'validated') {
            stats.validated++;
            execResult += ' ✓valid';
          } else if (vresult.status === 'rollback') {
            stats.rollbacks++;
            execResult += ' ↩rollback';
          }
        }
      }
    }

    // ── Print row ──
    const display = actionDisplay[action] || action;
    console.log(
      `  ${String(step + 1).padEnd(6)} ${String(windowEndUtc).padEnd(22)} ` +
      `${String(current).padEnd(6)} ${String(predicted).padEnd(6)} ` +
      `${display}  ${execResult.padEnd(20)} ${latency.toFixed(1)}ms`
    );

    // ── Wait ──
    if (intervalSec > 0 && step < totalSteps - 1) {
      await sleep(intervalSec * 1000);
    }
  }

  printSummary(stats);
}

// ============================================================
// 9. Summary printer
// ============================================================

function printSummary(stats) {
  const total = stats.predictions;
  if (total === 0) return;

  const avg = stats.latencies.reduce((a, b) => a + b, 0) / stats.latencies.length;
  const min = Math.min(...stats.latencies);
  const max = Math.max(...stats.latencies);

  const pct = (n) => `(${((n / total) * 100).toFixed(1)}%)`;

  console.log();
  console.log('='.repeat(76));
  console.log('  CONTROLLER SESSION SUMMARY');
  console.log('='.repeat(76));
  console.log(`  Total predictions:     ${total}`);
  console.log(`  Scale ups:             ${String(stats.scale_ups).padStart(4)}  ${pct(stats.scale_ups)}`);
  console.log(`  Scale downs:           ${String(stats.scale_downs).padStart(4)}  ${pct(stats.scale_downs)}`);
  console.log(`  No change:             ${String(stats.no_changes).padStart(4)}  ${pct(stats.no_changes)}`);
  console.log(`  Executor calls sent:   ${stats.executor_calls}`);
  console.log(`  Executor errors:       ${stats.executor_errors}`);
  console.log(`  Validations passed:    ${stats.validated}`);
  console.log(`  Rollbacks triggered:   ${stats.rollbacks}`);
  console.log(`  Avg prediction latency: ${avg.toFixed(1)}ms`);
  console.log(`  Min/Max latency:       ${min.toFixed(1)}ms / ${max.toFixed(1)}ms`);
  console.log('='.repeat(76));
  console.log();
}

// ============================================================
// Utility
// ============================================================

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ============================================================
// CLI Entry Point
// ============================================================

function parseArgs() {
  const args  = process.argv.slice(2);
  const opts  = {
    interval:    60,
    steps:       null,
    dryRun:      false,
    validate:    false,
    startOffset: 0,
    mlApi:       ML_API_URL,
    executor:    EXECUTOR_URL,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    switch (arg) {
      case '--interval':
        opts.interval = parseFloat(args[++i]);
        break;
      case '--steps':
        opts.steps = parseInt(args[++i], 10);
        break;
      case '--dry-run':
        opts.dryRun = true;
        break;
      case '--validate':
        opts.validate = true;
        break;
      case '--start-offset':
        opts.startOffset = parseInt(args[++i], 10);
        break;
      case '--ml-api':
        opts.mlApi = args[++i];
        break;
      case '--executor':
        opts.executor = args[++i];
        break;
      case '--help':
      case '-h':
        console.log(`
Smart Resource Allocation Controller (Node.js)
Connects ML Prediction API to K8s Scaling Executor.

Usage:
  node controller/controller.js [options]

Options:
  --interval <sec>       Seconds between cycles (default: 60). Use 0 for no delay.
  --steps <n>            Max prediction steps (default: all data)
  --dry-run              Log decisions but skip executor calls
  --validate             Re-query ML API after each scale to validate; rollback if conflict
  --start-offset <n>     Skip first N simulation rows (use ~770 for peak traffic)
  --ml-api <url>         ML API base URL (env: ML_API_URL)
  --executor <url>       Executor API URL (env: EXECUTOR_URL)

Examples:
  node controller/controller.js --dry-run --interval 0 --steps 20
  node controller/controller.js --dry-run --validate --interval 0 --steps 60 --start-offset 770
        `);
        process.exit(0);
    }
  }
  return opts;
}

// ── Main ──
(async () => {
  const opts = parseArgs();

  if (opts.mlApi !== ML_API_URL)     ML_API_URL   = opts.mlApi;
  if (opts.executor !== EXECUTOR_URL) EXECUTOR_URL = opts.executor;

  try {
    await runController({
      intervalSec:  opts.interval,
      maxSteps:     opts.steps,
      dryRun:       opts.dryRun,
      executorUrl:  opts.executor,
      startOffset:  opts.startOffset,
      validate:     opts.validate,
    });
  } catch (err) {
    log('ERROR', `Fatal: ${err.message}`);
    process.exit(1);
  }
})();
