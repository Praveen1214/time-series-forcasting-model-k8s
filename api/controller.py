"""
Smart Resource Allocation Controller
======================================
Connects the ML Prediction API ↔ K8s Scaling Executor to form
a closed-loop proactive autoscaling system.

Architecture:
                                                         
  ┌──────────────┐   48×21 window   ┌──────────────────┐
  │  Simulation   │ ───────────────► │   ML Prediction  │
  │  Data (CSV)   │                  │   API (FastAPI)  │
  └──────────────┘                   └────────┬─────────┘
                                              │ prediction
                                              ▼
                                     ┌──────────────────┐
                   ┌────────────────►│   This Controller │
                   │  every 60s      │   (Python)        │
                   └─────────────────┤                   │
                                     └────────┬─────────┘
                                              │ scale request
                                              ▼
                                     ┌──────────────────┐
                                     │  K8s Executor    │
                                     │  (Node.js)       │
                                     │  kubectl scale   │
                                     └──────────────────┘

Usage:
  python api/controller.py                          # Real-time (60s interval)
  python api/controller.py --interval 10            # Faster demo (10s)
  python api/controller.py --interval 0 --steps 20  # Instant, 20 steps
  python api/controller.py --dry-run                # Skip executor calls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

# ============================================================
# Configuration
# ============================================================

ML_API_URL = os.environ.get(
    "ML_API_URL",
    "https://k8s-model-api.whiteglacier-fec535bb.southeastasia.azurecontainerapps.io",
)
EXECUTOR_URL = os.environ.get(
    "EXECUTOR_URL",
    "http://localhost:6000/api/v1/scale-with-metrics",
)

LOOKBACK = 48  # Sliding window size (48 rows × 21 features)

# Column order must match the ML API expectation exactly
FEATURE_COLS = [
    "request_rate_rps", "latency_p95_ms", "latency_p99_ms", "error_rate_percent",
    "queue_length", "pod_cpu_usage_percent_avg", "pod_cpu_usage_percent_p95",
    "pod_memory_usage_mb_avg", "pod_memory_usage_mb_p95", "hour_sin", "hour_cos",
    "day_sin", "day_cos", "mesh_inbound_rps", "mesh_inbound_latency_p95",
    "mesh_inbound_error_rate", "degree_centrality", "eigenvector_centrality",
    "betweenness_centrality", "closeness_centrality"
]

# ============================================================
# 1. Data Fetching
# ============================================================

def fetch_simulation_data() -> List[Dict[str, Any]]:
    """
    Fetch simulation data from the ML API's /simulation-data endpoint.
    Returns raw row dicts with all feature columns + current_pod_count.
    """
    log("INFO", "Fetching simulation data from ML API...")
    resp = requests.get(f"{ML_API_URL}/simulation-data", timeout=60)
    resp.raise_for_status()
    body = resp.json()
    data = body["data"]
    log("INFO", f"Loaded {len(data)} simulation rows "
                f"(total dataset: {body['total_rows']})")
    return data


# ============================================================
# 2. Window Builder
# ============================================================

def build_window(rows: List[Dict[str, Any]]) -> Tuple[List[List[float]], str]:
    """
    Convert 48 raw data rows into a 48×21 numeric window
    and return the window_end timestamp.

    Column order: 20 feature columns + current_pod_count (last)
    """
    window: List[List[float]] = []
    for row in rows:
        values = [float(row[col]) for col in FEATURE_COLS]
        values.append(float(row["current_pod_count"]))
        window.append(values)
    window_end_utc = rows[-1]["timestamp"]
    return window, window_end_utc


# ============================================================
# 3. ML API Caller
# ============================================================

def call_predict(
    window_data: List[List[float]],
    window_end_utc: str,
    service_id: str = "Order",
) -> Dict[str, Any]:
    """
    Call POST /predict on the ML API and return the prediction response.
    """
    payload = {
        "window_data": window_data,
        "window_end_utc": window_end_utc,
        "service_id": service_id,
        "input_source": "controller",
    }
    resp = requests.post(f"{ML_API_URL}/predict", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ============================================================
# 4. Prediction → Executor Payload Converter
# ============================================================

def build_executor_payload(
    prediction: Dict[str, Any],
    raw_metrics: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Convert ML prediction + raw metrics into the executor's
    POST /api/v1/scale-with-metrics payload.

    Returns None if no scaling action is needed (no_change).
    """
    current_pods = prediction["current_pods"]
    predicted_pods = prediction["predicted_pods"]
    scale_action = prediction["scale_action"]

    if scale_action == "no_change":
        return None

    # Number of pods to add or remove
    request_pods = abs(predicted_pods - current_pods)

    # Convert error_rate_percent (e.g. 1.45 means 1.45%) to a 0–1 rate
    error_pct = float(raw_metrics.get("error_rate_percent", 0))
    error_rate = min(max(error_pct / 100.0, 0.0), 1.0)   # 1.45% → 0.0145
    success_rate = round(1.0 - error_rate, 4)              # → 0.9855

    cpu_pct = float(raw_metrics.get("pod_cpu_usage_percent_avg", 0))
    mem_pct = float(raw_metrics.get("pod_memory_usage_mb_p95", 0)) / 1024 * 100  # approx % of 1Gi
    mem_pct = min(round(mem_pct, 2), 100.0)

    payload = {
        "services": [
            {
                "deployment": prediction.get("service_id", "Order").lower(),
                "request_pods": request_pods,
                "scale_action": scale_action,
                "metrics": {
                    "successRate": success_rate,
                    "errorRate": round(error_rate, 4),
                    "p95LatencyBefore": round(float(raw_metrics.get("latency_p95_ms", 0)), 2),
                    "p95LatencyAfter": round(float(raw_metrics.get("latency_p95_ms", 0)) * 0.6, 2),  # estimated post-scale
                    "cpuPercent": round(cpu_pct, 2),
                    "memPercent": mem_pct,
                    "restartCount": 0,
                    "trafficRecovery": 0.95,
                },
            }
        ]
    }
    return payload


# ============================================================
# 5. Executor Caller
# ============================================================

def send_to_executor(payload: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """
    Send scaling request to the Node.js executor service.
    In dry-run mode, just logs what would be sent.
    """
    if dry_run:
        log("DRY-RUN", f"Would send to executor: {json.dumps(payload, indent=2)}")
        return {"status": "dry-run", "message": "Skipped (dry-run mode)"}

    try:
        resp = requests.post(EXECUTOR_URL, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        log("WARN", f"Executor not reachable at {EXECUTOR_URL} — is it running?")
        return {"status": "error", "message": "Executor service unreachable"}
    except Exception as e:
        log("ERROR", f"Executor call failed: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================
# 6. Logging Helper
# ============================================================

def log(level: str, message: str):
    """Simple timestamped logger."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    colors = {
        "INFO":     "\033[94m",   # Blue
        "PREDICT":  "\033[96m",   # Cyan
        "SCALE":    "\033[91m",   # Red
        "SKIP":     "\033[90m",   # Gray
        "OK":       "\033[92m",   # Green
        "WARN":     "\033[93m",   # Yellow
        "ERROR":    "\033[91m",   # Red
        "DRY-RUN":  "\033[95m",   # Magenta
        "VALIDATE": "\033[96m",   # Cyan
        "ROLLBACK": "\033[95m",   # Magenta
    }
    reset = "\033[0m"
    color = colors.get(level, "")
    print(f"  {color}[{ts}] [{level:^8}]{reset} {message}")


# ============================================================
# 6b. Validate + Rollback
# ============================================================

def validate_scale_action(
    data: List[Dict[str, Any]],
    step: int,
    original_prediction: Dict[str, Any],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    After a scaling action, re-query the ML API using the NEXT window
    (step+1) to validate whether the scale decision was correct.

    Validation logic:
      - If we scaled UP   but the next prediction says scale_down → ROLLBACK
      - If we scaled DOWN but the next prediction says scale_up  → ROLLBACK
      - Otherwise                                                → VALIDATED ✓

    Returns a dict with keys:
        status     : 'validated' | 'rollback' | 'skipped'
        message    : human-readable reason
        rollback_payload : rollback executor payload or None
    """
    next_start = step + 1
    next_end   = next_start + LOOKBACK

    # Not enough lookahead data to validate
    if next_end > len(data):
        return {"status": "skipped", "message": "Not enough lookahead data", "rollback_payload": None}

    next_rows = data[next_start:next_end]
    try:
        next_window, next_end_utc = build_window(next_rows)
        next_pred = call_predict(next_window, next_end_utc,
                                 service_id=original_prediction.get("service_id", "Order"))
    except Exception as e:
        log("WARN", f"Validate: ML API call failed — {e}")
        return {"status": "skipped", "message": f"ML API error: {e}", "rollback_payload": None}

    orig_action = original_prediction["scale_action"]
    next_action = next_pred["scale_action"]
    orig_pred_pods = original_prediction["predicted_pods"]
    next_curr_pods = next_pred["current_pods"]
    next_pred_pods = next_pred["predicted_pods"]

    # --- Conflict detection ---
    conflict = (
        (orig_action == "scale_up"   and next_action == "scale_down") or
        (orig_action == "scale_down" and next_action == "scale_up")
    )

    if conflict:
        # Build a rollback payload — reverse the original action
        rollback_action = "scale_down" if orig_action == "scale_up" else "scale_up"
        rollback_pods   = abs(orig_pred_pods - next_curr_pods)
        rollback_payload = {
            "services": [{
                "deployment": original_prediction.get("service_id", "Order").lower(),
                "request_pods": max(rollback_pods, 1),
                "scale_action": rollback_action,
                "metrics": {
                    "successRate": 0.95,
                    "errorRate":   0.05,
                    "p95LatencyBefore": 0.0,
                    "p95LatencyAfter":  0.0,
                    "cpuPercent": 0.0,
                    "memPercent": 0.0,
                    "restartCount": 0,
                    "trafficRecovery": 0.90,
                },
            }]
        }
        reason = (
            f"Original action={orig_action} (→{orig_pred_pods} pods) "
            f"but next step says {next_action} (→{next_pred_pods} pods) — ROLLING BACK"
        )
        log("ROLLBACK", reason)
        if dry_run:
            log("DRY-RUN", f"Rollback payload: {json.dumps(rollback_payload, indent=2)}")
        else:
            result = send_to_executor(rollback_payload, dry_run=False)
            log("ROLLBACK", f"Rollback executor result: {result}")
        return {"status": "rollback", "message": reason, "rollback_payload": rollback_payload}
    else:
        reason = (
            f"Confirmed: {orig_action} still valid — "
            f"next window predicts {next_action} ({next_pred_pods} pods)"
        )
        log("VALIDATE", f"✓ {reason}")
        return {"status": "validated", "message": reason, "rollback_payload": None}


# ============================================================
# 7. Main Controller Loop
# ============================================================

def run_controller(
    interval_sec: float = 60,
    max_steps: Optional[int] = None,
    dry_run: bool = False,
    executor_url: str = EXECUTOR_URL,
    start_offset: int = 0,
    validate: bool = False,
):
    global EXECUTOR_URL
    EXECUTOR_URL = executor_url

    # ── Banner ──
    print()
    print("=" * 76)
    print("  SMART RESOURCE ALLOCATION CONTROLLER")
    print("  ML Prediction API  ←→  K8s Scaling Executor")
    print("=" * 76)
    print(f"  ML API:     {ML_API_URL}")
    print(f"  Executor:   {EXECUTOR_URL}")
    print(f"  Interval:   {interval_sec}s" +
          (" (real-time)" if interval_sec == 60 else ""))
    print(f"  Dry-run:    {dry_run}")
    print(f"  Validate:   {validate}  (rollback on model conflict)")
    print(f"  Max steps:  {max_steps or 'unlimited'}")
    print("=" * 76)

    # ── Step 1: Fetch all simulation data once ──
    data = fetch_simulation_data()

    # Skip ahead if requested (to reach busier traffic periods)
    if start_offset > 0:
        data = data[start_offset:]
        log("INFO", f"Skipped first {start_offset} rows (--start-offset)")

    total_steps = len(data) - LOOKBACK
    if max_steps:
        total_steps = min(total_steps, max_steps)
    log("INFO", f"Ready to run {total_steps} prediction steps")
    print()

    # ── Tracking ──
    stats = {
        "predictions": 0,
        "scale_ups": 0,
        "scale_downs": 0,
        "no_changes": 0,
        "executor_calls": 0,
        "executor_errors": 0,
        "validated": 0,
        "rollbacks": 0,
        "latencies": [],
    }

    print(f"  {'Step':<6} {'Time (sim)':<22} {'Curr':<6} {'Pred':<6} "
          f"{'Action':<14} {'Exec Result':<20} {'Latency'}")
    print("  " + "-" * 84)

    try:
        for step in range(total_steps):
            # ── Slide the window ──
            window_rows = data[step : step + LOOKBACK]
            if len(window_rows) < LOOKBACK:
                break

            window_data, window_end_utc = build_window(window_rows)
            latest_row = window_rows[-1]

            # ── Call ML API ──
            try:
                prediction = call_predict(window_data, window_end_utc)
            except Exception as e:
                log("ERROR", f"Step {step + 1}: ML API failed — {e}")
                continue

            current = prediction["current_pods"]
            predicted = prediction["predicted_pods"]
            action = prediction["scale_action"]
            latency = prediction["latency_ms"]

            stats["predictions"] += 1
            stats["latencies"].append(latency)

            # ── Decide scaling ──
            exec_result = "—"

            if action == "no_change":
                stats["no_changes"] += 1
                exec_result = "skipped"
            else:
                # Build payload & send to executor
                payload = build_executor_payload(prediction, latest_row)
                if payload:
                    result = send_to_executor(payload, dry_run=dry_run)
                    stats["executor_calls"] += 1

                    if result.get("status") == "error":
                        stats["executor_errors"] += 1
                        exec_result = f"ERR: {result.get('message', '?')[:15]}"
                    elif dry_run:
                        exec_result = "dry-run OK"
                    else:
                        exec_result = "executed"

                    if action == "scale_up":
                        stats["scale_ups"] += 1
                    else:
                        stats["scale_downs"] += 1

                    # ── Validate + Rollback ──
                    if validate:
                        vresult = validate_scale_action(
                            data, step, prediction, dry_run=dry_run
                        )
                        if vresult["status"] == "validated":
                            stats["validated"] += 1
                            exec_result += " ✓valid"
                        elif vresult["status"] == "rollback":
                            stats["rollbacks"] += 1
                            exec_result += " ↩rollback"

            # ── Color the action ──
            action_colors = {
                "scale_up":   "\033[91m▲ SCALE UP\033[0m  ",
                "scale_down": "\033[92m▼ SCALE DOWN\033[0m",
                "no_change":  "\033[90m— NO CHANGE\033[0m ",
            }
            action_display = action_colors.get(action, action)

            print(f"  {step+1:<6} {window_end_utc:<22} {current:<6} {predicted:<6} "
                  f"{action_display}  {exec_result:<20} {latency:.1f}ms")

            # ── Wait ──
            if interval_sec > 0 and step < total_steps - 1:
                time.sleep(interval_sec)

    except KeyboardInterrupt:
        print("\n")
        log("INFO", "Controller stopped by user (Ctrl+C)")

    # ── Summary ──
    _print_summary(stats)


def _print_summary(stats: dict):
    total = stats["predictions"]
    if total == 0:
        return

    avg_lat = sum(stats["latencies"]) / len(stats["latencies"])
    min_lat = min(stats["latencies"])
    max_lat = max(stats["latencies"])

    print()
    print("=" * 76)
    print("  CONTROLLER SESSION SUMMARY")
    print("=" * 76)
    print(f"  Total predictions:     {total}")
    print(f"  Scale ups:             {stats['scale_ups']:>4}  "
          f"({stats['scale_ups']/total*100:.1f}%)")
    print(f"  Scale downs:           {stats['scale_downs']:>4}  "
          f"({stats['scale_downs']/total*100:.1f}%)")
    print(f"  No change:             {stats['no_changes']:>4}  "
          f"({stats['no_changes']/total*100:.1f}%)")
    print(f"  Executor calls sent:   {stats['executor_calls']}")
    print(f"  Executor errors:       {stats['executor_errors']}")
    print(f"  Validations passed:    {stats['validated']}")
    print(f"  Rollbacks triggered:   {stats['rollbacks']}")
    print(f"  Avg prediction latency:{avg_lat:>7.1f}ms")
    print(f"  Min/Max latency:       {min_lat:.1f}ms / {max_lat:.1f}ms")
    print("=" * 76)
    print()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smart Resource Allocation Controller — "
                    "connects ML Prediction API to K8s Scaling Executor"
    )
    parser.add_argument(
        "--interval", type=float, default=60,
        help="Seconds between prediction cycles (default: 60 = real-time). "
             "Use 0 for no delay."
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Max number of prediction steps (default: all available data)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log scaling decisions but don't call the executor"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="After each scale action, re-query ML API with the next window "
             "to verify the decision. Triggers rollback if the model disagrees."
    )
    parser.add_argument(
        "--start-offset", type=int, default=0,
        help="Skip this many rows into the simulation data "
             "(use ~400 to reach peak traffic)"
    )
    parser.add_argument(
        "--ml-api", type=str, default=ML_API_URL,
        help=f"ML API base URL (default: {ML_API_URL})"
    )
    parser.add_argument(
        "--executor", type=str, default=EXECUTOR_URL,
        help=f"Executor API URL (default: {EXECUTOR_URL})"
    )
    args = parser.parse_args()

    if args.ml_api != ML_API_URL:
        ML_API_URL = args.ml_api

    run_controller(
        interval_sec=args.interval,
        max_steps=args.steps,
        dry_run=args.dry_run,
        executor_url=args.executor,
        start_offset=args.start_offset,
        validate=args.validate,
    )
