"""
BiLSTM Pod Prediction Model - Evaluation Script
================================================
Tests the prediction API against historical CSV data.
Compares predicted pod count vs actual pod count at +5 minutes.

Usage:
    1. Start the API server first:
       cd informer-k8s-autoscaling
       python -m uvicorn api.main:app --port 8000
    
    2. Run this test:
       python test_from_csv.py
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests

# ============================================================
# Configuration
# ============================================================

LOOKBACK = 48           # Input window size (minutes)
HORIZON_MIN = 5         # Prediction horizon (minutes)
SAMPLE_INTERVAL = 60    # Sample every N minutes from CSV
MAX_SAMPLES = 200       # Maximum test samples
USE_LAST_N_ROWS = 10000  # Use last N rows from CSV for testing

API_URL = "http://localhost:8000/predict"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = REPO_ROOT / "data" / "data.csv"

# CRITICAL: Feature order MUST match training notebook exactly!
# Training used: feature_cols + ['y'] where 'y' is scaled pod_count
# So input is: [20 features] + [current_pod_count] (API will scale it)
FEATURE_COLS = [
    "request_rate_rps", "latency_p95_ms", "latency_p99_ms", "error_rate_percent",
    "queue_length", "pod_cpu_usage_percent_avg", "pod_cpu_usage_percent_p95",
    "pod_memory_usage_mb_avg", "pod_memory_usage_mb_p95", "hour_sin", "hour_cos",
    "day_sin", "day_cos", "mesh_inbound_rps", "mesh_inbound_latency_p95",
    "mesh_inbound_error_rate", "degree_centrality", "eigenvector_centrality",
    "betweenness_centrality", "closeness_centrality"
]
# Full input: 20 features + pod_count as last column
INPUT_COLS = FEATURE_COLS + ["current_pod_count"]

# ============================================================
# Data Classes
# ============================================================

@dataclass
class PredictionResult:
    timestamp: str
    current_pods: int
    predicted_pods: int
    actual_pods: int
    error: int
    latency_ms: float


# ============================================================
# Helper Functions
# ============================================================

def check_api_health() -> dict:
    """Check if API server is running and return health info."""
    try:
        r = requests.get(f"{API_URL.replace('/predict', '/health')}", timeout=5)
        return r.json() if r.ok else None
    except:
        return None


def load_data() -> pd.DataFrame:
    """Load and prepare CSV data (last N rows)."""
    if not CSV_PATH.exists():
        print(f"❌ CSV not found: {CSV_PATH}")
        sys.exit(1)
    
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Use last N rows for testing
    if len(df) > USE_LAST_N_ROWS:
        df = df.iloc[-USE_LAST_N_ROWS:].reset_index(drop=True)
    
    return df


def make_prediction(window_data: List[List[float]], timestamp: str) -> Tuple[int, float, float]:
    """Call prediction API. Returns (predicted_pods, raw_pred, latency_ms)."""
    payload = {
        "service_id": "Order",
        "window_end_utc": timestamp,
        "input_source": "csv_test",
        "window_data": window_data
    }
    
    start = time.perf_counter()
    r = requests.post(API_URL, json=payload, timeout=20)
    latency = (time.perf_counter() - start) * 1000
    
    r.raise_for_status()
    resp = r.json()
    return resp["predicted_pod_count"], resp.get("raw_prediction", 0), latency


def print_header(health: dict):
    """Print test header."""
    print("\n" + "=" * 70)
    print("  BiLSTM Pod Prediction - Model Evaluation")
    print("=" * 70)
    print(f"  CSV Data:    {CSV_PATH.name} (last {USE_LAST_N_ROWS:,} rows)")
    print(f"  API:         {API_URL}")
    if health:
        print(f"  Model:       {health.get('model_file', 'unknown')}")
        print(f"  Scalers:     {health.get('scalers_file', 'unknown')}")
    print(f"  Lookback:    {LOOKBACK} min")
    print(f"  Horizon:     {HORIZON_MIN} min")
    print("=" * 70 + "\n")


def print_result_row(i: int, total: int, result: PredictionResult):
    """Print single prediction result."""
    error_abs = abs(result.error)
    if error_abs == 0:
        status = "✅ EXACT"
    elif error_abs == 1:
        status = "✅ ±1"
    elif error_abs == 2:
        status = "⚠️  ±2"
    else:
        status = "❌ MISS"
    
    print(f"  [{i+1:3d}/{total}]  {result.timestamp[:16]}  "
          f"Current:{result.current_pods:2d}  Pred:{result.predicted_pods:2d}  "
          f"Actual:{result.actual_pods:2d}  Error:{result.error:+2d}  {status}")


def print_summary(results: List[PredictionResult]):
    """Print evaluation summary statistics."""
    if not results:
        print("❌ No results to summarize")
        return
    
    errors = np.array([r.error for r in results])
    abs_errors = np.abs(errors)
    latencies = np.array([r.latency_ms for r in results])
    
    exact_match = np.mean(errors == 0) * 100
    within_1 = np.mean(abs_errors <= 1) * 100
    within_2 = np.mean(abs_errors <= 2) * 100
    
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\n  📊 Accuracy Metrics:")
    print(f"     • Samples Tested:     {len(results)}")
    print(f"     • Mean Absolute Error: {np.mean(abs_errors):.2f} pods")
    print(f"     • Max Error:          {np.max(abs_errors):+d} pods")
    print(f"     • Std Dev:            {np.std(errors):.2f}")
    
    print(f"\n  🎯 Prediction Accuracy:")
    print(f"     • Exact Match:        {exact_match:5.1f}%  ({int(np.sum(errors == 0))}/{len(results)})")
    print(f"     • Within ±1 pod:      {within_1:5.1f}%  ({int(np.sum(abs_errors <= 1))}/{len(results)})")
    print(f"     • Within ±2 pods:     {within_2:5.1f}%  ({int(np.sum(abs_errors <= 2))}/{len(results)})")
    
    print(f"\n  ⚡ Latency:")
    print(f"     • Mean:               {np.mean(latencies):.1f} ms")
    print(f"     • P95:                {np.percentile(latencies, 95):.1f} ms")
    print(f"     • Max:                {np.max(latencies):.1f} ms")
    
    print("\n  📈 Error Distribution:")
    for e in range(-3, 4):
        count = np.sum(errors == e)
        pct = count / len(results) * 100
        bar = "█" * int(pct / 2)
        print(f"     {e:+2d}: {bar:<25} {count:4d} ({pct:5.1f}%)")
    
    # Over/under provisioning summary
    under = np.sum(errors < 0)
    over = np.sum(errors > 0)
    exact = np.sum(errors == 0)
    print(f"\n  📦 Provisioning:")
    print(f"     • Under-provisioned:  {under:4d} ({under/len(results)*100:5.1f}%)")
    print(f"     • Over-provisioned:   {over:4d} ({over/len(results)*100:5.1f}%)")
    print(f"     • Exact:              {exact:4d} ({exact/len(results)*100:5.1f}%)")
    
    print("\n" + "=" * 70 + "\n")


# ============================================================
# Main
# ============================================================

def main():
    # Check API
    print("\n  Checking API connection...", end=" ")
    health = check_api_health()
    if not health:
        print("❌ FAILED")
        print("\n  ⚠️  API server not running!")
        print("  Start it with:")
        print("    cd informer-k8s-autoscaling")
        print("    python -m uvicorn api.main:app --port 8000\n")
        sys.exit(1)
    print("✅ OK")
    
    # Load data
    print("  Loading CSV data...", end=" ")
    df = load_data()
    print(f"✅ {len(df):,} rows\n")
    
    print_header(health)
    
    # Generate test points
    max_start = len(df) - LOOKBACK - HORIZON_MIN
    test_indices = list(range(0, max_start, SAMPLE_INTERVAL))[:MAX_SAMPLES]
    
    print(f"  Running {len(test_indices)} predictions...\n")
    print("-" * 70)
    
    results: List[PredictionResult] = []
    
    for i, start_idx in enumerate(test_indices):
        end_idx = start_idx + LOOKBACK
        window_df = df.iloc[start_idx:end_idx]
        
        # CRITICAL: Use correct column order [20 features] + [pod_count]
        window_data = window_df[INPUT_COLS].astype(float).values.tolist()
        timestamp = window_df["timestamp"].iloc[-1].strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Get actual pod count at +5 min
        actual_idx = end_idx + HORIZON_MIN - 1
        actual_pods = int(df.iloc[actual_idx]["current_pod_count"])
        current_pods = int(window_df["current_pod_count"].iloc[-1])
        
        try:
            predicted_pods, raw_pred, latency = make_prediction(window_data, timestamp)
            
            result = PredictionResult(
                timestamp=timestamp,
                current_pods=current_pods,
                predicted_pods=predicted_pods,
                actual_pods=actual_pods,
                error=predicted_pods - actual_pods,
                latency_ms=latency
            )
            results.append(result)
            
            # Print every 10th result
            if (i + 1) % 10 == 0 or i == 0:
                print_result_row(i, len(test_indices), result)
                
        except requests.exceptions.ConnectionError:
            print(f"\n  ❌ Lost connection to API")
            break
        except Exception as e:
            print(f"\n  ❌ Error at sample {i}: {e}")
            continue
    
    print("-" * 70)
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()

