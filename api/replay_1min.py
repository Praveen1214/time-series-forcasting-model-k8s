import argparse
import asyncio
from datetime import datetime

import httpx
import pandas as pd

DEFAULT_URL = "http://localhost:8000/predict"
DEFAULT_SERVICE = "Order"


def row_to_payload(row: pd.Series) -> dict:
    return {
        "timestamp": pd.to_datetime(row["timestamp"]).isoformat(),
        "service_id": row["service_id"],
        "current_pod_count": int(row["current_pod_count"]),
        "request_rate_rps": float(row["request_rate_rps"]),
        "latency_p95_ms": float(row["latency_p95_ms"]),
        "latency_p99_ms": float(row["latency_p99_ms"]),
        "error_rate_percent": float(row["error_rate_percent"]),
        "queue_length": float(row["queue_length"]),
        "pod_cpu_usage_percent_avg": float(row["pod_cpu_usage_percent_avg"]),
        "pod_cpu_usage_percent_p95": float(row["pod_cpu_usage_percent_p95"]),
        "pod_memory_usage_mb_avg": float(row["pod_memory_usage_mb_avg"]),
        "pod_memory_usage_mb_p95": float(row["pod_memory_usage_mb_p95"]),
        "mesh_inbound_rps": float(row["mesh_inbound_rps"]),
        "mesh_inbound_latency_p95": float(row["mesh_inbound_latency_p95"]),
        "mesh_inbound_error_rate": float(row["mesh_inbound_error_rate"]),
        "degree_centrality": float(row["degree_centrality"]),
        "eigenvector_centrality": float(row["eigenvector_centrality"]),
        "betweenness_centrality": float(row["betweenness_centrality"]),
        "closeness_centrality": float(row["closeness_centrality"]),
        "horizon_minutes": 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay 1-minute metrics to /predict")
    parser.add_argument("--csv", default="../data/data.csv", help="Path to data.csv")
    parser.add_argument("--url", default=DEFAULT_URL, help="Prediction endpoint")
    parser.add_argument("--service", default=DEFAULT_SERVICE, help="Service id to filter")
    parser.add_argument("--start", help="Start timestamp (inclusive) e.g. '2025-11-30 18:04:00'")
    parser.add_argument("--end", help="End timestamp (inclusive) e.g. '2025-11-30 19:36:00'")
    parser.add_argument(
        "--sleep",
        type=float,
        default=60.0,
        help="Seconds to sleep between calls (default 60s to keep 1-minute spacing)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows")
    return parser.parse_args()


async def main():
    args = parse_args()
    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    df = df[df["service_id"] == args.service].copy()
    df = df.sort_values("timestamp")

    if args.start:
        df = df[df["timestamp"] >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df["timestamp"] <= pd.to_datetime(args.end)]
    if args.limit:
        df = df.head(args.limit)

    if df.empty:
        print("No rows to replay after filtering.")
        return

    print(f"Replaying {len(df)} rows for service={args.service} to {args.url}")

    async with httpx.AsyncClient(timeout=10.0) as client:
        for _, row in df.iterrows():
            payload = row_to_payload(row)
            try:
                resp = await client.post(args.url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    print(
                        f"{payload['timestamp']} -> predicted={data['predicted_pod_count']} "
                        f"(conf={data['confidence']}, baseline={data['prophet_baseline']:.4f}, "
                        f"residual={data['lstm_residual']:.4f})"
                    )
                else:
                    print(f"{payload['timestamp']} -> HTTP {resp.status_code}: {resp.text}")
            except Exception as exc:
                print(f"{payload['timestamp']} -> error: {exc}")

            if args.sleep > 0:
                await asyncio.sleep(args.sleep)


if __name__ == "__main__":
    asyncio.run(main())
