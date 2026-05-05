import urllib.request
import json
import urllib.error

url_sim = 'https://k8s-model-api.whiteglacier-fec535bb.southeastasia.azurecontainerapps.io/simulation-data'
with urllib.request.urlopen(url_sim) as response:
    sim_res = json.loads(response.read().decode('utf-8'))

data_list = sim_res.get('data', [])
if len(data_list) < 48:
    print("Not enough data to form a window of 48")
    exit(1)

# Take last 48 rows
window_dicts = data_list[-48:]

FEATURE_COLS = [
    "request_rate_rps", "latency_p95_ms", "latency_p99_ms", "error_rate_percent",
    "queue_length", "pod_cpu_usage_percent_avg", "pod_cpu_usage_percent_p95",
    "pod_memory_usage_mb_avg", "pod_memory_usage_mb_p95", "hour_sin", "hour_cos",
    "day_sin", "day_cos", "mesh_inbound_rps", "mesh_inbound_latency_p95",
    "mesh_inbound_error_rate", "degree_centrality", "eigenvector_centrality",
    "betweenness_centrality", "closeness_centrality"
]

window_data = []
for row in window_dicts:
    # Build list for this row: features + current_pod_count
    row_list = []
    for col in FEATURE_COLS:
        row_list.append(float(row.get(col, 0.0)))
    row_list.append(float(row.get("current_pod_count", 1.0)))
    window_data.append(row_list)

window_end_utc = window_dicts[-1].get("timestamp", "2026-03-10T00:00:00Z")

payload = {
    "window_data": window_data,
    "window_end_utc": window_end_utc,
    "service_id": "Order",
    "input_source": "simulation"
}

req_data = json.dumps(payload).encode('utf-8')

url_predict = 'https://k8s-model-api.whiteglacier-fec535bb.southeastasia.azurecontainerapps.io/predict'
req_predict = urllib.request.Request(url_predict, data=req_data, headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req_predict) as response:
        predict_data = json.loads(response.read().decode('utf-8'))
        print("Prediction Successful!")
        print(json.dumps(predict_data, indent=2))
except urllib.error.HTTPError as e:
    err_msg = e.read().decode('utf-8')
    print("HTTP Error:", e.code)
    try:
        err_json = json.loads(err_msg)
        print(json.dumps(err_json, indent=2))
    except:
        print(err_msg)
except Exception as e:
    print("Exception:", e)
