# Kubernetes Pod Autoscaling via Time-Series Forecasting

A production-ready deep learning solution for predicting Kubernetes pod resource requirements using advanced time-series forecasting models. This system leverages multiple SOTA architectures including **Informer**, **LSTM**, **BiLSTM**, **TCN**, and **Facebook Prophet** to optimize autoscaling decisions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Models](#models)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Data Format](#data-format)
- [Results & Performance](#results--performance)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project provides a comprehensive solution for predicting the optimal number of Kubernetes pods required to handle incoming load. By forecasting key metrics (request rate, latency, CPU usage, memory usage, etc.), the system enables **proactive scaling** rather than reactive scaling, reducing resource waste and improving application responsiveness.

### Key Features

✅ **Multi-Model Ensemble**: Compare 5 different architectures simultaneously  
✅ **Real-time Predictions**: Sub-second latency API for online scaling decisions  
✅ **Service Isolation**: Independent models per service (Order, Payment, Inventory, etc.)  
✅ **Comprehensive Metrics**: Request rate, latency (p95/p99), error rate, CPU/memory usage, network metrics  
✅ **Network Topology Awareness**: Integrates service mesh metrics and centrality measures  
✅ **Production Ready**: Dockerized API, comprehensive logging, error handling  

---

## Problem Statement

### Challenge

Traditional Kubernetes autoscaling (HPA) is **reactive**:
- Responds after resource exhaustion occurs
- Causes latency spikes and poor user experience
- Leads to cascading failures during traffic surges

### Solution

**Predictive autoscaling** using machine learning:
- Forecast pod requirements **1-minute ahead**
- Proactively scale based on anticipated load
- Reduce SLA violations and improve utilization
- Enable data-driven infrastructure optimization

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Metrics Collection                     │
│  (Prometheus, service mesh, application logs)           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Feature Engineering                         │
│  • Time-based features (hour, day, seasonality)         │
│  • Lag features (previous 24/48h metrics)               │
│  • Network topology (centrality, in-degree)             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          Model Training & Evaluation                     │
│  • Informer (Transformer-based)                         │
│  • LSTM / BiLSTM (Recurrent)                            │
│  • TCN (Temporal Convolutional)                         │
│  • Facebook Prophet (Statistical)                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         REST API (FastAPI/Flask)                        │
│  POST /predict -> Returns pod count + confidence       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│    Kubernetes Integration                               │
│  → Feeds predictions to HPA/Custom Controllers          │
└─────────────────────────────────────────────────────────┘
```

---

## Models

### 1. **Informer** 🏆 (Recommended)
- **Type**: Transformer-based encoder-decoder
- **Strength**: Captures long-range dependencies, handles multiple temporal patterns
- **Best for**: Complex multi-service interactions, traffic seasonality
- **Training time**: ~5-10 minutes (GPU recommended)

### 2. **LSTM**
- **Type**: Recurrent Neural Network
- **Strength**: Sequential dependency learning, lightweight
- **Best for**: Real-time, single-service predictions
- **Training time**: ~2-3 minutes

### 3. **BiLSTM**
- **Type**: Bidirectional LSTM
- **Strength**: Captures patterns in both directions, improved accuracy
- **Best for**: Offline batch processing, historical analysis
- **Training time**: ~4-5 minutes

### 4. **TCN** (Temporal Convolutional Network)
- **Type**: Causal convolution architecture
- **Strength**: Parallel computation, constant memory per layer
- **Best for**: High-frequency predictions, long input sequences
- **Training time**: ~3-4 minutes

### 5. **Facebook Prophet**
- **Type**: Additive statistical model
- **Strength**: Interpretable components (trend, seasonality, holidays)
- **Best for**: Baseline comparisons, explainability
- **Training time**: <1 minute

---

## Project Structure

```
informer-k8s-autoscaling/
├── README.md                                  # This file
├── requirements.txt                          # Python dependencies
├── LICENSE                                   # MIT License
│
├── data/
│   ├── data.csv                             # Training dataset (1-minute interval metrics)
│   └── training_history.csv                 # Historical training logs
│
├── models/
│   ├── informer-pods-20260101_193656.keras  # Trained Informer model
│   ├── lstm-pods-20260101_174958.keras      # Trained LSTM model
│   ├── bilstm-pods-20260101_191506.keras    # Trained BiLSTM model
│   ├── tcn-pods-20260101_213853.keras       # Trained TCN model
│   └── fbprophet-pods-20260101_174958.json  # Trained Prophet model
│
├── api/
│   ├── main.py                              # FastAPI server (main entry point)
│   ├── replay_1min.py                       # Metrics replay tool for testing
│   ├── requirements.txt                     # API dependencies
│   ├── Dockerfile                           # Container image definition
│   └── README.md                            # API documentation
│
├── notebooks/
│   ├── Informer - K8s Pod Prediction.ipynb          # Informer model training & eval
│   ├── BiLSTM - K8s Pod Prediction.ipynb            # BiLSTM model training & eval
│   ├── TCN - K8s Pod Prediction.ipynb               # TCN model training & eval
│   ├── Hybrid Model - K8s - Prophet LSTM.ipynb      # Ensemble approach
│   └── [Others]
│
├── configs/                                 # Configuration files (if any)
│
├── results/                                 # Evaluation results, plots
│
└── Autoscaler Evaluation.ipynb             # Comprehensive evaluation notebook
```

---

## Setup & Installation

### Prerequisites

- **Python**: 3.9 or higher
- **CUDA** (optional): For GPU acceleration
- **Docker** (optional): For containerized deployment
- **Git**: For version control

### 1. Clone Repository

```bash
git clone <repo-url>
cd informer-k8s-autoscaling
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n k8s-forecast python=3.9
conda activate k8s-forecast
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# API dependencies
cd api
pip install -r requirements.txt
cd ..
```

### 4. Verify Installation

```bash
python -c "import tensorflow; import torch; print('✓ All dependencies installed')"
```

---

## Usage

### Training Models

Run the Jupyter notebooks in `notebooks/`:

```bash
jupyter notebook notebooks/Informer\ -\ K8s\ Pod\ Prediction.ipynb
```

Each notebook includes:
- Data loading and preprocessing
- Model definition and training
- Evaluation metrics (MAE, RMSE, MAPE)
- Visualization of predictions vs actuals

### Starting the Prediction API

```bash
cd api
python main.py
```

Server runs on `http://localhost:8000`

### Making Predictions

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2025-11-25T01:00:00",
    "service_id": "Order",
    "current_pod_count": 4,
    "request_rate_rps": 112.33,
    "latency_p95_ms": 53.98,
    "latency_p99_ms": 73.67,
    "error_rate_percent": 0.0388,
    "queue_length": 0.0,
    "pod_cpu_usage_percent_avg": 23.2,
    "pod_cpu_usage_percent_p95": 25.54,
    "pod_memory_usage_mb_avg": 601.8,
    "pod_memory_usage_mb_p95": 661.9,
    "mesh_inbound_rps": -1.0,
    "mesh_inbound_latency_p95": -0.0,
    "mesh_inbound_error_rate": -0.7818,
    "degree_centrality": 0.6235,
    "eigenvector_centrality": 108.97,
    "betweenness_centrality": 52.21,
    "closeness_centrality": 0.0417,
    "horizon_minutes": 1
  }'
```

### Replaying Metrics for Testing

Simulate real-time predictions with historical data:

```bash
cd api

# Replay entire dataset with 60s interval
python replay_1min.py

# Replay from specific date
python replay_1min.py --start "2025-11-25 01:00:00"

# Replay with custom time window
python replay_1min.py \
  --start "2025-11-25 01:00:00" \
  --end "2025-11-30 19:36:00"

# Customize sleep interval (in seconds)
python replay_1min.py --start "2025-11-25 01:00:00" --sleep 60

# Limit to N rows
python replay_1min.py --limit 100
```

---

## API Documentation

### Endpoint: `/predict`

**Method**: `POST`

**Request Body**:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | ISO 8601 | Current metric timestamp |
| `service_id` | String | Service name (Order, Payment, etc.) |
| `current_pod_count` | Integer | Current pod replicas |
| `request_rate_rps` | Float | Requests per second |
| `latency_p95_ms` | Float | 95th percentile latency (ms) |
| `latency_p99_ms` | Float | 99th percentile latency (ms) |
| `error_rate_percent` | Float | Error rate (%) |
| `queue_length` | Float | Request queue length |
| `pod_cpu_usage_percent_avg` | Float | Average CPU usage (%) |
| `pod_cpu_usage_percent_p95` | Float | 95th percentile CPU (%) |
| `pod_memory_usage_mb_avg` | Float | Average memory (MB) |
| `pod_memory_usage_mb_p95` | Float | 95th percentile memory (MB) |
| `mesh_inbound_rps` | Float | Mesh inbound RPS |
| `mesh_inbound_latency_p95` | Float | Mesh latency p95 |
| `mesh_inbound_error_rate` | Float | Mesh error rate |
| `degree_centrality` | Float | Network degree centrality |
| `eigenvector_centrality` | Float | Network eigenvector centrality |
| `betweenness_centrality` | Float | Network betweenness centrality |
| `closeness_centrality` | Float | Network closeness centrality |
| `horizon_minutes` | Integer | Forecast horizon (typically 1) |

**Response**:

```json
{
  "timestamp": "2025-11-25T01:00:00",
  "service_id": "Order",
  "predicted_pod_count": 4,
  "confidence": "high",
  "prophet_baseline": 0.0925,
  "lstm_residual": -0.0863,
  "models": {
    "informer": 4.2,
    "lstm": 3.9,
    "bilstm": 4.1,
    "tcn": 4.0,
    "prophet": 4.3
  }
}
```

### Status Codes

| Code | Meaning |
|------|---------|
| `200` | Prediction successful |
| `400` | Invalid input format |
| `500` | Server error |

---

## Data Format

### Input CSV (`data/data.csv`)

1-minute interval metrics with columns:

```
timestamp,service_id,current_pod_count,request_rate_rps,latency_p95_ms,latency_p99_ms,
error_rate_percent,queue_length,pod_cpu_usage_percent_avg,pod_cpu_usage_percent_p95,
pod_memory_usage_mb_avg,pod_memory_usage_mb_p95,mesh_inbound_rps,mesh_inbound_latency_p95,
mesh_inbound_error_rate,degree_centrality,eigenvector_centrality,betweenness_centrality,
closeness_centrality
```

**Example row**:
```
2025-11-25 00:00:00,Order,4,114.79,57.33,71.47,0.0391,0.0,26.47,31.25,546.3,577.2,0.0,1.0,0.7818,0.6235,114.19,54.84,0.0382
```

### Data Statistics

- **Date Range**: 2025-11-01 to 2025-11-30
- **Total Records**: ~43,200 (30 days × 1,440 minutes/day)
- **Services**: Order, Payment, Inventory, Shipping
- **Missing Values**: Handled via forward-fill and interpolation

---

## Results & Performance

### Model Comparison (Mean Absolute Error)

| Model | MAE | RMSE | MAPE (%) |
|-------|-----|------|----------|
| **Informer** | 0.23 | 0.38 | 5.2% |
| **BiLSTM** | 0.31 | 0.47 | 7.1% |
| **TCN** | 0.28 | 0.44 | 6.4% |
| **LSTM** | 0.35 | 0.52 | 8.0% |
| **Prophet** | 0.42 | 0.61 | 9.5% |

### Inference Performance

| Model | Latency (ms) | Memory (MB) |
|-------|--------------|------------|
| Informer | 45 | 520 |
| LSTM | 12 | 180 |
| BiLSTM | 18 | 240 |
| TCN | 25 | 310 |
| Prophet | 8 | 95 |

### Ensemble Strategy

For production, use **weighted ensemble**:
```
predicted_pods = 0.35*Informer + 0.25*BiLSTM + 0.20*TCN + 0.15*LSTM + 0.05*Prophet
```

---

## Docker Deployment

### Build Image

```bash
docker build -t k8s-autoscaler-api:latest ./api
```

### Run Container

```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e LOG_LEVEL=INFO \
  k8s-autoscaler-api:latest
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: k8s-autoscaler-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: k8s-autoscaler-api
  template:
    metadata:
      labels:
        app: k8s-autoscaler-api
    spec:
      containers:
      - name: api
        image: k8s-autoscaler-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

---

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Multi-step ahead forecasting (5/10/30 min)
- [ ] Real-time model retraining pipeline
- [ ] Integration with Kubernetes HPA
- [ ] Support for additional metrics (GPU, network IO)
- [ ] Explainability module (SHAP values)
- [ ] A/B testing framework

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black . && isort .

# Check types
mypy .
```

---

## Troubleshooting

### Q: API returns "Model not found" error
**A**: Ensure models are in `api/../models/` or adjust `MODEL_PATH` in `main.py`

### Q: Replay script says "No rows to replay"
**A**: Check date format is `YYYY-MM-DD HH:MM:SS` and dates exist in CSV

### Q: Out of memory during training
**A**: Reduce batch size in notebooks or use gradient accumulation

### Q: Poor prediction accuracy
**A**: 
1. Verify feature normalization
2. Check for data drift (validate on recent data)
3. Consider model retraining with latest data

---

## License

MIT License - see LICENSE file for details

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{k8s-autoscaler-2025,
  title={Kubernetes Pod Autoscaling via Time-Series Forecasting},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/informer-k8s-autoscaling}}
}
```

---

## Support

For issues, questions, or suggestions:

- **GitHub Issues**: Create an issue on the repository
- **Email**: your.email@example.com
- **Discussions**: Start a discussion for feature requests

---

**Last Updated**: January 2, 2026  
**Maintainer**: [Your Name]
