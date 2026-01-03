# K8s Autoscaler Prediction API

FastAPI-based REST API for Kubernetes pod count prediction using a hybrid Prophet+LSTM model.

## Features

- **Single Prediction**: Predict pod count for a specific timestamp
- **Batch Prediction**: Predict multiple timestamps in one request
- **Health Check**: Monitor API and model status
- **Auto-generated Docs**: Swagger UI at `/docs`

## Quick Start

### Local Development

```bash
# Navigate to project root
cd informer-k8s-autoscaling

# Install dependencies
pip install -r api/requirements.txt

# Run the API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
# Build image
docker build -t k8s-autoscaler-api -f api/Dockerfile .

# Run container
docker run -p 8000:8000 k8s-autoscaler-api
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
    "timestamp": "2025-11-25T10:30:00",
    "pod_cpu_usage_percent_avg": 65.5,
    "pod_memory_usage_percent_avg": 45.2,
    "requests_per_second": 150.0,
    "response_time_ms_avg": 25.5
}
```

**Response:**
```json
{
    "timestamp": "2025-11-25T10:30:00",
    "predicted_pods": 5,
    "prophet_baseline": 0.42,
    "lstm_residual": 0.03,
    "confidence": "high"
}
```

### Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
    "data": [
        {
            "timestamp": "2025-11-25T10:30:00",
            "pod_cpu_usage_percent_avg": 65.5,
            "pod_memory_usage_percent_avg": 45.2
        },
        {
            "timestamp": "2025-11-25T10:31:00",
            "pod_cpu_usage_percent_avg": 70.0,
            "pod_memory_usage_percent_avg": 48.0
        }
    ]
}
```

### Model Info
```bash
GET /model/info
```

## API Documentation

Once running, access interactive documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Model Architecture

The API uses a hybrid prediction approach:

1. **Prophet Model**: Captures trends, seasonality, and holiday effects
2. **LSTM Model**: Learns residual patterns and short-term dynamics

Final prediction = Prophet baseline + LSTM residual → Inverse scaled to pod count

## Production Deployment

For production, use gunicorn with uvicorn workers:

```bash
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | API port |
| `WORKERS` | 4 | Number of worker processes |
