"""
Prediction-Based Autoscaler API
Prophet + LSTM Hybrid Model for Kubernetes Pod Count Prediction

Handles continuous 1-minute interval metrics from Kubernetes monitoring.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.preprocessing import MinMaxScaler
from prophet.serialize import model_from_json
from keras.models import load_model
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# Configuration
# ============================================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LOOK_BACK = 1  # LSTM look-back window (must match training)
# The LSTM was trained for single-step (1-minute) residuals. Enforce horizon=1 to avoid
# misleading multi-step outputs that were never learned by the model.
DEFAULT_HORIZON_MINUTES = 1

# Feature columns (must match training order exactly)
FEATURE_COLS = [
    "request_rate_rps",
    "latency_p95_ms",
    "latency_p99_ms",
    "error_rate_percent",
    "queue_length",
    "pod_cpu_usage_percent_avg",
    "pod_cpu_usage_percent_p95",
    "pod_memory_usage_mb_avg",
    "pod_memory_usage_mb_p95",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "mesh_inbound_rps",
    "mesh_inbound_latency_p95",
    "mesh_inbound_error_rate",
    "degree_centrality",
    "eigenvector_centrality",
    "betweenness_centrality",
    "closeness_centrality",
]

# Columns from API input (before time feature derivation)
API_INPUT_COLS = [
    "request_rate_rps",
    "latency_p95_ms",
    "latency_p99_ms",
    "error_rate_percent",
    "queue_length",
    "pod_cpu_usage_percent_avg",
    "pod_cpu_usage_percent_p95",
    "pod_memory_usage_mb_avg",
    "pod_memory_usage_mb_p95",
    "mesh_inbound_rps",
    "mesh_inbound_latency_p95",
    "mesh_inbound_error_rate",
    "degree_centrality",
    "eigenvector_centrality",
    "betweenness_centrality",
    "closeness_centrality",
]


# ============================================================================
# Pydantic Models
# ============================================================================
class MetricsInput(BaseModel):
    """Input schema for Kubernetes metrics data (1-minute interval)"""
    timestamp: str = Field(..., description="ISO format timestamp, e.g., '2025-11-30T19:12:00'")
    service_id: str = Field(..., description="Service identifier, e.g., 'Order'")
    current_pod_count: int = Field(..., ge=1, description="Current number of running pods")
    request_rate_rps: float = Field(..., ge=0, description="Request rate in requests per second")
    latency_p95_ms: float = Field(..., ge=0, description="95th percentile latency in ms")
    latency_p99_ms: float = Field(..., ge=0, description="99th percentile latency in ms")
    error_rate_percent: float = Field(..., ge=0, le=100, description="Error rate percentage")
    queue_length: float = Field(..., ge=0, description="Request queue length")
    pod_cpu_usage_percent_avg: float = Field(..., ge=0, description="Average CPU usage percentage")
    pod_cpu_usage_percent_p95: float = Field(..., ge=0, description="95th percentile CPU usage")
    pod_memory_usage_mb_avg: float = Field(..., ge=0, description="Average memory usage in MB")
    pod_memory_usage_mb_p95: float = Field(..., ge=0, description="95th percentile memory usage")
    mesh_inbound_rps: float = Field(..., ge=0, description="Service mesh inbound RPS")
    mesh_inbound_latency_p95: float = Field(..., ge=0, description="Mesh inbound p95 latency")
    mesh_inbound_error_rate: float = Field(..., ge=0, description="Mesh inbound error rate")
    degree_centrality: float = Field(..., ge=0, le=1, description="Graph degree centrality")
    eigenvector_centrality: float = Field(..., ge=0, le=1, description="Graph eigenvector centrality")
    betweenness_centrality: float = Field(..., ge=0, le=1, description="Graph betweenness centrality")
    closeness_centrality: float = Field(..., ge=0, le=1, description="Graph closeness centrality")
    horizon_minutes: int = Field(
        default=1,
        ge=1,
        le=1,
        description="Prediction horizon in minutes. Model is trained for single-step (1 minute) only.",
    )


class PredictionResponse(BaseModel):
    """Response schema for pod count prediction"""
    timestamp: str
    service_id: str
    current_pod_count: int
    predicted_pod_count: int
    prediction_for_timestamp: str  # The future timestamp this prediction is for
    horizon_minutes: int
    prophet_baseline: float
    lstm_residual: float
    confidence: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: bool
    history_size: int
    last_prediction_time: Optional[str] = None
    total_predictions: int = 0


class BatchMetricsInput(BaseModel):
    """Batch input for multiple metrics"""
    metrics: List[MetricsInput]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    processed_count: int


# ============================================================================
# Model Manager (Singleton with Thread Safety)
# ============================================================================
class HybridModelManager:
    """Manages Prophet + LSTM hybrid model for pod prediction (thread-safe)"""
    
    def __init__(self):
        self.prophet_model = None
        self.lstm_model = None
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler()
        self.history_buffer = []  # Stores recent scaled features + residuals
        self.is_loaded = False
        self._lock = Lock()  # Thread safety for concurrent requests
        self.last_prediction_time = None
        self.total_predictions = 0
        self.last_timestamp = None  # Track last processed timestamp
    
    def load_models(self):
        """Load Prophet and LSTM models, fit scalers on training data"""
        try:
            # Load Prophet model
            prophet_path = os.path.join(MODEL_DIR, "fbprophet-pods-20260101_174958.json")
            with open(prophet_path, "r") as f:
                self.prophet_model = model_from_json(f.read())
            logger.info(f"Prophet model loaded from {prophet_path}")
            
            # Load LSTM model
            lstm_path = os.path.join(MODEL_DIR, "lstm-pods-20260101_174958.keras")
            self.lstm_model = load_model(lstm_path)
            logger.info(f"LSTM model loaded from {lstm_path}")
            
            # Fit scalers on training data
            self._fit_scalers()
            
            self.is_loaded = True
            logger.info("Hybrid model ready for predictions")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _fit_scalers(self):
        """Fit scalers on full training dataset for consistency"""
        data_path = os.path.join(DATA_DIR, "data.csv")
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
        
        # Filter to primary service (Order)
        primary_service = df["service_id"].mode()[0]
        df = df[df["service_id"] == primary_service].copy()
        
        # Fit target scaler (without feature names to avoid warnings)
        self.target_scaler.fit(df[["current_pod_count"]].values)
        
        # Fit feature scaler on FEATURE_COLS in exact order (without feature names)
        self.feature_scaler.fit(df[FEATURE_COLS].values)
        
        logger.info(f"Scalers fitted on {len(df)} training samples (service: {primary_service})")
    
    def _derive_time_features(self, timestamp: datetime) -> dict:
        """Derive cyclical time features from timestamp"""
        hour = timestamp.hour + timestamp.minute / 60.0
        day_of_week = timestamp.weekday()
        
        return {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "day_sin": np.sin(2 * np.pi * day_of_week / 7),
            "day_cos": np.cos(2 * np.pi * day_of_week / 7),
        }
    
    def predict(self, metrics: MetricsInput) -> PredictionResponse:
        """
        Predict pod count N minutes ahead using Prophet + LSTM hybrid model (thread-safe)
        
        Flow:
        1. Parse timestamp and calculate future timestamp (+ horizon_minutes)
        2. Prophet predicts baseline for FUTURE timestamp (captures seasonality)
        3. Calculate current residual from actual vs Prophet baseline
        4. LSTM predicts residual correction based on current patterns
        5. Combine: final = Prophet future baseline + LSTM residual
        6. Inverse scale and round to get pod count
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded")
        
        with self._lock:  # Thread-safe prediction
            # Parse timestamp
            ts = datetime.fromisoformat(metrics.timestamp)
            horizon = getattr(metrics, 'horizon_minutes', DEFAULT_HORIZON_MINUTES)

            if horizon != 1:
                raise ValueError(
                    "horizon_minutes must be 1. The LSTM was trained for single-step (1-minute) predictions; "
                    "multi-step forecasting would require retraining or iterative rollout with predicted features."
                )
            
            # Calculate future timestamp for prediction
            future_ts = ts + pd.Timedelta(minutes=horizon)
            
            # Check for duplicate/out-of-order timestamps
            if self.last_timestamp and ts <= self.last_timestamp:
                logger.warning(f"Out-of-order timestamp: {ts} <= {self.last_timestamp}")
            
            # Step 1: Prophet baseline prediction for FUTURE timestamp
            prophet_input = pd.DataFrame({"ds": [future_ts]})
            prophet_forecast = self.prophet_model.predict(prophet_input)
            prophet_yhat_future = prophet_forecast["yhat"].values[0]  # Future scaled prediction
            
            # Also get Prophet prediction for current time (for residual calculation)
            prophet_current = self.prophet_model.predict(pd.DataFrame({"ds": [ts]}))
            prophet_yhat_current = prophet_current["yhat"].values[0]
            
            # Step 2: Scale current pod count
            current_scaled = self.target_scaler.transform([[metrics.current_pod_count]])[0, 0]
            
            # Step 3: Calculate actual residual (current actual vs current Prophet)
            actual_residual = current_scaled - prophet_yhat_current
            
            # Step 4: Prepare features for LSTM
            time_features = self._derive_time_features(ts)
            
            # Build feature vector in exact training order
            feature_dict = {col: getattr(metrics, col) for col in API_INPUT_COLS}
            feature_dict.update(time_features)
            
            # Create feature array in correct order
            feature_values = [feature_dict[col] for col in FEATURE_COLS]
            feature_array = np.array(feature_values).reshape(1, -1)
            
            # Scale features
            scaled_features = self.feature_scaler.transform(feature_array)[0]
            
            # Append residual to create LSTM input
            lstm_features = np.append(scaled_features, actual_residual)
            
            # Update history buffer
            self.history_buffer.append(lstm_features)
            if len(self.history_buffer) > LOOK_BACK:
                self.history_buffer = self.history_buffer[-LOOK_BACK:]
            
            # Step 5: LSTM residual prediction
            if len(self.history_buffer) >= LOOK_BACK:
                lstm_input = np.array(self.history_buffer[-LOOK_BACK:]).reshape(1, LOOK_BACK, -1)
                lstm_residual_pred = self.lstm_model.predict(lstm_input, verbose=0)[0, 0]
            else:
                # Not enough history, use actual residual as prediction
                lstm_residual_pred = actual_residual
            
            # Step 6: Combine predictions (Prophet FUTURE baseline + LSTM residual)
            # The residual captures the deviation pattern, applied to future baseline
            final_scaled = prophet_yhat_future + lstm_residual_pred
            
            # Step 7: Inverse transform to pod count
            predicted_pods = self.target_scaler.inverse_transform([[final_scaled]])[0, 0]
            predicted_pods = int(max(1, round(predicted_pods)))  # Ensure minimum 1 pod
            
            # Update tracking
            self.last_timestamp = ts
            self.last_prediction_time = datetime.now().isoformat()
            self.total_predictions += 1
            
            # Determine confidence based on history availability
            confidence = "high" if len(self.history_buffer) >= LOOK_BACK else "warming_up"
            
            # Log prediction
            logger.info(
                f"Prediction #{self.total_predictions} | "
                f"ts={metrics.timestamp} | horizon={horizon}min | "
                f"current={metrics.current_pod_count} | predicted={predicted_pods} | "
                f"for={future_ts.isoformat()}"
            )
            
            return PredictionResponse(
                timestamp=metrics.timestamp,
                service_id=metrics.service_id,
                current_pod_count=metrics.current_pod_count,
                predicted_pod_count=predicted_pods,
                prediction_for_timestamp=future_ts.isoformat(),
                horizon_minutes=horizon,
                prophet_baseline=round(float(prophet_yhat_future), 4),
                lstm_residual=round(float(lstm_residual_pred), 4),
                confidence=confidence,
            )
    
    def clear_history(self):
        """Clear the history buffer (useful for testing or service restarts)"""
        with self._lock:
            self.history_buffer = []
            self.last_timestamp = None
            logger.info("History buffer cleared")


# Global model manager instance
model_manager = HybridModelManager()


# ============================================================================
# FastAPI Application
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - load models on startup"""
    logger.info("=" * 60)
    logger.info("Starting Prediction-Based Autoscaler API")
    logger.info("=" * 60)
    model_manager.load_models()
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Prediction-Based Autoscaler API",
    description="Prophet + LSTM Hybrid Model for Kubernetes Pod Count Prediction",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status"""
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "unhealthy",
        models_loaded=model_manager.is_loaded,
        history_size=len(model_manager.history_buffer),
        last_prediction_time=model_manager.last_prediction_time,
        total_predictions=model_manager.total_predictions,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_pods(metrics: MetricsInput):
    """
    Predict the pod count N minutes ahead based on current metrics.
    
    **Default: 1 minute ahead** (the trained model supports only single-step forecasts)
    
    The hybrid model combines:
    - **Prophet**: Captures trend and seasonality patterns for the FUTURE timestamp
    - **LSTM**: Learns residual corrections from current short-term dynamics
    
    Multi-minute horizons require retraining or an iterative rollout strategy with
    predicted features, which is not enabled here to avoid misleading outputs.
    """
    try:
        return model_manager.predict(metrics)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_pods_batch(batch: BatchMetricsInput):
    """
    Process multiple metrics in order (for catching up after downtime).
    
    Metrics should be sorted by timestamp in ascending order.
    Each prediction updates the history buffer for subsequent predictions.
    """
    try:
        predictions = []
        for metrics in batch.metrics:
            pred = model_manager.predict(metrics)
            predictions.append(pred)
        
        return BatchPredictionResponse(
            predictions=predictions,
            processed_count=len(predictions),
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/reset")
async def reset_history():
    """Reset the prediction history buffer (useful after service restarts)"""
    model_manager.clear_history()
    return {"status": "success", "message": "History buffer cleared"}


@app.get("/stats")
async def get_stats():
    """Get prediction statistics"""
    return {
        "total_predictions": model_manager.total_predictions,
        "history_size": len(model_manager.history_buffer),
        "last_prediction_time": model_manager.last_prediction_time,
        "last_timestamp_processed": model_manager.last_timestamp.isoformat() if model_manager.last_timestamp else None,
        "models_loaded": model_manager.is_loaded,
    }


@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Prediction-Based Autoscaler API",
        "model": "Prophet + LSTM Hybrid",
        "version": "1.0.0",
        "description": "Handles 1-minute interval metrics for pod count prediction",
        "endpoints": {
            "/predict": "POST - Predict next pod count (single)",
            "/predict/batch": "POST - Predict for multiple metrics (catch-up)",
            "/health": "GET - Health check with stats",
            "/stats": "GET - Prediction statistics",
            "/reset": "POST - Reset history buffer",
            "/docs": "GET - OpenAPI documentation",
        },
    }


# ============================================================================
# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
