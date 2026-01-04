"""
K8s Proactive Scaler API
========================
Predicts pod count needed in +5 minutes using BiLSTM model.

CRITICAL: This API handles the full preprocessing pipeline:
1. Receives RAW metrics (unscaled)
2. Scales features using the trained MinMaxScaler
3. Runs model inference (outputs scaled 0-1)
4. Inverse transforms prediction to actual pod count
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

# ============================================================
# Configuration
# ============================================================

LOOKBACK = 48
N_FEATURES = 21  # 20 features + 1 target (current_pod_count)
PREDICTION_HORIZON_MIN = 5
SAFETY_BUFFER = 1.1

# Feature order MUST match training exactly
# First 20 are features, last one is the target (pod_count) used as input feature
FEATURE_COLS = [
    "request_rate_rps", "latency_p95_ms", "latency_p99_ms", "error_rate_percent",
    "queue_length", "pod_cpu_usage_percent_avg", "pod_cpu_usage_percent_p95",
    "pod_memory_usage_mb_avg", "pod_memory_usage_mb_p95", "hour_sin", "hour_cos",
    "day_sin", "day_cos", "mesh_inbound_rps", "mesh_inbound_latency_p95",
    "mesh_inbound_error_rate", "degree_centrality", "eigenvector_centrality",
    "betweenness_centrality", "closeness_centrality"
]
# Full input order: FEATURE_COLS + ['current_pod_count'] (scaled as 'y')

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = BASE_DIR / "notebooks" / "models" / "bilstm-pods-20260104_012142.keras"
DEFAULT_SCALERS = BASE_DIR / "notebooks" / "models" / "bilstm-scalers-20260104_012142.pkl"

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL))).expanduser().resolve()
SCALERS_PATH = Path(os.getenv("SCALERS_PATH", str(DEFAULT_SCALERS))).expanduser().resolve()

# ============================================================
# Pydantic Models
# ============================================================

class PredictRequest(BaseModel):
    """
    Input: RAW (unscaled) metrics window.
    Shape: 48 rows x 21 columns
    Column order: FEATURE_COLS (20) + current_pod_count (1)
    """
    window_data: List[List[float]] = Field(..., description="48x21 RAW feature window (unscaled)")
    window_end_utc: str = Field(..., description="ISO timestamp (UTC)")
    service_id: Optional[str] = Field(default="Order")
    input_source: Optional[str] = Field(default="unknown")

    @validator("window_data")
    def check_shape(cls, v):
        if len(v) != LOOKBACK:
            raise ValueError(f"Expected {LOOKBACK} rows, got {len(v)}")
        for i, row in enumerate(v):
            if len(row) != N_FEATURES:
                raise ValueError(f"Row {i}: expected {N_FEATURES} features, got {len(row)}")
        return v


class PredictResponse(BaseModel):
    service_id: str
    window_end_utc: str
    horizon_min: int
    
    current_pod_count: int
    raw_prediction: float        # Model output (scaled 0-1)
    predicted_pod_count: int     # Final prediction (inverse scaled + buffer)
    safety_buffer: float
    
    inference_ms: float
    total_ms: float
    
    # Debug info
    input_scaled_sample: List[float]  # Last row after scaling (for debugging)


# ============================================================
# Global State
# ============================================================

app = FastAPI(title="K8s Proactive Scaler API")

_model: Optional[tf.keras.Model] = None
_predict_fn = None
_feature_scaler = None  # MinMaxScaler for 20 features
_target_scaler = None   # MinMaxScaler for pod_count


# ============================================================
# Preprocessing Functions
# ============================================================

def scale_window(raw_window: np.ndarray) -> np.ndarray:
    """
    Scale raw input window using trained scalers.
    
    Input shape: (48, 21) where columns are [20 features, pod_count]
    Output shape: (48, 21) scaled to [0, 1]
    """
    if _feature_scaler is None or _target_scaler is None:
        raise RuntimeError("Scalers not loaded")
    
    # Split features and target
    features = raw_window[:, :-1]  # First 20 columns
    pod_count = raw_window[:, -1:]  # Last column (current_pod_count)
    
    # Scale separately (same as training)
    features_scaled = _feature_scaler.transform(features)
    pod_count_scaled = _target_scaler.transform(pod_count)
    
    # Combine back: [scaled_features, scaled_pod_count]
    return np.hstack([features_scaled, pod_count_scaled])


def inverse_scale_prediction(scaled_pred: float) -> float:
    """Convert model output (0-1) back to actual pod count."""
    if _target_scaler is None:
        raise RuntimeError("Target scaler not loaded")
    return float(_target_scaler.inverse_transform([[scaled_pred]])[0, 0])


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
def startup():
    global _model, _predict_fn, _feature_scaler, _target_scaler
    
    # Load model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    _model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH.name}")
    
    # Load scalers
    if not SCALERS_PATH.exists():
        raise RuntimeError(f"Scalers not found: {SCALERS_PATH}")
    scalers = joblib.load(SCALERS_PATH)
    _feature_scaler = scalers['feature_scaler']
    _target_scaler = scalers['target_scaler']
    print(f"✅ Scalers loaded: {SCALERS_PATH.name}")
    
    # Compile prediction function
    @tf.function
    def compiled_predict(x):
        return _model(x, training=False)
    _predict_fn = compiled_predict
    
    # Warm up
    dummy = tf.zeros((1, LOOKBACK, N_FEATURES), dtype=tf.float32)
    _ = _predict_fn(dummy)
    print("✅ Model warmed up")


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_file": MODEL_PATH.name,
        "scalers_file": SCALERS_PATH.name,
        "lookback": LOOKBACK,
        "n_features": N_FEATURES,
        "horizon_min": PREDICTION_HORIZON_MIN,
        "feature_order": FEATURE_COLS + ["current_pod_count"],
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    t0 = time.perf_counter()
    
    try:
        if _model is None or _predict_fn is None:
            raise RuntimeError("Model not loaded")
        if _feature_scaler is None or _target_scaler is None:
            raise RuntimeError("Scalers not loaded")
        
        # Get current pod count from raw data (last row, last column)
        raw_window = np.asarray(req.window_data, dtype=np.float32)
        current_pods = int(raw_window[-1, -1])  # Last column is pod_count
        
        # CRITICAL: Scale the input data
        scaled_window = scale_window(raw_window)
        X = scaled_window.reshape(1, LOOKBACK, N_FEATURES)
        
        # Run inference
        t1 = time.perf_counter()
        y_scaled = _predict_fn(tf.convert_to_tensor(X, dtype=tf.float32))
        inference_ms = (time.perf_counter() - t1) * 1000.0
        
        raw_pred = float(y_scaled.numpy()[0, 0])
        
        # CRITICAL: Inverse transform to get actual pod count
        pred_pods_float = inverse_scale_prediction(raw_pred)
        
        # Apply safety buffer and round up
        predicted_pods = max(1, int(np.ceil(pred_pods_float * SAFETY_BUFFER)))
        
        total_ms = (time.perf_counter() - t0) * 1000.0
        
        return PredictResponse(
            service_id=req.service_id or "unknown",
            window_end_utc=req.window_end_utc,
            horizon_min=PREDICTION_HORIZON_MIN,
            current_pod_count=current_pods,
            raw_prediction=round(pred_pods_float, 2),
            predicted_pod_count=predicted_pods,
            safety_buffer=SAFETY_BUFFER,
            inference_ms=round(inference_ms, 2),
            total_ms=round(total_ms, 2),
            input_scaled_sample=scaled_window[-1].tolist(),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
