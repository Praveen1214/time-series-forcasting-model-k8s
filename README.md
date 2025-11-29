# K8s Predictive Autoscaler

🚀 A predictive Kubernetes autoscaler using Transformer-based Informer model for proactive pod count prediction.

## Overview

This project implements a machine learning-based predictive autoscaler for Kubernetes workloads. Instead of reacting to current load, it predicts future resource requirements and scales proactively, reducing latency and improving resource utilization.

### Key Features

- **Transformer-based Informer Model**: Uses multi-head attention to capture complex temporal patterns
- **Multivariate Time-Series Prediction**: Leverages 20+ telemetry features including CPU, memory, latency, and service mesh metrics
- **Production-Ready Code**: Modular Python codebase with training, evaluation, and inference utilities
- **Kubernetes Integration Ready**: Designed for deployment as a custom autoscaler operator

## Architecture

### Components

1. **Data Collection** (notebooks/01_data_exploration.ipynb)
   - Kubernetes telemetry data (request rates, latency, CPU, memory, etc.)
   - Service mesh metrics (Istio/Linkerd)
   - Graph centrality features

2. **Model Training** (src/)
   - Informer model with positional encoding
   - Multi-head attention mechanism
   - Sequence-to-point prediction

3. **Deployment** (deployment/)
   - Future: Custom Kubernetes operator
   - Future: Real-time prediction service
   - Future: Integration with HPA/VPA

## Project Structure

```
k8s-predictive-autoscaler/
│
├── data/                     # Data storage
│   └── sample.csv           # Sample telemetry data
│
├── notebooks/                # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_informer_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/                      # Production Python code
│   ├── __init__.py
│   ├── dataset.py           # Data loading and sequence creation
│   ├── model.py             # Informer model definition
│   ├── train.py             # Training loop
│   ├── evaluate.py          # Evaluation and inference
│   └── utils.py             # Utilities
│
├── deployment/               # Kubernetes deployment configs
│   ├── autoscaler_operator/ # Future: Custom operator
│   └── manifests/           # K8s YAML files
│
├── configs/                  # Configuration files
│   └── config.yaml          # Training and model config
│
├── tests/                    # Unit tests
│   └── test_dataset.py
│
├── requirements.txt          # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.10+
- (Optional) CUDA-enabled GPU for faster training

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/k8s-predictive-autoscaler.git
cd k8s-predictive-autoscaler

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Data Exploration

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. Train the Model

```bash
cd src
python train.py --config ../configs/config.yaml
```

### 3. Evaluate

```bash
jupyter notebook notebooks/04_model_evaluation.ipynb
```

### 4. Inference

```python
from src.utils import load_model
from src.evaluate import predict_pod_count

# Load trained model
model, scaler_X, scaler_y, config = load_model('../models/informer_k8s')

# Predict pod count
pod_count = predict_pod_count(model, scaler_X, scaler_y, feature_cols, recent_data)
print(f"Predicted pod count: {pod_count}")
```

## Model Details

### Input Features (20 dimensions)

- **Pod Metrics**: Current pod count, CPU usage (avg, p95), memory usage (avg, p95)
- **Application Metrics**: Request rate, latency (p95), error rate, queue length
- **Service Mesh Metrics**: Inbound RPS, latency, error rate
- **Graph Features**: Degree, eigenvector, betweenness, closeness centrality
- **Temporal Features**: Hour (sin, cos), day (sin, cos)

### Model Architecture

- **Type**: Transformer-based Informer
- **Input**: Sequence of 10 timesteps × 20 features
- **Output**: Single value (next pod count)
- **Key Components**:
  - Positional encoding
  - Multi-head attention (4 heads)
  - 2 Transformer encoder blocks
  - Feed-forward networks
  - Global average pooling

### Training Configuration

- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping)
- **Train/Val/Test Split**: 70/15/15

## Results

Sample performance metrics on the Order service:

- **MAE**: ~0.3 pods
- **RMSE**: ~0.45 pods
- **MAPE**: ~8%
- **R²**: ~0.92

*(Note: Update these with your actual results)*

## Future Work

### Component 3: Kubernetes Operator Integration

- [ ] Build custom Kubernetes operator using Kubebuilder
- [ ] Deploy prediction service as FastAPI application
- [ ] Implement real-time telemetry collection
- [ ] Integrate with Kubernetes HPA/VPA
- [ ] Add monitoring and alerting (Prometheus/Grafana)
- [ ] Support multiple services and namespaces
- [ ] Implement rolling updates and model versioning

### Enhancements

- [ ] Multi-step-ahead prediction
- [ ] Multi-service prediction with dependency awareness
- [ ] Anomaly detection for unexpected load patterns
- [ ] Cost optimization integration
- [ ] Support for GPU workloads
- [ ] A/B testing framework

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```
@software{k8s_predictive_autoscaler,
  author = {Your Name},
  title = {K8s Predictive Autoscaler: Transformer-based Pod Count Prediction},
  year = {2025},
  url = {https://github.com/yourusername/k8s-predictive-autoscaler}
}
```

## Acknowledgments

- Based on the Informer architecture for time-series forecasting
- Inspired by Kubernetes HPA and VPA
- Built with TensorFlow and scikit-learn

## Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

⭐ If you find this project helpful, please give it a star!
