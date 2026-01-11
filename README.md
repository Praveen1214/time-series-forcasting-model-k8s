# Kubernetes Pod Autoscaling via Time-Series Forecasting

A production-ready deep learning solution for predictive Kubernetes autoscaling. This system forecasts pod resource requirements using advanced temporal models including **Informer**, **BiLSTM**, **TCN**, and **Hybrid Architectures**.

## 🚀 Key Features

*   **Proactive Scaling**: Forecasts load 1-minute ahead to prevent bottlenecks.
*   **Multi-Model Support**: Includes Informer, LSTM, BiLSTM, TCN, Prophet, and a **Hybrid (Prophet + LSTM)** model.
*   **Production API**: Fast, containerized API for real-time inference.
*   **Hybrid Evaluation**: Robust comparison against standard HPA metrics.

## 📂 Project Structure

```
informer-k8s-autoscaling/
├── api/                   # Inference API (FastAPI)
│   ├── main.py           # Server entry point
│   └── requirements.txt  # API dependencies
├── data/                  # Datasets & training logs
├── notebooks/             # Training & Analysis Notebooks
│   ├── Informer...ipynb
│   ├── BiLSTM...ipynb
│   ├── Hybrid Model...ipynb # Prophet + LSTM Ensemble
│   └── ...
├── BiLSTM_Evaluation.py   # Evaluation script
└── README.md
```

## 🛠️ Setup & Installation

1.  **Clone the repository**
    ```bash
    git clone <repo-url>
    cd informer-k8s-autoscaling
    ```

2.  **Install Dependencies**
    For the API and general usage:
    ```bash
    pip install -r api/requirements.txt
    ```
    *Note: GPU support requires appropriate CUDA versions for TensorFlow/PyTorch.*

## 💻 Usage

### 1. Training Models
Explore the `notebooks/` directory to train individual models. Each notebook handles data loading, preprocessing, model training, and evaluation.
*   `notebooks/Informer_K8s_Pod_Prediction.ipynb`
*   `notebooks/BiLSTM - K8s Pod Prediction.ipynb`
*   `notebooks/TCN - K8s Pod Prediction.ipynb`
*   `notebooks/Hybrid Model - K8s - Prophet LSTM.ipynb`

### 2. Running the API
The API provides endpoints for real-time pod count predictions.

```bash
cd api
uvicorn main:app --reload
```
Access docs at: `http://localhost:8000/docs`

### 3. Evaluation
Run the evaluation scripts to compare model performance.
```bash
python BiLSTM_Evaluation.py
```

## 📊 Model Performance (Overview)

| Model | MAE | RMSE | Best For |
|-------|-----|------|----------|
| **Informer** | 0.23 | 0.38 | Complex/Seasonal patterns |
| **Hybrid** | *0.25* | *0.40* | Trend + Residual learning |
| **BiLSTM** | 0.31 | 0.47 | General accuracy |
| **TCN** | 0.28 | 0.44 | Speed & Efficiency |
| **Prophet** | 0.42 | 0.61 | Baselines |

*Note: Performance metrics are approximate approximations based on validation sets.*

## 📜 License
MIT License
