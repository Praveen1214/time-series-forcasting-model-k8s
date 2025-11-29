# Kubernetes Deployment Configuration

This directory contains Kubernetes deployment configurations and integration with the autoscaler operator.

## Structure

- **autoscaler_operator/**: Custom operator for predictive autoscaling (Component 3 integration)
- **manifests/**: Kubernetes YAML manifests for deploying the prediction service

## Future Work

### Component 3: Autoscaler Operator Integration

The `autoscaler_operator/` directory will contain:

1. **Custom Kubernetes Operator**
   - Watches pods and collects telemetry metrics
   - Calls the Informer model prediction service
   - Adjusts pod replicas based on predictions

2. **Deployment Architecture**
   - Model inference service (FastAPI/Flask)
   - Metrics collector (Prometheus integration)
   - Operator controller (Kubernetes controller)

3. **Integration Steps**
   - Deploy prediction service as a pod
   - Install custom resource definitions (CRDs)
   - Deploy operator controller
   - Configure target services for autoscaling

## Quick Start

### Deploy Prediction Service

```bash
kubectl apply -f manifests/prediction-service.yaml
```

### Install Autoscaler Operator (Coming Soon)

```bash
kubectl apply -f autoscaler_operator/crd.yaml
kubectl apply -f autoscaler_operator/operator.yaml
```

## Requirements

- Kubernetes cluster (v1.20+)
- kubectl configured
- Prometheus (optional, for metrics collection)
- Istio or other service mesh (optional, for traffic metrics)

## Next Steps

1. Containerize the prediction model
2. Create FastAPI/Flask inference service
3. Build Kubernetes operator using Kubebuilder or Operator SDK
4. Implement telemetry collection pipeline
5. Add monitoring and alerting
