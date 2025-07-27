# ML Deployment with Kubernetes Auto-scaling

This directory contains Kubernetes manifests for deploying a complete ML pipeline with auto-scaling capabilities.

## 🏗️ Architecture

The deployment includes:

- **FastAPI Application**: ML model serving API with auto-scaling
- **MLflow Server**: Model tracking and registry
- **PostgreSQL**: MLflow backend database
- **MinIO**: Object storage for ML artifacts
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Visualization and dashboards
- **Horizontal Pod Autoscaler (HPA)**: Automatic scaling based on CPU and memory usage

## 🚀 Auto-scaling Features

### Horizontal Pod Autoscaler (HPA)

- **Min Replicas**: 2
- **Max Replicas**: 10
- **CPU Threshold**: 70% utilization
- **Memory Threshold**: 80% utilization
- **Scale Up**: Immediate (60s stabilization window)
- **Scale Down**: Gradual (300s stabilization window)

### Scaling Behavior

- **Scale Up**: Can double replicas every 15 seconds
- **Scale Down**: Can reduce by 10% every 60 seconds
- **Stabilization**: Prevents rapid scaling oscillations

## 📁 File Structure

```
k8s/
├── namespace.yaml              # Namespace definition
├── configmap.yaml              # Configuration values (auto-generated)
├── secret.yaml                 # Sensitive data (auto-generated)
├── postgres-deployment.yaml    # PostgreSQL database
├── minio-deployment.yaml       # MinIO object storage
├── mlflow-deployment.yaml      # MLflow tracking server
├── fastapi-deployment.yaml     # FastAPI app with HPA
├── prometheus-deployment.yaml  # Prometheus monitoring
├── grafana-deployment.yaml     # Grafana dashboards
├── ingress.yaml               # External access routing (path-based)
├── kustomization.yaml         # Kustomize configuration
├── setup-env.sh              # Environment setup script
├── generate-configmap.sh      # ConfigMap generator
├── generate-secrets.sh        # Secrets generator
├── deployment-scripts.sh      # Automated deployment script
├── cleanup-scripts.sh         # Cleanup script
├── load-test.sh              # Load testing for auto-scaling
└── README.md                 # This file
```

## 🛠️ Prerequisites

1. **Kubernetes Cluster**: Minikube, Docker Desktop, or cloud cluster
2. **kubectl**: Kubernetes command-line tool
3. **kustomize**: Kubernetes configuration management
4. **Docker**: For building container images

## 📦 Installation

### 1. Install Tools

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install kustomize
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
sudo mv kustomize /usr/local/bin/
```

### 2. Deploy the Application

```bash
# Make scripts executable
chmod +x k8s/*.sh

# Run the deployment script
cd k8s
./deployment-scripts.sh
```

### 3. Manual Deployment (Alternative)

```bash
# Build Docker images
docker build -t fastapi-app:latest ../app
docker build -t mlflow-server:latest ../mlflow_server
docker build -t model-trainer:latest ../model

# Apply Kubernetes manifests
kubectl apply -k k8s/
```

## 🔗 Accessing Services

### Via Ingress (Production)

```bash
# Get your server IP
SERVER_IP=$(curl -s ifconfig.me)

# Access services via path-based routing
curl http://$SERVER_IP/api/          # FastAPI Application
curl http://$SERVER_IP/mlflow/       # MLflow UI
curl http://$SERVER_IP/prometheus/   # Prometheus
curl http://$SERVER_IP/grafana/      # Grafana
curl http://$SERVER_IP/minio/        # MinIO Console
```

### Service URLs

- **FastAPI App**: `http://<server-ip>/api/`
- **MLflow UI**: `http://<server-ip>/mlflow/`
- **Prometheus**: `http://<server-ip>/prometheus/`
- **Grafana**: `http://<server-ip>/grafana/` (admin/your_grafana_password)
- **MinIO Console**: `http://<server-ip>/minio/` (minio_prod/your_minio_password)

### Port Forwarding (Development)

```bash
# For local development/testing
kubectl port-forward -n ml-deployment svc/fastapi-service 8000:8000
kubectl port-forward -n ml-deployment svc/mlflow-service 5000:5000
kubectl port-forward -n ml-deployment svc/prometheus-service 9090:9090
kubectl port-forward -n ml-deployment svc/grafana-service 3000:3000
kubectl port-forward -n ml-deployment svc/minio-service 9001:9001
```

## 🔍 Monitoring Auto-scaling

### Check HPA Status

```bash
# View HPA details
kubectl get hpa -n ml-deployment

# Detailed HPA information
kubectl describe hpa fastapi-hpa -n ml-deployment

# View scaling events
kubectl get events -n ml-deployment --sort-by='.lastTimestamp'
```

### Monitor Pod Scaling

```bash
# Watch pod scaling in real-time
kubectl get pods -n ml-deployment -l app=fastapi-app -w

# Check resource usage
kubectl top pods -n ml-deployment
```

## 🧪 Testing Auto-scaling

### Load Testing

```bash
# Run automated load test
cd k8s
./load-test.sh
```

### Manual Testing

```bash
# Scale manually
kubectl scale deployment fastapi-app --replicas=5 -n ml-deployment

# Generate load with curl (using ingress)
SERVER_IP=$(curl -s ifconfig.me)
for i in {1..100}; do
  curl -X POST http://$SERVER_IP/api/predict \
    -H "Content-Type: application/json" \
    -d '[{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]' &
done
```

## 📊 Metrics and Monitoring

### Prometheus Metrics

The FastAPI application exposes these custom metrics:

- `ml_predictions_total`: Total number of predictions
- `ml_model_accuracy`: Current model accuracy
- `ml_prediction_confidence`: Prediction confidence distribution

### Grafana Dashboards

Pre-configured dashboards include:

- **ML API Dashboard**: Prediction metrics, model accuracy, confidence distribution
- **System Metrics**: CPU, memory, and network usage

## ⚙️ Configuration

### HPA Configuration

Edit `fastapi-deployment.yaml` to modify auto-scaling behavior:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 2 # Minimum pods
  maxReplicas: 10 # Maximum pods
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70 # CPU threshold
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80 # Memory threshold
```

### Resource Limits

Adjust resource requests and limits in deployment files:

```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

## 🧹 Cleanup

### Remove Deployment

```bash
# Automated cleanup
cd k8s
./cleanup-scripts.sh

# Manual cleanup
kubectl delete -k k8s/
```

## 🔧 Troubleshooting

### Common Issues

1. **Pods not scaling up**:

   ```bash
   kubectl describe hpa fastapi-hpa -n ml-deployment
   kubectl get events -n ml-deployment
   ```

2. **Metrics not available**:

   ```bash
   kubectl get pods -n ml-deployment -l app=fastapi-app
   kubectl logs deployment/fastapi-app -n ml-deployment
   ```

3. **Resource constraints**:
   ```bash
   kubectl describe nodes
   kubectl top nodes
   ```

### Debug Commands

```bash
# Check all resources
kubectl get all -n ml-deployment

# View logs
kubectl logs -f deployment/fastapi-app -n ml-deployment

# Check events
kubectl get events -n ml-deployment --sort-by='.lastTimestamp'

# Access pod shell
kubectl exec -it deployment/fastapi-app -n ml-deployment -- /bin/bash
```

## 📈 Performance Optimization

### Auto-scaling Tuning

1. **Adjust thresholds**: Lower CPU/memory thresholds for faster scaling
2. **Modify behavior**: Change stabilization windows and scaling policies
3. **Resource limits**: Optimize resource requests and limits
4. **Metrics**: Add custom metrics for better scaling decisions

### Monitoring Best Practices

1. **Set up alerts**: Configure Prometheus alerts for scaling events
2. **Monitor costs**: Track resource usage and scaling patterns
3. **Performance testing**: Regular load testing to validate scaling
4. **Capacity planning**: Monitor trends to plan resource allocation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the deployment
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.
