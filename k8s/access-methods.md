# Access Methods for ML Deployment

This document explains different ways to access your ML deployment services via IP address.

## 🎯 Access Methods Overview

### 1. **Ingress with Path-based Routing** (Recommended for production)

- **File**: `ingress.yaml`
- **Access**: `http://<server-ip>/api`, `http://<server-ip>/mlflow`, etc.
- **Pros**: Single entry point, clean URLs, SSL termination
- **Cons**: Requires ingress controller

### 2. **LoadBalancer Services** (Cloud environments)

- **File**: `loadbalancer-services.yaml`
- **Access**: `http://<loadbalancer-ip>:8000`, `http://<loadbalancer-ip>:5000`, etc.
- **Pros**: Direct access, no path routing needed
- **Cons**: Requires cloud LoadBalancer, additional costs

### 3. **NodePort Services** (Local/Development)

- **File**: `nodeport-services.yaml`
- **Access**: `http://<node-ip>:30080`, `http://<node-ip>:30500`, etc.
- **Pros**: Works everywhere, no external dependencies
- **Cons**: Port conflicts possible, less secure

### 4. **Port Forwarding** (Development/Testing)

- **Command**: `kubectl port-forward`
- **Access**: `http://localhost:8000`, `http://localhost:5000`, etc.
- **Pros**: Simple, secure, no external exposure
- **Cons**: Only local access, not suitable for production

## 🚀 Quick Start - Choose Your Method

### Option A: Ingress (Production Ready)

```bash
# Deploy with ingress
kubectl apply -f k8s/ingress.yaml

# Get your server IP
kubectl get nodes -o wide

# Access services
curl http://<server-ip>/api/
curl http://<server-ip>/mlflow/
curl http://<server-ip>/prometheus/
curl http://<server-ip>/grafana/
curl http://<server-ip>/minio/
```

### Option B: NodePort (Development)

```bash
# Deploy with NodePort services
kubectl apply -f k8s/nodeport-services.yaml

# Get your node IP
kubectl get nodes -o wide

# Access services
curl http://<node-ip>:30080/  # FastAPI
curl http://<node-ip>:30500/  # MLflow
curl http://<node-ip>:30909/  # Prometheus
curl http://<node-ip>:30300/  # Grafana
curl http://<node-ip>:30901/  # MinIO Console
```

### Option C: LoadBalancer (Cloud)

```bash
# Deploy with LoadBalancer services
kubectl apply -f k8s/loadbalancer-services.yaml

# Get LoadBalancer IPs
kubectl get svc -n ml-deployment

# Access services
curl http://<lb-ip>:8000/  # FastAPI
curl http://<lb-ip>:5000/  # MLflow
curl http://<lb-ip>:9090/  # Prometheus
curl http://<lb-ip>:3000/  # Grafana
curl http://<lb-ip>:9001/  # MinIO Console
```

## 🔧 Configuration Examples

### Ingress Configuration (Path-based)

```yaml
# Access via: http://<server-ip>/api/
# Access via: http://<server-ip>/mlflow/
# Access via: http://<server-ip>/prometheus/
# Access via: http://<server-ip>/grafana/
# Access via: http://<server-ip>/minio/
```

### NodePort Configuration

```yaml
# FastAPI: http://<node-ip>:30080
# MLflow:  http://<node-ip>:30500
# Prometheus: http://<node-ip>:30909
# Grafana: http://<node-ip>:30300
# MinIO Console: http://<node-ip>:30901
```

## 🌐 Finding Your Server IP

### For Minikube

```bash
minikube ip
```

### For Docker Desktop

```bash
# Usually localhost or 127.0.0.1
```

### For Cloud Clusters

```bash
# Get node IPs
kubectl get nodes -o wide

# Get LoadBalancer IPs
kubectl get svc -n ml-deployment
```

### For Local Clusters

```bash
# Get your machine's IP
ip addr show | grep "inet " | grep -v 127.0.0.1
```

## 🔒 Security Considerations

### Ingress (Most Secure)

- SSL termination
- Path-based routing
- Rate limiting possible
- Authentication/authorization

### LoadBalancer

- Direct service access
- Network policies recommended
- SSL termination at service level

### NodePort

- Exposed on all nodes
- Use firewall rules
- Consider network policies

### Port Forwarding (Most Secure for Development)

- Only local access
- No external exposure
- Perfect for development

## 📝 Usage Examples

### Testing FastAPI Endpoints

```bash
# Health check
curl http://<server-ip>/api/

# Make prediction
curl -X POST http://<server-ip>/api/predict \
  -H "Content-Type: application/json" \
  -d '[{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]'
```

### Accessing MLflow UI

```bash
# Open in browser
open http://<server-ip>/mlflow/
# or
xdg-open http://<server-ip>/mlflow/
```

### Monitoring with Prometheus

```bash
# Check metrics
curl http://<server-ip>/prometheus/metrics

# Open Prometheus UI
open http://<server-ip>/prometheus/
```

### Grafana Dashboards

```bash
# Open Grafana
open http://<server-ip>/grafana/
# Login: admin/admin
```

## 🛠️ Troubleshooting

### Check Service Status

```bash
kubectl get svc -n ml-deployment
kubectl describe svc <service-name> -n ml-deployment
```

### Check Pod Status

```bash
kubectl get pods -n ml-deployment
kubectl logs <pod-name> -n ml-deployment
```

### Check Ingress Status

```bash
kubectl get ingress -n ml-deployment
kubectl describe ingress ml-ingress -n ml-deployment
```

### Test Connectivity

```bash
# Test from within cluster
kubectl run test-pod --image=curlimages/curl -i --rm --restart=Never -- curl http://fastapi-service:8000/

# Test from outside
curl -v http://<server-ip>/api/
```

## 🎯 Recommendation

- **Development**: Use NodePort or Port Forwarding
- **Testing**: Use Ingress with path-based routing
- **Production**: Use Ingress with proper SSL certificates
- **Cloud**: Use LoadBalancer for direct access
