#!/bin/bash

# ML Deployment Kubernetes Deployment Script
# This script automates the deployment of the ML application to Kubernetes

set -e

echo "🚀 Starting ML Deployment to Kubernetes..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Check if kustomize is installed
if ! command -v kustomize &> /dev/null; then
    print_warning "kustomize is not installed. Installing kustomize..."
    curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
    sudo mv kustomize /usr/local/bin/
fi

# Check if .env file exists and generate ConfigMap/Secrets
if [ ! -f "../.env" ]; then
    print_error ".env file not found!"
    print_status "Please run the setup script first:"
    echo "  cd .. && ./k8s/setup-env.sh"
    exit 1
fi

# Generate ConfigMap and Secrets from .env file
print_status "Generating ConfigMap and Secrets from .env file..."

# Make scripts executable
chmod +x generate-configmap.sh generate-secrets.sh

# Generate ConfigMap
./generate-configmap.sh

# Generate Secrets
./generate-secrets.sh

# Build Docker images
print_status "Building Docker images..."

# Build FastAPI app image
print_status "Building FastAPI app image..."
docker build -t fastapi-app:latest ../app

# Build MLflow server image
print_status "Building MLflow server image..."
docker build -t mlflow-server:latest ../mlflow_server

# Build model trainer image
print_status "Building model trainer image..."
docker build -t model-trainer:latest ../model

print_status "Docker images built successfully!"

# Create namespace and apply configurations
print_status "Creating namespace and applying configurations..."

# Apply all configurations using kustomize
kubectl apply -k .

# Wait for pods to be ready
print_status "Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n ml-deployment --timeout=300s
kubectl wait --for=condition=ready pod -l app=minio -n ml-deployment --timeout=300s
kubectl wait --for=condition=ready pod -l app=mlflow-server -n ml-deployment --timeout=300s
kubectl wait --for=condition=ready pod -l app=fastapi-app -n ml-deployment --timeout=300s
kubectl wait --for=condition=ready pod -l app=prometheus -n ml-deployment --timeout=300s
kubectl wait --for=condition=ready pod -l app=grafana -n ml-deployment --timeout=300s

print_status "All pods are ready!"

# Show service status
print_status "Deployment completed successfully!"
echo ""
print_status "Service endpoints:"
echo "  FastAPI App: http://localhost:8000 (via port-forward)"
echo "  MLflow UI: http://localhost:5000 (via port-forward)"
echo "  Prometheus: http://localhost:9090 (via port-forward)"
echo "  Grafana: http://localhost:3000 (via port-forward)"
echo "  MinIO Console: http://localhost:9001 (via port-forward)"
echo ""

print_status "To access services, run the following commands:"
echo "  kubectl port-forward -n ml-deployment svc/fastapi-service 8000:8000"
echo "  kubectl port-forward -n ml-deployment svc/mlflow-service 5000:5000"
echo "  kubectl port-forward -n ml-deployment svc/prometheus-service 9090:9090"
echo "  kubectl port-forward -n ml-deployment svc/grafana-service 3000:3000"
echo "  kubectl port-forward -n ml-deployment svc/minio-service 9001:9001"
echo ""

print_status "To check HPA status:"
echo "  kubectl get hpa -n ml-deployment"
echo ""

print_status "To view logs:"
echo "  kubectl logs -f deployment/fastapi-app -n ml-deployment"
echo "  kubectl logs -f deployment/mlflow-server -n ml-deployment"
echo ""

print_status "To scale the FastAPI app manually:"
echo "  kubectl scale deployment fastapi-app --replicas=5 -n ml-deployment"
echo ""

print_status "To delete the deployment:"
echo "  kubectl delete -k ." 