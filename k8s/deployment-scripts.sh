#!/bin/bash

# Comprehensive ML Deployment Kubernetes Script
# This script handles everything: setup, deployment, and image fixes

set -e

echo "🚀 Starting Comprehensive ML Deployment to Kubernetes..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_header() {
    echo -e "${BLUE}[STEP]${NC} $1"
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

# Step 1: Environment Setup
print_header "Step 1: Environment Setup"

# Check if .env file exists
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

# Create namespace first
print_status "Creating ml-deployment namespace..."
kubectl apply -f namespace.yaml

# Generate ConfigMap
./generate-configmap.sh

# Generate Secrets
./generate-secrets.sh

# Step 1.5: Generate Grafana and Prometheus ConfigMaps from original files
print_header "Step 1.5: Generating Grafana and Prometheus ConfigMaps"

# Make scripts executable
chmod +x generate-grafana-configmaps.sh generate-prometheus-configmap.sh

print_status "Generating Grafana ConfigMaps from /grafana directory..."
./generate-grafana-configmaps.sh

print_status "Generating Prometheus ConfigMap from /prometheus directory..."
./generate-prometheus-configmap.sh

# Step 2: Build Docker Images
print_header "Step 2: Building Docker Images"

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

# Step 3: Deploy to Kubernetes
print_header "Step 3: Deploying to Kubernetes"

# Apply Grafana and Prometheus ConfigMaps first
print_status "Applying Grafana and Prometheus ConfigMaps..."
kubectl apply -f grafana-datasources-configmap.yaml
kubectl apply -f grafana-dashboards-configmap.yaml
kubectl apply -f grafana-dashboard-files-configmap.yaml
kubectl apply -f prometheus-configmap.yaml

# Create namespace and apply configurations
print_status "Creating namespace and applying configurations..."
kubectl apply -k .

# Step 4: Fix Image Issues (for k3s)
print_header "Step 4: Fixing Image Issues"

# Check if we're using k3s
if kubectl get nodes | grep -q "k3s"; then
    print_status "Detected k3s cluster. Fixing image access issues..."
    
    # Import images into k3s
    print_status "Importing Docker images into k3s..."
    docker save fastapi-app:latest > fastapi-app.tar
    docker save mlflow-server:latest > mlflow-server.tar
    docker save model-trainer:latest > model-trainer.tar
    
    sudo k3s ctr images import fastapi-app.tar
    sudo k3s ctr images import mlflow-server.tar
    sudo k3s ctr images import model-trainer.tar
    
    # Clean up tar files
    rm -f fastapi-app.tar mlflow-server.tar model-trainer.tar
    
    # Restart deployments to pick up images
    print_status "Restarting deployments to use imported images..."
    kubectl rollout restart deployment/fastapi-app -n ml-deployment
    kubectl rollout restart deployment/mlflow-server -n ml-deployment
    kubectl rollout restart deployment/grafana -n ml-deployment
    kubectl rollout restart deployment/prometheus -n ml-deployment
fi

# Step 5: Wait for Deployment
print_header "Step 5: Waiting for Deployment"

# Wait for pods to be ready
print_status "Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n ml-deployment --timeout=300s
kubectl wait --for=condition=ready pod -l app=minio -n ml-deployment --timeout=300s
kubectl wait --for=condition=ready pod -l app=mlflow-server -n ml-deployment --timeout=300s
kubectl wait --for=condition=ready pod -l app=fastapi-app -n ml-deployment --timeout=300s
kubectl wait --for=condition=ready pod -l app=prometheus -n ml-deployment --timeout=300s
kubectl wait --for=condition=ready pod -l app=grafana -n ml-deployment --timeout=300s

print_status "All pods are ready!"

# Step 6: Setup PostgreSQL NodePort
print_header "Step 6: Setting up PostgreSQL NodePort"

print_status "Setting up PostgreSQL NodePort for external access..."
print_status "This will allow you to connect to PostgreSQL from anywhere without port-forwarding"

# Get server public IP
PUBLIC_IP=$(curl -s ifconfig.me)
print_status "Server Public IP: $PUBLIC_IP"

# Get NodePort
NODEPORT=$(kubectl get svc postgres-nodeport -n ml-deployment -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null)
if [ -n "$NODEPORT" ]; then
    print_status "✅ PostgreSQL NodePort: $NODEPORT"
    print_status "✅ PostgreSQL NodePort is ready!"
else
    print_warning "⚠️  NodePort not available yet"
    print_status "You can check manually with:"
    echo "  kubectl get svc postgres-nodeport -n ml-deployment"
    print_status "Or use port-forwarding as fallback:"
    echo "  kubectl port-forward -n ml-deployment svc/postgres-service 5432:5432"
fi

# Step 7: Show Results
print_header "Step 7: Deployment Complete"

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

print_status "PostgreSQL NodePort (External Access):"
if [ -n "$NODEPORT" ] && [ -n "$PUBLIC_IP" ]; then
    echo "  Host: $PUBLIC_IP"
    echo "  Port: $NODEPORT"
    echo "  Database: mlflow"
    echo "  Username: mlflowuser"
    echo "  Password: mlflowpassword"
    echo "  Connection String: jdbc:postgresql://$PUBLIC_IP:$NODEPORT/mlflow"
else
    echo "  Check NodePort status: kubectl get svc postgres-nodeport -n ml-deployment"
    echo "  Or use port-forward: kubectl port-forward -n ml-deployment svc/postgres-service 5432:5432"
fi
echo ""

print_status "To check HPA status:"
echo "  kubectl get hpa -n ml-deployment"
echo ""

print_status "To view logs:"
echo "  kubectl logs -f deployment/fastapi-app -n ml-deployment"
echo "  kubectl logs -f deployment/mlflow-server -n ml-deployment"
echo "  kubectl logs -f deployment/grafana -n ml-deployment"
echo "  kubectl logs -f deployment/prometheus -n ml-deployment"
echo ""
print_status "Dashboard yang tersedia di Grafana:"
echo "  - ML API Monitoring (menggunakan file dari /grafana)"
echo ""
print_status "Prometheus configuration:"
echo "  - Menggunakan file dari /prometheus/prometheus.yml"
echo ""

print_status "To scale the FastAPI app manually:"
echo "  kubectl scale deployment fastapi-app --replicas=5 -n ml-deployment"
echo ""

print_status "PostgreSQL NodePort Security:"
echo "  ⚠️  PostgreSQL is now exposed to the internet!"
echo "  🔒 Recommended security measures:"
echo "     - Use VPN for access"
echo "     - Setup firewall rules"
echo "     - Enable SSL/TLS"
echo "     - Restrict IP access"
echo "     - Change default passwords"
echo ""

print_status "To monitor PostgreSQL NodePort:"
echo "  kubectl get svc postgres-nodeport -n ml-deployment"
echo "  kubectl describe svc postgres-nodeport -n ml-deployment"
echo ""

print_status "To delete the deployment:"
echo "  ./cleanup.sh" 