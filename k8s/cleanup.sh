#!/bin/bash

# Comprehensive Cleanup Script for ML Deployment
# This script removes all Kubernetes resources and Docker images

set -e

echo "🧹 Starting comprehensive cleanup..."

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

# Confirm deletion
echo -e "${YELLOW}This will delete ALL resources in the ml-deployment namespace and remove Docker images.${NC}"
read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_status "Cleanup cancelled."
    exit 0
fi

# Step 1: Remove Kubernetes resources
print_status "Step 1: Removing Kubernetes resources..."

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

# Remove all resources in namespace
if kubectl get namespace ml-deployment &> /dev/null; then
    print_status "Removing all resources in ml-deployment namespace..."
    kubectl delete namespace ml-deployment --timeout=300s
    print_status "Namespace ml-deployment removed successfully!"
else
    print_warning "Namespace ml-deployment not found."
fi

# Step 2: Remove Docker images
print_status "Step 2: Removing Docker images..."

# Remove project-specific images
print_status "Removing project Docker images..."
docker rmi fastapi-app:latest 2>/dev/null || print_warning "fastapi-app:latest not found"
docker rmi mlflow-server:latest 2>/dev/null || print_warning "mlflow-server:latest not found"
docker rmi model-trainer:latest 2>/dev/null || print_warning "model-trainer:latest not found"

# Remove tar files if they exist
print_status "Removing image tar files..."
rm -f fastapi-app.tar 2>/dev/null || true
rm -f mlflow-server.tar 2>/dev/null || true
rm -f model-trainer.tar 2>/dev/null || true

# Step 3: Clean up generated files
print_status "Step 3: Cleaning up generated files..."

# Remove generated Kubernetes files
rm -f configmap.yaml 2>/dev/null || true
rm -f secret.yaml 2>/dev/null || true

# Step 4: Clean up Docker system
print_status "Step 4: Cleaning up Docker system..."

# Remove unused containers, networks, and images
docker system prune -f

# Optional: Remove all unused volumes
echo ""
read -p "Do you want to remove all unused Docker volumes? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Removing unused Docker volumes..."
    docker volume prune -f
fi

# Step 5: Clean up k3s images (if using k3s)
print_status "Step 5: Cleaning up k3s images..."

# Remove images from k3s containerd
sudo k3s ctr images rm docker.io/library/fastapi-app:latest 2>/dev/null || true
sudo k3s ctr images rm docker.io/library/mlflow-server:latest 2>/dev/null || true
sudo k3s ctr images rm docker.io/library/model-trainer:latest 2>/dev/null || true

print_status "Comprehensive cleanup completed successfully! 🎉"
print_status "All Kubernetes resources, Docker images, and generated files have been removed."