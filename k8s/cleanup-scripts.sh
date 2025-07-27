#!/bin/bash

# ML Deployment Kubernetes Cleanup Script
# This script removes the ML application from Kubernetes

set -e

echo "🧹 Starting ML Deployment cleanup from Kubernetes..."

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

# Confirm deletion
echo -e "${YELLOW}This will delete all resources in the ml-deployment namespace.${NC}"
read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_status "Cleanup cancelled."
    exit 0
fi

# Delete all resources using kustomize
print_status "Deleting all resources..."
kubectl delete -k .

# Wait for resources to be deleted
print_status "Waiting for resources to be deleted..."
kubectl wait --for=delete namespace/ml-deployment --timeout=300s 2>/dev/null || true

# Clean up Docker images (optional)
echo -e "${YELLOW}Do you want to remove the Docker images as well? (y/N):${NC}"
read -p "" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Removing Docker images..."
    docker rmi fastapi-app:latest 2>/dev/null || true
    docker rmi mlflow-server:latest 2>/dev/null || true
    docker rmi model-trainer:latest 2>/dev/null || true
    print_status "Docker images removed!"
fi

print_status "Cleanup completed successfully!" 