#!/bin/bash

# Script untuk mengkonversi file prometheus.yml menjadi ConfigMap Kubernetes
# Menggunakan file asli dari direktori /prometheus

set -e

echo "📁 Mengkonversi file prometheus menjadi ConfigMap Kubernetes..."

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

# Check if prometheus directory exists
if [ ! -d "../prometheus" ]; then
    print_error "Direktori ../prometheus tidak ditemukan!"
    exit 1
fi

print_status "Membuat ConfigMap untuk prometheus.yml..."
cat > prometheus-configmap.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: ml-deployment
data:
EOF

# Add prometheus.yml file
if [ -f "../prometheus/prometheus.yml" ]; then
    print_status "Menambahkan prometheus.yml..."
    echo "  prometheus.yml: |" >> prometheus-configmap.yaml
    # Fix the target for Kubernetes
    sed 's|fastapi_app:8000|fastapi-service:8000|g' ../prometheus/prometheus.yml | sed 's/^/    /' >> prometheus-configmap.yaml
else
    print_warning "File prometheus.yml tidak ditemukan!"
fi

print_status "✅ Prometheus ConfigMap berhasil dibuat!"
echo ""
print_status "File yang dibuat:"
echo "  - prometheus-configmap.yaml"
echo ""
print_status "Untuk menerapkan ConfigMap:"
echo "  kubectl apply -f prometheus-configmap.yaml" 