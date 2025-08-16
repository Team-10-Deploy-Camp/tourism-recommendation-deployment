#!/bin/bash

# Script untuk mengkonversi file dari direktori grafana menjadi ConfigMap Kubernetes
# Menggunakan file asli dari direktori /grafana

set -e

echo "📁 Mengkonversi file grafana menjadi ConfigMap Kubernetes..."

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

# Check if grafana directory exists
if [ ! -d "../grafana" ]; then
    print_error "Direktori ../grafana tidak ditemukan!"
    exit 1
fi

print_status "Membuat ConfigMap untuk datasource..."
cat > grafana-datasources-configmap.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-datasources
  namespace: ml-deployment
data:
EOF

# Add datasource file
if [ -f "../grafana/provisioning/datasources/datasource.yml" ]; then
    print_status "Menambahkan datasource.yml..."
    echo "  datasource.yml: |" >> grafana-datasources-configmap.yaml
    # Use the datasource file as-is since it now includes Kubernetes-compatible configurations
    sed 's/^/    /' ../grafana/provisioning/datasources/datasource.yml >> grafana-datasources-configmap.yaml
else
    print_warning "File datasource.yml tidak ditemukan!"
fi

print_status "Membuat ConfigMap untuk dashboard configuration..."
cat > grafana-dashboards-configmap.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboards
  namespace: ml-deployment
data:
EOF

# Add dashboard configuration
if [ -f "../grafana/provisioning/dashboards/dashboard.yml" ]; then
    print_status "Menambahkan dashboard.yml..."
    echo "  dashboard.yml: |" >> grafana-dashboards-configmap.yaml
    sed 's/^/    /' ../grafana/provisioning/dashboards/dashboard.yml >> grafana-dashboards-configmap.yaml
else
    print_warning "File dashboard.yml tidak ditemukan!"
fi

print_status "Membuat ConfigMap untuk dashboard files..."
cat > grafana-dashboard-files-configmap.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard-files
  namespace: ml-deployment
data:
EOF

# Add dashboard JSON files
if [ -f "../grafana/provisioning/dashboards/ml_api_dashboard.json" ]; then
    print_status "Menambahkan ml_api_dashboard.json..."
    echo "  ml_api_dashboard.json: |" >> grafana-dashboard-files-configmap.yaml
    sed 's/^/    /' ../grafana/provisioning/dashboards/ml_api_dashboard.json >> grafana-dashboard-files-configmap.yaml
else
    print_warning "File ml_api_dashboard.json tidak ditemukan!"
fi

print_status "✅ ConfigMap files berhasil dibuat!"
echo ""
print_status "Files yang dibuat:"
echo "  - grafana-datasources-configmap.yaml"
echo "  - grafana-dashboards-configmap.yaml"
echo "  - grafana-dashboard-files-configmap.yaml"
echo ""
print_status "Untuk menerapkan ConfigMap:"
echo "  kubectl apply -f grafana-datasources-configmap.yaml"
echo "  kubectl apply -f grafana-dashboards-configmap.yaml"
echo "  kubectl apply -f grafana-dashboard-files-configmap.yaml"