#!/bin/bash

# Script untuk setup domain adityawidiyanto.my.id dengan Cloudflare
# Pilih antara subdomain atau path-based routing

set -e

echo "🌐 Setup Domain adityawidiyanto.my.id dengan Cloudflare..."

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

# Check if namespace exists
if ! kubectl get namespace ml-deployment &> /dev/null; then
    print_error "Namespace ml-deployment tidak ditemukan. Jalankan deployment terlebih dahulu."
    exit 1
fi

print_header "Pilih metode routing:"
echo "1. Subdomain (api.adityawidiyanto.my.id, mlflow.adityawidiyanto.my.id, dll)"
echo "2. Path-based (adityawidiyanto.my.id/api, adityawidiyanto.my.id/mlflow, dll)"
echo ""
read -p "Pilih opsi (1 atau 2): " choice

case $choice in
    1)
        print_status "Menggunakan subdomain routing..."
        kubectl apply -f ingress-domain.yaml
        print_status "✅ Ingress subdomain berhasil diterapkan!"
        echo ""
        print_status "Subdomain yang akan dibuat:"
        echo "  - api.adityawidiyanto.my.id → FastAPI"
        echo "  - mlflow.adityawidiyanto.my.id → MLflow"
        echo "  - grafana.adityawidiyanto.my.id → Grafana"
        echo "  - prometheus.adityawidiyanto.my.id → Prometheus"
        echo "  - minio.adityawidiyanto.my.id → MinIO Console"
        ;;
    2)
        print_status "Menggunakan path-based routing..."
        kubectl apply -f ingress-path-based.yaml
        print_status "✅ Ingress path-based berhasil diterapkan!"
        echo ""
        print_status "Path yang akan tersedia:"
        echo "  - adityawidiyanto.my.id/api → FastAPI"
        echo "  - adityawidiyanto.my.id/mlflow → MLflow"
        echo "  - adityawidiyanto.my.id/grafana → Grafana"
        echo "  - adityawidiyanto.my.id/prometheus → Prometheus"
        echo "  - adityawidiyanto.my.id/minio → MinIO Console"
        ;;
    *)
        print_error "Pilihan tidak valid!"
        exit 1
        ;;
esac

print_header "Setup Cloudflare DNS Records"

# Get external IP
print_status "Mendapatkan External IP..."
EXTERNAL_IP=$(kubectl get nodes -o wide | grep -v "INTERNAL-IP" | head -1 | awk '{print $6}')

if [ -z "$EXTERNAL_IP" ]; then
    print_warning "Tidak dapat mendapatkan External IP otomatis."
    read -p "Masukkan External IP server Anda: " EXTERNAL_IP
fi

print_status "External IP: $EXTERNAL_IP"
echo ""

print_header "Langkah-langkah setup Cloudflare:"
echo ""
print_status "1. Login ke Cloudflare Dashboard"
echo "2. Pilih domain: adityawidiyanto.my.id"
echo "3. Buka tab 'DNS'"
echo "4. Tambahkan record berikut:"
echo ""

if [ "$choice" = "1" ]; then
    echo "Type: A"
    echo "Name: api"
    echo "Content: $EXTERNAL_IP"
    echo "Proxy: Yes (Orange Cloud)"
    echo ""
    echo "Type: A"
    echo "Name: mlflow"
    echo "Content: $EXTERNAL_IP"
    echo "Proxy: Yes (Orange Cloud)"
    echo ""
    echo "Type: A"
    echo "Name: grafana"
    echo "Content: $EXTERNAL_IP"
    echo "Proxy: Yes (Orange Cloud)"
    echo ""
    echo "Type: A"
    echo "Name: prometheus"
    echo "Content: $EXTERNAL_IP"
    echo "Proxy: Yes (Orange Cloud)"
    echo ""
    echo "Type: A"
    echo "Name: minio"
    echo "Content: $EXTERNAL_IP"
    echo "Proxy: Yes (Orange Cloud)"
else
    echo "Type: A"
    echo "Name: @ (atau kosong)"
    echo "Content: $EXTERNAL_IP"
    echo "Proxy: Yes (Orange Cloud)"
fi

echo ""
print_status "5. Tunggu propagasi DNS (biasanya 5-10 menit)"
echo ""
print_status "6. Test akses:"
if [ "$choice" = "1" ]; then
    echo "   curl http://api.adityawidiyanto.my.id/"
    echo "   curl http://mlflow.adityawidiyanto.my.id/"
else
    echo "   curl http://adityawidiyanto.my.id/api/"
    echo "   curl http://adityawidiyanto.my.id/mlflow/"
fi

print_status "✅ Setup domain selesai!" 