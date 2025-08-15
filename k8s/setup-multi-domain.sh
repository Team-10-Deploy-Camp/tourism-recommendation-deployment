#!/bin/bash

# Script untuk setup multi-domain dengan Cloudflare
# Mendukung adityawidiyanto.my.id dan team10deploycamp.online

set -e

echo "�� Setup Multi-Domain dengan Cloudflare..."

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

print_header "Setup Multi-Domain Ingress"

# Apply multi-domain ingress
print_status "Menerapkan ingress multi-domain..."
kubectl apply -f ingress-multi-domain.yaml
print_status "✅ Ingress multi-domain berhasil diterapkan!"

print_header "Setup Cloudflare DNS Records"

# Get external IP
print_status "Mendapatkan External IP..."

# Try multiple services to get public IP
SERVICES=(
    "ifconfig.me"
    "ipinfo.io/ip"
    "icanhazip.com"
    "checkip.amazonaws.com"
)

EXTERNAL_IP=""

for service in "${SERVICES[@]}"; do
    print_status "Mencoba $service..."
    ip=$(curl -s --max-time 5 "$service" 2>/dev/null)
    
    if [[ $ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        EXTERNAL_IP=$ip
        print_status "✅ IP Public ditemukan: $EXTERNAL_IP"
        break
    else
        print_warning "Gagal mendapatkan IP dari $service"
    fi
done

if [ -z "$EXTERNAL_IP" ]; then
    print_warning "Tidak dapat mendapatkan IP public otomatis."
    read -p "Masukkan IP public server Anda: " EXTERNAL_IP
fi

print_status "External IP: $EXTERNAL_IP"

# Check if it's a private IP
if [[ $EXTERNAL_IP =~ ^(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.) ]]; then
    print_warning "⚠️  IP $EXTERNAL_IP adalah IP private!"
    echo ""
    print_status "Untuk Cloudflare Proxy (Orange Cloud), Anda perlu IP public."
    print_status "Silakan masukkan IP public yang benar:"
    read -p "Masukkan IP public server Anda: " EXTERNAL_IP
    print_status "IP Public yang akan digunakan: $EXTERNAL_IP"
else
    print_status "✅ IP $EXTERNAL_IP adalah IP public yang valid!"
fi

echo ""

print_header "Langkah-langkah setup Cloudflare untuk kedua domain:"
echo ""

print_status "Domain 1: adityawidiyanto.my.id"
echo "1. Login ke Cloudflare Dashboard"
echo "2. Pilih domain: adityawidiyanto.my.id"
echo "3. Buka tab 'DNS'"
echo "4. Tambahkan record berikut:"
echo "   Type: A, Name: api, Content: $EXTERNAL_IP, Proxy: Yes (Orange Cloud)"
echo "   Type: A, Name: mlflow, Content: $EXTERNAL_IP, Proxy: Yes (Orange Cloud)"
echo "   Type: A, Name: grafana, Content: $EXTERNAL_IP, Proxy: Yes (Orange Cloud)"
echo "   Type: A, Name: prometheus, Content: $EXTERNAL_IP, Proxy: Yes (Orange Cloud)"
echo "   Type: A, Name: minio, Content: $EXTERNAL_IP, Proxy: Yes (Orange Cloud)"
echo ""

print_status "Domain 2: team10deploycamp.online"
echo "1. Login ke Cloudflare Dashboard"
echo "2. Pilih domain: team10deploycamp.online"
echo "3. Buka tab 'DNS'"
echo "4. Tambahkan record berikut:"
echo "   Type: A, Name: api, Content: $EXTERNAL_IP, Proxy: Yes (Orange Cloud)"
echo "   Type: A, Name: mlflow, Content: $EXTERNAL_IP, Proxy: Yes (Orange Cloud)"
echo "   Type: A, Name: grafana, Content: $EXTERNAL_IP, Proxy: Yes (Orange Cloud)"
echo "   Type: A, Name: prometheus, Content: $EXTERNAL_IP, Proxy: Yes (Orange Cloud)"
echo "   Type: A, Name: minio, Content: $EXTERNAL_IP, Proxy: Yes (Orange Cloud)"
echo ""

print_status "5. Tunggu propagasi DNS (biasanya 5-10 menit)"
echo ""
print_status "6. Test akses untuk kedua domain:"
echo "   # Domain 1"
echo "   curl http://api.adityawidiyanto.my.id/"
echo "   curl http://mlflow.adityawidiyanto.my.id/"
echo ""
echo "   # Domain 2"
echo "   curl http://api.team10deploycamp.online/"
echo "   curl http://mlflow.team10deploycamp.online/"

print_status "✅ Setup multi-domain selesai!" 