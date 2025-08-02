#!/bin/bash

# Setup Environment for Kubernetes Deployment
# This script sets up the .env file and generates Kubernetes ConfigMap and Secrets

set -e

echo "🚀 Setting up environment for Kubernetes deployment..."

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
    echo -e "${BLUE}[SETUP]${NC} $1"
}

# Check if we're in the right directory
if [ ! -d "k8s" ]; then
    print_error "This script must be run from the project root directory!"
    print_status "Please run: cd /path/to/ml_deployment && ./k8s/setup-env.sh"
    exit 1
fi

# Step 1: Setup .env file
print_header "Step 1: Setting up .env file"

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        print_status "Creating .env file from .env.example..."
        cp .env.example .env
        print_warning "Please edit .env file with your actual values before continuing."
        print_status "You can edit it with: nano .env"
        
        echo ""
        print_status "Required variables to set in .env file:"
        echo "  - POSTGRES_USER: Database username"
        echo "  - POSTGRES_PASSWORD: Database password"
        echo "  - MINIO_ROOT_USER: MinIO username"
        echo "  - MINIO_ROOT_PASSWORD: MinIO password"
        echo "  - GRAFANA_ADMIN_USER: Grafana admin username"
        echo "  - GRAFANA_ADMIN_PASSWORD: Grafana admin password"
        echo ""
        
        read -p "Press Enter after you've edited the .env file..."
    else
        print_error ".env.example file not found!"
        exit 1
    fi
else
    print_status ".env file already exists."
fi

# Step 2: Create Namespace
print_header "Step 2: Creating Kubernetes Namespace"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed or not in PATH"
    print_status "Please install kubectl first:"
    echo "  curl -LO \"https://dl.k8s.io/release/\$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl\""
    echo "  sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl"
    exit 1
fi

# Navigate to k8s directory first
print_status "Navigating to k8s directory..."
cd k8s

# Create namespace first
print_status "Creating ml-deployment namespace..."
kubectl apply -f namespace.yaml

if [ $? -eq 0 ]; then
    print_status "Namespace created successfully!"
else
    print_warning "Namespace might already exist or there was an issue creating it."
fi

# Step 3: Generate ConfigMap and Secrets
print_header "Step 3: Generating Kubernetes ConfigMap and Secrets"

# Make scripts executable
chmod +x generate-configmap.sh generate-secrets.sh

# Generate ConfigMap
print_status "Generating ConfigMap..."
./generate-configmap.sh

# Generate Secrets
print_status "Generating Secrets..."
./generate-secrets.sh

# Step 4: Verify generated files
print_header "Step 4: Verifying generated files"

if [ -f "configmap.yaml" ] && [ -f "secret.yaml" ]; then
    print_status "✅ ConfigMap and Secret files generated successfully!"
    
    echo ""
    print_status "Generated files:"
    echo "  - configmap.yaml: Kubernetes ConfigMap"
    echo "  - secret.yaml: Kubernetes Secret"
    
    echo ""
    print_status "Next steps:"
    echo "  1. Review the generated files:"
    echo "     cat configmap.yaml"
    echo "     cat secret.yaml"
    echo ""
    echo "  2. Apply the configuration:"
    echo "     kubectl apply -f configmap.yaml"
    echo "     kubectl apply -f secret.yaml"
    echo ""
    echo "  3. Deploy the application:"
    echo "     kubectl apply -k ."
    echo ""
    echo "  4. Or use the deployment script:"
    echo "     ./deployment-scripts.sh"
    
else
    print_error "Failed to generate ConfigMap or Secret files!"
    exit 1
fi

print_status "Environment setup completed successfully! 🎉" 