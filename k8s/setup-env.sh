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
        echo ""
        print_status "📊 Database Configuration:"
        echo "  - POSTGRES_USER: Database username"
        echo "  - POSTGRES_PASSWORD: Database password"
        echo "  - POSTGRES_DB: Database name"
        echo "  - POSTGRES_PORT: Database port"
        echo ""
        print_status "🗄️  MinIO/S3 Configuration:"
        echo "  - MINIO_ROOT_USER: MinIO username"
        echo "  - MINIO_ROOT_PASSWORD: MinIO password"
        echo "  - MINIO_PORT: MinIO API port"
        echo "  - MINIO_CONSOLE_PORT: MinIO console port"
        echo ""
        print_status "🔐 AWS Credentials (for MLflow):"
        echo "  - AWS_ACCESS_KEY_ID: S3 access key (usually same as MINIO_ROOT_USER)"
        echo "  - AWS_SECRET_ACCESS_KEY: S3 secret key (usually same as MINIO_ROOT_PASSWORD)"
        echo "  - MLFLOW_S3_ENDPOINT_URL: S3 endpoint URL"
        echo ""
        print_status "📈 MLflow Configuration:"
        echo "  - MLFLOW_TRACKING_URI: MLflow tracking server URI"
        echo "  - MLFLOW_PORT: MLflow server port"
        echo ""
        print_status "🏗️  ClickHouse Configuration (for training data):"
        echo "  - clickhouse_host: ClickHouse server host"
        echo "  - clickhouse_port: ClickHouse server port"
        echo "  - clickhouse_user: ClickHouse username"
        echo "  - clickhouse_database: ClickHouse database name"
        echo "  - clickhouse_table: ClickHouse table name"
        echo ""
        print_status "🌐 Service Ports:"
        echo "  - FASTAPI_PORT: FastAPI application port"
        echo "  - PROMETHEUS_PORT: Prometheus monitoring port"
        echo "  - GRAFANA_PORT: Grafana dashboard port"
        echo ""
        print_status "📊 Grafana Configuration:"
        echo "  - GRAFANA_ADMIN_USER: Grafana admin username"
        echo "  - GRAFANA_ADMIN_PASSWORD: Grafana admin password"
        echo ""
        print_status "🤖 Model Configuration:"
        echo "  - MODEL_NAME: Name of the ML model"
        echo "  - MODEL_STAGE: Model deployment stage"
        echo ""
        
        read -p "Press Enter after you've edited the .env file..."
    else
        print_error ".env.example file not found!"
        print_status "Creating .env file with default values..."
        
        cat > .env << EOF
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000

# ClickHouse Configuration
clickhouse_host=localhost
clickhouse_port=8123
clickhouse_user=default
clickhouse_database=tourism_db
clickhouse_table=tourism_data

# S3/MinIO Configuration
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

# PostgreSQL Configuration
POSTGRES_USER=mlflowuser
POSTGRES_PASSWORD=mlflowpassword
POSTGRES_DB=mlflow
POSTGRES_PORT=5432

# MinIO Configuration
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001

# Service Ports
MLFLOW_PORT=5000
FASTAPI_PORT=8000
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Grafana Configuration
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin

# Model Configuration
MODEL_NAME=tourism-recommendation-model
MODEL_STAGE=production
EOF
        
        print_warning "Created .env file with default values. Please review and edit if needed."
        print_status "You can edit it with: nano .env"
        read -p "Press Enter after you've reviewed the .env file..."
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
    echo "  - configmap.yaml: Kubernetes ConfigMap (includes ClickHouse config)"
    echo "  - secret.yaml: Kubernetes Secret (includes AWS credentials & Grafana)"
    
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
print_status "All environment variables including ClickHouse, AWS credentials, and Grafana are now configured." 