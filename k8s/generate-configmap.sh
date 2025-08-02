#!/bin/bash

# Generate Kubernetes ConfigMap from .env file
# This script reads the .env file and creates Kubernetes ConfigMap automatically

set -e

echo "⚙️ Generating Kubernetes ConfigMap from .env file..."

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

# Check if .env file exists
if [ ! -f "../.env" ]; then
    print_error ".env file not found!"
    print_status "Creating .env file from .env.example..."
    
    if [ -f "../.env.example" ]; then
        cp ../.env.example ../.env
        print_warning "Please edit ../.env file with your actual values before continuing."
        print_status "You can edit it with: nano ../.env"
        exit 1
    else
        print_error ".env.example file not found either!"
        exit 1
    fi
fi

# Function to get value from .env file
get_env_value() {
    local key=$1
    local value=$(grep "^${key}=" ../.env | cut -d '=' -f2- | tr -d '"' | tr -d "'")
    echo "$value"
}

# List of variables for ConfigMap (non-sensitive data)
CONFIGMAP_VARS=(
    "POSTGRES_DB"
    "POSTGRES_PORT"
    "MINIO_PORT"
    "MINIO_CONSOLE_PORT"
    "MLFLOW_PORT"
    "FASTAPI_PORT"
    "PROMETHEUS_PORT"
    "GRAFANA_PORT"
    "MODEL_NAME"
    "MODEL_STAGE"
)

print_status "Reading configuration values from .env file..."

# Create the Kubernetes ConfigMap YAML
print_status "Creating Kubernetes ConfigMap YAML..."

cat > configmap.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-config
  namespace: ml-deployment
data:
EOF

# Add each variable to the ConfigMap
for var in "${CONFIGMAP_VARS[@]}"; do
    value=$(get_env_value "$var")
    if [ -n "$value" ]; then
        echo "  ${var}: \"${value}\"" >> configmap.yaml
    else
        print_warning "Variable ${var} not found in .env file, using default value"
        case $var in
            "POSTGRES_DB")
                echo "  ${var}: \"mlflow\"" >> configmap.yaml
                ;;
            "POSTGRES_PORT")
                echo "  ${var}: \"5432\"" >> configmap.yaml
                ;;
            "MINIO_PORT")
                echo "  ${var}: \"9000\"" >> configmap.yaml
                ;;
            "MINIO_CONSOLE_PORT")
                echo "  ${var}: \"9001\"" >> configmap.yaml
                ;;
            "MLFLOW_PORT")
                echo "  ${var}: \"5000\"" >> configmap.yaml
                ;;
            "FASTAPI_PORT")
                echo "  ${var}: \"8000\"" >> configmap.yaml
                ;;
            "PROMETHEUS_PORT")
                echo "  ${var}: \"9090\"" >> configmap.yaml
                ;;
            "GRAFANA_PORT")
                echo "  ${var}: \"3000\"" >> configmap.yaml
                ;;
            "MODEL_NAME")
                echo "  ${var}: \"iris-classifier\"" >> configmap.yaml
                ;;
            "MODEL_STAGE")
                echo "  ${var}: \"production\"" >> configmap.yaml
                ;;
        esac
    fi
done

print_status "Kubernetes ConfigMap YAML created successfully!"
print_status "ConfigMap file: configmap.yaml"

print_status "ConfigMap YAML created successfully!"

print_status "ConfigMap generation completed!" 