#!/bin/bash

# Generate Kubernetes Secrets from .env file
# This script reads the .env file and creates Kubernetes secrets automatically

set -e

echo "🔐 Generating Kubernetes Secrets from .env file..."

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

# Function to encode value to base64
encode_base64() {
    echo -n "$1" | base64
}

# Function to get value from .env file
get_env_value() {
    local key=$1
    local value=$(grep "^${key}=" ../.env | cut -d '=' -f2- | tr -d '"' | tr -d "'")
    echo "$value"
}

# Check if required environment variables are set
print_status "Checking required environment variables..."

# List of required variables for secrets
REQUIRED_VARS=(
    "POSTGRES_USER"
    "POSTGRES_PASSWORD"
    "MINIO_ROOT_USER"
    "MINIO_ROOT_PASSWORD"
    "GRAFANA_ADMIN_USER"
    "GRAFANA_ADMIN_PASSWORD"
)

# Check if any required variables are missing or have placeholder values
MISSING_VARS=()
for var in "${REQUIRED_VARS[@]}"; do
    value=$(get_env_value "$var")
    if [ -z "$value" ] || [[ "$value" == *"your_"* ]] || [[ "$value" == *"here"* ]]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    print_error "The following variables need to be set in your .env file:"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    print_status "Please edit ../.env file and set proper values for these variables."
    print_status "You can edit it with: nano ../.env"
    exit 1
fi

print_status "All required environment variables are set!"

# Generate base64 encoded values
print_status "Generating base64 encoded values..."

POSTGRES_USER_B64=$(encode_base64 "$(get_env_value 'POSTGRES_USER')")
POSTGRES_PASSWORD_B64=$(encode_base64 "$(get_env_value 'POSTGRES_PASSWORD')")
MINIO_ROOT_USER_B64=$(encode_base64 "$(get_env_value 'MINIO_ROOT_USER')")
MINIO_ROOT_PASSWORD_B64=$(encode_base64 "$(get_env_value 'MINIO_ROOT_PASSWORD')")
GRAFANA_ADMIN_USER_B64=$(encode_base64 "$(get_env_value 'GRAFANA_ADMIN_USER')")
GRAFANA_ADMIN_PASSWORD_B64=$(encode_base64 "$(get_env_value 'GRAFANA_ADMIN_PASSWORD')")

# Create the Kubernetes secret YAML
print_status "Creating Kubernetes secret YAML..."

cat > secret.yaml << EOF
apiVersion: v1
kind: Secret
metadata:
  name: ml-secrets
  namespace: ml-deployment
type: Opaque
data:
  POSTGRES_USER: ${POSTGRES_USER_B64}
  POSTGRES_PASSWORD: ${POSTGRES_PASSWORD_B64}
  MINIO_ROOT_USER: ${MINIO_ROOT_USER_B64}
  MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD_B64}
  GRAFANA_ADMIN_USER: ${GRAFANA_ADMIN_USER_B64}
  GRAFANA_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD_B64}
EOF

print_status "Kubernetes secret YAML created successfully!"
print_status "Secret file: secret.yaml"

print_status "Secret YAML created successfully!"

print_status "Secret generation completed!" 