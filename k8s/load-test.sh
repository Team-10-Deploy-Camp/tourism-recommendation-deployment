#!/bin/bash

# Load Testing Script for ML API Auto-scaling
# This script generates load to test the Horizontal Pod Autoscaler

set -e

echo "🔥 Starting Load Test for ML API Auto-scaling..."

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

# Check if curl is installed
if ! command -v curl &> /dev/null; then
    print_error "curl is not installed. Please install curl first."
    exit 1
fi

# Configuration
API_URL="https://api.adityawidiyanto.my.id"
DURATION=300  # 5 minutes
CONCURRENT_USERS=50  # Much higher concurrent users
REQUESTS_PER_SECOND=20  # Much higher requests per second

print_status "Load test configuration:"
echo "  API URL: $API_URL"
echo "  Domain: adityawidiyanto.my.id"
echo "  Endpoint: /predict (heavy computational load)"
echo "  Duration: ${DURATION}s"
echo "  Concurrent users: $CONCURRENT_USERS"
echo "  Requests per second: $REQUESTS_PER_SECOND"
echo "  Expected total requests: $((CONCURRENT_USERS * REQUESTS_PER_SECOND * DURATION))"
echo "  Target: Auto-scaling test with intensive ML prediction workload"
echo ""

# Sample prediction data - larger dataset for heavier load
PREDICTION_DATA='[
  {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  },
  {
    "sepal_length": 6.3,
    "sepal_width": 3.3,
    "petal_length": 4.7,
    "petal_width": 1.6
  },
  {
    "sepal_length": 6.7,
    "sepal_width": 3.0,
    "petal_length": 5.2,
    "petal_width": 2.3
  },
  {
    "sepal_length": 5.8,
    "sepal_width": 2.7,
    "petal_length": 4.1,
    "petal_width": 1.0
  },
  {
    "sepal_length": 7.1,
    "sepal_width": 3.0,
    "petal_length": 5.9,
    "petal_width": 2.1
  },
  {
    "sepal_length": 4.9,
    "sepal_width": 3.1,
    "petal_length": 1.5,
    "petal_width": 0.1
  },
  {
    "sepal_length": 6.5,
    "sepal_width": 3.0,
    "petal_length": 5.8,
    "petal_width": 2.2
  },
  {
    "sepal_length": 5.5,
    "sepal_width": 2.4,
    "petal_length": 3.8,
    "petal_width": 1.1
  }
]'

# Check if API is accessible
print_status "Testing API connectivity..."
if curl -s "$API_URL/" > /dev/null 2>&1; then
    print_status "✅ API is accessible at $API_URL"
    API_RESPONSE=$(curl -s "$API_URL/")
    print_status "API Status: $API_RESPONSE"

    # Test prediction endpoint
    print_status "Testing prediction endpoint..."
    if curl -s -X POST -H "Content-Type: application/json" -d "$PREDICTION_DATA" "$API_URL/predict" > /dev/null 2>&1; then
        print_status "✅ Prediction endpoint is working"
    else
        print_warning "⚠️  Prediction endpoint is not working"
        print_status "This might affect the load test results"
    fi
else
    print_warning "⚠️  API is not accessible at $API_URL"
    print_status "Make sure:"
    echo "  1. Domain is properly configured in Cloudflare"
    echo "  2. Ingress is working correctly"
    echo "  3. FastAPI pods are running"
    echo ""
    print_status "Checking pod status..."
    kubectl get pods -n ml-deployment -l app=fastapi-app
    echo ""
    read -p "Press Enter to continue anyway..."
fi

# Function to make a prediction request
make_prediction() {
    local response=$(curl -s -w "%{http_code}" -o /tmp/response.json \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$PREDICTION_DATA" \
        "$API_URL/predict")
    
    local status_code=${response: -3}
    echo "$status_code"
}

# Function to make a health check request
make_health_check() {
    local response=$(curl -s -w "%{http_code}" -o /tmp/health_response.json \
        "$API_URL/")
    
    local status_code=${response: -3}
    echo "$status_code"
}

# Function to check HPA status
check_hpa_status() {
    echo "=== HPA Status ==="
    kubectl get hpa -n ml-deployment
    echo ""
    echo "=== Pod Status ==="
    kubectl get pods -n ml-deployment -l app=fastapi-app
    echo ""
}

# Start monitoring in background
print_status "Starting HPA monitoring..."
(
    while true; do
        check_hpa_status
        sleep 30
    done
) &
MONITOR_PID=$!

# Cleanup function
cleanup() {
    print_status "Stopping load test..."
    kill $MONITOR_PID 2>/dev/null || true
    print_status "Load test completed!"
    echo ""
    print_status "Final HPA status:"
    check_hpa_status
}

# Set trap to cleanup on exit
trap cleanup EXIT

print_status "Starting load test..."
print_status "Press Ctrl+C to stop the test early"

# Load test loop
start_time=$(date +%s)
request_count=0
success_count=0
error_count=0

print_status "Starting intensive load test with /predict endpoint..."
print_status "This will generate heavy computational load to trigger auto-scaling"

while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    if [ $elapsed -ge $DURATION ]; then
        break
    fi
    
    # Make concurrent prediction requests (heavier load)
    for i in $(seq 1 $CONCURRENT_USERS); do
        (
            status=$(make_prediction)
            if [ "$status" = "200" ]; then
                echo "SUCCESS" >> /tmp/load_test_results
            else
                echo "ERROR:$status" >> /tmp/load_test_results
            fi
        ) &
    done
    
    # Wait for all requests to complete
    wait
    
    # Count results
    if [ -f /tmp/load_test_results ]; then
        success_count=$((success_count + $(grep -c "SUCCESS" /tmp/load_test_results)))
        error_count=$((error_count + $(grep -c "ERROR" /tmp/load_test_results)))
        rm /tmp/load_test_results
    fi
    
    request_count=$((request_count + CONCURRENT_USERS))
    
    # Progress update - more frequent updates
    if [ $((elapsed % 10)) -eq 0 ]; then
        print_status "Progress: ${elapsed}s/${DURATION}s - Prediction Requests: $request_count, Success: $success_count, Errors: $error_count"
        print_status "Current RPS: ~$((request_count / (elapsed + 1))) requests/second"
    fi
    
    # Rate limiting - calculate sleep time correctly
    sleep_time=$(echo "scale=3; 1 / $REQUESTS_PER_SECOND" | bc -l 2>/dev/null || echo "0.05")
    sleep $sleep_time
done

# Final statistics
total_requests=$((success_count + error_count))
success_rate=$(echo "scale=2; $success_count * 100 / $total_requests" | bc -l 2>/dev/null || echo "0")

print_status "Load test completed!"
echo "  Total requests: $total_requests"
echo "  Successful requests: $success_count"
echo "  Failed requests: $error_count"
echo "  Success rate: ${success_rate}%"
echo "  Duration: ${elapsed}s"
echo ""
print_status "API Endpoint tested: $API_URL/predict"
print_status "Model Status: $(curl -s "$API_URL/" | grep -o '"model_status":"[^"]*"' | cut -d'"' -f4)"
print_status "Model Accuracy: $(curl -s "$API_URL/" | grep -o '"model_accuracy":"[^"]*"' | cut -d'"' -f4)"
print_status "Load Type: ML Prediction (computationally intensive)"
echo ""
print_status "To monitor HPA in real-time:"
echo "  kubectl get hpa -n ml-deployment -w"
echo ""
print_status "To check pod scaling:"
echo "  kubectl get pods -n ml-deployment -l app=fastapi-app -w" 