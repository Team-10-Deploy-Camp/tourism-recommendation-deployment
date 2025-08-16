# Tourism Recommendation API - ML Deployment System

A comprehensive, production-ready machine learning deployment system for tourism place recommendations. This project demonstrates advanced ML model deployment, monitoring, and scaling capabilities using modern DevOps practices and cloud-native technologies.

## 🌟 Key Features

### 🤖 Advanced Machine Learning
- **Hybrid Gradient Boosting Models**: Advanced ensemble methods with enhanced feature engineering
- **Intelligent Feature Engineering**: User preferences, price sensitivity, and contextual features
- **Model Versioning**: Complete MLflow integration for experiment tracking and model lifecycle management
- **Real-time Predictions**: High-performance API for instant tourism recommendations

### 🚀 Production-Ready Infrastructure
- **FastAPI Backend**: Modern, async Python API with automatic documentation
- **Containerized Architecture**: Docker Compose and Kubernetes deployment options
- **Auto-scaling**: Horizontal Pod Autoscaler (HPA) for dynamic scaling based on load
- **Load Balancing**: NGINX ingress controller for optimal traffic distribution

### 📊 Comprehensive Monitoring
- **Prometheus Metrics**: Application and infrastructure monitoring
- **Grafana Dashboards**: Beautiful visualizations for system insights
- **Health Checks**: Kubernetes liveness and readiness probes
- **Distributed Tracing**: Complete observability stack

### 💾 Robust Data Management
- **PostgreSQL**: Reliable metadata storage for MLflow
- **MinIO**: S3-compatible object storage for ML artifacts
- **ClickHouse Integration**: Large-scale training data support
- **Persistent Storage**: Kubernetes PVCs for data durability

### 🔒 Enterprise Security
- **Role-Based Access Control (RBAC)**: Kubernetes security policies
- **Network Policies**: Secure pod-to-pod communication
- **SSL/TLS Support**: Cert-manager integration for automated certificates
- **Secrets Management**: Kubernetes secrets for sensitive configuration

- **Live API**: Access the production API at [https://api.team10deploycamp.online/docs](https://api.team10deploycamp.online/docs)

## 🏗️ System Architecture

This project implements a microservices architecture with the following components:

### Core Services
- **FastAPI Application** (`app/`): High-performance REST API serving ML models
  - Asynchronous request handling with uvicorn
  - Pydantic models for data validation
  - Prometheus metrics integration
  - Health checks and graceful shutdown

- **MLflow Server** (`mlflow_server/`): ML lifecycle management
  - Experiment tracking and model registry
  - Model versioning and staging
  - Artifact storage integration
  - Web UI for model management

- **Model Training Pipeline** (`model/`): Automated ML training
  - Advanced hybrid gradient boosting models
  - Feature engineering with statistical aggregations
  - ClickHouse data source integration
  - Automated model registration and deployment

### Infrastructure Services
- **PostgreSQL**: Persistent metadata storage for MLflow experiments
- **MinIO**: S3-compatible object storage for ML artifacts and models
- **Prometheus**: Time-series metrics collection and alerting
- **Grafana**: Comprehensive dashboards and visualization
- **NGINX Ingress**: Load balancing and traffic routing

### Deployment Options
- **Docker Compose** (`docker-compose.yml`): Local development and testing
- **Kubernetes** (`k8s/`): Production deployment with auto-scaling
  - Horizontal Pod Autoscaler (HPA)
  - Persistent Volume Claims (PVCs)
  - ConfigMaps and Secrets management
  - Network policies and RBAC

### Monitoring & Observability
- **Application Metrics**: Request latency, throughput, error rates
- **ML Metrics**: Model performance, prediction confidence, feature importance
- **Infrastructure Metrics**: CPU, memory, disk, network usage
- **Custom Dashboards**: Business KPIs and operational insights

## 📋 Prerequisites

- **Kubernetes cluster** (k3s, MicroK8s, or full Kubernetes)
- **kubectl** installed and configured
- **kustomize** installed
- **Git** installed
- **Ubuntu server** (18.04, 20.04, or 22.04 LTS recommended)
- At least 4GB RAM and 2 CPU cores
- 20GB free disk space

## 🚀 Quick Start

### 1. Server Setup

Connect to your Ubuntu server and install required components:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y curl wget git unzip software-properties-common

# Install k3s (Lightweight Kubernetes)
curl -sfL https://get.k3s.io | sh -

# Setup kubeconfig
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER:$USER ~/.kube/config

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install kustomize
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
sudo mv kustomize /usr/local/bin/

# Verify installations
kubectl get nodes
kustomize version
```

### 2. Clone and Setup Project

```bash
# Navigate to suitable directory
cd /opt

# Clone your repository
sudo git clone <repository-url> ml_deployment
sudo chown -R $USER:$USER ml_deployment
cd ml_deployment

# Setup environment automatically
./k8s/setup-env.sh
```

The setup script will create a `.env` file from `.env.example` and prompt you to edit it with your configuration:

```bash
# Database Configuration
POSTGRES_USER=mlflow_prod
POSTGRES_PASSWORD=your_strong_password_here
POSTGRES_DB=mlflow
POSTGRES_PORT=5432

# MinIO Configuration
MINIO_ROOT_USER=minio_prod
MINIO_ROOT_PASSWORD=your_strong_minio_password
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001

# MLflow Configuration
MLFLOW_PORT=5000

# FastAPI Configuration
FASTAPI_PORT=8000

# Monitoring Configuration
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=your_strong_grafana_password

# Model Configuration
MODEL_NAME=tourism-advanced-hybrid-gb
MODEL_ALIAS=production

# ClickHouse Configuration (for training data)
clickhouse_host=your-clickhouse-host
clickhouse_port=8123
clickhouse_user=default
clickhouse_database=tourism_db
clickhouse_table=user_place_ratings
```

### 3. Deploy to Kubernetes

Deploy all services to your Kubernetes cluster:

```bash
# Navigate to k8s directory
cd k8s

# Make deployment scripts executable
chmod +x *.sh

# Deploy using the automated script
./deployment-scripts.sh

# Check deployment status
kubectl get all -n ml-deployment

# Wait for all pods to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n ml-deployment --timeout=300s
kubectl wait --for=condition=ready pod -l app=minio -n ml-deployment --timeout=300s
kubectl wait --for=condition=ready pod -l app=mlflow-server -n ml-deployment --timeout=300s
kubectl wait --for=condition=ready pod -l app=fastapi-app -n ml-deployment --timeout=300s
kubectl wait --for=condition=ready pod -l app=prometheus -n ml-deployment --timeout=300s
kubectl wait --for=condition=ready pod -l app=grafana -n ml-deployment --timeout=300s
```

### 4. Access Your Services

Get your server IP and access the services:

```bash
# Get your server's external IP
SERVER_IP=$(curl -s ifconfig.me)
echo "Server IP: $SERVER_IP"
```

Services will be available at:

- **Tourism API**: `http://<server-ip>/api/`
- **API Documentation**: `http://<server-ip>/api/docs`
- **MLflow UI**: `http://<server-ip>/mlflow/`
- **MinIO Console**: `http://<server-ip>/minio/`
- **Prometheus**: `http://<server-ip>/prometheus/`
- **Grafana**: `http://<server-ip>/grafana/`

### 5. Initialize MinIO Storage

1. Access MinIO Console at `http://<server-ip>/minio/`
2. Login with your `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD`
3. Create a bucket named `mlflow`

### 6. Train and Register Model

Train the tourism recommendation model using Kubernetes Job:

```bash
# Create and apply model training job
kubectl apply -f k8s/model-training-job.yaml

# Check job status
kubectl get jobs -n ml-deployment

# Monitor job progress
kubectl get pods -n ml-deployment | grep model-training

# View training logs
kubectl logs job/model-training-job -n ml-deployment

# Clean up job after completion
kubectl delete job model-training-job -n ml-deployment
```

This will:
- Load tourism data from ClickHouse
- Train the advanced hybrid gradient boosting model
- Register the model in MLflow
- Store model artifacts in MinIO

### 7. Promote Model to Production

1. Open MLflow UI at `http://<server-ip>/mlflow/`
2. Navigate to **Models** → `tourism-advanced-hybrid-gb`
3. Select the latest model version
4. Set alias to `production`

```bash
# Restart FastAPI deployment to load new model
kubectl rollout restart deployment/fastapi-app -n ml-deployment
```

### 8. Test the API

Test the tourism recommendation API:

```bash
# Get your server IP
SERVER_IP=$(curl -s ifconfig.me)

# Test the API
curl -X POST "http://$SERVER_IP/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user": {
      "user_age": 28,
      "preferred_category": "Cultural",
      "preferred_city": "Jakarta",
      "budget_range": "medium"
    },
    "places": [
      {
        "place_id": "place_001",
        "place_category": "Cultural",
        "place_city": "Jakarta",
        "place_price": 50000,
        "place_average_rating": 4.5,
        "place_visit_duration_minutes": 120,
        "place_description": "Historical museum with cultural artifacts"
      },
      {
        "place_id": "place_002",
        "place_category": "Nature",
        "place_city": "Bandung",
        "place_price": 25000,
        "place_average_rating": 4.2,
        "place_visit_duration_minutes": 180,
        "place_description": "Beautiful mountain view and hiking trails"
      }
    ]
  }'
```

Expected response:

```json
{
  "predictions": [
    {
      "place_id": "place_001",
      "predicted_rating": 4.3,
      "confidence_score": 0.87,
      "recommendation_rank": 1
    },
    {
      "place_id": "place_002",
      "predicted_rating": 3.8,
      "confidence_score": 0.82,
      "recommendation_rank": 2
    }
  ],
  "model_used": "tourism-advanced-hybrid-gb",
  "prediction_timestamp": "2024-01-15T10:30:00Z",
  "total_places_evaluated": 2,
  "top_recommendation": {
    "place_id": "place_001",
    "predicted_rating": 4.3,
    "confidence_score": 0.87,
    "recommendation_rank": 1
  }
}
```

## 📊 API Endpoints

### Core Endpoints

| Method | Endpoint | Description | Request/Response |
|--------|----------|-------------|------------------|
| `GET` | `/` | API root and welcome message | Basic API information |
| `GET` | `/health` | Comprehensive health check | Model status, API version, timestamp |
| `GET` | `/health/ready` | Kubernetes readiness probe | Service readiness status |
| `GET` | `/health/live` | Kubernetes liveness probe | Service health status |
| `GET` | `/model/info` | Current model information | Model metadata, metrics, features |
| `POST` | `/predict` | Get rating predictions for places | User preferences + places → rated predictions |
| `POST` | `/recommend` | Get top-K recommendations | User preferences + places → ranked recommendations |
| `GET` | `/model/reload` | Reload model from registry | Background model refresh |
| `GET` | `/metrics` | Prometheus metrics endpoint | Application and ML metrics |

### Request/Response Examples

#### Prediction Request
```json
{
  "user": {
    "user_age": 28,
    "preferred_category": "Cultural",
    "preferred_city": "Jakarta",
    "budget_range": "medium"
  },
  "places": [
    {
      "place_id": "place_001",
      "place_category": "Cultural",
      "place_city": "Jakarta",
      "place_price": 50000,
      "place_average_rating": 4.5,
      "place_visit_duration_minutes": 120,
      "place_description": "Historical museum"
    }
  ]
}
```

#### Prediction Response
```json
{
  "predictions": [
    {
      "place_id": "place_001",
      "predicted_rating": 4.3,
      "confidence_score": 0.87,
      "recommendation_rank": 1
    }
  ],
  "model_used": "tourism-advanced-hybrid-gb",
  "prediction_timestamp": "2024-01-15T10:30:00Z",
  "total_places_evaluated": 1,
  "top_recommendation": {
    "place_id": "place_001",
    "predicted_rating": 4.3,
    "confidence_score": 0.87,
    "recommendation_rank": 1
  }
}
```

### Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs` - Interactive API testing
- **ReDoc**: `http://localhost:8000/redoc` - Comprehensive API documentation
- **Live API**: [https://api.team10deploycamp.online/docs](https://api.team10deploycamp.online/docs)

## 🔧 Model Management

### Model Training & Deployment

#### Using Kubernetes (Production)
```bash
# Train new model using Kubernetes Job
kubectl apply -f k8s/model-training-job.yaml

# Monitor training progress
kubectl logs job/model-training-job -n ml-deployment -f

# Check model registration in MLflow
# Access MLflow UI at http://<server-ip>/mlflow/

# Promote model to production in MLflow UI
# Set model alias to "production"

# Restart API to load new model
kubectl rollout restart deployment/fastapi-app -n ml-deployment
```

#### Using Docker Compose (Development)
```bash
# Retrain model
docker-compose up --build model_trainer

# Restart API to load new model
docker-compose restart fastapi_app
```

### Model Versioning & Lifecycle

The system supports comprehensive model lifecycle management:

- **Experiment Tracking**: All training runs logged to MLflow
- **Model Registry**: Centralized model storage with versioning
- **Stage Management**: Development → Staging → Production promotion
- **A/B Testing**: Multiple model versions for comparison
- **Rollback Capability**: Quick reversion to previous model versions

### Model Performance Monitoring

#### Grafana Dashboards
Access Grafana at `http://localhost:3000` (admin/your_password) to monitor:

- **Model Performance**: Accuracy, RMSE, MAE metrics over time
- **Prediction Quality**: Confidence score distributions and trends
- **API Performance**: Request/response times, throughput, error rates
- **Business Metrics**: Recommendation diversity, user satisfaction
- **System Resources**: CPU, memory, disk usage per service

#### Real-time Metrics
The API automatically tracks and exposes:

- **ML Metrics**: `ml_recommendations_total`, `ml_prediction_confidence`
- **Model Quality**: `ml_model_rmse`, `ml_recommendation_diversity`
- **API Metrics**: Request duration, status codes, concurrent users
- **Infrastructure**: Pod scaling events, resource utilization

## 📈 Scaling Options

### Vertical Scaling

Increase resources for the ML app:

```bash
# Update resource limits in k8s/fastapi-app-deployment.yaml
kubectl patch deployment fastapi-app -n ml-deployment -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "fastapi-app",
          "resources": {
            "limits": {
              "cpu": "2000m",
              "memory": "4Gi"
            },
            "requests": {
              "cpu": "1000m",
              "memory": "2Gi"
            }
          }
        }]
      }
    }
  }
}'
```

### Horizontal Scaling

Scale the FastAPI application:

```bash
# Scale to 3 replicas
kubectl scale deployment fastapi-app --replicas=3 -n ml-deployment

# Enable Horizontal Pod Autoscaler (HPA)
kubectl autoscale deployment fastapi-app --cpu-percent=70 --min=2 --max=10 -n ml-deployment

# Check HPA status
kubectl get hpa -n ml-deployment
```

## ☸️ Kubernetes Deployment

For production deployment with auto-scaling, see [KUBERNETES_DEPLOYMENT.md](KUBERNETES_DEPLOYMENT.md) for detailed instructions including:

- Kubernetes cluster setup
- Ingress configuration
- Horizontal Pod Autoscaling (HPA)
- SSL/TLS certificates
- Production monitoring

### Quick Kubernetes Deploy

```bash
# Setup environment and generate configs
./k8s/setup-env.sh

# Deploy to Kubernetes
./k8s/deployment-scripts.sh

# Check deployment status
kubectl get all -n ml-deployment
```

## 🔍 Monitoring & Observability

### Metrics Available

- **API Metrics**: Request count, latency, error rates
- **ML Metrics**: Prediction confidence, model performance
- **System Metrics**: CPU, memory, disk usage
- **Business Metrics**: Recommendation diversity, user satisfaction

### Grafana Dashboards

- **ML API Monitoring**: Comprehensive API and model metrics
- **System Overview**: Infrastructure and resource monitoring
- **Business Intelligence**: Tourism recommendation insights

## 🛠️ Development Guide

### Local Development Setup

#### Option 1: Port-forward from Kubernetes
```bash
# Port-forward services for local access
kubectl port-forward service/fastapi-app 8000:8000 -n ml-deployment &
kubectl port-forward service/mlflow-server 5000:5000 -n ml-deployment &
kubectl port-forward service/grafana 3000:3000 -n ml-deployment &
kubectl port-forward service/prometheus 9090:9090 -n ml-deployment &

# Access services locally
# FastAPI: http://localhost:8000
# MLflow: http://localhost:5000
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
```

#### Option 2: Docker Compose Development
```bash
# Start development environment
docker-compose up -d

# Install development dependencies
pip install -r requirements.txt
pip install -r app/requirements.txt

# Run application in development mode
cd app && python app.py
```

### Project Structure
```
learning-ml-deployment/
├── app/                    # FastAPI application
│   ├── Dockerfile         # API container definition
│   ├── app.py            # Main FastAPI application
│   └── requirements.txt   # Python dependencies
├── model/                 # ML training pipeline
│   ├── Dockerfile        # Training container
│   ├── train.py         # Model training script
│   └── requirements.txt  # Training dependencies
├── mlflow_server/        # MLflow server configuration
├── k8s/                  # Kubernetes manifests
│   ├── *.yaml           # Deployment configurations
│   ├── *.sh            # Automation scripts
│   └── kustomization.yaml
├── grafana/             # Monitoring dashboards
├── prometheus/          # Metrics configuration
├── nginx/              # Load balancer config
└── docker-compose.yml  # Development environment
```

### Model Development & Testing

#### Interactive Development Pod
```bash
# Create development environment in Kubernetes
kubectl run model-dev --image=python:3.9 -it --rm --restart=Never -n ml-deployment \
  --env="MLFLOW_TRACKING_URI=http://mlflow-service:5000" \
  --env="AWS_ACCESS_KEY_ID=$(kubectl get secret ml-secrets -n ml-deployment -o jsonpath='{.data.MINIO_ROOT_USER}' | base64 -d)" \
  --env="AWS_SECRET_ACCESS_KEY=$(kubectl get secret ml-secrets -n ml-deployment -o jsonpath='{.data.MINIO_ROOT_PASSWORD}' | base64 -d)" \
  --env="MLFLOW_S3_ENDPOINT_URL=http://minio-service:9000" \
  -- bash

# Inside the pod
pip install mlflow scikit-learn pandas numpy clickhouse-connect
python -c "import mlflow; print('MLflow version:', mlflow.__version__)"
```

#### Local Model Training
```bash
# Set up environment
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

# Train model locally
cd model
python train.py
```

### Testing & Quality Assurance

#### Unit Testing
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run API tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

#### Integration Testing
```bash
# Test full pipeline
./k8s/load-test.sh

# Test API endpoints
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user": {"user_age": 25}, "places": [{"place_id": "test"}]}'
```

#### Performance Testing
```bash
# Load testing with Apache Bench
ab -n 1000 -c 10 http://localhost:8000/health

# Stress testing with K6
k6 run --vus 50 --duration 5m tests/load-test.js
```

### Debugging & Troubleshooting

#### Application Logs
```bash
# FastAPI application logs
kubectl logs -f deployment/fastapi-app -n ml-deployment

# MLflow server logs
kubectl logs -f deployment/mlflow-server -n ml-deployment

# All application logs
kubectl logs -f -l app=fastapi-app -n ml-deployment
```

#### Debug Mode
```bash
# Enable debug logging
kubectl set env deployment/fastapi-app LOG_LEVEL=DEBUG -n ml-deployment

# Access application shell
kubectl exec -it deployment/fastapi-app -n ml-deployment -- /bin/bash
```

## 🔒 Security Considerations

- Change default passwords in production
- Use strong authentication for all services
- Configure CORS appropriately
- Enable SSL/TLS with cert-manager and Let's Encrypt
- Implement proper network policies in Kubernetes
- Enable RBAC (Role-Based Access Control)
- Use Kubernetes Secrets for sensitive data
- Regular security updates and monitoring

```bash
# Install cert-manager for SSL certificates
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Apply network policies
kubectl apply -f k8s/network-policies.yaml
```

## 📝 Configuration

### Configuration Management

The system uses a multi-layered configuration approach:

#### Environment Variables (.env file)
```bash
# Database Configuration
POSTGRES_USER=mlflow_prod
POSTGRES_PASSWORD=secure_password_123
POSTGRES_DB=mlflow
POSTGRES_PORT=5432

# MinIO Configuration
MINIO_ROOT_USER=minio_admin
MINIO_ROOT_PASSWORD=secure_minio_password
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001

# MLflow Configuration
MLFLOW_PORT=5000
MODEL_NAME=tourism-advanced-hybrid-gb
MODEL_ALIAS=production

# API Configuration
FASTAPI_PORT=8000
CORS_ORIGINS=["*"]
LOG_LEVEL=INFO

# Monitoring Configuration
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=secure_grafana_password

# ClickHouse Configuration (for training)
clickhouse_host=your-clickhouse-host
clickhouse_port=8123
clickhouse_user=default
clickhouse_database=tourism_db
clickhouse_table=user_place_ratings
```

#### Kubernetes Configuration Management
```bash
# Generate ConfigMaps and Secrets from .env
./k8s/generate-configmap.sh
./k8s/generate-secrets.sh

# View current configuration
kubectl get configmap -n ml-deployment -o yaml
kubectl get secrets -n ml-deployment

# Update configuration dynamically
kubectl create configmap app-config --from-env-file=.env -n ml-deployment --dry-run=client -o yaml | kubectl apply -f -

# Restart deployments to pick up config changes
kubectl rollout restart deployment/fastapi-app -n ml-deployment
```

#### Configuration Categories

- **Database**: PostgreSQL connection, backup settings, performance tuning
- **Storage**: MinIO/S3 configuration, bucket policies, retention settings
- **ML Pipeline**: Model registry, experiment tracking, training parameters
- **API Gateway**: CORS policies, rate limiting, authentication
- **Monitoring**: Metrics collection, alerting rules, dashboard configuration
- **Security**: TLS certificates, network policies, RBAC rules

### Model Configuration & Feature Engineering

#### Model Architecture
- **Algorithm**: Advanced Hybrid Gradient Boosting
- **Features**: 22 engineered features including:
  - User statistical features (mean, std, count, range)
  - Place popularity and rating features
  - Category and city preference modeling
  - Price sensitivity analysis
  - Contextual interaction features

#### Model Lifecycle
- **Training Data**: ClickHouse integration for large-scale datasets
- **Validation**: Train/test split with temporal considerations
- **Registration**: Automatic MLflow model registration
- **Deployment**: Production alias-based deployment
- **Monitoring**: Real-time performance tracking

## 🤝 Contributing

We welcome contributions to improve this ML deployment system! Here's how to contribute:

### Getting Started
1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch** from `main`
4. **Set up development environment** using Docker Compose

### Development Workflow
```bash
# Clone your fork
git clone https://github.com/your-username/learning-ml-deployment.git
cd learning-ml-deployment

# Create feature branch
git checkout -b feature/your-feature-name

# Set up development environment
docker-compose up -d

# Make your changes and test
pytest tests/
./k8s/load-test.sh

# Commit and push
git add .
git commit -m "feat: your feature description"
git push origin feature/your-feature-name
```

### Contribution Guidelines
- **Code Quality**: Follow PEP 8 for Python code
- **Documentation**: Update README and inline docs for new features
- **Testing**: Add unit and integration tests for new functionality
- **Kubernetes**: Test deployments in kind/minikube before submitting
- **Security**: Follow security best practices, never commit secrets

### Areas for Contribution
- **ML Models**: Improve model architectures and feature engineering
- **Performance**: Optimize API response times and resource usage
- **Monitoring**: Add new Grafana dashboards and metrics
- **Documentation**: Improve setup guides and troubleshooting
- **CI/CD**: Enhance automation and deployment pipelines
- **Security**: Implement additional security measures

### Pull Request Process
1. **Update documentation** for any new features
2. **Add tests** that cover your changes
3. **Ensure all tests pass** in CI/CD pipeline
4. **Request review** from maintainers
5. **Address feedback** and iterate as needed

### Community Guidelines
- Be respectful and constructive in discussions
- Help newcomers get started with the project
- Share knowledge about ML deployment best practices
- Report bugs with detailed reproduction steps

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

### Third-Party Licenses
- **FastAPI**: MIT License
- **MLflow**: Apache License 2.0
- **Prometheus**: Apache License 2.0
- **Grafana**: AGPL License (OSS)
- **PostgreSQL**: PostgreSQL License
- **MinIO**: AGPL License

## 🆘 Troubleshooting

### Common Issues

1. **Pod not starting**: Check resource limits and node capacity
2. **Service not accessible**: Verify Ingress configuration and firewall rules
3. **Persistent volume issues**: Check storage class and PV/PVC status
4. **Model loading errors**: Verify MLflow connection and model registration
5. **API errors**: Check logs with `kubectl logs deployment/fastapi-app -n ml-deployment`
6. **Training failures**: Verify ClickHouse connection and data availability
7. **Network connectivity**: Check service discovery and network policies

### Debug Commands

```bash
# Check pod status and logs
kubectl get pods -n ml-deployment
kubectl logs deployment/fastapi-app -n ml-deployment
kubectl logs deployment/mlflow-server -n ml-deployment

# Describe problematic pods
kubectl describe pod <pod-name> -n ml-deployment

# Check service endpoints
kubectl get endpoints -n ml-deployment

# Check ingress status
kubectl get ingress -n ml-deployment
kubectl describe ingress ml-deployment-ingress -n ml-deployment

# Check resource usage
kubectl top pods -n ml-deployment
kubectl top nodes

# Restart deployments
kubectl rollout restart deployment/fastapi-app -n ml-deployment
kubectl rollout restart deployment/mlflow-server -n ml-deployment

# Check events for troubleshooting
kubectl get events -n ml-deployment --sort-by='.lastTimestamp'
```

### Health Checks

```bash
# Get server IP
SERVER_IP=$(curl -s ifconfig.me)

# API health check
curl http://$SERVER_IP/api/health

# MLflow health check
curl http://$SERVER_IP/mlflow/health

# Check all services
kubectl get all -n ml-deployment

# Check service health from within cluster
kubectl run curl-test --image=curlimages/curl -i --rm --restart=Never -- curl http://fastapi-app:8000/health
```

### Getting Help

- Check service logs: `kubectl logs deployment/<service-name> -n ml-deployment`
- Monitor health endpoints: `/health`, `/health/ready`, `/health/live`
- Review Grafana dashboards for system metrics
- Consult the [Kubernetes deployment guide](KUBERNETES_DEPLOYMENT.md) for production issues

## 📚 Additional Resources

### Official Documentation
- **[MLflow Documentation](https://mlflow.org/docs/latest/index.html)** - Model lifecycle management
- **[FastAPI Documentation](https://fastapi.tiangolo.com/)** - Modern Python web framework
- **[Kubernetes Documentation](https://kubernetes.io/docs/)** - Container orchestration
- **[Prometheus Documentation](https://prometheus.io/docs/)** - Monitoring and alerting
- **[Grafana Documentation](https://grafana.com/docs/)** - Observability platform

### Deployment & Infrastructure
- **[k3s Documentation](https://docs.k3s.io/)** - Lightweight Kubernetes
- **[Kustomize Documentation](https://kustomize.io/)** - Kubernetes configuration management
- **[cert-manager Documentation](https://cert-manager.io/docs/)** - TLS certificate automation
- **[NGINX Ingress Controller](https://kubernetes.github.io/ingress-nginx/)** - Load balancing

### ML & Data Engineering
- **[Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)** - Machine learning algorithms
- **[ClickHouse Documentation](https://clickhouse.com/docs)** - Analytical database
- **[MinIO Documentation](https://docs.min.io/)** - Object storage

### Best Practices & Tutorials
- **[ML Engineering Best Practices](https://ml-ops.org/)** - MLOps methodology
- **[Kubernetes Best Practices](https://kubernetes.io/docs/concepts/best-practices/)** - Production deployment
- **[FastAPI Best Practices](https://github.com/zhanymkanov/fastapi-best-practices)** - API development
- **[Monitoring Best Practices](https://prometheus.io/docs/practices/)** - Observability

### Community & Support
- **[MLflow Community](https://mlflow.org/community)** - Forums and discussions
- **[Kubernetes Community](https://kubernetes.io/community/)** - CNCF ecosystem
- **[FastAPI Community](https://fastapi.tiangolo.com/help-fastapi/)** - GitHub and Discord

---

## 🌐 Live Demo

**🚀 Production API**: [https://api.team10deploycamp.online/docs](https://api.team10deploycamp.online/docs)

**📖 Interactive Documentation**: Complete API documentation with live testing capabilities available at the `/docs` endpoint

**🔍 Health Check**: Monitor system status at `/health` endpoint

---

*This project demonstrates production-ready ML deployment practices and serves as a comprehensive template for deploying machine learning systems at scale.*
