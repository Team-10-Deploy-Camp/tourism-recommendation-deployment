# Kubernetes Deployment Guide for SSH Server

This guide will walk you through deploying your ML deployment project on a fresh Ubuntu server using Kubernetes with ingress for auto-scaling capabilities.

## Prerequisites

- An Ubuntu server (18.04, 20.04, or 22.04 LTS recommended)
- SSH access to your server
- At least 4GB RAM and 2 CPU cores
- 20GB free disk space
- A domain name (optional, for production)

## Step 1: Connect to Your Server

### Via SSH

```bash
ssh username@your-server-ip
```

### Via SSH Key (Recommended)

```bash
ssh -i /path/to/your/private-key username@your-server-ip
```

## Step 2: Update System Packages

```bash
# Update package list
sudo apt update

# Upgrade existing packages
sudo apt upgrade -y

# Install essential tools
sudo apt install -y curl wget git unzip software-properties-common apt-transport-https ca-certificates gnupg lsb-release
```

## Step 3: Install Git

```bash
# Install Git
sudo apt install -y git

# Configure Git (replace with your details)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify installation
git --version
```

## Step 4: Install Docker

### Add Docker's official GPG key

```bash
# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
```

### Add Docker repository

```bash
# Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### Install Docker Engine

```bash
# Update package list
sudo apt update

# Install Docker Engine
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to docker group (replace 'username' with your actual username)
sudo usermod -aG docker $USER

# Verify installation
docker --version
docker compose version
```

**Important**: Log out and log back in for the group changes to take effect, or run:

```bash
newgrp docker
```

## Step 5: Install k3s (Recommended Kubernetes Distribution)

### Install k3s (Lightweight Kubernetes)

```bash
# Install k3s with default configuration
curl -sfL https://get.k3s.io | sh -

# Wait for k3s to start (usually 30-60 seconds)
sudo systemctl status k3s

# Verify k3s is running
sudo systemctl is-active k3s

# Get kubeconfig
sudo cat /etc/rancher/k3s/k3s.yaml

# Copy kubeconfig to your user directory
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER:$USER ~/.kube/config

# Verify installation
kubectl get nodes

# Check k3s version
kubectl version --short
```

### Alternative: Install MicroK8s (Ubuntu's Kubernetes)

```bash
# Install MicroK8s
sudo snap install microk8s --classic

# Add user to microk8s group
sudo usermod -a -G microk8s $USER

# Start MicroK8s
sudo microk8s start

# Get kubeconfig
sudo microk8s config > ~/.kube/config

# Verify installation
kubectl get nodes
```

## Step 6: Install kubectl

```bash
# Download kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

# Install kubectl
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Verify installation
kubectl version --client
```

## Step 7: Install kustomize

```bash
# Install kustomize
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
sudo mv kustomize /usr/local/bin/

# Verify installation
kustomize version
```

## Step 8: Install Ingress Controller

### For k3s (Traefik is pre-installed)

```bash
# Check if Traefik is running
kubectl get pods -n kube-system | grep traefik

# If not running, enable it
sudo systemctl restart k3s
```

### For MicroK8s

```bash
# Enable ingress addon
sudo microk8s enable ingress

# Wait for ingress to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=ingress-nginx -n ingress-nginx --timeout=300s
```

### For other Kubernetes distributions

```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/baremetal/deploy.yaml

# Wait for ingress controller to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=ingress-nginx -n ingress-nginx --timeout=300s
```

## Step 9: Clone Your Project

```bash
# Navigate to a suitable directory
cd /opt

# Clone your repository
sudo git clone https://github.com/yourusername/ml_deployment.git

# Change ownership to your user
sudo chown -R $USER:$USER ml_deployment

# Navigate to project directory
cd ml_deployment
```

## Step 10: Setup Environment and Generate Kubernetes Resources

### Automatic Setup (Recommended)

```bash
# Navigate to project root
cd /opt/ml_deployment

# Run the setup script (this will create .env from .env.example and generate ConfigMap/Secrets)
./k8s/setup-env.sh
```

The setup script will:

1. Create `.env` file from `.env.example` if it doesn't exist
2. Prompt you to edit the `.env` file with your actual values
3. Generate `configmap.yaml` and `secret.yaml` from your `.env` file
4. Optionally apply them to Kubernetes

### Manual Setup (Alternative)

If you prefer to do it manually:

```bash
# Navigate to k8s directory
cd k8s

# Generate ConfigMap from .env file
./generate-configmap.sh

# Generate Secrets from .env file
./generate-secrets.sh
```

### Environment Variables to Set

Edit your `.env` file with these values:

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

# Prometheus Configuration
PROMETHEUS_PORT=9090

# Grafana Configuration
GRAFANA_PORT=3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=your_strong_grafana_password

# Model Configuration
MODEL_NAME=iris-classifier
MODEL_STAGE=production
```

## Step 11: Build Docker Images

```bash
# Navigate back to project root
cd /opt/ml_deployment

# Build FastAPI app image
docker build -t fastapi-app:latest ./app

# Build MLflow server image
docker build -t mlflow-server:latest ./mlflow_server

# Build model trainer image
docker build -t model-trainer:latest ./model

# Verify images are built
docker images | grep -E "(fastapi-app|mlflow-server|model-trainer)"
```

## Step 12: Configure Firewall

```bash
# Install UFW if not already installed
sudo apt install -y ufw

# Allow SSH
sudo ufw allow ssh

# Allow HTTP and HTTPS (for ingress)
sudo ufw allow 80
sudo ufw allow 443

# Allow Kubernetes API (if needed)
sudo ufw allow 6443

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status
```

## Step 13: Deploy to Kubernetes

### Deploy the Application

```bash
# Navigate to k8s directory
cd /opt/ml_deployment/k8s

# Make deployment scripts executable
chmod +x *.sh

# Deploy using the automated script
./deployment-scripts.sh
```

### Manual Deployment (Alternative)

```bash
# Apply all configurations using kustomize
kubectl apply -k .

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

## Step 14: Get Your Server IP

```bash
# Get your server's external IP
curl -s ifconfig.me

# Or get your server's internal IP
ip addr show | grep "inet " | grep -v 127.0.0.1

# For cloud providers, you might need to check your cloud console
```

## Step 15: Access Your Services

### Via Ingress (Path-based routing)

```bash
# Get your server IP
SERVER_IP=$(curl -s ifconfig.me)

# Test FastAPI application
curl http://$SERVER_IP/api/

# Test MLflow UI
curl http://$SERVER_IP/mlflow/

# Test Prometheus
curl http://$SERVER_IP/prometheus/

# Test Grafana
curl http://$SERVER_IP/grafana/

# Test MinIO Console
curl http://$SERVER_IP/minio/
```

### Service URLs

- **FastAPI App**: `http://<server-ip>/api/`
- **MLflow UI**: `http://<server-ip>/mlflow/`
- **Prometheus**: `http://<server-ip>/prometheus/`
- **Grafana**: `http://<server-ip>/grafana/` (admin/your_grafana_password)
- **MinIO Console**: `http://<server-ip>/minio/` (minio_prod/your_minio_password)

## Step 16: Set Up MinIO Bucket

```bash
# Access MinIO console at http://<server-ip>/minio/
# Login with your MINIO_ROOT_USER and MINIO_ROOT_PASSWORD
# Create a bucket named 'mlflow'
```

## Step 17: Train Your Model

### Method 1: Using Kubernetes Job (Recommended)

First, create a Job YAML file for model training:

```bash
# Create model training job file
cat > k8s/model-training-job.yaml << 'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: model-training-job
  namespace: ml-deployment
spec:
  template:
    spec:
      containers:
      - name: model-trainer
        image: model-trainer:latest
        imagePullPolicy: Never
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-service:5000"
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: ml-secrets
              key: MINIO_ROOT_USER
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: ml-secrets
              key: MINIO_ROOT_PASSWORD
        - name: MLFLOW_S3_ENDPOINT_URL
          value: "http://minio-service:9000"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
      restartPolicy: Never
  backoffLimit: 3
EOF

# Apply the job
kubectl apply -f k8s/model-training-job.yaml

# Check job status
kubectl get jobs -n ml-deployment

# Monitor job progress
kubectl get pods -n ml-deployment | grep model-training

# View training logs
kubectl logs job/model-training-job -n ml-deployment
```

### Method 2: Manual Training (Alternative)

```bash
# Run training manually with correct credentials
kubectl run model-trainer --image=model-trainer:latest -n ml-deployment --rm -i --tty --restart=Never \
  --env="MLFLOW_TRACKING_URI=http://mlflow-service:5000" \
  --env="AWS_ACCESS_KEY_ID=minioadminteam10" \
  --env="AWS_SECRET_ACCESS_KEY=minioadminteam10" \
  --env="MLFLOW_S3_ENDPOINT_URL=http://minio-service:9000"
```

### Verify Training Success

```bash
# Check if job completed successfully
kubectl get jobs -n ml-deployment

# View detailed logs
kubectl logs job/model-training-job -n ml-deployment --tail=50

# Check if model was registered in MLflow
# Access MLflow UI at http://<server-ip>/mlflow/
# Look for the "iris-classifier" model in the Model Registry
```

### Access MLflow to View Training Results

```bash
# Port forward MLflow service for local access
kubectl port-forward service/mlflow-service 5000:5000 -n ml-deployment

# Or access via ingress
curl http://<server-ip>/mlflow/
```

**MLflow Features to Explore:**

- **Experiments**: View training runs and metrics
- **Model Registry**: See registered models and versions
- **Artifacts**: Download model files and artifacts
- **Metrics**: Compare model performance across runs

### Clean Up Training Job

```bash
# Delete the training job after completion
kubectl delete job model-training-job -n ml-deployment

# Or delete all completed jobs
kubectl delete jobs --field-selector status.successful=1 -n ml-deployment
```

### Troubleshooting Model Training

**Common Issues:**

1. **Job fails to start:**

   ```bash
   # Check if image exists
   docker images | grep model-trainer

   # Check pod events
   kubectl describe pod -l job-name=model-training-job -n ml-deployment
   ```

2. **Training fails with connection errors:**

   ```bash
   # Check if MLflow service is running
   kubectl get pods -n ml-deployment | grep mlflow

   # Check MLflow logs
   kubectl logs deployment/mlflow-server -n ml-deployment
   ```

3. **MinIO connection issues:**

   ```bash
   # Check MinIO service
   kubectl get pods -n ml-deployment | grep minio

   # Verify MinIO credentials
   kubectl get secret ml-secrets -n ml-deployment -o jsonpath='{.data.MINIO_ROOT_USER}' | base64 -d
   ```

4. **Model not appearing in MLflow:**
   ```bash
   # Check if bucket exists in MinIO
   # Access MinIO console at http://<server-ip>/minio/
   # Create 'mlflow' bucket if it doesn't exist
   ```

## Step 18: Test Auto-scaling

### Check HPA Status

```bash
# View HPA details
kubectl get hpa -n ml-deployment

# Detailed HPA information
kubectl describe hpa fastapi-hpa -n ml-deployment
```

### Load Testing

```bash
# Navigate to k8s directory
cd /opt/ml_deployment/k8s

# Run load test to trigger auto-scaling
./load-test.sh
```

### Manual Scaling Test

```bash
# Generate load manually
for i in {1..100}; do
  curl -X POST http://<server-ip>/api/predict \
    -H "Content-Type: application/json" \
    -d '[{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]' &
done

# Watch pods scale up
kubectl get pods -n ml-deployment -l app=fastapi-app -w
```

## Step 19: Monitor Your Deployment

### Check Resource Usage

```bash
# Check pod status
kubectl get pods -n ml-deployment

# Check resource usage
kubectl top pods -n ml-deployment

# Check node resource usage
kubectl top nodes
```

### View Logs

```bash
# View FastAPI logs
kubectl logs -f deployment/fastapi-app -n ml-deployment

# View MLflow logs
kubectl logs -f deployment/mlflow-server -n ml-deployment

# View all logs
kubectl logs -f -l app=fastapi-app -n ml-deployment
```

### Check Events

```bash
# View recent events
kubectl get events -n ml-deployment --sort-by='.lastTimestamp'

# View HPA events
kubectl get events -n ml-deployment | grep -i hpa
```

## Step 20: Set Up Domain and SSL (Optional)

### Install Cert-Manager

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Wait for cert-manager to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=cert-manager -n cert-manager --timeout=300s
```

### Create ClusterIssuer for Let's Encrypt

```bash
# Create ClusterIssuer
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### Update Ingress with SSL

```bash
# Update ingress.yaml to include SSL annotations
nano ingress.yaml
```

Add these annotations to your ingress:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-ingress
  namespace: ml-deployment
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /$2
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/use-regex: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
    - hosts:
        - your-domain.com
      secretName: ml-tls
  rules:
    - host: your-domain.com
      http:
        paths:
          - path: /api(/|$)(.*)
            pathType: Prefix
            backend:
              service:
                name: fastapi-service
                port:
                  number: 8000
        # ... other paths
```

### Apply Updated Ingress

```bash
# Apply the updated ingress
kubectl apply -f ingress.yaml

# Check certificate status
kubectl get certificate -n ml-deployment
```

## Step 21: Backup and Maintenance

### Backup Database

```bash
# Create backup directory
mkdir -p /opt/backups

# Backup PostgreSQL data
kubectl exec -n ml-deployment deployment/postgres -- pg_dump -U mlflow_prod mlflow > /opt/backups/mlflow_backup_$(date +%Y%m%d_%H%M%S).sql
```

### Update Application

```bash
# Pull latest changes
cd /opt/ml_deployment
git pull origin main

# Rebuild images
docker build -t fastapi-app:latest ./app
docker build -t mlflow-server:latest ./mlflow_server
docker build -t model-trainer:latest ./model

# Restart deployments
kubectl rollout restart deployment/fastapi-app -n ml-deployment
kubectl rollout restart deployment/mlflow-server -n ml-deployment

# Wait for rollout to complete
kubectl rollout status deployment/fastapi-app -n ml-deployment
kubectl rollout status deployment/mlflow-server -n ml-deployment
```

### Monitor Auto-scaling

```bash
# Check HPA status
kubectl get hpa -n ml-deployment

# View scaling events
kubectl get events -n ml-deployment | grep -i hpa

# Monitor pod scaling in real-time
kubectl get pods -n ml-deployment -l app=fastapi-app -w
```

## Troubleshooting

### Common Issues

1. **Pods not starting**

   ```bash
   # Check pod status
   kubectl get pods -n ml-deployment

   # Check pod events
   kubectl describe pod <pod-name> -n ml-deployment

   # Check logs
   kubectl logs <pod-name> -n ml-deployment
   ```

2. **Ingress not working**

   ```bash
   # Check ingress status
   kubectl get ingress -n ml-deployment

   # Check ingress controller
   kubectl get pods -n kube-system | grep ingress

   # Check ingress logs
   kubectl logs -n kube-system -l app.kubernetes.io/name=ingress-nginx
   ```

3. **HPA not scaling**

   ```bash
   # Check HPA status
   kubectl describe hpa fastapi-hpa -n ml-deployment

   # Check metrics server
   kubectl get pods -n kube-system | grep metrics-server

   # Check resource usage
   kubectl top pods -n ml-deployment
   ```

4. **Storage issues**

   ```bash
   # Check PVC status
   kubectl get pvc -n ml-deployment

   # Check storage class
   kubectl get storageclass

   # Check persistent volumes
   kubectl get pv
   ```

### Useful Commands

```bash
# Get all resources in namespace
kubectl get all -n ml-deployment

# Check resource usage
kubectl top pods -n ml-deployment
kubectl top nodes

# View events
kubectl get events -n ml-deployment --sort-by='.lastTimestamp'

# Access pod shell
kubectl exec -it deployment/fastapi-app -n ml-deployment -- /bin/bash

# Port forward for debugging
kubectl port-forward -n ml-deployment svc/fastapi-service 8000:8000

# Check ingress controller logs
kubectl logs -n kube-system -l app.kubernetes.io/name=ingress-nginx
```

## Cleanup

### Remove Deployment

```bash
# Navigate to k8s directory
cd /opt/ml_deployment/k8s

# Remove all resources
kubectl delete -k .

# Or use the cleanup script
./cleanup-scripts.sh
```

### Complete Cleanup

```bash
# Remove namespace
kubectl delete namespace ml-deployment

# Remove Docker images
docker rmi fastapi-app:latest
docker rmi mlflow-server:latest
docker rmi model-trainer:latest

# Clean up Docker
docker system prune -a --volumes

# Remove project files
sudo rm -rf /opt/ml_deployment
```

## Security Recommendations

1. **Change default passwords** in your secret.yaml file
2. **Use strong passwords** for all services
3. **Limit firewall access** to necessary ports only
4. **Regular updates** - keep system and Kubernetes updated
5. **Monitor logs** for suspicious activity
6. **Backup regularly** - database and configuration files
7. **Use SSL/TLS** for production deployments
8. **Network policies** - restrict pod-to-pod communication
9. **RBAC** - implement proper role-based access control
10. **Secrets management** - use external secrets management for production

## Performance Optimization

1. **Resource limits** - Set appropriate memory and CPU limits
2. **HPA tuning** - Adjust scaling thresholds and behavior
3. **Database optimization** - Configure PostgreSQL for your workload
4. **Caching** - Consider Redis for caching if needed
5. **Monitoring** - Use Prometheus and Grafana for metrics
6. **Node affinity** - Place pods on appropriate nodes
7. **Pod disruption budgets** - Ensure high availability during updates

The system will automatically scale your FastAPI application based on CPU and memory usage, providing high availability and optimal performance for your ML workloads.
