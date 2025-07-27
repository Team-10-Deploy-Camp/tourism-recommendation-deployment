# Ubuntu Server Deployment Guide

This guide will walk you through deploying your ML deployment project on a fresh Ubuntu server, from initial SSH connection to running your application.

## Prerequisites

- An Ubuntu server (18.04, 20.04, or 22.04 LTS recommended)
- SSH access to your server
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

## Step 5: Clone Your Project

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

## Step 6: Set Up Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit the environment file with your production values
nano .env
```

### Recommended Production Environment Variables

```bash
# Database Configuration
POSTGRES_USER=mlflow_prod
POSTGRES_PASSWORD=your_strong_password_here
POSTGRES_DB=mlflow_prod
POSTGRES_PORT=2345

# MinIO Configuration
MINIO_ROOT_USER=minio_prod
MINIO_ROOT_PASSWORD=your_strong_minio_password
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001

# MLflow Configuration
MLFLOW_PORT=5001

# FastAPI Configuration
FASTAPI_PORT=8000

# Prometheus Configuration
PROMETHEUS_PORT=9090

# Grafana Configuration
GRAFANA_PORT=3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=your_strong_grafana_password
```

## Step 7: Configure Firewall (Optional but Recommended)

```bash
# Install UFW if not already installed
sudo apt install -y ufw

# Allow SSH
sudo ufw allow ssh

# Allow HTTP and HTTPS (if you plan to use a reverse proxy)
sudo ufw allow 80
sudo ufw allow 443

# Allow your application ports
sudo ufw allow 8000  # FastAPI
sudo ufw allow 5001  # MLflow
sudo ufw allow 9001  # MinIO Console
sudo ufw allow 9090  # Prometheus
sudo ufw allow 3000  # Grafana

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status
```

## Step 8: Deploy Your Application

```bash
# Build and start all services
docker compose up --build -d

# Check if all services are running
docker compose ps

# View logs if needed
docker compose logs -f
```

## Step 9: Set Up MinIO Bucket

```bash
# Access MinIO console at http://your-server-ip:9001
# Login with your MINIO_ROOT_USER and MINIO_ROOT_PASSWORD
# Create a bucket named 'mlflow'
```

## Step 10: Train Your Model

```bash
# Train the model
docker compose up --build model_trainer
```

## Step 11: Verify Deployment

### Check Service Status

```bash
# Check all containers are running
docker compose ps

# Check individual service logs
docker compose logs fastapi_app
docker compose logs mlflow_server
```

### Test Your API

```bash
# Test the prediction endpoint
curl -X 'POST' \
  'http://your-server-ip:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
  }]'
```

## Step 12: Set Up Domain and SSL (Optional)

### Install Nginx

```bash
sudo apt install -y nginx

# Create Nginx configuration
sudo nano /etc/nginx/sites-available/ml_deployment
```

### Nginx Configuration Example

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /mlflow {
        proxy_pass http://localhost:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Enable the Site

```bash
# Create symbolic link
sudo ln -s /etc/nginx/sites-available/ml_deployment /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

### Install SSL with Let's Encrypt

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add this line: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Step 13: Monitoring and Maintenance

### View Logs

```bash
# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f fastapi_app
docker compose logs -f mlflow_server
```

### Update Application

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker compose down
docker compose up --build -d
```

### Backup Database

```bash
# Create backup directory
mkdir -p /opt/backups

# Backup PostgreSQL data
docker compose exec db pg_dump -U $POSTGRES_USER $POSTGRES_DB > /opt/backups/mlflow_backup_$(date +%Y%m%d_%H%M%S).sql
```

### Monitor Resources

```bash
# Check disk usage
df -h

# Check memory usage
free -h

# Check running containers
docker stats
```

## Troubleshooting

### Common Issues

1. **Port already in use**

   ```bash
   # Check what's using the port
   sudo netstat -tulpn | grep :8000

   # Kill the process or change port in .env
   ```

2. **Permission denied**

   ```bash
   # Make sure your user is in docker group
   groups $USER

   # If not, add and relogin
   sudo usermod -aG docker $USER
   ```

3. **Container won't start**

   ```bash
   # Check logs
   docker compose logs service_name

   # Check if ports are available
   sudo netstat -tulpn
   ```

4. **Out of disk space**

   ```bash
   # Clean up Docker
   docker system prune -a

   # Check disk usage
   df -h
   ```

### Useful Commands

```bash
# Stop all services
docker compose down

# Start all services
docker compose up -d

# Restart specific service
docker compose restart fastapi_app

# View resource usage
docker stats

# Access container shell
docker compose exec fastapi_app bash

# Update and restart everything
git pull && docker compose down && docker compose up --build -d
```

## Security Recommendations

1. **Change default passwords** in your `.env` file
2. **Use strong passwords** for all services
3. **Limit firewall access** to necessary ports only
4. **Regular updates** - keep system and Docker updated
5. **Monitor logs** for suspicious activity
6. **Backup regularly** - database and configuration files
7. **Use SSL/TLS** for production deployments

## Performance Optimization

1. **Resource limits** - Set memory and CPU limits in docker-compose.yml
2. **Database optimization** - Configure PostgreSQL for your workload
3. **Caching** - Consider Redis for caching if needed
4. **Load balancing** - Use Nginx for multiple API instances
5. **Monitoring** - Use Prometheus and Grafana for metrics

## Complete Cleanup and Removal

If you need to completely remove the deployed application from your server, follow these steps:

### Step 1: Stop and Remove Docker Containers

```bash
# Navigate to your project directory
cd /opt/ml_deployment

# Stop all running containers
docker compose down

# Remove all containers, networks, and volumes
docker compose down -v

# Verify no containers are running
docker ps -a
```

### Step 2: Remove Docker Images

```bash
# List all images
docker images

# Remove project-specific images
docker rmi ml_deployment_fastapi_app:latest
docker rmi ml_deployment_mlflow_server:latest
docker rmi ml_deployment_model_trainer:latest

# Or remove all unused images
docker image prune -a
```

### Step 3: Remove Docker Volumes

```bash
# List all volumes
docker volume ls

# Remove project volumes
docker volume rm ml_deployment_mlflow_db_data
docker volume rm ml_deployment_mlflow_minio_data
docker volume rm ml_deployment_prometheus_data
docker volume rm ml_deployment_grafana_data

# Or remove all unused volumes
docker volume prune
```

### Step 4: Remove Project Files

```bash
# Navigate to parent directory
cd /opt

# Remove the entire project directory
sudo rm -rf ml_deployment

# Verify removal
ls -la /opt
```

### Step 5: Remove Nginx Configuration (if installed)

```bash
# Remove Nginx site configuration
sudo rm /etc/nginx/sites-enabled/ml_deployment
sudo rm /etc/nginx/sites-available/ml_deployment

# Test and reload Nginx
sudo nginx -t
sudo systemctl reload nginx
```

### Step 6: Remove Firewall Rules (if configured)

```bash
# Remove application-specific firewall rules
sudo ufw delete allow 8000
sudo ufw delete allow 5001
sudo ufw delete allow 9001
sudo ufw delete allow 9090
sudo ufw delete allow 3000

# Check firewall status
sudo ufw status
```

### Step 7: Remove SSL Certificates (if installed)

```bash
# Remove Let's Encrypt certificates
sudo certbot delete --cert-name your-domain.com

# Remove Certbot
sudo apt remove -y certbot python3-certbot-nginx
```

### Step 8: Clean Up System Packages (Optional)

```bash
# Remove Docker (if you want to completely remove Docker)
sudo apt remove -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo apt autoremove -y

# Remove Docker GPG key and repository
sudo rm -f /usr/share/keyrings/docker-archive-keyring.gpg
sudo rm -f /etc/apt/sources.list.d/docker.list

# Remove Docker data directory
sudo rm -rf /var/lib/docker
```

### Step 9: Remove Backups (if created)

```bash
# Remove backup directory
sudo rm -rf /opt/backups
```

### Step 10: Final Cleanup

```bash
# Clean up any remaining Docker resources
docker system prune -a --volumes

# Update package list
sudo apt update

# Clean up package cache
sudo apt autoremove -y
sudo apt autoclean
```

### Complete Removal Script

You can create a cleanup script for easy removal:

```bash
#!/bin/bash
# cleanup.sh - Complete removal script

echo "Stopping and removing containers..."
cd /opt/ml_deployment
docker compose down -v

echo "Removing Docker images..."
docker rmi ml_deployment_fastapi_app:latest 2>/dev/null || true
docker rmi ml_deployment_mlflow_server:latest 2>/dev/null || true
docker rmi ml_deployment_model_trainer:latest 2>/dev/null || true

echo "Removing project files..."
cd /opt
sudo rm -rf ml_deployment

echo "Removing Nginx configuration..."
sudo rm -f /etc/nginx/sites-enabled/ml_deployment
sudo rm -f /etc/nginx/sites-available/ml_deployment

echo "Removing firewall rules..."
sudo ufw delete allow 8000 2>/dev/null || true
sudo ufw delete allow 5001 2>/dev/null || true
sudo ufw delete allow 9001 2>/dev/null || true
sudo ufw delete allow 9090 2>/dev/null || true
sudo ufw delete allow 3000 2>/dev/null || true

echo "Cleaning up Docker..."
docker system prune -a --volumes -f

echo "Cleanup complete!"
```

Make it executable and run:

```bash
chmod +x cleanup.sh
sudo ./cleanup.sh
```

### Verification

After cleanup, verify everything is removed:

```bash
# Check no containers are running
docker ps -a

# Check no project images exist
docker images | grep ml_deployment

# Check no project volumes exist
docker volume ls | grep ml_deployment

# Check project directory is gone
ls -la /opt/ml_deployment 2>/dev/null || echo "Project directory removed"

# Check firewall rules
sudo ufw status
```

### Important Notes

⚠️ **Warning**: This will permanently delete:

- All your ML models and data
- Database contents
- MinIO artifacts
- All application configurations

💾 **Backup First**: If you want to keep any data, backup before cleanup:

```bash
# Backup database
docker compose exec db pg_dump -U $POSTGRES_USER $POSTGRES_DB > backup.sql

# Backup MinIO data
# Download important files from MinIO console first
```

Your ML deployment is now ready to serve predictions! 🚀
