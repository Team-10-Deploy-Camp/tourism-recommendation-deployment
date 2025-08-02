# Setup Domain adityawidiyanto.my.id dengan Cloudflare

Dokumentasi ini menjelaskan cara mengimplementasikan domain `adityawidiyanto.my.id` dengan Cloudflare untuk mengakses services Kubernetes.

## 🎯 Opsi Implementasi

### Opsi 1: Subdomain Routing (Recommended)

- **api.adityawidiyanto.my.id** → FastAPI
- **mlflow.adityawidiyanto.my.id** → MLflow
- **grafana.adityawidiyanto.my.id** → Grafana
- **prometheus.adityawidiyanto.my.id** → Prometheus
- **minio.adityawidiyanto.my.id** → MinIO Console

### Opsi 2: Path-based Routing

- **adityawidiyanto.my.id/api** → FastAPI
- **adityawidiyanto.my.id/mlflow** → MLflow
- **adityawidiyanto.my.id/grafana** → Grafana
- **adityawidiyanto.my.id/prometheus** → Prometheus
- **adityawidiyanto.my.id/minio** → MinIO Console

## 🚀 Quick Setup

### 1. Jalankan Script Setup

```bash
cd k8s
chmod +x setup-domain.sh
./setup-domain.sh
```

### 2. Pilih Metode Routing

Script akan meminta Anda memilih antara:

- `1` untuk subdomain routing
- `2` untuk path-based routing

## 🔧 Manual Setup

### Step 1: Deploy Ingress

#### Untuk Subdomain Routing:

```bash
kubectl apply -f ingress-domain.yaml
```

#### Untuk Path-based Routing:

```bash
kubectl apply -f ingress-path-based.yaml
```

### Step 2: Dapatkan External IP

```bash
kubectl get nodes -o wide
```

Catat External IP dari node Anda.

### Step 3: Setup Cloudflare DNS

1. **Login ke Cloudflare Dashboard**
2. **Pilih domain**: `adityawidiyanto.my.id`
3. **Buka tab 'DNS'**
4. **Tambahkan DNS Records**

#### Untuk Subdomain Routing:

```
Type: A
Name: api
Content: [YOUR_EXTERNAL_IP]
Proxy: Yes (Orange Cloud)

Type: A
Name: mlflow
Content: [YOUR_EXTERNAL_IP]
Proxy: Yes (Orange Cloud)

Type: A
Name: grafana
Content: [YOUR_EXTERNAL_IP]
Proxy: Yes (Orange Cloud)

Type: A
Name: prometheus
Content: [YOUR_EXTERNAL_IP]
Proxy: Yes (Orange Cloud)

Type: A
Name: minio
Content: [YOUR_EXTERNAL_IP]
Proxy: Yes (Orange Cloud)
```

#### Untuk Path-based Routing:

```
Type: A
Name: @ (atau kosong)
Content: [YOUR_EXTERNAL_IP]
Proxy: Yes (Orange Cloud)
```

### Step 4: Test Akses

#### Subdomain Routing:

```bash
# Test FastAPI
curl http://api.adityawidiyanto.my.id/

# Test MLflow
curl http://mlflow.adityawidiyanto.my.id/

# Test Grafana
curl http://grafana.adityawidiyanto.my.id/

# Test Prometheus
curl http://prometheus.adityawidiyanto.my.id/

# Test MinIO
curl http://minio.adityawidiyanto.my.id/
```

#### Path-based Routing:

```bash
# Test FastAPI
curl http://adityawidiyanto.my.id/api/

# Test MLflow
curl http://adityawidiyanto.my.id/mlflow/

# Test Grafana
curl http://adityawidiyanto.my.id/grafana/

# Test Prometheus
curl http://adityawidiyanto.my.id/prometheus/

# Test MinIO
curl http://adityawidiyanto.my.id/minio/
```

## 🔒 Security dengan Cloudflare

### SSL/TLS

- Cloudflare menyediakan SSL/TLS gratis
- Aktifkan "Always Use HTTPS" di Cloudflare
- Gunakan "Full" atau "Full (Strict)" SSL mode

### Firewall Rules

```javascript
// Block access to admin panels from specific countries
(http.request.uri.path contains "/grafana" or http.request.uri.path contains "/prometheus") and ip.geoip.country ne "ID"
```

### Rate Limiting

- Aktifkan rate limiting untuk API endpoints
- Set limit: 100 requests per minute per IP

## 📊 Monitoring

### Cloudflare Analytics

- Monitor traffic di Cloudflare Dashboard
- Lihat bandwidth usage
- Track security threats

### Kubernetes Monitoring

```bash
# Check ingress status
kubectl get ingress -n ml-deployment

# Check ingress logs
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller

# Monitor services
kubectl get svc -n ml-deployment
```

## 🛠️ Troubleshooting

### DNS Propagation

```bash
# Check DNS propagation
nslookup api.adityawidiyanto.my.id
dig api.adityawidiyanto.my.id

# Check from different locations
curl -I http://api.adityawidiyanto.my.id/
```

### Ingress Issues

```bash
# Check ingress status
kubectl describe ingress ml-ingress-domain -n ml-deployment

# Check ingress controller logs
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller

# Test internal connectivity
kubectl run test-pod --image=curlimages/curl -i --rm --restart=Never -- curl http://fastapi-service:8000/
```

### Cloudflare Issues

1. **Check DNS Records**: Pastikan semua record sudah benar
2. **Proxy Status**: Pastikan proxy aktif (orange cloud)
3. **SSL Mode**: Gunakan "Full" atau "Full (Strict)"
4. **Firewall Rules**: Periksa apakah ada rule yang memblokir

## 🔄 Update Domain

### Menambah Service Baru

1. Update file ingress yang sesuai
2. Tambahkan DNS record di Cloudflare
3. Apply perubahan:

```bash
kubectl apply -f ingress-domain.yaml
```

### Mengubah IP Server

1. Update semua DNS records di Cloudflare
2. Tunggu propagasi DNS (5-10 menit)
3. Test akses

## 📝 Best Practices

1. **Gunakan Subdomain**: Lebih bersih dan mudah di-manage
2. **Aktifkan Cloudflare Proxy**: Untuk SSL dan security
3. **Monitor Traffic**: Gunakan Cloudflare Analytics
4. **Backup DNS**: Simpan konfigurasi DNS
5. **Test Regularly**: Test akses secara berkala

## 🎯 Contoh Penggunaan

### FastAPI Prediction

```bash
curl -X POST http://api.adityawidiyanto.my.id/predict \
  -H "Content-Type: application/json" \
  -d '[{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]'
```

### MLflow UI

```bash
# Buka browser
open http://mlflow.adityawidiyanto.my.id/
```

### Grafana Dashboard

```bash
# Buka browser
open http://grafana.adityawidiyanto.my.id/
# Login: admin/admin
```
