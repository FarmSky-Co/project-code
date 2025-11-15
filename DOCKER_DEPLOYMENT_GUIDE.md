# Farmer Credit Scoring System - Production Deployment Guide

## 🐋 Docker Production Setup

This guide helps you deploy the Farmer Credit Scoring System to production using Docker.

### 📋 Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 4GB+ RAM
- 20GB+ storage
- Linux/Windows Server

### 🚀 Quick Start

1. **Clone and prepare:**
```bash
git clone <your-repo>
cd farmer-credit-scoring-prod
cp .env.example .env
```

2. **Configure environment:**
Edit `.env` file with your settings:
```bash
# Update these values
SECRET_KEY=your-production-secret
DATABASE_URL=postgresql://user:pass@postgres:5432/farmsky
REDIS_URL=redis://redis:6379/0
```

3. **Deploy:**
```bash
# Make deploy script executable (Linux/Mac)
chmod +x deploy.sh

# Deploy to production
./deploy.sh production

# Or use Docker Compose directly
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│     Nginx       │────│   Streamlit  │────│   Redis     │
│  (Port 80/443)  │    │  (Port 8501) │    │ (Cache)     │
└─────────────────┘    └──────────────┘    └─────────────┘
                              │
                       ┌──────────────┐    ┌─────────────┐
                       │ PostgreSQL   │────│ Monitoring  │
                       │ (Database)   │    │ (Optional)  │
                       └──────────────┘    └─────────────┘
```

### 📦 Docker Images

**Main Application:**
- Base: `python:3.10-slim`
- Size: ~800MB
- Includes: Streamlit, ML libraries, credit scoring engine

**Services:**
- `farmsky-app`: Main application
- `redis`: Caching layer
- `nginx`: Reverse proxy
- `postgres`: Database (production)
- `prometheus`: Monitoring (optional)
- `grafana`: Dashboards (optional)

### 🔧 Configuration

#### Environment Variables (.env)
```bash
# Application
STREAMLIT_SERVER_PORT=8501
PYTHONPATH=/app

# Database
DATABASE_URL=postgresql://user:pass@postgres:5432/farmsky

# Cache
REDIS_URL=redis://redis:6379/0

# Security
SECRET_KEY=your-secret-key
JWT_SECRET=jwt-secret

# Performance
MAX_WORKERS=4
CACHE_TTL=3600
```

#### Resource Limits
```yaml
# Production limits
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
    reservations:
      cpus: '1.0'
      memory: 2G
```

### 🌐 Networking

**Ports:**
- `80`: HTTP (Nginx)
- `443`: HTTPS (Nginx, with SSL)
- `8501`: Streamlit (internal)
- `6379`: Redis (internal)
- `5432`: PostgreSQL (internal)
- `9090`: Prometheus (optional)
- `3000`: Grafana (optional)

**Health Checks:**
- App: `GET /_stcore/health`
- Redis: `redis-cli ping`
- Postgres: `pg_isready`

### 💾 Data Persistence

**Volumes:**
```yaml
volumes:
  - ./data:/app/data          # Training data
  - ./models:/app/models      # ML models
  - ./results:/app/results    # Analysis results
  - ./logs:/app/logs          # Application logs
  - postgres_data:/var/lib/postgresql/data
  - redis_data:/data
```

### 🔒 Security

**Implemented:**
- Non-root container user
- Security headers (Nginx)
- Rate limiting
- HTTPS ready
- Secret management

**SSL Setup (Production):**
1. Obtain SSL certificate
2. Place in `./ssl/` directory
3. Update `nginx.conf` paths
4. Uncomment HTTPS server block

### 📊 Monitoring

**Included (Optional):**
- Prometheus metrics collection
- Grafana dashboards
- Application health checks
- Container resource monitoring

**Access:**
- Grafana: `http://your-domain:3000`
- Prometheus: `http://your-domain:9090`

### 🚀 Deployment Commands

```bash
# Build and deploy
./deploy.sh production

# View logs
docker-compose logs -f farmsky-app

# Scale application
docker-compose up -d --scale farmsky-app=3

# Update application
docker-compose pull farmsky-app
docker-compose up -d farmsky-app

# Backup data
docker-compose exec postgres pg_dump farmsky > backup.sql

# Shell access
docker-compose exec farmsky-app /bin/bash
```

### 🔧 Maintenance

**Daily:**
- Monitor logs: `docker-compose logs --tail=100`
- Check health: `curl http://localhost/_health`

**Weekly:**
- Update images: `docker-compose pull`
- Backup database
- Review metrics

**Monthly:**
- Security updates
- Performance optimization
- Capacity planning

### ⚠️ Troubleshooting

**Common Issues:**

1. **App won't start:**
```bash
docker-compose logs farmsky-app
# Check environment variables and dependencies
```

2. **Database connection failed:**
```bash
docker-compose exec farmsky-app ping postgres
# Verify DATABASE_URL in .env
```

3. **High memory usage:**
```bash
docker stats
# Consider scaling or increasing limits
```

4. **SSL issues:**
```bash
# Verify certificate paths in nginx.conf
# Check SSL certificate validity
```

### 📈 Production Checklist

**Before Deployment:**
- [ ] Update `.env` with production values
- [ ] Configure SSL certificates
- [ ] Set up monitoring
- [ ] Test backup procedures
- [ ] Configure log rotation
- [ ] Set resource limits
- [ ] Update domain in nginx.conf

**After Deployment:**
- [ ] Verify all services running
- [ ] Test application functionality
- [ ] Check SSL certificate
- [ ] Confirm monitoring active
- [ ] Test backup/restore
- [ ] Document access credentials

### 🆘 Support

For issues:
1. Check logs: `docker-compose logs`
2. Verify configuration
3. Review health checks
4. Check resource usage
5. Consult documentation

---

**Production URL:** `http://your-domain.com`  
**Admin Access:** Configure in application settings  
**Monitoring:** Grafana dashboard at port 3000