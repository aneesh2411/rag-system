# Docker Deployment Guide

This project supports two Docker deployment modes:

## 🌐 RunPod Hybrid (Recommended)
Uses RunPod for Elasticsearch and LLM, runs API and UI locally in containers.

### Setup
```bash
# 1. Ensure .env file exists with RunPod configuration
cp .env.example .env  # Edit with your RunPod URLs and API keys

# 2. Run setup script
./setup-runpod.sh
```

### Configuration (.env file)
```bash
ELASTICSEARCH_URL=https://your-pod-id-9200.proxy.runpod.net
OPENAI_BASE_URL=https://api.runpod.ai/v2/your-pod-id/openai/v1
OPENAI_API_KEY=your-runpod-api-key
OPENAI_MODEL=google/gemma-3-1b-it
LLM_PROVIDER=openai
TOKENIZERS_PARALLELISM=false
```

### Files Used
- `docker-compose.yml` (RunPod hybrid)
- `Dockerfile.api` (FastAPI backend)
- `Dockerfile.ui` (Streamlit frontend)

## 🏠 Local Development
Runs everything locally including Elasticsearch and Ollama.

### Setup
```bash
# Run local setup script
./setup.sh
```

### Files Used
- `docker-compose.local.yml` (Full local stack)
- `Dockerfile.api` (FastAPI backend)
- `Dockerfile.ui` (Streamlit frontend)

## 🔧 Common Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Restart a service
docker-compose restart api

# Rebuild and start
docker-compose build --no-cache && docker-compose up -d

# Check service health
curl http://localhost:8000/healthz
curl http://localhost:8000/metrics
```

## 📊 Service Ports

| Service | Port | Purpose |
|---------|------|---------|
| Streamlit UI | 8501 | Web interface |
| FastAPI | 8000 | API backend |
| Elasticsearch | 9200 | Search engine (local only) |
| Ollama | 11434 | LLM server (local only) |

## 🛡️ Production Considerations

The containers include:
- ✅ Health checks
- ✅ Resource limits
- ✅ Restart policies
- ✅ Security optimizations
- ✅ Performance tuning

## 🐛 Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs service-name

# Check resource usage
docker stats

# Rebuild container
docker-compose build --no-cache service-name
```

### API connection issues
```bash
# Verify environment variables
docker-compose exec api env | grep -E "(ELASTICSEARCH|OPENAI)"

# Test connectivity
docker-compose exec api curl -s http://localhost:8000/healthz
```

### Performance issues
```bash
# Monitor metrics
curl http://localhost:8000/metrics | jq

# Check resource limits
docker-compose config
```
