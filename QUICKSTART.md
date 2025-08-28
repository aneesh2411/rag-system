# 🚀 Quick Start Guide

## Choose Your Setup

### 🏠 Local Development (No Docker)
```bash
# 1. Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install 'httpx[http2]'
brew install tesseract ghostscript  # macOS

# 2. Start Elasticsearch (separate terminal)
curl -O https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.14.3-darwin-x86_64.tar.gz
tar -xzf elasticsearch-8.14.3-darwin-x86_64.tar.gz && cd elasticsearch-8.14.3
echo "xpack.security.enabled: false" >> config/elasticsearch.yml
./bin/elasticsearch

# 3. Start Ollama (separate terminal)
ollama serve &
ollama pull phi3.5:3.8b-mini-instruct-q4_0

# 4. Configure .env
echo "ELASTICSEARCH_URL=http://localhost:9200" > .env
echo "OLLAMA_URL=http://localhost:11434" >> .env
echo "LLM_PROVIDER=ollama" >> .env

# 5. Deploy ELSER
curl -X PUT "localhost:9200/_ml/trained_models/.elser_model_2_linux-x86_64" -H 'Content-Type: application/json' -d '{"input":{"field_names":["text_field"]}}'
curl -X POST "localhost:9200/_ml/trained_models/.elser_model_2_linux-x86_64/deployment/_start"

# 6. Start services
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
streamlit run ui/app.py --server.port 8501
```

### ☁️ Docker + RunPod (Recommended)
```bash
# 1. Setup RunPod (get URLs and API keys)
# 2. Create .env with RunPod configuration
echo "ELASTICSEARCH_URL=https://your-pod-id-9200.proxy.runpod.net" > .env
echo "OPENAI_BASE_URL=https://api.runpod.ai/v2/your-pod-id/openai/v1" >> .env
echo "OPENAI_API_KEY=your-runpod-api-key" >> .env
echo "LLM_PROVIDER=openai" >> .env

# 3. Deploy ELSER to RunPod ES
curl -X PUT "https://your-pod-id-9200.proxy.runpod.net/_ml/trained_models/.elser_model_2_linux-x86_64" -H 'Content-Type: application/json' -d '{"input":{"field_names":["text_field"]}}'
curl -X POST "https://your-pod-id-9200.proxy.runpod.net/_ml/trained_models/.elser_model_2_linux-x86_64/deployment/_start"

# 4. Start with Docker
./setup-runpod.sh
```

### 🏠 Docker Local (Full Stack)
```bash
# 1. Start everything
./setup.sh

# 2. Setup Ollama model
docker-compose -f docker-compose.local.yml exec ollama ollama pull phi3.5:3.8b-mini-instruct-q4_0

# 3. Deploy ELSER
curl -X PUT "localhost:9200/_ml/trained_models/.elser_model_2_linux-x86_64" -H 'Content-Type: application/json' -d '{"input":{"field_names":["text_field"]}}'
curl -X POST "localhost:9200/_ml/trained_models/.elser_model_2_linux-x86_64/deployment/_start"
```

## 🎯 Access Points

- **UI**: http://localhost:8501
- **API**: http://localhost:8000/docs  
- **Health**: http://localhost:8000/healthz
- **Metrics**: http://localhost:8000/metrics

## 🔧 Common Commands

```bash
# Health check
curl http://localhost:8000/healthz

# View metrics
curl http://localhost:8000/metrics | jq

# Clear caches
curl -X POST http://localhost:8000/admin/cache/clear

# Docker logs
docker-compose logs -f

# Restart services
docker-compose restart
```

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| Services won't start | Check logs: `docker-compose logs -f` |
| ES connection failed | Verify URL in .env, test: `curl $ELASTICSEARCH_URL/_cluster/health` |
| OCR not working | Install: `brew install tesseract ghostscript` |
| Dense search failing | Check embedding model in metrics |
| Memory issues | Increase Docker memory limits |

For detailed instructions, see [README.md](README.md)
