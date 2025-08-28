# 🚀 Enterprise RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with hybrid search capabilities, comprehensive safety guardrails, and industry-standard performance optimizations.

## ✨ Features

- **🔍 Hybrid Search**: ELSER + BM25 + Dense vector search with RRF fusion
- **🛡️ Safety Guardrails**: 18 safety patterns across 7 categories
- **⚡ Performance**: Connection pooling, embedding caching, circuit breakers
- **📊 Monitoring**: Comprehensive metrics and health checks
- **🔄 OCR Support**: Handles handwritten documents with Tesseract + Ghostscript
- **☁️ Cloud Ready**: RunPod integration for scalable inference
- **🐳 Docker Support**: Multiple deployment options

## 🎥 Demo

Watch the system in action:

https://github.com/your-username/rag-system/assets/your-user-id/RAG.mp4

*This demo showcases the RAG system's document ingestion, hybrid search capabilities, and safety guardrails in action.*

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   FastAPI API   │────│  Elasticsearch  │
│   (Frontend)    │    │   (Backend)     │    │   + ELSER       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   LLM Service   │
                       │ (RunPod/Ollama) │
                       └─────────────────┘
```

## 📋 Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- Git

## 🚀 Quick Start

Choose your deployment method:

### Option 1: Local Development (No Docker, No RunPod)
### Option 2: Docker + RunPod Hybrid (Recommended)
### Option 3: Docker Local (Full Local Stack)

---

## 🏠 Option 1: Local Development

Run everything locally without Docker or RunPod dependencies.

### 1. Clone and Setup

```bash
# Clone repository
git clone <your-repo-url>
cd rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install 'httpx[http2]'
pip install elasticsearch==8.11.1
```

### 2. Install System Dependencies

**macOS:**
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install tesseract
brew install ghostscript
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils ghostscript
```

### 3. Setup Local Services

**Install and Start Elasticsearch:**
```bash
# Download Elasticsearch 8.14.3
curl -O https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.14.3-darwin-x86_64.tar.gz
tar -xzf elasticsearch-8.14.3-darwin-x86_64.tar.gz
cd elasticsearch-8.14.3

# Configure for development
echo "xpack.security.enabled: false" >> config/elasticsearch.yml
echo "xpack.security.enrollment.enabled: false" >> config/elasticsearch.yml

# Start Elasticsearch
./bin/elasticsearch
```

**Install and Start Ollama:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve &

# Pull a model
ollama pull phi3.5:3.8b-mini-instruct-q4_0
```

### 4. Configure Environment

Create `.env` file:
```bash
# Elasticsearch (local)
ELASTICSEARCH_URL=http://localhost:9200

# Ollama (local)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=phi3.5:3.8b-mini-instruct-q4_0
LLM_PROVIDER=ollama

# Performance
TOKENIZERS_PARALLELISM=false

# Google Drive (optional)
DRIVE_FOLDER_ID=your-drive-folder-id
GOOGLE_API_KEY=your-google-api-key
```

### 5. Deploy ELSER Model

```bash
# Wait for Elasticsearch to start, then deploy ELSER
curl -X PUT "localhost:9200/_ml/trained_models/.elser_model_2_linux-x86_64?pretty" \
  -H 'Content-Type: application/json' \
  -d '{"input":{"field_names":["text_field"]}}'

curl -X POST "localhost:9200/_ml/trained_models/.elser_model_2_linux-x86_64/deployment/_start?pretty"
```

### 6. Start Services

```bash
# Terminal 1: Start FastAPI
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start Streamlit
source venv/bin/activate
streamlit run ui/app.py --server.port 8501
```

### 7. Access Application

- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/healthz
- **Metrics**: http://localhost:8000/metrics

---

## ☁️ Option 2: Docker + RunPod Hybrid (Recommended)

Uses RunPod for Elasticsearch and LLM, Docker for API and UI.

### 1. Setup RunPod Services

**Elasticsearch Pod:**
```bash
# Create Elasticsearch pod on RunPod
# Use template: Elasticsearch 8.x
# Expose port: 9200
# Note your pod URL: https://your-pod-id-9200.proxy.runpod.net
```

**LLM Pod:**
```bash
# Create LLM pod on RunPod  
# Use template: OpenAI Compatible API
# Model: google/gemma-3-1b-it
# Note your endpoint: https://api.runpod.ai/v2/your-pod-id/openai/v1
```

**Deploy ELSER Model:**
```bash
# Deploy ELSER to your Elasticsearch pod
curl -X PUT "https://your-pod-id-9200.proxy.runpod.net/_ml/trained_models/.elser_model_2_linux-x86_64?pretty" \
  -H 'Content-Type: application/json' \
  -d '{"input":{"field_names":["text_field"]}}'

curl -X POST "https://your-pod-id-9200.proxy.runpod.net/_ml/trained_models/.elser_model_2_linux-x86_64/deployment/_start?pretty"
```

### 2. Configure Environment

Create `.env` file:
```bash
# RunPod Elasticsearch
ELASTICSEARCH_URL=https://your-pod-id-9200.proxy.runpod.net

# RunPod LLM
OPENAI_BASE_URL=https://api.runpod.ai/v2/your-pod-id/openai/v1
OPENAI_API_KEY=your-runpod-api-key
OPENAI_MODEL=google/gemma-3-1b-it
LLM_PROVIDER=openai

# Performance
TOKENIZERS_PARALLELISM=false

# Google Drive (optional)
DRIVE_FOLDER_ID=your-drive-folder-id
```

### 3. Deploy with Docker

```bash
# Clone repository
git clone <your-repo-url>
cd rag-system

# Run setup script
./setup-runpod.sh

# Or manually:
docker-compose build --no-cache
docker-compose up -d
```

### 4. Verify Deployment

```bash
# Check services
docker-compose ps

# View logs
docker-compose logs -f

# Test health
curl http://localhost:8000/healthz
curl http://localhost:8000/metrics
```

### 5. Access Application

- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs
- **Metrics Dashboard**: http://localhost:8000/metrics

---

## 🏠 Option 3: Docker Local (Full Local Stack)

Everything runs locally in Docker containers.

### 1. Deploy Full Stack

```bash
# Clone repository
git clone <your-repo-url>
cd rag-system

# Run local setup
./setup.sh

# Or manually:
docker-compose -f docker-compose.local.yml up -d
```

### 2. Setup Ollama Model

```bash
# Enter Ollama container
docker-compose -f docker-compose.local.yml exec ollama bash

# Pull model
ollama pull phi3.5:3.8b-mini-instruct-q4_0
```

### 3. Deploy ELSER Model

```bash
# Wait for Elasticsearch to be healthy, then:
curl -X PUT "http://localhost:9200/_ml/trained_models/.elser_model_2_linux-x86_64?pretty" \
  -H 'Content-Type: application/json' \
  -d '{"input":{"field_names":["text_field"]}}'

curl -X POST "http://localhost:9200/_ml/trained_models/.elser_model_2_linux-x86_64/deployment/_start?pretty"
```

### 4. Access Application

- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs  
- **Elasticsearch**: http://localhost:9200
- **Ollama**: http://localhost:11434

---

## 📖 Usage Guide

### 1. Document Ingestion

1. Open Streamlit UI at http://localhost:8501
2. In sidebar, enter Google Drive Folder ID
3. Check "Reindex all documents" for first run
4. Click "🔄 Start Ingestion"
5. Wait for completion (progress shown)

### 2. Querying Documents

1. Type your question in the chat interface
2. Select retrieval mode:
   - **hybrid**: ELSER + BM25 + Dense (recommended)
   - **elser**: Sparse vector search only
3. Adjust "Number of Results" slider
4. View answer with citations

### 3. Monitoring

- **Health**: `GET /healthz`
- **Metrics**: `GET /metrics`
- **Cache Clear**: `POST /admin/cache/clear`

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ELASTICSEARCH_URL` | Elasticsearch endpoint | `http://localhost:9200` |
| `OPENAI_BASE_URL` | LLM API endpoint | `""` |
| `OPENAI_API_KEY` | LLM API key | `""` |
| `OPENAI_MODEL` | LLM model name | `google/gemma-3-1b-it` |
| `LLM_PROVIDER` | LLM provider (`ollama`/`openai`) | `ollama` |
| `OLLAMA_URL` | Ollama endpoint | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model name | `phi3.5:3.8b-mini-instruct-q4_0` |

### Performance Tuning

- **Embedding Cache**: 1000 entries, 1-hour TTL
- **Connection Pool**: 20 keepalive, 100 max connections
- **Circuit Breaker**: 5 failure threshold, 60s recovery
- **Batch Processing**: 20 chunks for ELSER, 4 files for ingestion

## 🛡️ Safety Features

### Guardrails Categories

1. **Harmful Content**: Violence, self-harm, weapons
2. **Illegal Activities**: Hacking, fraud, drugs
3. **Hate Speech**: Discrimination, supremacy
4. **PII Requests**: Personal data, credentials
5. **Inappropriate Content**: Explicit material
6. **Misinformation**: Conspiracy theories, fake advice
7. **System Manipulation**: Prompt injection attempts

### Response Validation

- Safety checking of LLM responses
- Hallucination detection
- Grounding enforcement
- Quality control

## 📊 Monitoring & Metrics

### Health Endpoints

```bash
# Overall health
curl http://localhost:8000/healthz

# Detailed metrics
curl http://localhost:8000/metrics | jq

# Cache statistics
curl http://localhost:8000/metrics | jq '.embedding_cache'

# Circuit breaker status
curl http://localhost:8000/metrics | jq '.circuit_breaker'
```

### Performance Metrics

- Embedding cache hit rates
- Circuit breaker statistics
- Connection pool utilization
- Guardrails safety stats
- System operational status

## 🐛 Troubleshooting

### Common Issues

**Services won't start:**
```bash
# Check logs
docker-compose logs -f

# Restart services
docker-compose restart

# Rebuild containers
docker-compose build --no-cache
```

**Elasticsearch connection failed:**
```bash
# Test connectivity
curl http://localhost:9200/_cluster/health

# Check ELSER model
curl http://localhost:9200/_ml/trained_models/.elser_model_2_linux-x86_64/_stats
```

**OCR not working:**
```bash
# Check dependencies
which tesseract
which gs

# Install missing packages
brew install tesseract ghostscript  # macOS
sudo apt-get install tesseract-ocr ghostscript  # Ubuntu
```

**Dense search failing:**
```bash
# Check embedding model initialization
curl http://localhost:8000/metrics | jq '.embedding_cache'

# Restart API service
docker-compose restart api
```

### Performance Issues

```bash
# Monitor resource usage
docker stats

# Check metrics
curl http://localhost:8000/metrics | jq

# Clear caches
curl -X POST http://localhost:8000/admin/cache/clear
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs: `docker-compose logs -f`
3. Test health endpoints
4. Check configuration files

## 📚 Additional Resources

- [Docker Deployment Guide](DOCKER.md)
- [API Documentation](http://localhost:8000/docs)
- [RunPod Documentation](https://docs.runpod.io/)
- [Elasticsearch ELSER Guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/semantic-search-elser.html)