#!/bin/bash

# RAG System Setup Script for RunPod Hybrid Deployment
# This script sets up the RAG system using RunPod for ES and LLM

set -e

echo "🚀 Starting RAG System Setup (RunPod Hybrid)..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "   Please create a .env file with your RunPod configuration:"
    echo "   ELASTICSEARCH_URL=https://your-pod-id-9200.proxy.runpod.net"
    echo "   OPENAI_BASE_URL=https://api.runpod.ai/v2/your-pod-id/openai/v1"
    echo "   OPENAI_API_KEY=your-runpod-api-key"
    echo "   OPENAI_MODEL=google/gemma-3-1b-it"
    echo "   LLM_PROVIDER=openai"
    exit 1
fi

echo "✅ Environment configuration found"

# Create docs directory if it doesn't exist
mkdir -p docs

echo "📦 Building and starting services with Docker Compose..."
docker-compose build --no-cache
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 45

# Function to check service health
check_service() {
    local service_name=$1
    local health_url=$2
    local max_attempts=20
    local attempt=1

    echo "🔍 Checking $service_name..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$health_url" > /dev/null 2>&1; then
            echo "✅ $service_name is healthy"
            return 0
        fi
        
        echo "⏳ Attempt $attempt/$max_attempts: Waiting for $service_name..."
        sleep 15
        ((attempt++))
    done
    
    echo "❌ $service_name failed to start after $max_attempts attempts"
    echo "   Check logs with: docker-compose logs $service_name"
    return 1
}

# Check service health
echo "🏥 Performing health checks..."

check_service "API" "http://localhost:8000/healthz"

echo "🎉 API service is healthy!"

# Wait a bit more for UI
sleep 15
check_service "UI" "http://localhost:8501/_stcore/health"

echo "🎉 All services are healthy!"

echo ""
echo "🌟 RAG System (RunPod Hybrid) is ready!"
echo ""
echo "📱 Access points:"
echo "   • Streamlit UI: http://localhost:8501"
echo "   • FastAPI Docs: http://localhost:8000/docs"
echo "   • API Metrics: http://localhost:8000/metrics"
echo "   • RunPod Elasticsearch: (configured via .env)"
echo "   • RunPod LLM: (configured via .env)"
echo ""
echo "🔧 Features enabled:"
echo "   ✅ Hybrid Search (ELSER + BM25 + Dense)"
echo "   ✅ Connection Pooling & Caching"
echo "   ✅ Circuit Breaker Protection"
echo "   ✅ Comprehensive Safety Guardrails"
echo "   ✅ OCR Support for Handwritten Notes"
echo "   ✅ Performance Metrics & Monitoring"
echo ""
echo "📖 Next steps:"
echo "   1. Open the Streamlit UI at http://localhost:8501"
echo "   2. Use the sidebar to ingest documents from Google Drive"
echo "   3. Start asking questions in the chat interface!"
echo ""
echo "🔧 Useful commands:"
echo "   • View logs: docker-compose logs -f"
echo "   • Stop system: docker-compose down"
echo "   • Restart: docker-compose restart"
echo "   • View metrics: curl http://localhost:8000/metrics"
echo ""
echo "📚 For more information, see README.md"
