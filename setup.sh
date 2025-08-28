#!/bin/bash

# RAG System Setup Script
# This script sets up and starts the complete RAG system

set -e

echo "🚀 Starting RAG System Setup..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create docs directory if it doesn't exist
mkdir -p docs

echo "📦 Starting services with Docker Compose..."
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 30

# Function to check service health
check_service() {
    local service_name=$1
    local health_url=$2
    local max_attempts=30
    local attempt=1

    echo "🔍 Checking $service_name..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$health_url" > /dev/null 2>&1; then
            echo "✅ $service_name is healthy"
            return 0
        fi
        
        echo "⏳ Attempt $attempt/$max_attempts: Waiting for $service_name..."
        sleep 10
        ((attempt++))
    done
    
    echo "❌ $service_name failed to start after $max_attempts attempts"
    return 1
}

# Check service health
echo "🏥 Performing health checks..."

check_service "Elasticsearch" "http://localhost:9200/_cluster/health"
check_service "Ollama" "http://localhost:11434/api/tags"
check_service "API" "http://localhost:8000/healthz"

echo "🎉 All services are healthy!"

echo ""
echo "🌟 RAG System is ready!"
echo ""
echo "📱 Access points:"
echo "   • Streamlit UI: http://localhost:8501"
echo "   • FastAPI Docs: http://localhost:8000/docs"
echo "   • Elasticsearch: http://localhost:9200"
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
echo ""
echo "📚 For more information, see README.md"
