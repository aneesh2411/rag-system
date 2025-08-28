"""
FastAPI main application with RAG endpoints.

This module provides a production-ready RAG (Retrieval-Augmented Generation) API
with industry-standard features including:
- Connection pooling for optimal performance
- Circuit breaker pattern for resilience
- Comprehensive safety guardrails
- Embedding caching for efficiency
- Performance metrics and monitoring

The API supports hybrid search (ELSER + BM25 + Dense vectors) and integrates
with RunPod for scalable cloud inference.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from contextlib import asynccontextmanager
import logging
import asyncio
import httpx

from .ingest import PDFIngester
from .retrieval import HybridRetriever
from .llm import RunPodLLM
from .guardrails import GuardrailsFilter
from . import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connection pooling lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with connection pooling."""
    # Startup: Create shared HTTP client with connection pooling
    logger.info("Starting application with connection pooling...")
    
    # Configure connection limits for optimal performance
    limits = httpx.Limits(
        max_keepalive_connections=20,  # Keep 20 connections alive
        max_connections=100,           # Max 100 total connections
        keepalive_expiry=30.0         # Keep connections alive for 30 seconds
    )
    
    # Create shared HTTP client with timeout configuration
    timeout = httpx.Timeout(
        connect=10.0,    # 10 seconds to establish connection
        read=300.0,      # 5 minutes for long-running requests (ingestion)
        write=10.0,      # 10 seconds to send data
        pool=5.0         # 5 seconds to get connection from pool
    )
    
    app.state.http_client = httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        http2=True  # Enable HTTP/2 for better performance
    )
    
    logger.info("Connection pool initialized successfully")
    
    yield
    
    # Shutdown: Close HTTP client
    logger.info("Shutting down connection pool...")
    await app.state.http_client.aclose()
    logger.info("Application shutdown complete")

app = FastAPI(
    title="RAG System API",
    description="Elasticsearch + ELSER + RunPod Gemma RAG System",
    version="1.0.0",
    lifespan=lifespan  # Enable connection pooling
)

# Initialize components
ingester = PDFIngester()
retriever = HybridRetriever()
llm = RunPodLLM()
guardrails = GuardrailsFilter()

# Request/Response models
class IngestRequest(BaseModel):
    drive_folder_id: str = Field(default=settings.DRIVE_FOLDER_ID)
    reindex: bool = Field(default=True, description="Whether to clear and reindex all documents")

class IngestResponse(BaseModel):
    documents_indexed: int
    chunks: int

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    mode: Literal["elser", "hybrid"] = Field(default="hybrid")
    top_k: int = Field(default=settings.DEFAULT_TOP_K, ge=1, le=20)

class Citation(BaseModel):
    title: str
    link: str
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    used_mode: str

class HealthResponse(BaseModel):
    status: str
    elasticsearch: str
    ollama: str

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    try:
        # Check Elasticsearch connection
        await retriever.initialize()
        logger.info("Elasticsearch connection established")
        
        # Check Ollama connection
        await llm.initialize()
        logger.info("Ollama connection established")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check Elasticsearch
        es_status = "ok" if await retriever.health_check() else "error"
        
        # Check Ollama
        ollama_status = "ok" if await llm.health_check() else "error"
        
        overall_status = "ok" if es_status == "ok" and ollama_status == "ok" else "error"
        
        return HealthResponse(
            status=overall_status,
            elasticsearch=es_status,
            ollama=ollama_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/metrics")
async def get_metrics():
    """Get system metrics and performance statistics."""
    try:
        # Get embedding cache stats
        cache_stats = retriever.get_cache_stats()
        
        # Get circuit breaker stats
        cb_stats = llm.get_circuit_breaker_stats()
        
        # Get connection pool stats (if available)
        pool_stats = {}
        if hasattr(app.state, 'http_client') and hasattr(app.state.http_client, '_pool'):
            pool = app.state.http_client._pool
            pool_stats = {
                "active_connections": len(pool._connections),
                "idle_connections": len([c for c in pool._connections if c.is_available()]),
                "max_connections": pool._max_connections,
                "max_keepalive": pool._max_keepalive_connections
            }
        
        # Get guardrails stats
        safety_stats = guardrails.get_safety_stats()
        
        return {
            "embedding_cache": cache_stats,
            "circuit_breaker": cb_stats,
            "connection_pool": pool_stats,
            "guardrails": safety_stats,
            "system": {
                "status": "operational"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@app.post("/admin/cache/clear")
async def clear_caches():
    """Clear all caches (admin endpoint)."""
    try:
        retriever.clear_cache()
        return {"status": "success", "message": "All caches cleared"}
    except Exception as e:
        logger.error(f"Failed to clear caches: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear caches")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """Ingest PDFs from Google Drive folder."""
    try:
        logger.info(f"Starting ingestion from folder: {request.drive_folder_id}")
        
        if request.reindex:
            await retriever.clear_index()
            logger.info("Cleared existing index")
        
        # Ingest documents
        result = await ingester.ingest_from_drive(
            folder_id=request.drive_folder_id,
            reindex=request.reindex
        )
        
        logger.info(f"Ingestion completed: {result['documents_indexed']} docs, {result['chunks']} chunks")
        
        return IngestResponse(
            documents_indexed=result["documents_indexed"],
            chunks=result["chunks"]
        )
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/ingest/stream")
async def ingest_documents_stream(request: IngestRequest):
    """Stream ingestion progress in real-time."""
    import json
    
    async def generate_progress():
        try:
            yield f"data: {json.dumps({'status': 'started', 'message': 'Starting ingestion...', 'progress': 0})}\n\n"
            
            if request.reindex:
                await retriever.clear_index()
                yield f"data: {json.dumps({'status': 'progress', 'message': 'Cleared existing index', 'progress': 5})}\n\n"
            
            # For now, let's just run the regular ingestion and provide periodic updates
            # TODO: Implement proper streaming in the ingester
            result = await ingester.ingest_from_drive(
                folder_id=request.drive_folder_id,
                reindex=request.reindex
            )
            
            yield f"data: {json.dumps({'status': 'completed', 'documents_indexed': result['documents_indexed'], 'chunks': result['chunks'], 'progress': 100})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming ingestion failed: {e}")
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using RAG."""
    try:
        # Apply guardrails
        if not guardrails.is_safe_query(request.question):
            return QueryResponse(
                answer="I cannot answer that question as it may involve unsafe, harmful, or inappropriate content.",
                citations=[],
                used_mode=request.mode
            )
        
        # Retrieve relevant documents
        chunks = await retriever.search(
            query=request.question,
            mode=request.mode,
            top_k=request.top_k
        )
        
        # Check if we have sufficient evidence
        if not chunks or not guardrails.has_sufficient_evidence(chunks):
            return QueryResponse(
                answer="I don't know. I couldn't find sufficient information in the documents to answer your question.",
                citations=[],
                used_mode=request.mode
            )
        
        # Generate answer using LLM
        answer = await llm.generate_answer(
            question=request.question,
            chunks=chunks
        )
        
        # Validate response for safety and groundedness
        is_valid, validated_answer = guardrails.validate_response(answer, request.question)
        if not is_valid:
            logger.warning(f"Response validation failed for query: {request.question[:50]}...")
            answer = validated_answer
        
        # Format citations
        citations = [
            Citation(
                title=chunk["filename"],
                link=chunk["drive_url"],
                snippet=chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"]
            )
            for chunk in chunks
        ]
        
        return QueryResponse(
            answer=answer,
            citations=citations,
            used_mode=request.mode
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
