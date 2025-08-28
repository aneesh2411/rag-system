"""FastAPI main application with RAG endpoints."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import logging
import asyncio

from .ingest import PDFIngester
from .retrieval import HybridRetriever
from .llm import OllamaLLM
from .guardrails import GuardrailsFilter
from . import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG System API",
    description="Elasticsearch + ELSER + Ollama RAG System",
    version="1.0.0"
)

# Initialize components
ingester = PDFIngester()
retriever = HybridRetriever()
llm = OllamaLLM()
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
