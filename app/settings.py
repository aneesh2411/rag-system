"""Configuration settings for the RAG system."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Elasticsearch settings
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ELASTICSEARCH_INDEX = "rag_documents"

# LLM settings - supports both Ollama and OpenAI-compatible APIs
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3.5:3.8b-mini-instruct-q4_0")  # fallback: llama3:8b

# OpenAI-compatible API settings (for RunPod, etc.)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")  # RunPod or other OpenAI-compatible endpoint
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "google/gemma-3-1b-it")  # Default model name

# LLM provider selection: "ollama" or "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama" if not OPENAI_BASE_URL else "openai")

# Google Drive settings
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "1h6GptTW3DPCdhu7q5tY-83CXrpV8TmY_")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")  # Optional for public folders

# Document processing settings
CHUNK_SIZE = 300  # tokens
CHUNK_OVERLAP = 0.2  # 20% overlap
MAX_CHUNKS_PER_DOC = 100

# Retrieval settings
DEFAULT_TOP_K = 5
RRF_K = 60  # Reciprocal Rank Fusion parameter

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# OCR settings
OCR_LANGUAGE = "eng"
MIN_TEXT_THRESHOLD = 100  # Minimum characters before triggering OCR

# Local docs fallback
LOCAL_DOCS_PATH = "/app/docs"
