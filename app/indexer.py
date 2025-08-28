"""Elasticsearch indexing with ELSER and dense vectors."""

import asyncio
from typing import List, Dict, Any
from elasticsearch import AsyncElasticsearch
from sentence_transformers import SentenceTransformer
import logging

from . import settings

logger = logging.getLogger(__name__)

class ElasticsearchIndexer:
    """Handles Elasticsearch indexing with ELSER and dense vectors."""
    
    def __init__(self):
        self.es = AsyncElasticsearch([settings.ELASTICSEARCH_URL])
        self.embedding_model = None
        self.index_name = settings.ELASTICSEARCH_INDEX
        
    async def initialize(self):
        """Initialize the indexer and embedding model."""
        try:
            # Load embedding model
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info(f"Loaded embedding model: {settings.EMBEDDING_MODEL}")
            
            # Ensure ELSER model is deployed
            await self._deploy_elser_model()
            
            # Create index with proper mapping
            await self._create_index()
            
            logger.info("Elasticsearch indexer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize indexer: {e}")
            raise
    
    async def _deploy_elser_model(self):
        """Deploy ELSER model if not already deployed."""
        try:
            # Check if ELSER model exists
            model_id = ".elser_model_2"
            
            try:
                model_info = await self.es.ml.get_trained_models(model_id=model_id)
                logger.info("ELSER model already exists")
                return
            except Exception:
                logger.info("ELSER model not found, deploying...")
            
            # Deploy ELSER model
            await self.es.ml.put_trained_model(
                model_id=model_id,
                body={
                    "description": "Elastic Learned Sparse EncodeR model",
                    "model_type": "pytorch",
                    "inference_config": {
                        "text_expansion": {
                            "tokenization": {
                                "bert": {
                                    "with_special_tokens": False
                                }
                            }
                        }
                    }
                }
            )
            
            # Start the model deployment
            await self.es.ml.start_trained_model_deployment(
                model_id=model_id,
                wait_for="started"
            )
            
            logger.info("ELSER model deployed and started successfully")
            
        except Exception as e:
            logger.error(f"Failed to deploy ELSER model: {e}")
            # Continue without ELSER for development
            logger.warning("Continuing without ELSER deployment")
    
    async def _create_index(self):
        """Create the index with proper mapping."""
        mapping = {
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "text_expansion": {
                        "type": "rank_features"
                    },
                    "vector": {
                        "type": "dense_vector",
                        "dims": settings.EMBEDDING_DIM,
                        "similarity": "cosine"
                    },
                    "filename": {
                        "type": "keyword"
                    },
                    "drive_url": {
                        "type": "keyword"
                    },
                    "chunk_id": {
                        "type": "keyword"
                    }
                }
            }
        }
        
        # Create index if it doesn't exist
        if not await self.es.indices.exists(index=self.index_name):
            await self.es.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Created index: {self.index_name}")
        else:
            logger.info(f"Index {self.index_name} already exists")
    
    async def clear_index(self):
        """Clear all documents from the index."""
        try:
            if await self.es.indices.exists(index=self.index_name):
                await self.es.delete_by_query(
                    index=self.index_name,
                    body={"query": {"match_all": {}}}
                )
                logger.info(f"Cleared index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            raise
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings for texts."""
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
        
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    async def _generate_elser_expansions(self, texts: List[str]) -> List[Dict[str, float]]:
        """Generate ELSER text expansions."""
        try:
            # Use inference API for ELSER
            expansions = []
            for text in texts:
                response = await self.es.ml.infer_trained_model(
                    model_id=".elser_model_2",
                    body={
                        "docs": [{"text_field": text}]
                    }
                )
                
                if response and "inference_results" in response:
                    expansion = response["inference_results"][0].get("predicted_value", {})
                    expansions.append(expansion)
                else:
                    expansions.append({})
            
            return expansions
            
        except Exception as e:
            logger.warning(f"ELSER expansion failed: {e}")
            # Return empty expansions if ELSER is not available
            return [{}] * len(texts)
    
    async def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Index document chunks with ELSER and dense vectors."""
        try:
            if not chunks:
                return 0
            
            # Extract texts for embedding
            texts = [chunk["content"] for chunk in chunks]
            
            # Generate dense embeddings
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self._generate_embeddings(texts)
            
            # Generate ELSER expansions
            logger.info(f"Generating ELSER expansions for {len(texts)} chunks...")
            elser_expansions = await self._generate_elser_expansions(texts)
            
            # Prepare bulk indexing
            actions = []
            for i, chunk in enumerate(chunks):
                # Action metadata
                actions.append({
                    "index": {
                        "_index": self.index_name,
                        "_id": chunk["chunk_id"]
                    }
                })
                # Document source
                actions.append({
                    "content": chunk["content"],
                    "text_expansion": elser_expansions[i],
                    "vector": embeddings[i],
                    "filename": chunk["filename"],
                    "drive_url": chunk["drive_url"],
                    "chunk_id": chunk["chunk_id"]
                })
            
            # Bulk index
            logger.info(f"Bulk indexing {len(actions)//2} chunks...")
            response = await self.es.bulk(body=actions)
            
            # Check for errors
            if response.get("errors"):
                error_count = sum(1 for item in response["items"] if "error" in item.get("index", {}))
                logger.warning(f"Bulk indexing had {error_count} errors")
            
            # Refresh index
            await self.es.indices.refresh(index=self.index_name)
            
            indexed_count = len([item for item in response["items"] if "error" not in item.get("index", {})])
            logger.info(f"Successfully indexed {indexed_count} chunks")
            
            return indexed_count
            
        except Exception as e:
            logger.error(f"Failed to index chunks: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if Elasticsearch is healthy."""
        try:
            health = await self.es.cluster.health()
            return health["status"] in ["green", "yellow"]
        except Exception:
            return False
    
    async def close(self):
        """Close the Elasticsearch connection."""
        await self.es.close()
