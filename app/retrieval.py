"""Hybrid retrieval with ELSER, BM25, and dense vectors using RRF."""

from typing import List, Dict, Any, Literal
from elasticsearch import AsyncElasticsearch
from sentence_transformers import SentenceTransformer
import logging
import asyncio
from collections import defaultdict

from . import settings

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Handles hybrid retrieval with ELSER, BM25, and dense vectors."""
    
    def __init__(self):
        self.es = AsyncElasticsearch([settings.ELASTICSEARCH_URL])
        self.embedding_model = None
        self.index_name = settings.ELASTICSEARCH_INDEX
        
    async def initialize(self):
        """Initialize the retriever and embedding model."""
        try:
            # Load embedding model
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info(f"Loaded embedding model: {settings.EMBEDDING_MODEL}")
            
            # Verify Elasticsearch connection
            if not await self.health_check():
                raise ConnectionError("Cannot connect to Elasticsearch")
            
            logger.info("Hybrid retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if Elasticsearch is healthy."""
        try:
            health = await self.es.cluster.health()
            return health["status"] in ["green", "yellow"]
        except Exception:
            return False
    
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
    
    def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query."""
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
        
        embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        return embedding[0].tolist()
    
    async def _search_elser(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using ELSER text expansion."""
        try:
            # Use text expansion query
            search_body = {
                "query": {
                    "text_expansion": {
                        "text_expansion": {
                            "model_id": ".elser_model_2",
                            "model_text": query
                        }
                    }
                },
                "size": top_k,
                "_source": ["content", "filename", "drive_url", "chunk_id"]
            }
            
            response = await self.es.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "content": hit["_source"]["content"],
                    "filename": hit["_source"]["filename"],
                    "drive_url": hit["_source"]["drive_url"],
                    "chunk_id": hit["_source"]["chunk_id"],
                    "score": hit["_score"],
                    "rank": len(results) + 1
                })
            
            return results
            
        except Exception as e:
            logger.warning(f"ELSER search failed: {e}")
            # Return empty results if ELSER is not available
            return []
    
    async def _search_bm25(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using BM25."""
        try:
            search_body = {
                "query": {
                    "match": {
                        "content": {
                            "query": query,
                            "operator": "or"
                        }
                    }
                },
                "size": top_k,
                "_source": ["content", "filename", "drive_url", "chunk_id"]
            }
            
            response = await self.es.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "content": hit["_source"]["content"],
                    "filename": hit["_source"]["filename"],
                    "drive_url": hit["_source"]["drive_url"],
                    "chunk_id": hit["_source"]["chunk_id"],
                    "score": hit["_score"],
                    "rank": len(results) + 1
                })
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    async def _search_dense(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using dense vectors."""
        try:
            # Generate query embedding
            query_vector = self._generate_query_embedding(query)
            
            search_body = {
                "query": {
                    "knn": {
                        "field": "vector",
                        "query_vector": query_vector,
                        "k": top_k,
                        "num_candidates": top_k * 2
                    }
                },
                "size": top_k,
                "_source": ["content", "filename", "drive_url", "chunk_id"]
            }
            
            response = await self.es.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "content": hit["_source"]["content"],
                    "filename": hit["_source"]["filename"],
                    "drive_url": hit["_source"]["drive_url"],
                    "chunk_id": hit["_source"]["chunk_id"],
                    "score": hit["_score"],
                    "rank": len(results) + 1
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    def _reciprocal_rank_fusion(self, 
                               elser_results: List[Dict[str, Any]], 
                               bm25_results: List[Dict[str, Any]], 
                               dense_results: List[Dict[str, Any]], 
                               k: int = 60) -> List[Dict[str, Any]]:
        """Fuse results using Reciprocal Rank Fusion."""
        # Collect all unique documents
        doc_scores = defaultdict(float)
        doc_info = {}
        
        # Process ELSER results
        for i, doc in enumerate(elser_results):
            doc_id = doc["chunk_id"]
            doc_scores[doc_id] += 1.0 / (k + i + 1)
            doc_info[doc_id] = doc
        
        # Process BM25 results
        for i, doc in enumerate(bm25_results):
            doc_id = doc["chunk_id"]
            doc_scores[doc_id] += 1.0 / (k + i + 1)
            doc_info[doc_id] = doc
        
        # Process dense results
        for i, doc in enumerate(dense_results):
            doc_id = doc["chunk_id"]
            doc_scores[doc_id] += 1.0 / (k + i + 1)
            doc_info[doc_id] = doc
        
        # Sort by RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top documents with RRF scores
        fused_results = []
        for doc_id, rrf_score in sorted_docs:
            doc = doc_info[doc_id].copy()
            doc["rrf_score"] = rrf_score
            fused_results.append(doc)
        
        return fused_results
    
    async def search(self, 
                    query: str, 
                    mode: Literal["elser", "hybrid"] = "hybrid", 
                    top_k: int = 5) -> List[Dict[str, Any]]:
        """Search documents using specified mode."""
        try:
            if mode == "elser":
                # ELSER-only search
                results = await self._search_elser(query, top_k)
                logger.info(f"ELSER search returned {len(results)} results")
                return results[:top_k]
            
            elif mode == "hybrid":
                # Hybrid search with RRF
                logger.info(f"Starting hybrid search for: {query}")
                
                # Run all three searches concurrently
                elser_task = self._search_elser(query, top_k * 2)  # Get more candidates
                bm25_task = self._search_bm25(query, top_k * 2)
                dense_task = self._search_dense(query, top_k * 2)
                
                elser_results, bm25_results, dense_results = await asyncio.gather(
                    elser_task, bm25_task, dense_task
                )
                
                logger.info(f"Search results: ELSER={len(elser_results)}, BM25={len(bm25_results)}, Dense={len(dense_results)}")
                
                # Fuse results using RRF
                fused_results = self._reciprocal_rank_fusion(
                    elser_results, bm25_results, dense_results, k=settings.RRF_K
                )
                
                logger.info(f"RRF fusion returned {len(fused_results)} results")
                return fused_results[:top_k]
            
            else:
                raise ValueError(f"Unknown search mode: {mode}")
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def close(self):
        """Close the Elasticsearch connection."""
        await self.es.close()
