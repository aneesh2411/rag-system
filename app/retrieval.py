"""Hybrid retrieval with ELSER, BM25, and dense vectors using RRF."""

from typing import List, Dict, Any, Literal, Optional
from elasticsearch import AsyncElasticsearch
from sentence_transformers import SentenceTransformer
import logging
import asyncio
from collections import defaultdict
import hashlib
import time

from . import settings

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """LRU cache for embeddings with TTL (Time To Live)."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
    
    def _hash_text(self, text: str) -> str:
        """Create a hash key for the text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.creation_times:
            return True
        return time.time() - self.creation_times[key] > self.ttl_seconds
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_key(lru_key)
    
    def _remove_key(self, key: str):
        """Remove key from all tracking dictionaries."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.creation_times.pop(key, None)
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        key = self._hash_text(text)
        
        # Check if exists and not expired
        if key in self.cache and not self._is_expired(key):
            self.access_times[key] = time.time()
            logger.debug(f"Cache hit for text hash: {key[:8]}...")
            return self.cache[key]
        
        # Remove if expired
        if key in self.cache:
            self._remove_key(key)
        
        return None
    
    def put(self, text: str, embedding: List[float]):
        """Store embedding in cache."""
        key = self._hash_text(text)
        
        # Evict if at capacity
        while len(self.cache) >= self.max_size:
            self._evict_lru()
        
        # Store new embedding
        current_time = time.time()
        self.cache[key] = embedding
        self.access_times[key] = current_time
        self.creation_times[key] = current_time
        
        logger.debug(f"Cached embedding for text hash: {key[:8]}...")
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.access_times.clear()
        self.creation_times.clear()
        logger.info("Embedding cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hit_rate": getattr(self, '_hits', 0) / max(getattr(self, '_requests', 1), 1)
        }

class HybridRetriever:
    """Handles hybrid retrieval with ELSER, BM25, and dense vectors."""
    
    def __init__(self):
        self.es = AsyncElasticsearch([settings.ELASTICSEARCH_URL])
        self.embedding_model = None
        self.index_name = settings.ELASTICSEARCH_INDEX
        # Initialize embedding cache (1000 embeddings, 1 hour TTL)
        self.embedding_cache = EmbeddingCache(max_size=1000, ttl_seconds=3600)
        self._cache_hits = 0
        self._cache_requests = 0
        
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
        """Generate embedding for query with caching."""
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
        
        # Track cache requests
        self._cache_requests += 1
        
        # Try to get from cache first
        cached_embedding = self.embedding_cache.get(query)
        if cached_embedding is not None:
            self._cache_hits += 1
            logger.debug(f"Embedding cache hit for query: {query[:50]}...")
            return cached_embedding
        
        # Generate new embedding
        start_time = time.time()
        embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        embedding_list = embedding[0].tolist()
        generation_time = time.time() - start_time
        
        # Cache the result
        self.embedding_cache.put(query, embedding_list)
        
        logger.debug(f"Generated embedding in {generation_time:.3f}s for query: {query[:50]}...")
        return embedding_list
    
    async def _search_elser(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using ELSER text expansion."""
        try:
            # Use text expansion query
            search_body = {
                "query": {
                    "text_expansion": {
                        "text_expansion": {
                            "model_id": ".elser_model_2_linux-x86_64",
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
                "knn": {
                    "field": "vector",
                    "query_vector": query_vector,
                    "k": top_k,
                    "num_candidates": top_k * 2
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
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        cache_stats = self.embedding_cache.stats()
        cache_stats.update({
            "requests": self._cache_requests,
            "hits": self._cache_hits,
            "hit_rate": self._cache_hits / max(self._cache_requests, 1)
        })
        return cache_stats
    
    def clear_cache(self):
        """Clear embedding cache."""
        self.embedding_cache.clear()
        self._cache_hits = 0
        self._cache_requests = 0
        logger.info("Embedding cache cleared and stats reset")
    
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
