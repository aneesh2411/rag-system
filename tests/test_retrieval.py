"""Unit tests for hybrid retrieval functionality."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from app.retrieval import HybridRetriever
from app import settings

class TestHybridRetriever:
    """Test cases for HybridRetriever."""
    
    @pytest.fixture
    def retriever(self):
        """Create a HybridRetriever instance for testing."""
        with patch('app.retrieval.AsyncElasticsearch') as mock_es:
            with patch('app.retrieval.SentenceTransformer') as mock_st:
                retriever = HybridRetriever()
                retriever.es = mock_es.return_value
                retriever.embedding_model = mock_st.return_value
                return retriever
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing."""
        return [
            {
                "_score": 1.5,
                "_source": {
                    "content": "This is the first test document content.",
                    "filename": "doc1.pdf",
                    "drive_url": "https://drive.google.com/file/d/doc1/view",
                    "chunk_id": "doc1_0_abc123"
                }
            },
            {
                "_score": 1.2,
                "_source": {
                    "content": "This is the second test document content.",
                    "filename": "doc2.pdf", 
                    "drive_url": "https://drive.google.com/file/d/doc2/view",
                    "chunk_id": "doc2_0_def456"
                }
            }
        ]
    
    def test_generate_query_embedding(self, retriever):
        """Test query embedding generation."""
        query = "test query"
        mock_embedding = [[0.1, 0.2, 0.3, 0.4]]
        
        retriever.embedding_model.encode.return_value = mock_embedding
        
        result = retriever._generate_query_embedding(query)
        
        assert result == [0.1, 0.2, 0.3, 0.4]
        retriever.embedding_model.encode.assert_called_once_with([query], convert_to_tensor=False)
    
    def test_generate_query_embedding_no_model(self, retriever):
        """Test embedding generation without model."""
        retriever.embedding_model = None
        
        with pytest.raises(ValueError, match="Embedding model not initialized"):
            retriever._generate_query_embedding("test")
    
    @pytest.mark.asyncio
    async def test_search_bm25_success(self, retriever, sample_search_results):
        """Test successful BM25 search."""
        query = "test query"
        top_k = 5
        
        # Mock Elasticsearch response
        mock_response = {
            "hits": {
                "hits": sample_search_results
            }
        }
        retriever.es.search = AsyncMock(return_value=mock_response)
        
        results = await retriever._search_bm25(query, top_k)
        
        assert len(results) == 2
        assert results[0]["content"] == "This is the first test document content."
        assert results[0]["score"] == 1.5
        assert results[0]["rank"] == 1
        assert results[1]["rank"] == 2
    
    @pytest.mark.asyncio
    async def test_search_bm25_failure(self, retriever):
        """Test BM25 search failure."""
        retriever.es.search = AsyncMock(side_effect=Exception("ES error"))
        
        results = await retriever._search_bm25("test query", 5)
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_search_dense_success(self, retriever, sample_search_results):
        """Test successful dense vector search."""
        query = "test query"
        top_k = 5
        
        # Mock embedding generation
        retriever._generate_query_embedding = Mock(return_value=[0.1, 0.2, 0.3])
        
        # Mock Elasticsearch response
        mock_response = {
            "hits": {
                "hits": sample_search_results
            }
        }
        retriever.es.search = AsyncMock(return_value=mock_response)
        
        results = await retriever._search_dense(query, top_k)
        
        assert len(results) == 2
        assert results[0]["content"] == "This is the first test document content."
        
        # Verify the search was called with knn query
        retriever.es.search.assert_called_once()
        call_args = retriever.es.search.call_args
        assert "knn" in call_args[1]["body"]["query"]
    
    @pytest.mark.asyncio
    async def test_search_dense_failure(self, retriever):
        """Test dense search failure."""
        retriever._generate_query_embedding = Mock(side_effect=Exception("Embedding error"))
        
        results = await retriever._search_dense("test query", 5)
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_search_elser_success(self, retriever, sample_search_results):
        """Test successful ELSER search."""
        query = "test query"
        top_k = 5
        
        # Mock Elasticsearch response
        mock_response = {
            "hits": {
                "hits": sample_search_results
            }
        }
        retriever.es.search = AsyncMock(return_value=mock_response)
        
        results = await retriever._search_elser(query, top_k)
        
        assert len(results) == 2
        
        # Verify the search was called with text_expansion query
        retriever.es.search.assert_called_once()
        call_args = retriever.es.search.call_args
        assert "text_expansion" in call_args[1]["body"]["query"]
    
    @pytest.mark.asyncio
    async def test_search_elser_failure(self, retriever):
        """Test ELSER search failure (graceful degradation)."""
        retriever.es.search = AsyncMock(side_effect=Exception("ELSER not available"))
        
        results = await retriever._search_elser("test query", 5)
        
        assert results == []  # Should return empty list, not raise exception
    
    def test_reciprocal_rank_fusion_basic(self, retriever):
        """Test basic RRF functionality."""
        # Create test results from different search methods
        elser_results = [
            {"chunk_id": "doc1_chunk1", "content": "content1", "filename": "doc1.pdf", "drive_url": "url1"},
            {"chunk_id": "doc2_chunk1", "content": "content2", "filename": "doc2.pdf", "drive_url": "url2"}
        ]
        
        bm25_results = [
            {"chunk_id": "doc2_chunk1", "content": "content2", "filename": "doc2.pdf", "drive_url": "url2"},
            {"chunk_id": "doc3_chunk1", "content": "content3", "filename": "doc3.pdf", "drive_url": "url3"}
        ]
        
        dense_results = [
            {"chunk_id": "doc1_chunk1", "content": "content1", "filename": "doc1.pdf", "drive_url": "url1"},
            {"chunk_id": "doc4_chunk1", "content": "content4", "filename": "doc4.pdf", "drive_url": "url4"}
        ]
        
        fused_results = retriever._reciprocal_rank_fusion(
            elser_results, bm25_results, dense_results, k=60
        )
        
        # Should have unique documents
        chunk_ids = [result["chunk_id"] for result in fused_results]
        assert len(chunk_ids) == len(set(chunk_ids))  # All unique
        
        # Should have RRF scores
        assert all("rrf_score" in result for result in fused_results)
        
        # Results should be sorted by RRF score (descending)
        rrf_scores = [result["rrf_score"] for result in fused_results]
        assert rrf_scores == sorted(rrf_scores, reverse=True)
    
    def test_reciprocal_rank_fusion_overlap_bonus(self, retriever):
        """Test that documents appearing in multiple results get higher RRF scores."""
        # Document appearing in all three searches should rank highest
        shared_doc = {"chunk_id": "shared_doc", "content": "shared", "filename": "shared.pdf", "drive_url": "url"}
        unique_doc = {"chunk_id": "unique_doc", "content": "unique", "filename": "unique.pdf", "drive_url": "url"}
        
        elser_results = [shared_doc, unique_doc]
        bm25_results = [shared_doc]
        dense_results = [shared_doc]
        
        fused_results = retriever._reciprocal_rank_fusion(
            elser_results, bm25_results, dense_results, k=60
        )
        
        # Shared document should rank first due to appearing in all searches
        assert fused_results[0]["chunk_id"] == "shared_doc"
        assert fused_results[0]["rrf_score"] > fused_results[1]["rrf_score"]
    
    @pytest.mark.asyncio
    async def test_search_elser_mode(self, retriever, sample_search_results):
        """Test search with ELSER-only mode."""
        query = "test query"
        
        # Mock ELSER search
        retriever._search_elser = AsyncMock(return_value=[
            {"chunk_id": "doc1", "content": "content1", "filename": "doc1.pdf", "drive_url": "url1"}
        ])
        
        results = await retriever.search(query, mode="elser", top_k=5)
        
        assert len(results) == 1
        retriever._search_elser.assert_called_once_with(query, 5)
    
    @pytest.mark.asyncio
    async def test_search_hybrid_mode(self, retriever):
        """Test search with hybrid mode."""
        query = "test query"
        
        # Mock all search methods
        retriever._search_elser = AsyncMock(return_value=[
            {"chunk_id": "doc1", "content": "content1", "filename": "doc1.pdf", "drive_url": "url1"}
        ])
        retriever._search_bm25 = AsyncMock(return_value=[
            {"chunk_id": "doc2", "content": "content2", "filename": "doc2.pdf", "drive_url": "url2"}
        ])
        retriever._search_dense = AsyncMock(return_value=[
            {"chunk_id": "doc3", "content": "content3", "filename": "doc3.pdf", "drive_url": "url3"}
        ])
        
        results = await retriever.search(query, mode="hybrid", top_k=5)
        
        assert len(results) <= 5  # Should respect top_k
        
        # All search methods should have been called
        retriever._search_elser.assert_called_once()
        retriever._search_bm25.assert_called_once()
        retriever._search_dense.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_invalid_mode(self, retriever):
        """Test search with invalid mode."""
        with pytest.raises(ValueError, match="Unknown search mode"):
            await retriever.search("test query", mode="invalid", top_k=5)
    
    @pytest.mark.asyncio
    async def test_search_no_results(self, retriever):
        """Test search when no results are found."""
        # Mock all search methods to return empty results
        retriever._search_elser = AsyncMock(return_value=[])
        retriever._search_bm25 = AsyncMock(return_value=[])
        retriever._search_dense = AsyncMock(return_value=[])
        
        results = await retriever.search("nonexistent query", mode="hybrid", top_k=5)
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, retriever):
        """Test health check when Elasticsearch is healthy."""
        retriever.es.cluster.health = AsyncMock(return_value={"status": "green"})
        
        is_healthy = await retriever.health_check()
        
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_health_check_yellow(self, retriever):
        """Test health check when Elasticsearch is yellow (still considered healthy)."""
        retriever.es.cluster.health = AsyncMock(return_value={"status": "yellow"})
        
        is_healthy = await retriever.health_check()
        
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, retriever):
        """Test health check when Elasticsearch is unhealthy."""
        retriever.es.cluster.health = AsyncMock(return_value={"status": "red"})
        
        is_healthy = await retriever.health_check()
        
        assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_health_check_exception(self, retriever):
        """Test health check when Elasticsearch raises exception."""
        retriever.es.cluster.health = AsyncMock(side_effect=Exception("Connection error"))
        
        is_healthy = await retriever.health_check()
        
        assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_clear_index(self, retriever):
        """Test index clearing functionality."""
        retriever.es.indices.exists = AsyncMock(return_value=True)
        retriever.es.delete_by_query = AsyncMock()
        
        await retriever.clear_index()
        
        retriever.es.delete_by_query.assert_called_once()
        call_args = retriever.es.delete_by_query.call_args
        assert call_args[1]["body"]["query"]["match_all"] == {}
    
    @pytest.mark.asyncio
    async def test_clear_index_not_exists(self, retriever):
        """Test clearing index that doesn't exist."""
        retriever.es.indices.exists = AsyncMock(return_value=False)
        retriever.es.delete_by_query = AsyncMock()
        
        await retriever.clear_index()
        
        # Should not call delete_by_query if index doesn't exist
        retriever.es.delete_by_query.assert_not_called()
    
    def test_rrf_k_parameter(self, retriever):
        """Test that RRF uses the correct k parameter."""
        # Test with different k values
        results_a = [{"chunk_id": "doc1", "content": "content1", "filename": "doc1.pdf", "drive_url": "url1"}]
        results_b = []
        results_c = []
        
        # With k=1, should get score = 1/(1+1) = 0.5
        fused_k1 = retriever._reciprocal_rank_fusion(results_a, results_b, results_c, k=1)
        expected_score_k1 = 1.0 / (1 + 1)  # rank is 1-based, so rank=1
        
        # With k=60, should get score = 1/(60+1) ≈ 0.016
        fused_k60 = retriever._reciprocal_rank_fusion(results_a, results_b, results_c, k=60)
        expected_score_k60 = 1.0 / (60 + 1)
        
        assert abs(fused_k1[0]["rrf_score"] - expected_score_k1) < 0.001
        assert abs(fused_k60[0]["rrf_score"] - expected_score_k60) < 0.001
        assert fused_k1[0]["rrf_score"] > fused_k60[0]["rrf_score"]
