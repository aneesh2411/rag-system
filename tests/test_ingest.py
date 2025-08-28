"""Unit tests for PDF ingestion functionality."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile
import os

from app.ingest import PDFIngester
from app import settings

class TestPDFIngester:
    """Test cases for PDFIngester."""
    
    @pytest.fixture
    def ingester(self):
        """Create a PDFIngester instance for testing."""
        with patch('app.ingest.ElasticsearchIndexer'):
            return PDFIngester()
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF text content."""
        return """
        This is a sample document for testing the RAG system.
        It contains multiple paragraphs to test text extraction and chunking.
        
        The system should be able to process this content correctly
        and create appropriate chunks for indexing.
        
        This content includes enough text to trigger proper chunking
        with the configured chunk size and overlap parameters.
        """
    
    def test_create_mock_pdf_content(self, ingester):
        """Test mock PDF content creation."""
        filename = "test_document.pdf"
        content = ingester._create_mock_pdf_content(filename)
        
        assert isinstance(content, bytes)
        assert filename in content.decode('utf-8')
        assert len(content) > 100  # Should have substantial content
    
    def test_extract_text_from_mock_pdf(self, ingester, sample_pdf_content):
        """Test text extraction from mock PDF."""
        filename = "sample_document_test.pdf"
        pdf_data = sample_pdf_content.encode('utf-8')
        
        extracted_text = ingester._extract_text_from_pdf(pdf_data, filename)
        
        assert extracted_text.strip() == sample_pdf_content.strip()
        assert "sample document" in extracted_text.lower()
    
    def test_chunk_text_basic(self, ingester):
        """Test basic text chunking functionality."""
        text = "This is a test document. " * 100  # Create longer text
        filename = "test.pdf"
        drive_url = "https://drive.google.com/file/d/test/view"
        
        chunks = ingester._chunk_text(text, filename, drive_url)
        
        assert len(chunks) > 0
        assert all(chunk["filename"] == filename for chunk in chunks)
        assert all(chunk["drive_url"] == drive_url for chunk in chunks)
        assert all("chunk_id" in chunk for chunk in chunks)
        assert all(len(chunk["content"]) > 0 for chunk in chunks)
    
    def test_chunk_text_empty(self, ingester):
        """Test chunking with empty text."""
        chunks = ingester._chunk_text("", "test.pdf", "http://test.com")
        assert chunks == []
    
    def test_chunk_text_short(self, ingester):
        """Test chunking with short text."""
        text = "Short text."
        chunks = ingester._chunk_text(text, "test.pdf", "http://test.com")
        
        assert len(chunks) == 1
        assert chunks[0]["content"] == text
    
    def test_chunk_metadata(self, ingester):
        """Test that chunks contain proper metadata."""
        text = "Test content for chunking with metadata validation."
        filename = "metadata_test.pdf"
        drive_url = "https://drive.google.com/file/d/metadata_test/view"
        
        chunks = ingester._chunk_text(text, filename, drive_url)
        
        assert len(chunks) > 0
        
        for chunk in chunks:
            assert "content" in chunk
            assert "filename" in chunk
            assert "drive_url" in chunk
            assert "chunk_id" in chunk
            assert chunk["filename"] == filename
            assert chunk["drive_url"] == drive_url
            assert filename in chunk["chunk_id"]
    
    @pytest.mark.asyncio
    async def test_list_drive_files_fallback(self, ingester):
        """Test fallback Drive file listing."""
        folder_id = "test_folder_id"
        
        with patch.object(ingester, '_get_drive_service', return_value=None):
            files = await ingester._list_drive_files_fallback(folder_id)
        
        assert isinstance(files, list)
        # Should return mock files for demo
        if files:  # If mock files are returned
            assert all("id" in file for file in files)
            assert all("name" in file for file in files)
            assert all("url" in file for file in files)
    
    @pytest.mark.asyncio
    async def test_process_local_docs_no_folder(self, ingester):
        """Test processing local docs when folder doesn't exist."""
        with patch.object(Path, 'exists', return_value=False):
            with patch.object(Path, 'mkdir') as mock_mkdir:
                chunks = await ingester._process_local_docs()
        
        assert chunks == []
        mock_mkdir.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_local_docs_with_files(self, ingester, sample_pdf_content):
        """Test processing local docs with PDF files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test PDF file
            pdf_path = Path(temp_dir) / "test.pdf"
            pdf_path.write_bytes(sample_pdf_content.encode('utf-8'))
            
            # Mock the settings to use our temp directory
            with patch.object(settings, 'LOCAL_DOCS_PATH', temp_dir):
                with patch.object(ingester, '_extract_text_from_pdf', return_value=sample_pdf_content):
                    chunks = await ingester._process_local_docs()
            
            assert len(chunks) > 0
            assert any("test.pdf" in chunk["filename"] for chunk in chunks)
    
    def test_ocr_trigger_conditions(self, ingester):
        """Test conditions that trigger OCR."""
        # Short text should trigger OCR path (though OCR itself is mocked)
        short_text = "A"
        filename = "scanned.pdf"
        
        with patch.object(ingester, '_ocr_pdf', return_value="OCR extracted text") as mock_ocr:
            # Mock PyPDF2 and PyMuPDF to return short text
            with patch('PyPDF2.PdfReader') as mock_pypdf:
                mock_page = Mock()
                mock_page.extract_text.return_value = short_text
                mock_pypdf.return_value.pages = [mock_page]
                
                with patch('fitz.open') as mock_fitz:
                    mock_doc = Mock()
                    mock_page_fitz = Mock()
                    mock_page_fitz.get_text.return_value = short_text
                    mock_doc.__iter__ = Mock(return_value=iter([mock_page_fitz]))
                    mock_fitz.return_value = mock_doc
                    
                    result = ingester._extract_text_from_pdf(b"fake_pdf_data", filename)
            
            # OCR should have been called due to short text
            mock_ocr.assert_called_once()
            assert result == "OCR extracted text"
    
    @pytest.mark.asyncio
    async def test_ingest_from_drive_no_files(self, ingester):
        """Test ingestion when no files are found."""
        with patch.object(ingester, 'initialize', new_callable=AsyncMock):
            with patch.object(ingester, '_list_drive_files', return_value=[]):
                with patch.object(ingester, '_process_local_docs', return_value=[]):
                    
                    result = await ingester.ingest_from_drive("test_folder_id")
        
        assert result["documents_indexed"] == 0
        assert result["chunks"] == 0
    
    @pytest.mark.asyncio
    async def test_ingest_from_drive_with_files(self, ingester, sample_pdf_content):
        """Test successful ingestion with files."""
        mock_files = [
            {"id": "file1", "name": "doc1.pdf", "url": "http://drive.google.com/file1"},
            {"id": "file2", "name": "doc2.pdf", "url": "http://drive.google.com/file2"}
        ]
        
        with patch.object(ingester, 'initialize', new_callable=AsyncMock):
            with patch.object(ingester, '_list_drive_files', return_value=mock_files):
                with patch.object(ingester, '_download_pdf', return_value=sample_pdf_content.encode()):
                    with patch.object(ingester, '_extract_text_from_pdf', return_value=sample_pdf_content):
                        with patch.object(ingester.indexer, 'index_chunks', new_callable=AsyncMock, return_value=10):
                            
                            result = await ingester.ingest_from_drive("test_folder_id")
        
        assert result["documents_indexed"] == 2
        assert result["chunks"] > 0
    
    def test_chunk_size_configuration(self, ingester):
        """Test that chunking respects configuration settings."""
        # Create text that's longer than chunk size
        text = "Word " * 1000  # Should create multiple chunks
        
        chunks = ingester._chunk_text(text, "test.pdf", "http://test.com")
        
        # Should create multiple chunks for long text
        assert len(chunks) > 1
        
        # Check that chunks are roughly the expected size
        # (allowing for tokenization differences)
        for chunk in chunks:
            token_count = len(ingester.tokenizer.encode(chunk["content"]))
            assert token_count <= settings.CHUNK_SIZE * 1.2  # Allow some variance
    
    def test_chunk_overlap(self, ingester):
        """Test that chunks have proper overlap."""
        # Create text with distinctive markers
        text = " ".join([f"marker_{i}" for i in range(200)])
        
        chunks = ingester._chunk_text(text, "test.pdf", "http://test.com")
        
        if len(chunks) > 1:
            # Check that consecutive chunks share some content
            for i in range(len(chunks) - 1):
                chunk1_words = set(chunks[i]["content"].split())
                chunk2_words = set(chunks[i + 1]["content"].split())
                overlap = chunk1_words.intersection(chunk2_words)
                
                # Should have some overlap between consecutive chunks
                assert len(overlap) > 0
