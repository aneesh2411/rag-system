"""PDF ingestion from Google Drive with OCR support."""

import os
import io
import hashlib
from typing import List, Dict, Any
from pathlib import Path
import logging
import asyncio
import tiktoken

import PyPDF2
import fitz  # pymupdf
import ocrmypdf
from PIL import Image
import pytesseract

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import httpx

from .indexer import ElasticsearchIndexer
from . import settings

logger = logging.getLogger(__name__)

class PDFIngester:
    """Handles PDF ingestion from Google Drive with OCR support."""
    
    def __init__(self):
        self.indexer = ElasticsearchIndexer()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT tokenizer
        
    async def initialize(self):
        """Initialize the ingester."""
        await self.indexer.initialize()
    
    def _get_drive_service(self):
        """Get Google Drive API service."""
        try:
            # For public folders, we can use API key authentication
            if settings.GOOGLE_API_KEY:
                service = build('drive', 'v3', developerKey=settings.GOOGLE_API_KEY)
            else:
                # For development, we'll use direct HTTP requests
                service = None
            return service
        except Exception as e:
            logger.warning(f"Could not initialize Drive service: {e}")
            return None
    
    async def _list_drive_files(self, folder_id: str) -> List[Dict[str, str]]:
        """List PDF files in Google Drive public folder."""
        try:
            logger.info(f"Accessing public Google Drive folder: {folder_id}")
            
            # For public folders, we can try using a public API key or create one
            # Let's try the direct API approach first
            api_url = "https://www.googleapis.com/drive/v3/files"
            params = {
                'q': f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false",
                'fields': 'files(id,name,webViewLink)',
                'key': 'AIzaSyBjeUZKh1iL9LI6plwjrUQRTX79q9KZBnQ'
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    response = await client.get(api_url, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        files = []
                        for file in data.get('files', []):
                            files.append({
                                'id': file['id'],
                                'name': file['name'],
                                'url': file.get('webViewLink', f"https://drive.google.com/file/d/{file['id']}/view")
                            })
                        
                        if files:
                            logger.info(f"Found {len(files)} PDF files via API")
                            return files
                    else:
                        logger.warning(f"API returned status {response.status_code}: {response.text}")
                        
                except Exception as e:
                    logger.warning(f"API call failed: {e}")
            
            # If API fails, try alternative approaches
            logger.info("Trying alternative methods to access public folder")
            
            # Method 2: Try to access the RSS feed (if available)
            rss_url = f"https://drive.google.com/drive/folders/{folder_id}?usp=sharing"
            
            try:
                response = await client.get(rss_url)
                if response.status_code == 200:
                    import re
                    # Look for any file IDs and PDF names in the response
                    file_ids = re.findall(r'([a-zA-Z0-9_-]{25,44})', response.text)
                    pdf_names = re.findall(r'([^/"]*\.pdf)', response.text, re.IGNORECASE)
                    
                    if file_ids and pdf_names:
                        # Try to match IDs with names (this is approximate)
                        files = []
                        min_len = min(len(file_ids), len(pdf_names))
                        for i in range(min_len):
                            if len(file_ids[i]) >= 25:  # Valid Drive file ID length
                                files.append({
                                    'id': file_ids[i],
                                    'name': pdf_names[i],
                                    'url': f"https://drive.google.com/file/d/{file_ids[i]}/view"
                                })
                        
                        if files:
                            logger.info(f"Found {len(files)} PDF files via alternative method")
                            return files
                            
            except Exception as e:
                logger.warning(f"Alternative method failed: {e}")
            
            logger.error("Could not access public Google Drive folder with any method")
            return []
                
        except Exception as e:
            logger.error(f"Failed to list Drive files: {e}")
            return []
    
    async def _list_drive_files_api_public(self, folder_id: str) -> List[Dict[str, str]]:
        """Try to access public folder via Drive API without authentication."""
        try:
            # Some public folders can be accessed without API key
            api_url = f"https://www.googleapis.com/drive/v3/files"
            params = {
                'q': f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false",
                'fields': 'files(id,name,webViewLink)'
            }
            
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(api_url, params=params, timeout=10.0)
                    if response.status_code == 200:
                        data = response.json()
                        files = []
                        for file in data.get('files', []):
                            files.append({
                                'id': file['id'],
                                'name': file['name'],
                                'url': file.get('webViewLink', f"https://drive.google.com/file/d/{file['id']}/view")
                            })
                        
                        if files:
                            logger.info(f"Found {len(files)} files via API")
                            return files
                except Exception as api_error:
                    logger.warning(f"Public API access failed: {api_error}")
            
            logger.error("Could not access public Google Drive folder - no files found")
            return []
            
        except Exception as e:
            logger.error(f"API public access failed: {e}")
            return []
    
    async def _download_pdf(self, file_id: str, file_name: str) -> bytes:
        """Download PDF from Google Drive public file."""
        try:
            # For public files, use direct download URL
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            logger.info(f"Downloading {file_name} from Google Drive...")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(download_url, follow_redirects=True)
                
                if response.status_code == 200:
                    content = response.content
                    if len(content) > 1000:  # Reasonable PDF size check
                        logger.info(f"Successfully downloaded {file_name} ({len(content)} bytes)")
                        return content
                    else:
                        logger.warning(f"Downloaded file {file_name} seems too small ({len(content)} bytes)")
                
                # If direct download doesn't work, try alternative URL
                alt_url = f"https://drive.google.com/file/d/{file_id}/view"
                logger.info(f"Trying alternative download for {file_name}")
                
                response = await client.get(alt_url)
                if response.status_code == 200:
                    # Look for direct download link in the response
                    import re
                    download_match = re.search(r'"downloadUrl":"([^"]+)"', response.text)
                    if download_match:
                        real_download_url = download_match.group(1).replace('\\u003d', '=').replace('\\u0026', '&')
                        download_response = await client.get(real_download_url)
                        if download_response.status_code == 200:
                            logger.info(f"Successfully downloaded {file_name} via alternative method")
                            return download_response.content
                
                logger.error(f"Could not download {file_name} - file may not be publicly accessible")
                raise Exception(f"Download failed for {file_name}")
                
        except Exception as e:
            logger.error(f"Failed to download {file_name}: {e}")
            raise
    

    
    def _extract_text_from_pdf(self, pdf_data: bytes, filename: str) -> str:
        """Extract text from PDF, using OCR if necessary."""
        try:
            # Try PyPDF2 first
            text = ""
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                logger.info(f"PyPDF2 extracted {len(text)} characters from {filename}")
            except Exception as e:
                logger.warning(f"PyPDF2 failed for {filename}: {e}")
            
            # If text is too short, try pymupdf
            if len(text.strip()) < settings.MIN_TEXT_THRESHOLD:
                try:
                    doc = fitz.open(stream=pdf_data, filetype="pdf")
                    text = ""
                    for page in doc:
                        text += page.get_text() + "\n"
                    doc.close()
                    logger.info(f"PyMuPDF extracted {len(text)} characters from {filename}")
                except Exception as e:
                    logger.warning(f"PyMuPDF failed for {filename}: {e}")
            
            # If still too short, use OCR
            if len(text.strip()) < settings.MIN_TEXT_THRESHOLD:
                logger.info(f"Using OCR for {filename} (current text length: {len(text.strip())})")
                text = self._ocr_pdf(pdf_data, filename)
            
            final_text = text.strip()
            logger.info(f"Final extracted text length for {filename}: {len(final_text)} characters")
            return final_text
            
        except Exception as e:
            logger.error(f"Text extraction failed for {filename}: {e}")
            return ""
    
    def _ocr_pdf(self, pdf_data: bytes, filename: str) -> str:
        """Perform OCR on PDF."""
        try:
            # Save PDF to temporary file
            temp_path = f"/tmp/{filename}"
            with open(temp_path, 'wb') as f:
                f.write(pdf_data)
            
            # Use ocrmypdf to add OCR layer
            ocr_path = f"/tmp/ocr_{filename}"
            ocrmypdf.ocr(
                temp_path,
                ocr_path,
                language=settings.OCR_LANGUAGE,
                force_ocr=True,
                skip_text=True
            )
            
            # Extract text from OCR'd PDF
            with open(ocr_path, 'rb') as f:
                ocr_data = f.read()
            
            doc = fitz.open(stream=ocr_data, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            
            # Clean up temporary files
            os.remove(temp_path)
            os.remove(ocr_path)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR failed for {filename}: {e}")
            return ""
    
    def _chunk_text(self, text: str, filename: str, drive_url: str) -> List[Dict[str, Any]]:
        """Chunk text into overlapping segments."""
        if not text.strip():
            return []
        
        logger.info(f"Chunking {len(text)} characters from {filename}")
        
        # Limit text size to prevent hanging (max ~100k chars)
        if len(text) > 100000:
            logger.warning(f"Text too long ({len(text)} chars), truncating to 100k chars for {filename}")
            text = text[:100000]
        
        try:
            # Tokenize text
            logger.info(f"Tokenizing text for {filename}...")
            tokens = self.tokenizer.encode(text)
            logger.info(f"Tokenized {len(tokens)} tokens for {filename}")
            
            chunks = []
            chunk_size = settings.CHUNK_SIZE
            overlap_size = int(chunk_size * settings.CHUNK_OVERLAP)
            
            # Ensure overlap is less than chunk size to prevent infinite loops
            if overlap_size >= chunk_size:
                overlap_size = chunk_size // 2
                logger.warning(f"Overlap too large, reduced to {overlap_size} for {filename}")
            
            start = 0
            chunk_idx = 0
            max_chunks = settings.MAX_CHUNKS_PER_DOC  # Safety limit
            
            while start < len(tokens) and chunk_idx < max_chunks:
                end = min(start + chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                
                if chunk_text.strip():
                    chunk_id = f"{filename}_{chunk_idx}_{hashlib.md5(chunk_text.encode()).hexdigest()[:8]}"
                    
                    chunks.append({
                        "content": chunk_text.strip(),
                        "filename": filename,
                        "drive_url": drive_url,
                        "chunk_id": chunk_id
                    })
                    
                    chunk_idx += 1
                
                # Move start position with overlap - ensure we always advance
                next_start = end - overlap_size
                if next_start <= start:  # Safety check to prevent infinite loops
                    next_start = start + max(1, chunk_size - overlap_size)
                
                start = next_start
                
                # Progress logging for very long texts
                if chunk_idx % 50 == 0 and chunk_idx > 0:
                    logger.info(f"Processed {chunk_idx} chunks for {filename}...")
            
            if chunk_idx >= max_chunks:
                logger.warning(f"Reached max chunks limit ({max_chunks}) for {filename}")
            
            logger.info(f"Created {len(chunks)} chunks for {filename}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk text for {filename}: {e}")
            return []
    
    async def _process_local_docs(self) -> List[Dict[str, Any]]:
        """Process PDFs from local docs folder as fallback."""
        chunks = []
        docs_path = Path(settings.LOCAL_DOCS_PATH)
        
        if not docs_path.exists():
            logger.info("Local docs path does not exist, creating it")
            docs_path.mkdir(parents=True, exist_ok=True)
            return chunks
        
        pdf_files = list(docs_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDFs in local docs folder")
        
        for pdf_file in pdf_files:
            try:
                with open(pdf_file, 'rb') as f:
                    pdf_data = f.read()
                
                text = self._extract_text_from_pdf(pdf_data, pdf_file.name)
                if text:
                    file_chunks = self._chunk_text(
                        text, 
                        pdf_file.name,
                        f"file://{pdf_file.absolute()}"
                    )
                    chunks.extend(file_chunks)
                    logger.info(f"Processed {pdf_file.name}: {len(file_chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
        
        return chunks
    
    async def ingest_from_drive(self, folder_id: str, reindex: bool = True) -> Dict[str, int]:
        """Ingest PDFs from Google Drive folder in batches."""
        try:
            await self.initialize()
            
            total_documents_processed = 0
            total_chunks = 0
            batch_size = 2  # Process 2 files at a time to avoid memory issues
            
            # Try to get files from Drive
            drive_files = await self._list_drive_files(folder_id)
            
            if drive_files:
                logger.info(f"Found {len(drive_files)} PDFs in Drive folder")
                logger.info(f"Processing in batches of {batch_size} files to avoid memory issues")
                
                # Process files in batches
                for i in range(0, len(drive_files), batch_size):
                    batch = drive_files[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    logger.info(f"Processing batch {batch_num}/{(len(drive_files) + batch_size - 1) // batch_size} ({len(batch)} files)")
                    
                    batch_chunks = []
                    batch_docs_processed = 0
                    
                    for file_info in batch:
                        try:
                            logger.info(f"Processing {file_info['name']}...")
                            
                            # Download and process PDF
                            pdf_data = await self._download_pdf(file_info['id'], file_info['name'])
                            text = self._extract_text_from_pdf(pdf_data, file_info['name'])
                            
                            if text:
                                logger.info(f"Starting chunking for {file_info['name']}...")
                                chunks = self._chunk_text(text, file_info['name'], file_info['url'])
                                logger.info(f"Chunking completed for {file_info['name']}: {len(chunks)} chunks")
                                batch_chunks.extend(chunks)
                                batch_docs_processed += 1
                                logger.info(f"✅ Processed {file_info['name']}: {len(chunks)} chunks")
                            else:
                                logger.warning(f"⚠️  No text extracted from {file_info['name']}")
                            
                        except Exception as e:
                            logger.error(f"❌ Failed to process {file_info['name']}: {e}")
                    
                    # Index this batch
                    if batch_chunks:
                        logger.info(f"Starting indexing for batch {batch_num} with {len(batch_chunks)} chunks...")
                        try:
                            indexed_count = await self.indexer.index_chunks(batch_chunks)
                            logger.info(f"✅ Indexed batch {batch_num}: {indexed_count} chunks from {batch_docs_processed} documents")
                        except Exception as e:
                            logger.error(f"❌ Failed to index batch {batch_num}: {e}")
                            raise
                        
                        total_documents_processed += batch_docs_processed
                        total_chunks += len(batch_chunks)
                    else:
                        logger.warning(f"⚠️  No chunks to index in batch {batch_num}")
                    
                    # Clean up memory and small delay between batches
                    if i + batch_size < len(drive_files):
                        logger.info("Cleaning up memory and waiting 3 seconds before next batch...")
                        import gc
                        gc.collect()  # Force garbage collection
                        await asyncio.sleep(3)
                        
            else:
                # Fallback to local docs
                logger.info("No Drive files found, checking local docs folder")
                local_chunks = await self._process_local_docs()
                if local_chunks:
                    indexed_count = await self.indexer.index_chunks(local_chunks)
                    total_chunks = len(local_chunks)
                    total_documents_processed = len(list(Path(settings.LOCAL_DOCS_PATH).glob("*.pdf"))) if Path(settings.LOCAL_DOCS_PATH).exists() else 0
            
            logger.info(f"🎉 Ingestion completed! Total: {total_documents_processed} documents, {total_chunks} chunks")
            
            return {
                "documents_indexed": total_documents_processed,
                "chunks": total_chunks
            }
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise
