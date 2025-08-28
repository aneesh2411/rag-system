# Local Documents Folder

This folder serves as a fallback for PDF documents when Google Drive access is not available.

## Usage

1. Place PDF files directly in this folder
2. The system will automatically process them during ingestion
3. Files will be chunked and indexed with local file:// URLs

## Supported Formats

- PDF files (.pdf extension)
- Both text-based and scanned PDFs (OCR will be applied automatically)

## File Processing

- Text extraction using PyPDF2 and PyMuPDF
- OCR with Tesseract for scanned documents
- Chunking with ~300 tokens and 20% overlap
- Dense embeddings with all-MiniLM-L6-v2
- ELSER text expansion for sparse retrieval
