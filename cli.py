#!/usr/bin/env python3
"""Command-line interface for RAG system testing."""

import asyncio
import httpx
import json
import sys
import argparse
from typing import Dict, Any

API_URL = "http://localhost:8000"

async def health_check() -> bool:
    """Check if the API is healthy."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_URL}/healthz", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ System Status: {data['status']}")
                print(f"   Elasticsearch: {data['elasticsearch']}")
                print(f"   Ollama: {data['ollama']}")
                return data['status'] == 'ok'
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

async def ingest_documents(folder_id: str, reindex: bool = True) -> Dict[str, Any]:
    """Ingest documents from Google Drive."""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{API_URL}/ingest",
                json={
                    "drive_folder_id": folder_id,
                    "reindex": reindex
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Ingestion successful!")
                print(f"   Documents indexed: {data['documents_indexed']}")
                print(f"   Total chunks: {data['chunks']}")
                return data
            else:
                print(f"❌ Ingestion failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return {}
                
    except Exception as e:
        print(f"❌ Ingestion error: {e}")
        return {}

async def query_documents(question: str, mode: str = "hybrid", top_k: int = 5) -> Dict[str, Any]:
    """Query documents using RAG."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{API_URL}/query",
                json={
                    "question": question,
                    "mode": mode,
                    "top_k": top_k
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"🤖 Answer ({data['used_mode']} mode):")
                print(f"   {data['answer']}")
                print()
                
                if data['citations']:
                    print("📚 Sources:")
                    for i, citation in enumerate(data['citations'], 1):
                        print(f"   {i}. {citation['title']}")
                        print(f"      {citation['link']}")
                        print(f"      Snippet: {citation['snippet'][:100]}...")
                        print()
                else:
                    print("📚 No sources found.")
                
                return data
            else:
                print(f"❌ Query failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return {}
                
    except Exception as e:
        print(f"❌ Query error: {e}")
        return {}

async def demo_queries():
    """Run demo queries to test the system."""
    demo_questions = [
        "What are the main topics covered in the documents?",
        "Can you summarize the key findings?",
        "What recommendations are mentioned?",
        "Are there any specific dates mentioned?",
        "What is this document about?"  # This should trigger "I don't know" if no docs
    ]
    
    print("🎯 Running demo queries...")
    print()
    
    for i, question in enumerate(demo_questions, 1):
        print(f"📝 Question {i}: {question}")
        await query_documents(question, mode="hybrid", top_k=3)
        print("-" * 80)
        print()

async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="RAG System CLI")
    parser.add_argument("command", choices=["health", "ingest", "query", "demo"], 
                       help="Command to execute")
    parser.add_argument("--question", "-q", help="Question to ask (for query command)")
    parser.add_argument("--mode", "-m", choices=["elser", "hybrid"], default="hybrid",
                       help="Retrieval mode")
    parser.add_argument("--top-k", "-k", type=int, default=5, 
                       help="Number of results to retrieve")
    parser.add_argument("--folder-id", "-f", 
                       default="1h6GptTW3DPCdhu7q5tY-83CXrpV8TmY_",
                       help="Google Drive folder ID")
    parser.add_argument("--reindex", action="store_true", 
                       help="Clear index before ingesting")
    
    args = parser.parse_args()
    
    print("🤖 RAG System CLI")
    print("=" * 50)
    
    if args.command == "health":
        await health_check()
    
    elif args.command == "ingest":
        print(f"📥 Starting ingestion from folder: {args.folder_id}")
        await ingest_documents(args.folder_id, args.reindex)
    
    elif args.command == "query":
        if not args.question:
            print("❌ Please provide a question with --question")
            sys.exit(1)
        
        print(f"🔍 Querying: {args.question}")
        await query_documents(args.question, args.mode, args.top_k)
    
    elif args.command == "demo":
        # First check health
        if not await health_check():
            print("❌ System is not healthy. Please start the services first.")
            sys.exit(1)
        
        print()
        print("📥 Starting ingestion...")
        await ingest_documents(args.folder_id, args.reindex)
        
        print()
        await demo_queries()

if __name__ == "__main__":
    asyncio.run(main())
