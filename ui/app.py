"""Streamlit UI for the RAG system."""

import streamlit as st
import httpx
import json
import os
from typing import Dict, List, Any

# Configure page
st.set_page_config(
    page_title="RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "last_response" not in st.session_state:
        st.session_state.last_response = None

def check_api_health() -> bool:
    """Check if the API is healthy."""
    try:
        response = httpx.get(f"{API_URL}/healthz", timeout=5.0)
        return response.status_code == 200 and response.json().get("status") == "ok"
    except Exception:
        return False

def call_ingest_api(drive_folder_id: str, reindex: bool = True) -> Dict[str, Any]:
    """Call the ingest API."""
    try:
        with httpx.Client(timeout=300.0) as client:  # 5 minute timeout for ingestion
            response = client.post(
                f"{API_URL}/ingest",
                json={
                    "drive_folder_id": drive_folder_id,
                    "reindex": reindex
                }
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
                
    except Exception as e:
        return {"success": False, "error": str(e)}

def call_query_api(question: str, mode: str, top_k: int) -> Dict[str, Any]:
    """Call the query API."""
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{API_URL}/query",
                json={
                    "question": question,
                    "mode": mode,
                    "top_k": top_k
                }
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
                
    except Exception as e:
        return {"success": False, "error": str(e)}

def render_sidebar():
    """Render the sidebar with controls."""
    st.sidebar.title("⚙️ Settings")
    
    # API Health Status
    st.sidebar.subheader("🏥 System Status")
    if check_api_health():
        st.sidebar.success("✅ API is healthy")
    else:
        st.sidebar.error("❌ API is not responding")
    
    st.sidebar.divider()
    
    # Ingestion Controls
    st.sidebar.subheader("📥 Document Ingestion")
    
    drive_folder_id = st.sidebar.text_input(
        "Google Drive Folder ID",
        value="1h6GptTW3DPCdhu7q5tY-83CXrpV8TmY_",
        help="ID of the Google Drive folder containing PDFs"
    )
    
    reindex = st.sidebar.checkbox("Reindex all documents", value=True)
    
    if st.sidebar.button("🔄 Start Ingestion", type="primary"):
        with st.spinner("Ingesting documents..."):
            result = call_ingest_api(drive_folder_id, reindex)
            
            if result["success"]:
                data = result["data"]
                st.sidebar.success(
                    f"✅ Ingested {data['documents_indexed']} documents "
                    f"({data['chunks']} chunks)"
                )
            else:
                st.sidebar.error(f"❌ Ingestion failed: {result['error']}")
    
    st.sidebar.divider()
    
    # Query Settings
    st.sidebar.subheader("🔍 Search Settings")
    
    mode = st.sidebar.selectbox(
        "Retrieval Mode",
        options=["hybrid", "elser"],
        index=0,
        help="Choose between ELSER-only or Hybrid (ELSER + BM25 + Dense) retrieval"
    )
    
    top_k = st.sidebar.slider(
        "Number of Results",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of document chunks to retrieve"
    )
    
    return mode, top_k

def render_citations(citations: List[Dict[str, str]]):
    """Render citations in an organized way."""
    if not citations:
        return
    
    st.subheader("📚 Sources")
    
    for i, citation in enumerate(citations, 1):
        with st.expander(f"📄 {citation['title']}", expanded=False):
            st.write(f"**Snippet:** {citation['snippet']}")
            
            if citation['link'].startswith('http'):
                st.markdown(f"🔗 [Open in Google Drive]({citation['link']})")
            else:
                st.write(f"📁 Local file: {citation['link']}")

def render_chat_messages():
    """Render the chat messages only."""
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "citations" in message:
                st.write(message["content"])
                if message["citations"]:
                    render_citations(message["citations"])
            else:
                st.write(message["content"])

def handle_chat_input(prompt, mode, top_k):
    """Handle chat input and generate response."""
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            result = call_query_api(prompt, mode, top_k)
            
            if result["success"]:
                data = result["data"]
                answer = data["answer"]
                citations = data.get("citations", [])
                used_mode = data.get("used_mode", mode)
                
                # Display answer
                st.write(answer)
                
                # Display mode used
                st.caption(f"*Used {used_mode} retrieval mode*")
                
                # Display citations
                if citations:
                    render_citations(citations)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "mode": used_mode
                })
                
            else:
                error_msg = f"❌ Query failed: {result['error']}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

def render_demo_page():
    """Render a demo page with example queries."""
    st.title("🎯 Demo & Examples")
    
    st.markdown("""
    ## How to Use This System
    
    1. **Ingest Documents**: Use the sidebar to ingest PDFs from Google Drive
    2. **Ask Questions**: Type questions in the chat interface
    3. **Review Sources**: Check the citations to see which documents were used
    4. **Try Different Modes**: Compare ELSER-only vs Hybrid retrieval
    
    ## Example Questions to Try
    """)
    
    example_questions = [
        "What are the main topics covered in the documents?",
        "Can you summarize the key findings?",
        "What recommendations are mentioned?",
        "Are there any specific dates or numbers mentioned?",
        "What are the conclusions drawn in the documents?"
    ]
    
    for i, question in enumerate(example_questions, 1):
        st.markdown(f"{i}. *{question}*")
    
    st.markdown("""
    ## Retrieval Modes
    
    - **ELSER**: Uses Elasticsearch's learned sparse encoder for semantic search
    - **Hybrid**: Combines ELSER, BM25 (keyword), and dense vector search using Reciprocal Rank Fusion
    
    ## System Architecture
    
    - **Frontend**: Streamlit UI
    - **Backend**: FastAPI with async processing
    - **Search**: Elasticsearch with ELSER model
    - **LLM**: Ollama with phi3.5-mini or llama3:8b
    - **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
    """)

def main():
    """Main application."""
    init_session_state()
    
    # Render sidebar (this returns the current settings)
    mode, top_k = render_sidebar()
    
    # Main chat interface
    st.title("🤖 RAG System Chat")
    st.markdown("Ask questions about the ingested documents!")
    
    # Render chat messages
    render_chat_messages()
    
    # Chat input must be outside of tabs/containers
    if prompt := st.chat_input("Ask a question about the documents..."):
        handle_chat_input(prompt, mode, top_k)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "RAG System with Elasticsearch + ELSER + Ollama | "
        "Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
