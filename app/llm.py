"""Ollama LLM integration for answer generation."""

import httpx
import json
import logging
from typing import List, Dict, Any

from . import settings

logger = logging.getLogger(__name__)

class OllamaLLM:
    """Handles interaction with Ollama LLM for answer generation."""
    
    def __init__(self):
        self.base_url = settings.OLLAMA_URL
        self.model = settings.OLLAMA_MODEL
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def initialize(self):
        """Initialize the LLM and ensure model is available."""
        try:
            # Check if Ollama is running
            if not await self.health_check():
                raise ConnectionError("Cannot connect to Ollama")
            
            # Check if model is available, pull if needed
            await self._ensure_model_available()
            
            logger.info(f"Ollama LLM initialized with model: {self.model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if Ollama is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    async def _ensure_model_available(self):
        """Ensure the specified model is available, pull if needed."""
        try:
            # Check if model exists
            response = await self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                
                if self.model in model_names:
                    logger.info(f"Model {self.model} is available")
                    return
            
            # Pull the model
            logger.info(f"Pulling model {self.model}...")
            pull_response = await self.client.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model}
            )
            
            if pull_response.status_code == 200:
                logger.info(f"Successfully pulled model {self.model}")
            else:
                logger.warning(f"Failed to pull model {self.model}, trying fallback")
                # Try fallback model
                self.model = "llama3:8b"
                await self._pull_fallback_model()
                
        except Exception as e:
            logger.error(f"Failed to ensure model availability: {e}")
            # Use a smaller fallback
            self.model = "llama3:8b"
            logger.info(f"Switched to fallback model: {self.model}")
    
    async def _pull_fallback_model(self):
        """Pull fallback model."""
        try:
            pull_response = await self.client.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model}
            )
            
            if pull_response.status_code == 200:
                logger.info(f"Successfully pulled fallback model {self.model}")
            else:
                logger.error(f"Failed to pull fallback model {self.model}")
                
        except Exception as e:
            logger.error(f"Failed to pull fallback model: {e}")
    
    def _create_prompt(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """Create prompt for the LLM with context and question."""
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"Document {i} ({chunk['filename']}):\n{chunk['content']}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are a helpful assistant that answers questions based on the provided documents. 

IMPORTANT RULES:
1. Only answer based on the information provided in the documents below
2. If the documents don't contain enough information to answer the question, say "I don't know"
3. Always cite which documents you used in your answer
4. Be concise and accurate
5. Do not make up information not present in the documents

DOCUMENTS:
{context}

QUESTION: {question}

ANSWER:"""
        
        return prompt
    
    async def generate_answer(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using Ollama LLM."""
        try:
            if not chunks:
                return "I don't know. No relevant documents were found to answer your question."
            
            # Create prompt
            prompt = self._create_prompt(question, chunks)
            
            # Generate response
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for factual responses
                        "top_p": 0.9,
                        "max_tokens": 500,
                        "stop": ["QUESTION:", "DOCUMENTS:"]
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                
                if not answer:
                    return "I don't know. I couldn't generate a proper response based on the provided documents."
                
                # Basic post-processing
                answer = self._post_process_answer(answer, chunks)
                
                logger.info(f"Generated answer of length {len(answer)}")
                return answer
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return "I don't know. There was an error generating the response."
                
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return "I don't know. There was an error processing your question."
    
    def _post_process_answer(self, answer: str, chunks: List[Dict[str, Any]]) -> str:
        """Post-process the generated answer."""
        # Remove any potential hallucinations or unwanted prefixes
        if answer.lower().startswith("answer:"):
            answer = answer[7:].strip()
        
        # Ensure the answer is grounded
        if self._is_likely_hallucination(answer, chunks):
            return "I don't know. I couldn't find sufficient information in the documents to provide a confident answer."
        
        return answer
    
    def _is_likely_hallucination(self, answer: str, chunks: List[Dict[str, Any]]) -> bool:
        """Simple heuristic to detect potential hallucinations."""
        # Check if answer is too short or generic
        if len(answer.strip()) < 10:
            return True
        
        # Check if answer contains common hallucination phrases
        hallucination_phrases = [
            "i don't have access",
            "i cannot access",
            "based on my training",
            "as an ai",
            "i'm not able to browse",
            "i don't have the ability"
        ]
        
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in hallucination_phrases):
            return True
        
        # Check if answer has some overlap with the provided chunks
        chunk_text = " ".join([chunk["content"] for chunk in chunks]).lower()
        answer_words = set(answer_lower.split())
        chunk_words = set(chunk_text.split())
        
        # If there's very little overlap, it might be a hallucination
        overlap = len(answer_words.intersection(chunk_words))
        if overlap < max(2, len(answer_words) * 0.1):  # At least 10% overlap
            return True
        
        return False
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
