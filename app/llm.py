"""RunPod Gemma LLM integration for answer generation."""

import httpx
import json
import logging
import time
from typing import List, Dict, Any
from enum import Enum

from . import settings

logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    """Circuit breaker for API resilience."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, request_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.request_timeout = request_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
    
    def can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker moving to HALF_OPEN state")
                return True
            return False
        
        # HALF_OPEN state - allow one test request
        return True
    
    def record_success(self):
        """Record successful request."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info("Circuit breaker moving to CLOSED state - service recovered")
        
        self.failure_count = 0
    
    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitBreakerState.OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker OPEN - {self.failure_count} failures exceeded threshold")
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker moving back to OPEN state - test request failed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout
        }

class RunPodLLM:
    """Handles interaction with RunPod Gemma LLM for answer generation."""
    
    def __init__(self):
        self.base_url = settings.OPENAI_BASE_URL
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
        self.client = httpx.AsyncClient(timeout=30.0)
        # Initialize circuit breaker (5 failures, 60s recovery, 30s timeout)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60, request_timeout=30)
        
    async def initialize(self):
        """Initialize the LLM and ensure API is accessible."""
        try:
            if not self.base_url or not self.api_key:
                raise ValueError("RunPod base URL and API key are required")
            
            # Check if RunPod API is accessible
            if not await self.health_check():
                raise ConnectionError("Cannot connect to RunPod API")
            
            logger.info(f"RunPod LLM initialized with model: {self.model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize RunPod LLM: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if RunPod API is healthy."""
        try:
            response = await self.client.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            return response.status_code == 200
        except Exception:
            return False
    

    
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
        """Generate answer using RunPod Gemma LLM with circuit breaker."""
        # Check circuit breaker before making request
        if not self.circuit_breaker.can_execute():
            logger.warning("Circuit breaker OPEN - rejecting request")
            return "I apologize, but the AI service is currently unavailable. Please try again later."
        
        try:
            if not chunks:
                return "I don't know. No relevant documents were found to answer your question."
            
            # Create prompt
            prompt = self._create_prompt(question, chunks)
            
            # Generate response using OpenAI-compatible chat completions API
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.1,  # Low temperature for factual responses
                    "stop": ["QUESTION:", "DOCUMENTS:"]
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                
                if not answer:
                    return "I don't know. I couldn't generate a proper response based on the provided documents."
                
                # Basic post-processing
                answer = self._post_process_answer(answer, chunks)
                
                # Record success for circuit breaker
                self.circuit_breaker.record_success()
                
                logger.info(f"Generated answer of length {len(answer)}")
                return answer
            else:
                # Record failure for circuit breaker
                self.circuit_breaker.record_failure()
                logger.error(f"RunPod API error: {response.status_code} - {response.text}")
                return "I don't know. There was an error generating the response."
                
        except Exception as e:
            # Record failure for circuit breaker
            self.circuit_breaker.record_failure()
            logger.error(f"Failed to generate answer: {e}")
            return "I don't know. There was an error processing your question."
    
    def get_circuit_breaker_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return self.circuit_breaker.get_stats()
    
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

# For backward compatibility, create an alias
OllamaLLM = RunPodLLM
