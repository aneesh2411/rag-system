"""Guardrails for safe query processing and evidence validation."""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class GuardrailsFilter:
    """Handles safety filtering and evidence validation."""
    
    def __init__(self):
        # Define unsafe content patterns (comprehensive safety rules)
        self.unsafe_patterns = [
            # Harmful content
            r'\b(how to (?:make|create|build).{0,20}(?:bomb|explosive|weapon|poison|drug))\b',
            r'\b(?:kill|murder|assassinate|harm|hurt).{0,20}(?:someone|person|people)\b',
            r'\b(?:suicide|self.?harm|self.?hurt|end.{0,10}life)\b',
            r'\b(?:violence|torture|abuse|assault)\b',
            
            # Illegal activities
            r'\b(?:hack|crack|break into|steal|pirate|illegal download)\b',
            r'\b(?:drug dealing|money laundering|fraud|scam|counterfeit)\b',
            r'\b(?:bypass|circumvent).{0,20}(?:security|firewall|protection)\b',
            
            # Hate speech patterns
            r'\b(?:hate|discriminat\w+|racist|sexist|homophobic|xenophobic)\b',
            r'\b(?:supremacy|genocide|ethnic.{0,10}cleansing)\b',
            
            # PII requests
            r'\b(?:ssn|social security|credit card|password|api key|private key)\b',
            r'\b(?:personal information|private data|confidential|classified)\b',
            r'\b(?:medical records|financial records|tax information)\b',
            
            # Inappropriate content
            r'\b(?:adult content|nsfw|explicit|sexual|pornographic)\b',
            r'\b(?:minors|children).{0,20}(?:inappropriate|sexual|explicit)\b',
            
            # Misinformation attempts
            r'\b(?:conspiracy|hoax|fake news|disinformation)\b',
            r'\b(?:medical advice|legal advice|financial advice).{0,20}(?:professional|certified)\b',
            
            # System manipulation
            r'\b(?:ignore|forget|disregard).{0,20}(?:previous|instructions|rules|guidelines)\b',
            r'\b(?:pretend|act as|role.?play).{0,20}(?:different|other|evil|harmful)\b',
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.unsafe_patterns]
        
        # Evidence quality thresholds
        self.min_chunk_length = 50  # Minimum characters per chunk
        self.min_chunks = 1  # Minimum number of chunks
        self.min_relevance_score = 0.1  # Minimum relevance score
    
    def is_safe_query(self, query: str) -> bool:
        """Check if a query is safe to process."""
        try:
            # Check against unsafe patterns
            for pattern in self.compiled_patterns:
                if pattern.search(query):
                    logger.warning(f"Unsafe query detected: {query[:100]}...")
                    return False
            
            # Additional heuristics
            if self._contains_excessive_profanity(query):
                logger.warning("Query contains excessive profanity")
                return False
            
            if self._is_spam_like(query):
                logger.warning("Query appears to be spam-like")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in safety check: {e}")
            # Default to safe if there's an error
            return True
    
    def _contains_excessive_profanity(self, text: str) -> bool:
        """Check for excessive profanity."""
        # Basic profanity filter
        profanity_words = [
            'damn', 'hell', 'shit', 'fuck', 'bitch', 'ass', 'bastard', 'crap'
        ]
        
        text_lower = text.lower()
        profanity_count = sum(1 for word in profanity_words if word in text_lower)
        
        # Flag if more than 20% of words are profanity or more than 3 total
        word_count = len(text.split())
        return profanity_count > 3 or (word_count > 0 and profanity_count / word_count > 0.2)
    
    def _is_spam_like(self, text: str) -> bool:
        """Check if text appears spam-like."""
        # Check for excessive repetition
        words = text.lower().split()
        if len(words) > 10:
            unique_words = set(words)
            repetition_ratio = len(words) / len(unique_words)
            if repetition_ratio > 3:  # Same word repeated too often
                return True
        
        # Check for excessive caps
        if len(text) > 20:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.7:  # More than 70% caps
                return True
        
        # Check for excessive punctuation
        punct_count = sum(1 for c in text if c in '!?.')
        if punct_count > len(text) * 0.3:  # More than 30% punctuation
            return True
        
        return False
    
    def has_sufficient_evidence(self, chunks: List[Dict[str, Any]]) -> bool:
        """Check if retrieved chunks provide sufficient evidence."""
        try:
            if not chunks:
                logger.info("No chunks provided")
                return False
            
            if len(chunks) < self.min_chunks:
                logger.info(f"Too few chunks: {len(chunks)} < {self.min_chunks}")
                return False
            
            # Check chunk quality
            valid_chunks = 0
            total_content_length = 0
            
            for chunk in chunks:
                content = chunk.get("content", "")
                if len(content.strip()) >= self.min_chunk_length:
                    valid_chunks += 1
                    total_content_length += len(content)
                
                # Check for relevance score if available
                score = chunk.get("score", chunk.get("rrf_score", 1.0))
                if score < self.min_relevance_score:
                    logger.info(f"Chunk has low relevance score: {score}")
            
            if valid_chunks == 0:
                logger.info("No valid chunks found")
                return False
            
            # Check total content length
            avg_content_length = total_content_length / valid_chunks
            if avg_content_length < self.min_chunk_length:
                logger.info(f"Average chunk length too short: {avg_content_length}")
                return False
            
            # Check for content diversity (avoid repetitive chunks)
            if len(chunks) > 1 and self._chunks_too_similar(chunks):
                logger.info("Chunks are too similar, may lack diversity")
                return False
            
            logger.info(f"Evidence is sufficient: {valid_chunks} valid chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error in evidence validation: {e}")
            # Default to having sufficient evidence if there's an error
            return len(chunks) > 0
    
    def _chunks_too_similar(self, chunks: List[Dict[str, Any]]) -> bool:
        """Check if chunks are too similar to each other."""
        try:
            contents = [chunk.get("content", "") for chunk in chunks]
            
            # Simple similarity check based on word overlap
            for i in range(len(contents)):
                for j in range(i + 1, len(contents)):
                    similarity = self._calculate_text_similarity(contents[i], contents[j])
                    if similarity > 0.8:  # More than 80% similar
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity based on word overlap."""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception:
            return 0.0
    
    def validate_query_length(self, query: str) -> bool:
        """Validate query length."""
        min_length = 3
        max_length = 1000
        
        query_length = len(query.strip())
        
        if query_length < min_length:
            logger.info(f"Query too short: {query_length} < {min_length}")
            return False
        
        if query_length > max_length:
            logger.info(f"Query too long: {query_length} > {max_length}")
            return False
        
        return True
    
    def sanitize_query(self, query: str) -> str:
        """Sanitize query by removing potentially harmful content."""
        try:
            # Remove excessive whitespace
            sanitized = re.sub(r'\s+', ' ', query.strip())
            
            # Remove HTML tags if any
            sanitized = re.sub(r'<[^>]+>', '', sanitized)
            
            # Remove URLs
            sanitized = re.sub(r'https?://\S+', '', sanitized)
            
            # Remove email addresses
            sanitized = re.sub(r'\S+@\S+\.\S+', '', sanitized)
            
            # Limit special characters
            sanitized = re.sub(r'[^\w\s\?\!\.\,\-\(\)]', '', sanitized)
            
            return sanitized.strip()
            
        except Exception as e:
            logger.error(f"Error sanitizing query: {e}")
            return query
    
    def should_refuse_answer(self, question: str, chunks: List[Dict[str, Any]]) -> tuple[bool, str]:
        """Determine if we should refuse to answer and provide reason."""
        # Check safety
        if not self.is_safe_query(question):
            return True, "I cannot answer that question as it may involve unsafe, harmful, or inappropriate content."
        
        # Check query length
        if not self.validate_query_length(question):
            return True, "Please provide a question that is between 3 and 1000 characters."
        
        # Check evidence sufficiency
        if not self.has_sufficient_evidence(chunks):
            return True, "I don't know. I couldn't find sufficient information in the documents to answer your question."
        
        return False, ""
    
    def validate_response(self, response: str, question: str) -> tuple[bool, str]:
        """Validate LLM response for safety and groundedness."""
        try:
            # Check if response is safe
            if not self.is_safe_query(response):
                return False, "I cannot provide that response as it may contain unsafe content."
            
            # Check for groundedness indicators
            grounded_phrases = [
                "according to the documents",
                "based on the provided information",
                "the document states",
                "as mentioned in",
                "from the documents",
                "i don't know",
                "i couldn't find",
                "not enough information"
            ]
            
            response_lower = response.lower()
            has_grounding = any(phrase in response_lower for phrase in grounded_phrases)
            
            # Check for hallucination indicators
            hallucination_indicators = [
                "i know that",
                "it is well known",
                "generally speaking",
                "in my experience",
                "based on my knowledge",
                "i believe",
                "i think"
            ]
            
            has_hallucination = any(indicator in response_lower for indicator in hallucination_indicators)
            
            # Warn if response might be hallucinated
            if has_hallucination and not has_grounding:
                logger.warning(f"Response may contain hallucinated content for query: {question[:50]}...")
                return False, "I can only answer based on the information provided in the documents."
            
            # Check response length (too short might indicate poor quality)
            if len(response.strip()) < 10:
                return False, "I don't know. I couldn't generate a proper response based on the provided documents."
            
            return True, response
            
        except Exception as e:
            logger.error(f"Error validating response: {e}")
            return True, response  # Default to allowing response if validation fails
    
    def get_safety_stats(self) -> Dict[str, Any]:
        """Get safety statistics."""
        return {
            "total_patterns": len(self.unsafe_patterns),
            "pattern_categories": {
                "harmful_content": 4,
                "illegal_activities": 3, 
                "hate_speech": 2,
                "pii_requests": 3,
                "inappropriate_content": 2,
                "misinformation": 2,
                "system_manipulation": 2
            },
            "evidence_thresholds": {
                "min_chunk_length": self.min_chunk_length,
                "min_chunks": self.min_chunks,
                "min_relevance_score": self.min_relevance_score
            }
        }
