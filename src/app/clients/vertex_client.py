
"""
Vertex AI client for interacting with Google's LLM models
"""
import json
import logging
import time
import hashlib
from typing import List, Dict, Any, Optional, Union
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

def count_tokens(text: str) -> int:
    """
    Count token estimate for text using a simple heuristic.
    This is a rough approximation, for accurate counting use tiktoken or similar.
    
    Args:
        text: Input text to count tokens
        
    Returns:
        Estimated token count
    """
    # Simple word-based tokenization (rough approximation)
    # Real LLMs use subword tokenization which can vary by model
    words = re.findall(r'\b\w+\b', text)
    # Average English word is ~1.3 tokens
    return int(len(words) * 1.3) + 1  

class VertexAIClient:
    """Client for interacting with Vertex AI models"""
    
    def __init__(self, project_id: str, location: str, enable_caching: bool = True):
        """
        Initialize Vertex AI client
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region (e.g., us-central1)
            enable_caching: Whether to cache responses to save on API costs
        """
        self.project_id = project_id
        self.location = location
        self.enable_caching = enable_caching
        self.cache = {}
        self.cost_tracking = {
            "total_tokens_input": 0,
            "total_tokens_output": 0,
            "total_api_calls": 0,
            "total_estimated_cost": 0.0,
            "calls_by_model": {}
        }
        
        try:
            # Import Vertex AI SDK
            import vertexai
            from vertexai.generative_models import GenerativeModel, GenerationConfig
            
            # Initialize Vertex AI
            vertexai.init(project=project_id, location=location)
            
            self.vertexai = vertexai
            self.GenerativeModel = GenerativeModel
            self.GenerationConfig = GenerationConfig
            
            logger.info(f"Initialized Vertex AI client for project {project_id} in {location}")
        except ImportError:
            logger.warning("vertexai package not installed. Some functionality will be limited.")
            self.vertexai = None
            self.GenerativeModel = None
            self.GenerationConfig = None
    
    def batch_predict(self, 
                     examples: List[Dict[str, str]], 
                     prompt_state: Dict[str, Any],
                     model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run batch prediction on multiple examples
        
        Args:
            examples: List of example dictionaries with at least 'user_input' key
            prompt_state: Dictionary with 'system_prompt' and 'output_prompt'
            model_name: Name of the model to use (e.g., gemini-1.5-flash-001)
            
        Returns:
            Dictionary with predictions and metadata
        """
        if not self.vertexai:
            raise RuntimeError("Vertex AI SDK not initialized")
        
        if not model_name:
            raise ValueError("Model name must be specified")
        
        system_prompt = prompt_state.get("system_prompt", "")
        output_prompt = prompt_state.get("output_prompt", "")
        
        predictions = []
        token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
        
        batch_start_time = time.time()
        
        # Initialize model stats if not already tracked
        if model_name not in self.cost_tracking["calls_by_model"]:
            self.cost_tracking["calls_by_model"][model_name] = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost": 0.0
            }
        
        for example in examples:
            user_input = example.get("user_input", "")
            
            try:
                # Prepare the prompt
                full_prompt = f"{system_prompt}\n\nUSER INPUT:\n{user_input}\n\n{output_prompt}"
                
                # Check cache if enabled
                cache_key = None
                if self.enable_caching:
                    cache_key = hashlib.md5(f"{model_name}:{full_prompt}".encode()).hexdigest()
                    if cache_key in self.cache:
                        logger.info(f"Cache hit for example input: {user_input[:30]}...")
                        predictions.append(self.cache[cache_key]["response"])
                        token_usage["input_tokens"] += self.cache[cache_key]["input_tokens"]
                        token_usage["output_tokens"] += self.cache[cache_key]["output_tokens"]
                        token_usage["total_tokens"] += self.cache[cache_key]["total_tokens"]
                        continue
                
                # Count input tokens
                input_tokens = count_tokens(full_prompt)
                token_usage["input_tokens"] += input_tokens
                
                # Get the model response
                response = self.generate_response(
                    model_name=model_name,
                    user_content=full_prompt,
                    temperature=0.2
                )
                
                # Count output tokens
                output_tokens = count_tokens(response)
                token_usage["output_tokens"] += output_tokens
                token_usage["total_tokens"] += input_tokens + output_tokens
                
                # Update cost tracking
                self.cost_tracking["total_tokens_input"] += input_tokens
                self.cost_tracking["total_tokens_output"] += output_tokens
                self.cost_tracking["total_api_calls"] += 1
                self.cost_tracking["calls_by_model"][model_name]["calls"] += 1
                self.cost_tracking["calls_by_model"][model_name]["input_tokens"] += input_tokens
                self.cost_tracking["calls_by_model"][model_name]["output_tokens"] += output_tokens
                
                # Calculate cost estimate based on model
                cost = self._estimate_cost(model_name, input_tokens, output_tokens)
                self.cost_tracking["total_estimated_cost"] += cost
                self.cost_tracking["calls_by_model"][model_name]["estimated_cost"] += cost
                
                # Store in cache if enabled
                if self.enable_caching and cache_key:
                    self.cache[cache_key] = {
                        "response": response,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                        "timestamp": time.time()
                    }
                
                predictions.append(response)
            except Exception as e:
                logger.error(f"Error in batch prediction: {str(e)}")
                predictions.append(f"ERROR: {str(e)}")
        
        batch_duration = time.time() - batch_start_time
        
        return {
            "predictions": predictions,
            "model_name": model_name,
            "timestamp": self.vertexai.utils._utils._current_timestamp_sec(),
            "count": len(predictions),
            "token_usage": token_usage,
            "duration_seconds": batch_duration
        }
    
    @lru_cache(maxsize=128)
    def generate_response(self, 
                         model_name: str, 
                         user_content: str,
                         temperature: float = 0.7, 
                         response_mime_type: Optional[str] = None) -> str:
        """
        Generate a single response from the LLM
        
        Args:
            model_name: Name of the model to use
            user_content: User's prompt content
            temperature: Temperature for generation (0.0-1.0)
            response_mime_type: Optional MIME type for response format
            
        Returns:
            Generated text response
        """
        if not self.vertexai:
            raise RuntimeError("Vertex AI SDK not initialized")
        
        # Check cache if enabled (using function-level cache with lru_cache)
        if self.enable_caching:
            cache_key = hashlib.md5(f"{model_name}:{user_content}:{temperature}".encode()).hexdigest()
            if cache_key in self.cache:
                logger.info(f"Cache hit for direct generate_response call")
                return self.cache[cache_key]["response"]
        
        try:
            # Create generation config
            generation_config = self.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1024,
                top_p=0.95,
                top_k=40
            )
            
            # Create model instance
            model = self.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config
            )
            
            # Generate content
            start_time = time.time()
            response = model.generate_content(user_content)
            end_time = time.time()
            
            # Extract text from response
            response_text = response.text
            
            # Store in cache if enabled
            if self.enable_caching and cache_key:
                input_tokens = count_tokens(user_content)
                output_tokens = count_tokens(response_text)
                
                self.cache[cache_key] = {
                    "response": response_text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "duration": end_time - start_time,
                    "timestamp": time.time()
                }
            
            return response_text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")
def _estimate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate the cost of a model call based on token usage
        
        Args:
            model_name: Name of the model
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        # Pricing in USD per million tokens (approximate, may need updating)
        pricing = {
            "gemini-1.5-flash": {"input": 0.35, "output": 1.05},
            "gemini-1.5-flash-001": {"input": 0.35, "output": 1.05},
            "gemini-1.5-flash-002": {"input": 0.35, "output": 1.05},
            "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
            "gemini-1.5-pro-001": {"input": 3.50, "output": 10.50},
            "gemini-1.5-pro-002": {"input": 3.50, "output": 10.50},
            "gemini-2.5-flash": {"input": 0.35, "output": 1.05},
            "gemini-2.5-pro": {"input": 3.50, "output": 10.50},
            "default": {"input": 0.35, "output": 1.05}  # Default to flash pricing
        }
        
        # Get price rates for the model or fall back to default
        model_key = model_name
        for key in pricing:
            if key in model_name:
                model_key = key
                break
        
        if model_key not in pricing:
            model_key = "default"
            
        rates = pricing[model_key]
        
        # Calculate cost: (tokens / 1M) * rate
        input_cost = (input_tokens / 1000000) * rates["input"]
        output_cost = (output_tokens / 1000000) * rates["output"]
        
        return input_cost + output_cost
    
    def clean_cache(self, max_age_seconds: int = 3600, max_entries: int = 1000):
        """
        Clean old entries from the cache
        
        Args:
            max_age_seconds: Maximum age of cache entries in seconds
            max_entries: Maximum number of entries to keep in cache
        """
        if not self.enable_caching or not self.cache:
            return
            
        current_time = time.time()
        
        # Remove old entries
        keys_to_remove = []
        for key, entry in self.cache.items():
            if current_time - entry["timestamp"] > max_age_seconds:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self.cache[key]
            
        # If still too many entries, remove least recently used
        if len(self.cache) > max_entries:
            sorted_entries = sorted(
                [(k, v["timestamp"]) for k, v in self.cache.items()],
                key=lambda x: x[1]
            )
            
            # Keep only the most recent entries
            entries_to_remove = len(self.cache) - max_entries
            for i in range(entries_to_remove):
                if i < len(sorted_entries):
                    del self.cache[sorted_entries[i][0]]
    
    def get_cost_report(self) -> Dict[str, Any]:
        """
        Get a report of token usage and estimated costs
        
        Returns:
            Dictionary with token usage and cost information
        """
        return {
            "total_api_calls": self.cost_tracking["total_api_calls"],
            "total_tokens": {
                "input": self.cost_tracking["total_tokens_input"],
                "output": self.cost_tracking["total_tokens_output"],
                "total": self.cost_tracking["total_tokens_input"] + self.cost_tracking["total_tokens_output"]
            },
            "total_estimated_cost_usd": round(self.cost_tracking["total_estimated_cost"], 4),
            "models": self.cost_tracking["calls_by_model"]
        }
    
    def reset_cost_tracking(self):
        """Reset all cost tracking data"""
        self.cost_tracking = {
            "total_tokens_input": 0,
            "total_tokens_output": 0,
            "total_api_calls": 0,
            "total_estimated_cost": 0.0,
            "calls_by_model": {}
        }
