
"""
Vertex AI client for interacting with Google's LLM models
"""
import json
import logging
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class VertexAIClient:
    """Client for interacting with Vertex AI models"""
    
    def __init__(self, project_id: str, location: str):
        """
        Initialize Vertex AI client
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region (e.g., us-central1)
        """
        self.project_id = project_id
        self.location = location
        
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
        
        for example in examples:
            user_input = example.get("user_input", "")
            
            try:
                # Prepare the prompt
                full_prompt = f"{system_prompt}\n\nUSER INPUT:\n{user_input}\n\n{output_prompt}"
                
                # Get the model response
                response = self.generate_response(
                    model_name=model_name,
                    user_content=full_prompt,
                    temperature=0.2
                )
                
                predictions.append(response)
            except Exception as e:
                logger.error(f"Error in batch prediction: {str(e)}")
                predictions.append(f"ERROR: {str(e)}")
        
        return {
            "predictions": predictions,
            "model_name": model_name,
            "timestamp": self.vertexai.utils._utils._current_timestamp_sec(),
            "count": len(predictions)
        }
    
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
            response = model.generate_content(user_content)
            
            # Extract text from response
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")
