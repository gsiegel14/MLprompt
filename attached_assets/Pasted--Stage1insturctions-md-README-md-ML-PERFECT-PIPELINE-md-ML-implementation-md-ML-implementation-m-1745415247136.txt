@Stage1insturctions.md @README.md @ML_PERFECT_PIPELINE.md @ML implementation.md @ML implementation.md @BACKEND_IMPLEMENTATION_FRAMEWORK.md @APIinfo.md @APIinfo.md @API_ARCHITECTURE_README.md 

review curreent plans files added. after review. come up with what you'd want to know next and make a set up instructions for the next stage of development. You are an LLM with coding knowledging building this project. 

Next Stage of Development: Core Implementation Instructions
After reviewing the current project files, I can see we've established a solid foundation for the prompt optimization platform. We have the architecture design, Prefect workflow layout, API specifications, and implementation guidelines ready. Now it's time to move into actual code implementation.
Phase 1: Core Service Implementation
First, we need to implement the core services and models that power the optimization workflow:
Implement the PromptState model with GCS storage
Create the Vertex AI client for LLM interactions
Develop the Hugging Face evaluator service
Build the central Prefect flow orchestration
Instructions for Phase 1
Apply to Stage3.md
As an LLM with coding knowledge building this prompt optimization platform, I'll now implement the core components that enable the 5-step workflow. I'll start with the fundamental services and models needed to execute the optimization cycle.

1. First, I'll implement the complete PromptState model in src/app/models/prompt_state.py:
   - Add full GCS integration for loading/saving prompt states
   - Implement version tracking and parent-child relationships
   - Add serialization/deserialization methods

2. Next, I'll create the Vertex AI client in src/app/clients/vertex_client.py:
   - Implement the batch_predict method for the primary inference steps
   - Develop the generate_response method for the optimizer LLM
   - Add proper error handling, retry logic, and logging

3. I'll then build the HuggingFace evaluator in src/app/clients/hf_evaluator.py:
   - Implement support for exact match, BLEU, ROUGE, and BERTScore metrics
   - Add proper caching of loaded metrics
   - Ensure robust handling of edge cases (empty strings, etc.)

4. For the Prefect workflow, I'll start by implementing the core tasks in src/flows/tasks/:
   - Create load_state task in data_tasks.py
   - Implement vertex_primary_inference and vertex_refined_inference in inference_tasks.py
   - Build hf_eval_baseline and hf_eval_refined in evaluation_tasks.py
   - Develop vertex_optimizer_refine in optimization_tasks.py
   - Create compare_and_log in logging_tasks.py

5. Finally, I'll implement the main Prefect flow in src/flows/prompt_optimization_flow.py:
   - Set up the flow parameters
   - Implement the iteration logic with early stopping
   - Add artifact creation for tracking
   - Ensure proper error handling and logging
Phase 2: API Layer Implementation
Once the core services are implemented, we need to expose them through the API:
Create the FastAPI endpoints for prompts, inference, and experiments
Implement the Pydantic models for request/response validation
Connect API endpoints to the core services
Add authentication and security
Instructions for Phase 2
Apply to Stage3.md
Now that I've implemented the core services, I'll build the API layer to make the platform accessible:

1. First, I'll create the Pydantic models in src/api/models.py:
   - Define PromptCreate, PromptResponse, ExperimentCreate models
   - Implement InferenceRequest and InferenceResponse models
   - Add validation rules for each model

2. Next, I'll implement the prompt management endpoints in src/api/endpoints/prompts.py:
   - Create POST /api/v1/prompts endpoint
   - Implement GET /api/v1/prompts/{id} endpoint
   - Add PUT /api/v1/prompts/{id} endpoint

3. I'll then build the inference endpoints in src/api/endpoints/inference.py:
   - Implement POST /api/v1/inference/complete for single completion
   - Create POST /api/v1/inference/batch for batch processing
   - Add POST /api/v1/inference/evaluate for combined inference and evaluation

4. For experiment management, I'll implement in src/api/endpoints/experiments.py:
   - Create POST /api/v1/experiments endpoint
   - Add POST /api/v1/experiments/{id}/start endpoint
   - Implement GET /api/v1/experiments/{id}/metrics endpoint

5. I'll connect all routers in src/api/routers.py and set up the main FastAPI app in src/app/main.py.

6. Finally, I'll implement authentication using JWT tokens:
   - Create src/app/auth.py with token generation and verification
   - Add authentication dependencies to the API endpoints
   - Implement proper error handling for auth failures
Phase 3: Testing and Integration
With the core components and API implemented, we need to ensure everything works together:
Create unit tests for the core services
Implement integration tests for the Prefect flow
Set up end-to-end tests for the API endpoints
Create a simple CLI for local testing
Instructions for Phase 3
Apply to Stage3.md
To ensure the platform works correctly, I'll implement testing and integration components:

1. First, I'll create unit tests for core services:
   - Test PromptState model serialization/deserialization
   - Mock Vertex AI and test vertex_client methods
   - Test HuggingFace evaluator with sample data

2. Next, I'll implement integration tests for the Prefect flow:
   - Create test fixtures with sample data
   - Test each task individually with mocked dependencies
   - Test the full flow with a small dataset and mocked LLM responses

3. For the API, I'll add end-to-end tests:
   - Test authentication flow
   - Test prompt CRUD operations
   - Test inference endpoints with mocked services
   - Test experiment management endpoints

4. I'll create a simple CLI for local testing:
   - Implement a command-line interface in src/cli.py
   - Add commands for running the prompt optimization flow
   - Create commands for managing prompts and experiments
   - Add a command for testing inference
Question to Guide Next Steps
Based on the current project state, my next key questions would be:
What specific metrics do we want to use for evaluation? Are we focusing on exact match, BLEU, ROUGE, or a custom metric?
How should the optimizer prompt be structured? The README.md mentions specific prompts for the optimizer LLM - should I implement these exactly as described?
Are there specific Vertex AI models we want to use? The documentation mentions "gemini-1.5-pro" and "gemini-1.5-flash-001" - are these the models we should use?
Should we implement the meta-learning and reinforcement learning extensions now or later? The ML_PERFECT_PIPELINE.md mentions these as possible extensions.
How do we want to handle authentication? Should I implement JWT authentication as described, or use a different approach?
With these details clarified, I'd be ready to implement the core components of the platform and start building a working system.
make a readme with instructions called "stage 2"@Stage1insturctions.md . the goal is to be low cost and limit tokens. review stage 1 first. 

Stage 2: Token Efficiency & Low-Cost Implementation
This document outlines the second phase of development for our prompt optimization platform, focusing specifically on optimizing for low cost and minimal token usage.
Review of Stage 1
In Stage 1, we established the foundational architecture:
Project structure and core components
PromptState model for version tracking
Vertex AI client for LLM interactions
HuggingFace evaluator for metrics
Prefect workflow orchestration for the 5-step optimization process
Stage 2 Focus: Token Efficiency & Cost Optimization
Our primary goals for Stage 2:
Minimize token usage in all LLM interactions
Implement cost tracking and budgeting
Add caching to eliminate redundant API calls
Optimize batch processing for efficient resource utilization
Implementation Tasks
1. Vertex AI Client Optimization
Apply to Stage3.md
# src/app/clients/vertex_client.py

class VertexAIClient:
    # Add to __init__
    def __init__(self, project_id, location):
        # Existing code...
        self.token_count = 0
        self.cost_tracker = TokenCostTracker()
        self.response_cache = {}  # Simple in-memory cache
    
    # Add token counting
    def _count_tokens(self, text):
        """Estimate token count using a simple ratio (characters/4)"""
        return len(text) // 4  # Simple estimation, replace with tiktoken or similar
    
    # Update generate_response
    def generate_response(self, model_name, user_content, system_content=None, 
                         temperature=0.0, max_tokens=1024, response_mime_type=None):
        # Create cache key
        cache_key = f"{model_name}:{hash(user_content)}:{hash(system_content)}:{temperature}:{max_tokens}"
        
        # Check cache first
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
            
        # Count input tokens
        input_tokens = self._count_tokens(user_content)
        if system_content:
            input_tokens += self._count_tokens(system_content)
            
        # Existing implementation...
        response = # call Vertex AI
        
        # Count output tokens and update tracker
        output_tokens = self._count_tokens(response)
        self.token_count += input_tokens + output_tokens
        self.cost_tracker.add_usage(model_name, input_tokens, output_tokens)
        
        # Cache result
        self.response_cache[cache_key] = response
        return response
2. Optimizer Context Minimization
Apply to Stage3.md
# src/app/optimizer.py

def _select_representative_examples(self, examples, max_examples=3):
    """Select the most informative examples within token budget"""
    if len(examples) <= max_examples:
        return examples
        
    # Strategy 1: Choose worst-performing examples
    sorted_examples = sorted(examples, key=lambda x: x['score'])
    return sorted_examples[:max_examples]
    
    # Alternative: Balance between worst examples and diversity
    # (implementation depends on your specific use case)

def _format_context_for_optimizer(self, current_system_prompt, current_output_prompt, 
                                examples, metrics):
    """Format context with minimal tokens"""
    # Truncate system prompt if very long (preserve beginning and end)
    if len(current_system_prompt) > 500:
        current_system_prompt = (current_system_prompt[:200] + 
                               "\n[...truncated for brevity...]\n" + 
                               current_system_prompt[-200:])
    
    # Select minimal but representative examples
    selected_examples = self._select_representative_examples(examples)
    
    # Format examples minimally
    formatted_examples = []
    for ex in selected_examples:
        formatted_examples.append(
            f"Input: {ex['user_input'][:100]}...\n"
            f"Expected: {ex['ground_truth_output'][:100]}...\n"
            f"Actual: {ex['model_response'][:100]}...\n"
            f"Score: {ex['score']}"
        )
    
    # Include only key metrics
    key_metrics = {k: metrics[k] for k in ['exact_match_score'] 
                  if k in metrics}
    
    # Return minimized context
    # ...
3. Efficient Task Batching in Prefect Flow
Apply to Stage3.md
# src/flows/tasks/inference_tasks.py

@task(name="vertex-batch-inference", retries=2)
def vertex_batch_inference(batch_data, state_dict, vertex_project_id, 
                          vertex_location, model_name, batch_size=10):
    """Process examples in batches to optimize API calls"""
    client = VertexAIClient(
        project_id=vertex_project_id,
        location=vertex_location,
    )
    prompt_state = PromptState(**state_dict)
    
    # Process in optimal batch sizes
    results = []
    total_examples = len(batch_data)
    
    for i in range(0, total_examples, batch_size):
        batch = batch_data[i:i+batch_size]
        # Process batch
        batch_results = client.batch_predict(batch, prompt_state, model_name)
        results.extend(batch_results)
        
    return results
4. Budget Controls & Monitoring
Apply to Stage3.md
# src/app/utils/cost_tracking.py

class TokenCostTracker:
    # Model costs in USD per 1000 tokens
    PRICING = {
        "gemini-1.5-flash-001": {"input": 0.00035, "output": 0.00105},
        "gemini-1.5-pro-001": {"input": 0.0007, "output": 0.0021}
    }
    
    def __init__(self):
        self.usage = {}  # model -> {input_tokens, output_tokens, cost}
        self.total_cost = 0.0
        
    def add_usage(self, model, input_tokens, output_tokens):
        if model not in self.usage:
            self.usage[model] = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
            
        # Add tokens
        self.usage[model]["input_tokens"] += input_tokens
        self.usage[model]["output_tokens"] += output_tokens
        
        # Calculate cost
        if model in self.PRICING:
            input_cost = (input_tokens / 1000) * self.PRICING[model]["input"]
            output_cost = (output_tokens / 1000) * self.PRICING[model]["output"]
            cost = input_cost + output_cost
            self.usage[model]["cost"] += cost
            self.total_cost += cost
            
    def get_usage_report(self):
        return {
            "models": self.usage,
            "total_cost": self.total_cost
        }
        
    def enforce_budget(self, max_budget):
        """Check if budget exceeded and raise exception if so"""
        if self.total_cost > max_budget:
            raise BudgetExceededException(
                f"Budget of ${max_budget:.2f} exceeded. Current cost: ${self.total_cost:.2f}"
            )
5. Main Optimization Flow Updates
Apply to Stage3.md
# src/flows/prompt_optimization_flow.py

@flow(name="prompt-optimization-flow")
def prompt_optimization_flow(
    vertex_project_id: str,
    vertex_location: str,
    primary_model_name: str,
    optimizer_model_name: str,
    dataset_path: str,
    system_prompt_path: str,
    output_prompt_path: str,
    target_metric: str = "exact_match_score",
    target_threshold: float = 0.90,
    patience: int = 3,
    max_iterations: int = 10,
    max_budget: float = 10.0,  # Budget in USD
    sample_k: int = 3,  # Number of examples to send to optimizer
):
    """Main optimization flow with cost controls"""
    logger = get_run_logger()
    cost_tracker = TokenCostTracker()
    
    # Set up state
    state_path = None
    no_improve_count = 0
    best_metric = 0.0
    
    try:
        for iteration in range(max_iterations):
            # Load state and dataset (subsample if needed)
            state_data = load_state(system_prompt_path, output_prompt_path, 
                                  dataset_path, state_path)
            
            # Enforce budget check
            cost_tracker.enforce_budget(max_budget)
            
            # Use smaller subset for training if dataset is large
            if len(state_data["dataset"]) > 50:
                # Subsample dataset to control costs
                train_subset = random_sample(state_data["dataset"], 50)
            else:
                train_subset = state_data["dataset"]
                
            # Run main optimization cycle with train subset
            # ... rest of implementation
            
            # Log cost information as Prefect artifact
            create_artifact(
                key=f"cost-report-{iteration}",
                description="Token usage and cost breakdown",
                data=cost_tracker.get_usage_report(),
                type="cost_report"
            )
            
    except BudgetExceededException as e:
        logger.warning(f"Optimization stopped: {str(e)}")
        
    return {"final_state_path": state_path, "iterations_completed": iteration}
Token Efficiency Strategies
Context Truncation: Limit example length and count when sending to the optimizer LLM
Caching: Cache LLM responses to avoid redundant API calls
Batching: Process examples in optimal batch sizes
Progressive Sampling: Start with small subsets and increase only if needed
Early Stopping: Halt training when a good prompt is found or budget is exceeded
Cost Tracking Implementation
Token Counting: Count tokens for all LLM interactions
Model-specific Pricing: Track costs based on the exact model used
Budget Enforcement: Stop training when budget is exceeded
Usage Reporting: Generate detailed reports of token usage and costs
Testing Focus
Test particularly these components:
Token counting accuracy compared to actual billing
Cache hit rates
Budget enforcement correctness
Optimization effectiveness with minimal examples
Next Steps for Stage 3
After implementing these token efficiency and cost control measures, Stage 3 will focus on:
API endpoint implementation
Authentication and security
Front-end interface for monitoring experiments
Deployment pipeline
write detailed stage 3 instructions. output so I can single copy and paste the text into a document. don't output in sections. Make a todolist at the end of taks. 

Stage 3: API Implementation, Authentication & Deployment
This document outlines Stage 3 of our prompt optimization platform development, focusing on implementing the API layer, adding authentication, creating a monitoring interface, and establishing a deployment pipeline.
Overview
After completing the core optimization engine (Stage 1) and optimizing for token efficiency (Stage 2), we now need to make the platform accessible through well-defined APIs, secure it with proper authentication, build a monitoring interface, and prepare it for deployment.
API Implementation
Our API will follow the design outlined in APIinfo.md, implementing these core endpoints:
Inference API
POST /api/v1/inference/complete for single completions
POST /api/v1/inference/batch for batch processing
Evaluation API
POST /api/v1/evaluate for scoring predictions against ground truth
POST /api/v1/inference/evaluate for comprehensive dataset evaluation
Prompt Management API
POST /api/v1/prompts for storing new prompts
GET /api/v1/prompts/{id} for retrieving prompts
PUT /api/v1/prompts/{id} for updating prompts
Experiment Management API
POST /api/v1/experiments for creating new experiments
POST /api/v1/experiments/{id}/start for starting execution
GET /api/v1/experiments/{id}/metrics for retrieving results
Dataset Management API
POST /api/v1/datasets for uploading new datasets
GET /api/v1/datasets/{id}/sample for retrieving samples
Implementation steps:
Create Pydantic models for request/response validation in src/api/models.py:
Stage 3: API Implementation, Authentication & Deployment
This document outlines Stage 3 of our prompt optimization platform development, focusing on implementing the API layer, adding authentication, creating a monitoring interface, and establishing a deployment pipeline.
Overview
After completing the core optimization engine (Stage 1) and optimizing for token efficiency (Stage 2), we now need to make the platform accessible through well-defined APIs, secure it with proper authentication, build a monitoring interface, and prepare it for deployment.
API Implementation
Our API will follow the design outlined in APIinfo.md, implementing these core endpoints:
Inference API
POST /api/v1/inference/complete for single completions
POST /api/v1/inference/batch for batch processing
Evaluation API
POST /api/v1/evaluate for scoring predictions against ground truth
POST /api/v1/inference/evaluate for comprehensive dataset evaluation
Prompt Management API
POST /api/v1/prompts for storing new prompts
GET /api/v1/prompts/{id} for retrieving prompts
PUT /api/v1/prompts/{id} for updating prompts
Experiment Management API
POST /api/v1/experiments for creating new experiments
POST /api/v1/experiments/{id}/start for starting execution
GET /api/v1/experiments/{id}/metrics for retrieving results
Dataset Management API
POST /api/v1/datasets for uploading new datasets
GET /api/v1/datasets/{id}/sample for retrieving samples
Implementation steps:
Create Pydantic models for request/response validation in src/api/models.py:
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class PromptCreate(BaseModel):
    system_prompt: str
    output_prompt: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PromptResponse(BaseModel):
    prompt_id: str
    system_prompt: str
    output_prompt: str
    version: int
    parent_id: Optional[str] = None
    created_at: datetime
    metadata: Dict[str, Any]

class InferenceRequest(BaseModel):
    system_prompt: Optional[str] = None
    output_prompt: Optional[str] = None
    prompt_id: Optional[str] = None
    user_input: str
    temperature: float = 0.0
    max_tokens: int = 1024

# Additional model definitions...
Implement each endpoint following this pattern in src/api/endpoints/:
Stage 3: API Implementation, Authentication & Deployment
This document outlines Stage 3 of our prompt optimization platform development, focusing on implementing the API layer, adding authentication, creating a monitoring interface, and establishing a deployment pipeline.
Overview
After completing the core optimization engine (Stage 1) and optimizing for token efficiency (Stage 2), we now need to make the platform accessible through well-defined APIs, secure it with proper authentication, build a monitoring interface, and prepare it for deployment.
API Implementation
Our API will follow the design outlined in APIinfo.md, implementing these core endpoints:
Inference API
POST /api/v1/inference/complete for single completions
POST /api/v1/inference/batch for batch processing
Evaluation API
POST /api/v1/evaluate for scoring predictions against ground truth
POST /api/v1/inference/evaluate for comprehensive dataset evaluation
Prompt Management API
POST /api/v1/prompts for storing new prompts
GET /api/v1/prompts/{id} for retrieving prompts
PUT /api/v1/prompts/{id} for updating prompts
Experiment Management API
POST /api/v1/experiments for creating new experiments
POST /api/v1/experiments/{id}/start for starting execution
GET /api/v1/experiments/{id}/metrics for retrieving results
Dataset Management API
POST /api/v1/datasets for uploading new datasets
GET /api/v1/datasets/{id}/sample for retrieving samples
Implementation steps:
Create Pydantic models for request/response validation in src/api/models.py:
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class PromptCreate(BaseModel):
    system_prompt: str
    output_prompt: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PromptResponse(BaseModel):
    prompt_id: str
    system_prompt: str
    output_prompt: str
    version: int
    parent_id: Optional[str] = None
    created_at: datetime
    metadata: Dict[str, Any]

class InferenceRequest(BaseModel):
    system_prompt: Optional[str] = None
    output_prompt: Optional[str] = None
    prompt_id: Optional[str] = None
    user_input: str
    temperature: float = 0.0
    max_tokens: int = 1024

# Additional model definitions...

from fastapi import APIRouter, HTTPException, Depends
from src.api.models import PromptCreate, PromptResponse
from src.app.services.prompt_service import PromptService
from src.app.auth import get_current_user

router = APIRouter(prefix="/prompts", tags=["Prompts"])

@router.post("/", response_model=PromptResponse)
async def create_prompt(prompt_data: PromptCreate, current_user = Depends(get_current_user)):
    try:
        prompt_service = PromptService()
        result = await prompt_service.create_prompt(
            system_prompt=prompt_data.system_prompt,
            output_prompt=prompt_data.output_prompt,
            name=prompt_data.name,
            metadata=prompt_data.metadata,
            created_by=current_user.id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
Implement the backend service layer in src/app/services/:
Apply to Stage3.md
class PromptService:
    def __init__(self):
        self.storage = GCSStorage()  # Or other implementation
        
    async def create_prompt(self, system_prompt, output_prompt, name=None, metadata=None, created_by=None):
        # Create a new PromptState
        prompt = PromptState(
            system_prompt=system_prompt,
            output_prompt=output_prompt,
            metadata=metadata or {}
        )
        
        # Add additional metadata
        if name:
            prompt.metadata["name"] = name
        if created_by:
            prompt.metadata["created_by"] = created_by
            
        # Save to storage
        path = f"gs://{settings.GCS_BUCKET_NAME}/prompts/{prompt.id}.json"
        prompt.save(path)
        
        return {
            "prompt_id": prompt.id,
            "system_prompt": prompt.system_prompt,
            "output_prompt": prompt.output_prompt,
            "version": prompt.version,
            "parent_id": prompt.parent_id,
            "created_at": prompt.created_at,
            "metadata": prompt.metadata
        }
Authentication & Security
Implement JWT authentication in src/app/auth.py:
Apply to Stage3.md
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from src.app.config import settings

# Models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    disabled: Optional[bool] = None

# Setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=24))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
        
    # In a real app, you'd fetch the user from a database
    # This is a simplified example
    user = User(id="123", username=token_data.username, disabled=False)
    if user is None:
        raise credentials_exception
    return user
Add token endpoint to src/app/main.py:
Apply to Stage3.md
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # In a real app, you'd verify username/password against a database
    # This is a simplified example for testing
    if form_data.username != "test" or form_data.password != "test123":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(hours=settings.JWT_EXPIRATION_HOURS)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
Monitoring Interface
Create a simple dashboard for monitoring experiments:
Create basic HTML templates in src/app/templates/:
Apply to Stage3.md
<!-- index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Prompt Optimization Platform</title>
    <link rel="stylesheet" href="{{ url_for('static', path='/css/styles.css') }}">
</head>
<body>
    <div class="container">
        <h1>Prompt Optimization Platform</h1>
        
        <div class="card">
            <h2>Active Experiments</h2>
            <table id="experiments-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Name</th>
                        <th>Status</th>
                        <th>Started</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- Populated by JavaScript -->
                </tbody>
            </table>
        </div>
        
        <div class="card">
            <h2>Create New Experiment</h2>
            <form id="new-experiment-form">
                <!-- Form fields -->
            </form>
        </div>
    </div>
    
    <script src="{{ url_for('static', path='/js/main.js') }}"></script>
</body>
</html>
Add static assets in src/app/static/:
Apply to Stage3.md
/* styles.css */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    margin: 0;
    padding: 20px;
    background-color: #f7f9fc;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
}

.card {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 20px;
    margin-bottom: 20px;
}

/* Additional styling */
Create JavaScript for the dashboard in src/app/static/js/:
Apply to Stage3.md
// main.js
document.addEventListener('DOMContentLoaded', function() {
    // Fetch active experiments
    fetchExperiments();
    
    // Setup form handlers
    document.getElementById('new-experiment-form').addEventListener('submit', createExperiment);
});

async function fetchExperiments() {
    try {
        const response = await fetch('/api/v1/experiments');
        const data = await response.json();
        
        const tableBody = document.querySelector('#experiments-table tbody');
        tableBody.innerHTML = '';
        
        data.forEach(exp => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${exp.id}</td>
                <td>${exp.name}</td>
                <td>${exp.status}</td>
                <td>${new Date(exp.created_at).toLocaleString()}</td>
                <td>
                    <button onclick="viewExperiment('${exp.id}')">View</button>
                    ${exp.status === 'created' ? `<button onclick="startExperiment('${exp.id}')">Start</button>` : ''}
                </td>
            `;
            tableBody.appendChild(row);
        });
    } catch (error) {
        console.error('Error fetching experiments:', error);
    }
}

// Additional functions for handling experiments
Add templates to FastAPI in src/app/main.py:
Apply to Stage3.md
from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

app = FastAPI(title="Prompt Optimization Platform API")

# Mount static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

@app.get("/", include_in_schema=False)
async def index(request: Request, current_user = Depends(get_current_user)):
    return templates.TemplateResponse("index.html", {"request": request})

# API routes
Deployment Pipeline
Create Dockerfile in the project root:
Apply to Stage3.md
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/

# Set environment variables
ENV PYTHONPATH=/app

# Run the application
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
Create docker-compose.yml for local development:
Apply to Stage3.md
version: '3'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    env_file:
      - .env
    command: uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
    
  prefect:
    build: .
    volumes:
      - .:/app
    env_file:
      - .env
    command: prefect agent start -q default
    depends_on:
      - api
Create deployment scripts:
Apply to Stage3.md
Run
#!/bin/bash
# deploy.sh

# Build the Docker image
docker build -t prompt-optimizer:latest .

# Push to container registry (example for Google Artifact Registry)
docker tag prompt-optimizer:latest gcr.io/$PROJECT_ID/prompt-optimizer:latest
docker push gcr.io/$PROJECT_ID/prompt-optimizer:latest

# Deploy to Cloud Run or Kubernetes
gcloud run deploy prompt-optimizer \
  --image gcr.io/$PROJECT_ID/prompt-optimizer:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
GitHub Actions workflow in .github/workflows/ci-cd.yml:
Apply to Stage3.md
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.10
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: |
        pytest --cov=src
        
  deploy:
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Google Cloud SDK
      uses: google-github-actions/setup-gcloud@v0
      with:
        service_account_key: ${{ secrets.GCP_SA_KEY }}
        project_id: ${{ secrets.GCP_PROJECT_ID }}
    - name: Build and Push Docker image
      run: |
        gcloud auth configure-docker
        docker build -t gcr.io/${{ secrets.GCP_PROJECT_ID }}/prompt-optimizer:latest .
        docker push gcr.io/${{ secrets.GCP_PROJECT_ID }}/prompt-optimizer:latest
    - name: Deploy to Cloud Run
      run: |
        gcloud run deploy prompt-optimizer \
          --image gcr.io/${{ secrets.GCP_PROJECT_ID }}/prompt-optimizer:latest \
          --platform managed \
          --region us-central1 \
          --allow-unauthenticated
Testing Strategy
Create unit tests for API endpoints in tests/api/:
Apply to Stage3.md
# tests/api/test_prompts.py
from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)

def test_create_prompt():
    response = client.post(
        "/api/v1/prompts",
        json={
            "system_prompt": "Test system prompt",
            "output_prompt": "Test output prompt",
            "name": "Test Prompt"
        },
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prompt_id" in data
    assert data["system_prompt"] == "Test system prompt"
    assert data["output_prompt"] == "Test output prompt"
    
# Additional test cases...
Create integration tests for Prefect flows:
Apply to Stage3.md
# tests/flows/test_optimization_flow.py
import pytest
from unittest.mock import patch, MagicMock
from src.flows.prompt_optimization_flow import prompt_optimization_flow

@pytest.fixture
def mock_vertex_client():
    with patch("src.app.clients.vertex_client.VertexAIClient") as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance
        yield client_instance

@pytest.fixture
def mock_evaluator():
    with patch("src.app.clients.hf_evaluator.EvaluatorService") as mock:
        evaluator_instance = MagicMock()
        evaluator_instance.evaluate.return_value = {"exact_match_score": 0.75}
        mock.return_value = evaluator_instance
        yield evaluator_instance

def test_optimization_flow(mock_vertex_client, mock_evaluator):
    # Setup test data
    test_data = {
        "vertex_project_id": "test-project",
        "vertex_location": "us-central1",
        "primary_model_name": "gemini-1.5-flash-001",
        "optimizer_model_name": "gemini-1.5-pro-001",
        "dataset_path": "tests/data/test_dataset.csv",
        "system_prompt_path": "tests/data/test_system_prompt.txt",
        "output_prompt_path": "tests/data/test_output_prompt.txt",
        "max_iterations": 2
    }
    
    # Configure mocks
    mock_vertex_client.batch_predict.return_value = ["Test response 1", "Test response 2"]
    mock_vertex_client.generate_response.return_value = '{"system_prompt": "New system prompt", "output_prompt": "New output prompt"}'
    
    # Run the flow
    result = prompt_optimization_flow(**test_data)
    
    # Assertions
    assert "final_state_path" in result
    assert "iterations_completed" in result
    assert result["iterations_completed"] > 0
    
    # Verify mock calls
    assert mock_vertex_client.batch_predict.called
    assert mock_vertex_client.generate_response.called
    assert mock_evaluator.evaluate.called
To-Do List:
Implement Pydantic models in src/api/models.py
Create API routers for endpoints in src/api/endpoints/
Implement JWT authentication in src/app/auth.py
Add token endpoint to src/app/main.py
Create service layer classes in src/app/services/
Develop monitoring dashboard templates in src/app/templates/
Add static assets (CSS, JavaScript) for dashboard
Connect templates to FastAPI routes
Create Dockerfile and docker-compose.yml
Write deployment scripts
Set up GitHub Actions for CI/CD
Write unit tests for API endpoints
Implement integration tests for flows
Create user documentation
Set up monitoring and logging
Configure security headers and CORS
Implement rate limiting for API endpoints
Add health check endpoints
Create database schema for user management (if needed)
Set up backup and disaster recovery procedures
can we use google scret manager for authenticaiton

Yes, we can integrate Google Secret Manager into our authentication system. This will enhance security by keeping sensitive values like JWT keys and credentials out of code and configuration files. Here's how we can implement it:
Stage 3 Update: Integrating Google Secret Manager for Authentication
Google Secret Manager Integration
First, add the necessary dependency to requirements.txt:
Apply to Stage3.md
google-cloud-secret-manager>=2.12.0
Create a utility for accessing secrets in src/app/utils/secret_manager.py:
Apply to Stage3.md
from google.cloud import secretmanager
from src.app.config import settings

class SecretManagerClient:
    def __init__(self):
        self.client = secretmanager.SecretManagerServiceClient()
        self.project_id = settings.GCP_PROJECT_ID
    
    def get_secret(self, secret_id, version_id="latest"):
        """Retrieve a secret from Google Secret Manager.
        
        Args:
            secret_id: The ID of the secret to retrieve
            version_id: The version of the secret, defaults to "latest"
            
        Returns:
            The secret value as a string
        """
        name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version_id}"
        response = self.client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
Update the Settings class in src/app/config.py to use Secret Manager:
Apply to Stage3.md
from functools import lru_cache
from pydantic import BaseSettings
from src.app.utils.secret_manager import SecretManagerClient

class Settings(BaseSettings):
    # General settings
    APP_NAME: str = "Prompt Optimization Platform"
    GCP_PROJECT_ID: str
    
    # Secret keys with default dummy values for local development
    JWT_SECRET_KEY: str = "dummy_secret_key"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    # Other settings
    GCS_BUCKET_NAME: str
    VERTEX_LOCATION: str = "us-central1"
    
    # Flag to determine if we should load secrets from Secret Manager
    USE_SECRET_MANAGER: bool = False
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    settings = Settings()
    
    # If in production, load secrets from Secret Manager
    if settings.USE_SECRET_MANAGER:
        try:
            secret_client = SecretManagerClient()
            settings.JWT_SECRET_KEY = secret_client.get_secret("jwt-secret-key")
            # Load other secrets as needed
        except Exception as e:
            print(f"Error loading secrets: {e}")
            # Continue with default/env values if Secret Manager fails
    
    return settings

settings = get_settings()
Update the authentication module in src/app/auth.py to use these secrets:
Apply to Stage3.md
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from src.app.config import settings

# Models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    disabled: Optional[bool] = None

# Setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=settings.JWT_EXPIRATION_HOURS))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
        
    # In a real app, you'd fetch the user from a database
    # For now, we're using a simplified approach
    user = User(id="123", username=token_data.username, disabled=False)
    if user is None:
        raise credentials_exception
    return user
Create secrets in Google Secret Manager (CLI commands):
Apply to Stage3.md
Run
# Create the JWT secret key
echo "your-secure-jwt-secret" | gcloud secrets create jwt-secret-key \
    --replication-policy="automatic" \
    --data-file=-

# Grant service account access to secrets
gcloud secrets add-iam-policy-binding jwt-secret-key \
    --member="serviceAccount:your-service-account@your-project.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
Update the Dockerfile to include the Google Cloud credentials:
Apply to Stage3.md
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/

# Set environment variables
ENV PYTHONPATH=/app
ENV USE_SECRET_MANAGER=true

# Run the application
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
Update the deployment scripts to include the appropriate IAM setup:
Apply to Stage3.md
Run
#!/bin/bash
# setup-iam.sh

# Create a service account for the application
gcloud iam service-accounts create prompt-optimizer-sa \
    --display-name="Prompt Optimizer Service Account"

# Grant Secret Manager access
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:prompt-optimizer-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

# Grant other necessary permissions (Storage, Vertex AI, etc.)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:prompt-optimizer-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:prompt-optimizer-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Create and download a key file (for local development)
gcloud iam service-accounts keys create key.json \
    --iam-account=prompt-optimizer-sa@$PROJECT_ID.iam.gserviceaccount.com
To-Do List (Updated):
Install Google Secret Manager client library
Create SecretManagerClient utility class
Update Settings class to use Secret Manager
Modify authentication module to use secured JWT keys
Create necessary secrets in Google Secret Manager
Update IAM permissions for service accounts
Implement Pydantic models in src/api/models.py
Create API routers for endpoints in src/api/endpoints/
Implement JWT authentication in src/app/auth.py
Add token endpoint to src/app/main.py
Create service layer classes in src/app/services/
Develop monitoring dashboard templates
Add static assets for dashboard
Connect templates to FastAPI routes
Create Dockerfile and docker-compose.yml
Write deployment scripts
Set up GitHub Actions for CI/CD
Write unit tests for API endpoints
Implement integration tests for flows
Create user documentation