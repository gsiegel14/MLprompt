README: ML Prompt Optimization Platform - 4-Call Workflow (Free Model Focus)
Version: 2.0 (4-Call Workflow)
Date: 2023-10-27
Goal: Build a system that iteratively refines a system_prompt and output_prompt for a Primary Task LLM. This refinement loop uses three additional LLM calls per cycle: an Evaluator LLM, an Optimizer LLM, and a Grader LLM. The primary objectives are to improve the Primary LLM's reasoning/accuracy while using free LLM resources (prioritizing local models like Ollama) and minimizing token usage within the constraints of the 4-call workflow.
Core Philosophy:
Four Distinct LLM Roles: Each cycle involves calls to: Primary, Evaluator, Optimizer, and Grader LLMs.
Free Model Priority: Implementation must support easily swappable LLM backends, prioritizing free options like local models via Ollama (e.g., Llama 3, Mistral, Phi-3) or rate-limited APIs. Cost is paramount.
Token Minimization: Strategies will be employed to reduce token count where possible, acknowledging the inherent overhead introduced by using LLMs for evaluation and grading.
Modular Design: Components (LLM clients, prompts, evaluation) should be easily configurable and testable.
The 4-Call Workflow Cycle:
README: ML Prompt Optimization Platform - 4-Call Workflow (Free Model Focus)
Version: 2.0 (4-Call Workflow)
Date: 2023-10-27
Goal: Build a system that iteratively refines a system_prompt and output_prompt for a Primary Task LLM. This refinement loop uses three additional LLM calls per cycle: an Evaluator LLM, an Optimizer LLM, and a Grader LLM. The primary objectives are to improve the Primary LLM's reasoning/accuracy while using free LLM resources (prioritizing local models like Ollama) and minimizing token usage within the constraints of the 4-call workflow.
Core Philosophy:
Four Distinct LLM Roles: Each cycle involves calls to: Primary, Evaluator, Optimizer, and Grader LLMs.
Free Model Priority: Implementation must support easily swappable LLM backends, prioritizing free options like local models via Ollama (e.g., Llama 3, Mistral, Phi-3) or rate-limited APIs. Cost is paramount.
Token Minimization: Strategies will be employed to reduce token count where possible, acknowledging the inherent overhead introduced by using LLMs for evaluation and grading.
Modular Design: Components (LLM clients, prompts, evaluation) should be easily configurable and testable.
The 4-Call Workflow Cycle:
Detailed Workflow Steps (Single Cycle):
Load State: Load the current system_prompt (vN) and output_prompt (vN). Load a batch of training data (user_input, ground_truth_output).
Primary LLM Execution (Loop): For each data row:
Call 1: Send system_prompt, output_prompt, and user_input to the Primary LLM. Receive model_response.
Call 2: Send model_response, ground_truth_output, and a specific evaluation prompt (e.g., "Does the response accurately match the ground truth? Score 0-1.") to the Evaluator LLM. Receive evaluation_result (e.g., score, critique text).
Store the tuple (user_input, ground_truth_output, model_response, evaluation_result).
Aggregate Evaluations: Collect all stored results for the batch. Calculate aggregate statistics if needed (e.g., average LLM-assigned score).
Optimizer LLM Execution:
Call 3: Prepare context including the aggregated evaluations, a sample of specific examples (input, GT, output, evaluation), the current prompts (vN), and the optimizer prompt (e.g., reasoning_improver_v2.txt). Send this to the Optimizer LLM. Receive proposed_system_prompt (vN+1) and proposed_output_prompt (vN+1).
Grader LLM Execution:
Call 4: Prepare context including the proposed_system_prompt (vN+1), proposed_output_prompt (vN+1), and a specific prompt grading prompt (e.g., "Rate the clarity, conciseness, and likely effectiveness of these prompts. Score 0-1."). Send this to the Grader LLM. Receive prompt_quality_score or feedback.
Acceptance & Update:
Implement logic based on the prompt_quality_score and potentially the change from previous validation scores (if using validation within the loop, though not strictly required by the description). A simple approach: accept if the quality score is above a threshold (e.g., > 0.5).
If accepted, the proposed_prompts (vN+1) become the current prompts for the next cycle.
If rejected, retain the prompts from the start of the cycle (vN).
Logging: Log all inputs, outputs, evaluations, proposed prompts, grading scores, and acceptance decisions to W&B.
LLM Roles & Specific Prompts Needed:
Primary LLM: Executes the main task. Uses the evolving system_prompt and output_prompt.
Evaluator LLM: Scores/critiques the Primary LLM's output against ground truth. Requires its own specific prompt defining the evaluation criteria and desired output format (e.g., a score, a JSON object, critique text). This is a major source of token cost.
Optimizer LLM: Analyzes batch performance and proposes improved prompts. Uses the reasoning_improver_v2.txt template (or similar) combined with runtime data.
Grader LLM: Assesses the quality of the proposed prompts. Requires its own specific prompt defining prompt quality criteria (clarity, non-overfitting, conciseness) and desired output format (e.g., score, structured feedback). This adds token cost per cycle.
Technology Stack Adaptation:
LLM Client (app/llm_client.py): Crucial Change: Use a library like litellm to create a unified interface capable of calling various backends (Ollama, Hugging Face, Groq, Vertex AI, OpenAI, Anthropic etc.) based on configuration. The DSPyVertexAI wrapper needs to be replaced or generalized using litellm as the backend for dspy.LM.
Apply to APIworflow
]
Detailed Workflow Steps (Single Cycle):
Load State: Load the current system_prompt (vN) and output_prompt (vN). Load a batch of training data (user_input, ground_truth_output).
Primary LLM Execution (Loop): For each data row:
Call 1: Send system_prompt, output_prompt, and user_input to the Primary LLM. Receive model_response.
Call 2: Send model_response, ground_truth_output, and a specific evaluation prompt (e.g., "Does the response accurately match the ground truth? Score 0-1.") to the Evaluator LLM. Receive evaluation_result (e.g., score, critique text).
Store the tuple (user_input, ground_truth_output, model_response, evaluation_result).
Aggregate Evaluations: Collect all stored results for the batch. Calculate aggregate statistics if needed (e.g., average LLM-assigned score).
Optimizer LLM Execution:
Call 3: Prepare context including the aggregated evaluations, a sample of specific examples (input, GT, output, evaluation), the current prompts (vN), and the optimizer prompt (e.g., reasoning_improver_v2.txt). Send this to the Optimizer LLM. Receive proposed_system_prompt (vN+1) and proposed_output_prompt (vN+1).
Grader LLM Execution:
Call 4: Prepare context including the proposed_system_prompt (vN+1), proposed_output_prompt (vN+1), and a specific prompt grading prompt (e.g., "Rate the clarity, conciseness, and likely effectiveness of these prompts. Score 0-1."). Send this to the Grader LLM. Receive prompt_quality_score or feedback.
Acceptance & Update:
Implement logic based on the prompt_quality_score and potentially the change from previous validation scores (if using validation within the loop, though not strictly required by the description). A simple approach: accept if the quality score is above a threshold (e.g., > 0.5).
If accepted, the proposed_prompts (vN+1) become the current prompts for the next cycle.
If rejected, retain the prompts from the start of the cycle (vN).
Logging: Log all inputs, outputs, evaluations, proposed prompts, grading scores, and acceptance decisions to W&B.
LLM Roles & Specific Prompts Needed:
Primary LLM: Executes the main task. Uses the evolving system_prompt and output_prompt.
Evaluator LLM: Scores/critiques the Primary LLM's output against ground truth. Requires its own specific prompt defining the evaluation criteria and desired output format (e.g., a score, a JSON object, critique text). This is a major source of token cost.
Optimizer LLM: Analyzes batch performance and proposes improved prompts. Uses the reasoning_improver_v2.txt template (or similar) combined with runtime data.
Grader LLM: Assesses the quality of the proposed prompts. Requires its own specific prompt defining prompt quality criteria (clarity, non-overfitting, conciseness) and desired output format (e.g., score, structured feedback). This adds token cost per cycle.
Technology Stack Adaptation:
LLM Client (app/llm_client.py): Crucial Change: Use a library like litellm to create a unified interface capable of calling various backends (Ollama, Hugging Face, Groq, Vertex AI, OpenAI, Anthropic etc.) based on configuration. The DSPyVertexAI wrapper needs to be replaced or generalized using litellm as the backend for dspy.LM.    # Example using litellm within a DSPy LM wrapper
    import dspy
    from litellm import completion
    import time

    class DSPyLitellm(dspy.LM):
        def __init__(self, model: str, api_base: str | None = None, api_key: str | None = None, **kwargs):
            super().__init__(model)
            self.model = model
            self.api_base = api_base # e.g., http://localhost:11434 for Ollama
            self.api_key = api_key # If needed for HF Inference Endpoints etc.
            self.kwargs = kwargs # Default generation args (temp, max_tokens)

        def basic_request(self, prompt: str, **kwargs):
            request_kwargs = {**self.kwargs, **kwargs}
            # Structure messages for chat models if needed
            messages = [{"role": "user", "content": prompt}]
            start_time = time.time()
            response = completion(
                model=self.model,
                messages=messages,
                api_base=self.api_base,
                api_key=self.api_key,
                **request_kwargs
            )
            end_time = time.time()
            # Extract text - adjust based on litellm response structure
            response_text = response.choices[0].message.content
            self.history.append({
                 "prompt": prompt, "response": response_text, "kwargs": request_kwargs,
                 "response_time_ms": (end_time-start_time)*1000
            })
            return {"choices": [{"text": response_text}]} # Simple format for DSPy

        def __call__(self, prompt: str, **kwargs):
            response = self.basic_request(prompt, **kwargs)
            return [choice["text"] for choice in response["choices"]]Configuration (app/config.py): Add sections for different LLM configurations (model names per role, API bases/keys if not local).    # Example additions to config.py
    class LLMConfig(BaseSettings):
        model: str # e.g., "ollama/llama3", "huggingface/mistralai/Mistral-7B-Instruct-v0.1"
        api_base: Optional[str] = None
        api_key: Optional[SecretStr] = None # Use SecretStr for keys
        # Add default generation args here
        temperature: float = 0.0
        max_tokens: int = 512

    class LLMRolesConfig(BaseSettings):
        primary: LLMConfig
        evaluator: LLMConfig
        optimizer: LLMConfig
        grader: LLMConfig

    class AppConfig(BaseSettings):
        llm_roles: LLMRolesConfig = Field(...)
        # ... other configs ...
        Orchestration (Prefect): Remains suitable for managing the multi-step cycle.
Tracking (W&B): Remains suitable for logging metrics, prompts, and outputs from all 4 LLM calls.
Token Minimization Strategies (Within 4-Call Constraint):
Optimizer Context: Send very few examples (optimizer_examples_k=1-3). Prioritize failures. Consider summarizing metrics instead of sending raw scores if feasible.
Prompt Conciseness: Keep the instructions for the Evaluator, Optimizer, and Grader LLMs themselves as short and direct as possible.
Primary Prompts: Encourage the Optimizer to generate concise system_prompt and output_prompt. The Grader can check for this.
Model Choice: Use smaller, faster free models (e.g., Phi-3 Mini, Gemma 2B via Ollama) for Evaluator and Grader roles if their task complexity allows, saving tokens/time compared to larger models.
Quantization: Use quantized versions of local models (e.g., Q4_K_M GGUF) via Ollama to reduce memory/potentially speed up local inference.
(Advanced): Explore prompt compression techniques if token limits become severe (requires more complex implementation).
Free Model Considerations:
Setup: Requires installing and running Ollama locally, pulling desired models (ollama pull llama3). Document this clearly.
Consistency: Free/local models, especially smaller ones, may be less consistent than large commercial models in nuanced tasks like evaluation, optimization, and grading. Expect more variability and potentially slower convergence.
Quality Ceiling: The quality of the final optimized prompts may be limited by the capabilities of the free models used, particularly the Optimizer.
Resource Needs: Running multiple local models requires significant RAM/VRAM.
Implementation Phases (Revised):
Phase 0: Setup (include Ollama installation/model pulling, litellm dependency). Configure LLMRolesConfig.
Phase 1: Build litellm-based DSPyLitellm wrapper. Implement data_module.
Phase 2: Define DSPy Signature/Module for Primary task. Implement Programmatic Evaluator (evaluator.py) first as a baseline. Define draft prompts for LLM Evaluator, Optimizer (reasoning_improver_v2.txt), and Grader roles (save as text files).
Phase 3: Implement the 4-call workflow logic within a Prefect flow (scripts/compile_prompts.py), using the DSPyLitellm wrapper to call the configured models for each role. Add the acceptance logic based on the Grader output.
Phase 4: Integrate W&B logging for all 4 LLM call details, intermediate results, final prompts, and acceptance decisions.
Phase 5: Testing: Critical for this workflow.
Unit test core components.
Test the litellm client against actual local/free endpoints.
Compare results using the LLM Evaluator vs. the Programmatic Evaluator on a small dataset. Assess consistency and cost.
Test the Grader logic.
Phase 6: Implement run_inference.py for separate validation.
Trade-offs & Alternatives:
LLM Evaluation (Call 2) vs. Programmatic: Using an LLM for evaluation is token-intensive (call per example) and potentially less reliable/consistent than code, especially with smaller free models. Strongly consider keeping a programmatic evaluator (exact_match, regex, embedding similarity) as the default or a configurable option for significant token/cost savings and improved reliability.
LLM Grading (Call 4): Adds complexity and token cost for potentially subjective feedback. Its necessity depends on whether the Optimizer frequently produces poor-quality prompts. Could be made optional or use a very small/fast model.
Performance: Local models will be slower than cloud APIs unless running on high-end hardware. The 4-call sequence adds significant latency per cycle compared to a 2-call (Primary+Optimizer) loop.
