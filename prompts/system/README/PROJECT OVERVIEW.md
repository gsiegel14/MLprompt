Cursor Rules: ML Prompt Optimization Platform
Project Structure
Directory Organization
All database models must reside in src/app/models/database_models.py
All repositories must be in src/app/repositories/ directory
ML components must be stored in src/app/ml/ directory
Prefect flows and tasks must be organized in src/flows/
Configuration Management
Store all environment variables in .env file (never commit to repository)
Use config.py for accessing configuration, never direct environment variables
Database Rules
Schema Management
All schema changes require Alembic migrations
Run python -m src.cli create_migration "descriptive_name" for changes
Document all table relationships in code comments
PostgreSQL Best Practices
Include appropriate indexes on frequently queried columns
Use UUID type for all primary keys
Store JSON data in JSONB columns, not TEXT
Include created_at timestamp on all tables
ML Model Management
Feature Engineering
Extract features using PromptFeatureExtractor class only
Store all feature values as floating point numbers
Always save feature vectors to database for reproducibility
Limit embedding dimensions to 20 features maximum
Meta-Learning Rules
Train LightGBM models with minimum 5 data points
Use early stopping with 10-round patience
Save models to disk AND record paths in database
Include model metrics in artifact creation
Reinforcement Learning Rules
Limit RL actions to the 10 predefined edit types
Set reasonable action limits (max 5 edits per prompt)
Train PPO models for minimum 5000 timesteps
Save both model and metrics after training
Prefect Workflow
Task Design
Each task must have descriptive name with name= parameter
Include retries parameter for all API-calling tasks
Always use get_run_logger() for logging
Return structured dictionaries, never raw objects
Flow Structure
Use the 5-step workflow pattern for all optimization flows
Include early stopping logic based on metric improvement
Save artifacts after significant steps
Pass object IDs between tasks, not full objects
API Integration
LLM Services
Use retry mechanism for all Vertex AI calls
Cache responses when possible to reduce costs
Include timeout handling for all external API calls
Track token usage and cost metrics
Request/Response Handling
Validate all incoming requests with Pydantic models
Handle API errors with appropriate HTTP status codes
Return consistent response formats across all endpoints
Include metrics and timing information in responses
Testing & Monitoring
Testing Requirements
Write unit tests for all repository functions
Create integration tests for database migrations
Test ML components with small synthetic datasets
Verify Prefect flows with mock API responses
Monitoring Rules
Log all ML training metrics to Weights & Biases
Create Prefect artifacts for all model versions
Implement database backup before major operations
Track performance metrics between prompt versions
Documentation
Code Documentation
Include docstrings for all functions and classes
Document parameter types and return values
Add examples for complex functions
Explain database relationships in models
User Instructions
Create README with setup instructions
Document CLI commands with examples
Provide troubleshooting guides for common issues
Include sample workflows for new users
Security & Compliance
Security Practices
Never store API keys in code
Use environment variables for sensitive configuration
Implement proper authentication for API endpoints
Sanitize all user inputs before processing
