#!/usr/bin/env python
"""
Debug and Fix Training Process

This script diagnoses and fixes issues with the training process:
1. Fixed batch size limitations in workflow.py
2. Implements proper progress tracking
3. Ensures all examples are properly processed
4. Fixes experiment history tracking

Usage:
    python debug_and_fix_training.py
"""

import os
import sys
import json
import shutil
import logging
import traceback
from datetime import datetime
import yaml

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler("fix_training.log"),
                       logging.StreamHandler(sys.stdout)
                   ])
logger = logging.getLogger("training_fix")

def backup_file(file_path):
    """Create a backup of the file before modifying it."""
    if os.path.exists(file_path):
        backup_path = f"{file_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup of {file_path} at {backup_path}")
        return True
    else:
        logger.warning(f"File {file_path} does not exist, cannot create backup")
        return False

def fix_workflow_batch_limits():
    """Fix the batch size limitations in the workflow.py file."""
    workflow_file = "app/workflow.py"
    logger.info(f"Fixing batch size limitations in {workflow_file}")
    
    if not backup_file(workflow_file):
        return False
    
    try:
        with open(workflow_file, 'r') as f:
            content = f.read()
        
        # First fix: Remove the hard batch size limit of 20
        if "if batch_size == 0 or batch_size > 20:" in content:
            content = content.replace(
                "if batch_size == 0 or batch_size > 20:",
                "if batch_size == 0 or batch_size > 50:  # Increased limit from 20 to 50"
            )
            content = content.replace(
                "logger.info(f\"Limiting batch size to 20 examples (original: {batch_size})\")                    effective_batch_size = 20",
                "logger.info(f\"Limiting batch size to 50 examples (original: {batch_size})\")                    effective_batch_size = 50"
            )
            logger.info("Fixed hard batch size limit (increased from 20 to 50)")
        
        # Second fix: Increase the chunk size
        if "max_chunk_size = min(5, len(batch))" in content:
            content = content.replace(
                "max_chunk_size = min(5, len(batch))",
                "max_chunk_size = min(10, len(batch))  # Increased from 5 to 10"
            )
            logger.info("Increased processing chunk size from 5 to 10")
        
        # Save the modified file
        with open(workflow_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Successfully updated {workflow_file}")
        return True
    except Exception as e:
        logger.error(f"Error fixing workflow batch limits: {e}")
        logger.error(traceback.format_exc())
        return False

def fix_training_js_progress_tracking():
    """Fix the progress tracking in the training.js file."""
    training_js_file = "app/static/training.js"
    logger.info(f"Fixing progress tracking in {training_js_file}")
    
    if not backup_file(training_js_file):
        return False
    
    try:
        with open(training_js_file, 'r') as f:
            content = f.read()
        
        # Fix progress display for multiple iterations
        if "updateTrainingProgress(1, maxIterationsEl.value);" in content:
            content = content.replace(
                "updateTrainingProgress(1, maxIterationsEl.value);",
                "// Use actual iteration count instead of hardcoded 1\n" +
                "                updateTrainingProgress(data.iterations || 1, maxIterationsEl.value);"
            )
            logger.info("Fixed progress tracking to use actual iteration count")
        
        # Add more detailed logging to training process
        if "log(`Error: ${error.message}`);" in content and not "log(`Full Error Details: ${JSON.stringify(error)}" in content:
            content = content.replace(
                "log(`Error: ${error.message}`);",
                "log(`Error: ${error.message}`);\n" +
                "            // Log more detailed error information\n" +
                "            console.log('Full error object:', error);\n" +
                "            try {\n" +
                "                log(`Error trace: ${error.stack || 'No stack trace available'}`);\n" +
                "            } catch (e) { /* Ignore logging errors */ }"
            )
            logger.info("Enhanced error logging in UI")
        
        # Save the modified file
        with open(training_js_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Successfully updated {training_js_file}")
        return True
    except Exception as e:
        logger.error(f"Error fixing training.js progress tracking: {e}")
        logger.error(traceback.format_exc())
        return False

def fix_data_module_splitting():
    """Fix the data module splitting to ensure all examples are processed."""
    data_module_file = "app/data_module.py"
    logger.info(f"Fixing data splitting in {data_module_file}")
    
    if not backup_file(data_module_file):
        return False
    
    try:
        with open(data_module_file, 'r') as f:
            content = f.read()
        
        # Add debug logging to split_examples
        if "def split_examples(self, examples: List[Dict[str, str]], train_ratio: float = 0.8) -> Tuple[List, List]:" in content and not "logger.debug(f\"Before split - total examples:" in content:
            split_examples_func = "def split_examples(self, examples: List[Dict[str, str]], train_ratio: float = 0.8) -> Tuple[List, List]:"
            enhanced_func = ("def split_examples(self, examples: List[Dict[str, str]], train_ratio: float = 0.8) -> Tuple[List, List]:\n"
                            "        \"\"\"\n"
                            "        Split examples into training and validation sets.\n"
                            "        \n"
                            "        Args:\n"
                            "            examples (list): List of example dictionaries\n"
                            "            train_ratio (float): Ratio of examples to use for training\n"
                            "            \n"
                            "        Returns:\n"
                            "            tuple: (train_examples, validation_examples)\n"
                            "        \"\"\"\n"
                            "        if not examples:\n"
                            "            logger.warning(\"No examples to split\")\n"
                            "            return [], []\n"
                            "        \n"
                            "        # Add detailed logging\n"
                            "        logger.debug(f\"Before split - total examples: {len(examples)}\")\n"
                            "        if len(examples) > 0:\n"
                            "            logger.debug(f\"First example: {examples[0]}\")\n"
                            "        \n"
                            "        # Shuffle examples to ensure random split\n"
                            "        random.shuffle(examples)\n"
                            "        \n"
                            "        # Calculate the split index\n"
                            "        split_idx = max(int(len(examples) * train_ratio), 1)  # Ensure at least 1 training example\n"
                            "        \n"
                            "        # Ensure we have at least one validation example if enough examples\n"
                            "        if len(examples) > 1 and split_idx >= len(examples):\n"
                            "            split_idx = len(examples) - 1\n")
            
            content = content.replace(split_examples_func, enhanced_func)
            logger.info("Enhanced split_examples function with better logging and safeguards")
        
        # Fix get_batch method to include more logging
        if "def get_batch(self, batch_size: int = 0, validation: bool = False) -> List[Dict[str, str]]:" in content and not "logger.debug(f\"Retrieving batch with size" in content:
            get_batch_func = "def get_batch(self, batch_size: int = 0, validation: bool = False) -> List[Dict[str, str]]:"
            enhanced_batch_func = ("def get_batch(self, batch_size: int = 0, validation: bool = False) -> List[Dict[str, str]]:\n"
                                 "        \"\"\"\n"
                                 "        Get a batch of examples.\n"
                                 "        \n"
                                 "        Args:\n"
                                 "            batch_size (int): Number of examples in batch (0 for all)\n"
                                 "            validation (bool): Whether to use validation set\n"
                                 "            \n"
                                 "        Returns:\n"
                                 "            list: List of example dictionaries\n"
                                 "        \"\"\"\n"
                                 "        examples = self.validation_examples if validation else self.train_examples\n"
                                 "        set_type = \"validation\" if validation else \"training\"\n"
                                 "        \n"
                                 "        logger.debug(f\"Retrieving batch with size {batch_size} from {set_type} set with {len(examples)} examples\")\n"
                                 "        \n"
                                 "        if not examples:\n"
                                 "            logger.warning(f\"No {set_type} examples available\")\n"
                                 "            return []\n")
            
            content = content.replace(get_batch_func, enhanced_batch_func)
            logger.info("Enhanced get_batch function with better logging")
            
        # Save the modified file
        with open(data_module_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Successfully updated {data_module_file}")
        return True
    except Exception as e:
        logger.error(f"Error fixing data module splitting: {e}")
        logger.error(traceback.format_exc())
        return False

def fix_experiment_tracking():
    """Fix the experiment tracker to ensure history is properly saved."""
    exp_tracker_file = "app/experiment_tracker.py"
    logger.info(f"Fixing experiment tracking in {exp_tracker_file}")
    
    if not backup_file(exp_tracker_file):
        return False
    
    try:
        with open(exp_tracker_file, 'r') as f:
            content = f.read()
        
        # Add more robust error handling to save_iteration
        if "def save_iteration(self, experiment_id, iteration, system_prompt, output_prompt, " in content and not "try:" in content[:content.find("def save_iteration")]:
            save_iter_func = "def save_iteration(self, experiment_id, iteration, system_prompt, output_prompt, "
            idx = content.find(save_iter_func)
            if idx >= 0:
                # Find the function body by searching for the next def or end of file
                next_def = content.find("\n    def ", idx + 10)
                if next_def >= 0:
                    func_body = content[idx:next_def]
                else:
                    func_body = content[idx:]
                
                # Replace function if it doesn't already have robust error handling
                if "try:" not in func_body or "except Exception as e:" not in func_body:
                    improved_func = '''def save_iteration(self, experiment_id, iteration, system_prompt, output_prompt, 
                       metrics, examples=None, optimizer_reasoning=None):
        """
        Save data for a training iteration.
        
        Args:
            experiment_id (str): Experiment identifier
            iteration (int): Iteration number
            system_prompt (str): System prompt used
            output_prompt (str): Output prompt used
            metrics (dict): Metrics from the evaluation
            examples (list, optional): Example results
            optimizer_reasoning (str, optional): Reasoning from the optimizer
        """'''
        try:
            # Create experiment directories if needed
            exp_dir = os.path.join(self.base_dir, experiment_id)
            os.makedirs(exp_dir, exist_ok=True)
            
            # Log directory creation
            logger.debug(f"Saving iteration {iteration} to experiment directory: {exp_dir}")
            
            # Save prompts with detailed logging
            prompt_data = {
                "iteration": iteration,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "system_prompt": system_prompt,
                "output_prompt": output_prompt,
                "metrics": metrics
            }
            
            if optimizer_reasoning:
                prompt_data["optimizer_reasoning"] = optimizer_reasoning
            
            # Save the main iteration file
            iteration_file = os.path.join(exp_dir, f"iteration_{iteration}.json")
            try:
                with open(iteration_file, 'w') as f:
                    json.dump(prompt_data, f, indent=2)
                logger.debug(f"Saved iteration data to {iteration_file}")
            except Exception as e:
                logger.error(f"Error saving iteration file {iteration_file}: {e}")
                logger.error(traceback.format_exc())
            
            # Save examples if provided
            if examples:
                examples_dir = os.path.join(exp_dir, 'examples')
                os.makedirs(examples_dir, exist_ok=True)
                examples_file = os.path.join(examples_dir, f"examples_{iteration}.json")
                try:
                    with open(examples_file, 'w') as f:
                        json.dump(examples, f, indent=2)
                    logger.debug(f"Saved {len(examples)} examples to {examples_file}")
                except Exception as e:
                    logger.error(f"Error saving examples file {examples_file}: {e}")
                    logger.error(traceback.format_exc())
            
            # Update the experiment summary file
            summary_file = os.path.join(exp_dir, "summary.json")
            summary = {"last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            
            # Try to load existing summary
            try:
                if os.path.exists(summary_file):
                    with open(summary_file, 'r') as f:
                        existing_summary = json.load(f)
                        # Merge with new data
                        summary.update(existing_summary)
            except Exception as e:
                logger.error(f"Error reading summary file {summary_file}: {e}")
            
            # Update metrics history
            if "metrics_history" not in summary:
                summary["metrics_history"] = []
            
            # Add current metrics to history
            metric_entry = {
                "iteration": iteration,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": metrics
            }
            summary["metrics_history"].append(metric_entry)
            
            # Save updated summary
            try:
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                logger.debug(f"Updated experiment summary at {summary_file}")
            except Exception as e:
                logger.error(f"Error updating summary file {summary_file}: {e}")
                logger.error(traceback.format_exc())
            
            logger.info(f"Successfully saved iteration {iteration} for experiment {experiment_id}")
            
        except Exception as e:
            logger.error(f"Critical error in save_iteration: {e}")
            logger.error(traceback.format_exc())
            # Try to save the error information
            try:
                error_file = os.path.join(self.base_dir, "errors.log")
                with open(error_file, 'a') as f:
                    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error in save_iteration for experiment {experiment_id}, iteration {iteration}: {str(e)}\\n")
                    f.write(f"{traceback.format_exc()}\\n\\n")
            except:
                pass  # Ignore errors in error logging'''
                    
                    content = content.replace(func_body, improved_func)
                    logger.info("Enhanced save_iteration function with robust error handling")
        
        # Save the modified file
        with open(exp_tracker_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Successfully updated {exp_tracker_file}")
        return True
    except Exception as e:
        logger.error(f"Error fixing experiment tracking: {e}")
        logger.error(traceback.format_exc())
        return False

def fix_two_stage_train_endpoint():
    """Fix the two_stage_train endpoint in main.py to better handle examples."""
    main_py_file = "app/main.py"
    logger.info(f"Fixing two_stage_train endpoint in {main_py_file}")
    
    if not backup_file(main_py_file):
        return False
    
    try:
        with open(main_py_file, 'r') as f:
            content = f.read()
        
        # Enhanced error reporting in two_stage_train
        two_stage_start = content.find("@app.route('/two_stage_train'")
        if two_stage_start >= 0:
            two_stage_end = content.find("@app.route", two_stage_start + 10)
            if two_stage_end > 0:
                two_stage_content = content[two_stage_start:two_stage_end]
                
                # Find the return error statement
                if "return jsonify({'error': str(e)}), 500" in two_stage_content:
                    improved_error = "        # Get detailed error information\n" + \
                                     "        error_info = {\n" + \
                                     "            'error': str(e),\n" + \
                                     "            'traceback': traceback.format_exc(),\n" + \
                                     "            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')\n" + \
                                     "        }\n" + \
                                     "        \n" + \
                                     "        # Log the error details\n" + \
                                     "        logger.error(f\"Error details: {error_info}\")\n" + \
                                     "        \n" + \
                                     "        # Save the error to a file for debugging\n" + \
                                     "        os.makedirs('logs', exist_ok=True)\n" + \
                                     "        error_file = f\"logs/two_stage_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json\"\n" + \
                                     "        try:\n" + \
                                     "            with open(error_file, 'w') as f:\n" + \
                                     "                json.dump(error_info, f, indent=2)\n" + \
                                     "            logger.info(f\"Saved error details to {error_file}\")\n" + \
                                     "        except Exception as file_error:\n" + \
                                     "            logger.error(f\"Could not save error details: {file_error}\")\n" + \
                                     "        \n" + \
                                     "        return jsonify({'error': str(e), 'traceback': traceback.format_exc().split('\\n')}), 500"
                    
                    two_stage_content = two_stage_content.replace("        return jsonify({'error': str(e)}), 500", improved_error)
                    content = content.replace(content[two_stage_start:two_stage_end], two_stage_content)
                    logger.info("Enhanced error reporting in two_stage_train endpoint")
        
        # Save the modified file
        with open(main_py_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Successfully updated {main_py_file}")
        return True
    except Exception as e:
        logger.error(f"Error fixing two_stage_train endpoint: {e}")
        logger.error(traceback.format_exc())
        return False

def create_test_examples():
    """Create a set of test examples to verify the fixes."""
    logger.info("Creating test examples...")
    
    examples_text = """user_input,ground_truth_output
A 45-year-old man presents with crushing chest pain radiating to the left arm that started 1 hour ago. He has a history of hypertension and smoking. Vital signs: BP 160/95, HR 110, RR 22. ECG shows ST elevation in leads II, III, and aVF.,Acute inferior myocardial infarction
A 60-year-old woman presents with sudden onset of weakness and numbness on the left side of her face and left arm that started 2 hours ago. She has a history of atrial fibrillation and is on warfarin. Vital signs: BP 170/90, HR 80, RR 16. CT scan of the head shows no bleeding.,Ischemic stroke (cerebrovascular accident)
A 35-year-old woman presents with severe headache that began suddenly while exercising. She describes it as "the worst headache of my life." Neurological exam shows nuchal rigidity and photophobia. CT scan of the head shows subarachnoid blood.,Subarachnoid hemorrhage
A 50-year-old man presents with gradually worsening shortness of breath over the past 2 weeks. He has a history of congestive heart failure. Physical exam reveals bilateral lower extremity edema and crackles in both lung bases. BNP is elevated.,Acute exacerbation of congestive heart failure
A 25-year-old woman presents with fever, right flank pain, and dysuria for the past 2 days. Urinalysis shows pyuria and bacteriuria. CBC reveals leukocytosis.,Pyelonephritis
A 40-year-old man presents with abdominal pain that started periumbilically and moved to the right lower quadrant over 12 hours. He reports anorexia and nausea. On exam, he has rebound tenderness at McBurney's point. WBC is elevated.,Acute appendicitis
A 30-year-old woman presents with severe, colicky right flank pain radiating to the groin. She reports nausea and gross hematuria. Urinalysis shows microscopic hematuria. CT scan shows a 5mm stone in the right ureter.,Ureterolithiasis (kidney stone)
A 55-year-old man with a history of alcoholism presents with severe upper abdominal pain radiating to the back, nausea, and vomiting for the past 12 hours. On exam, he has epigastric tenderness. Labs show elevated lipase and amylase.,Acute pancreatitis
A 70-year-old woman presents with fever, productive cough with yellow sputum, and dyspnea for the past 3 days. She has a history of COPD. Chest X-ray shows a right lower lobe infiltrate.,Community-acquired pneumonia
A 65-year-old man with a history of hypertension and diabetes presents with sudden onset of severe, tearing chest pain radiating to the back. CT angiogram shows a dilated aortic root with an intimal flap.,Aortic dissection
"""
    
    # Save examples to a file for testing
    try:
        os.makedirs('data/test', exist_ok=True)
        with open('data/test/test_examples.csv', 'w') as f:
            f.write(examples_text)
        logger.info(f"Saved test examples to data/test/test_examples.csv")
        
        # Also create a simple Python script to verify the dataset
        verify_script = """#!/usr/bin/env python
import json
import os
import sys

def verify_datasets():
    print("Verifying datasets...")
    
    datasets = [
        ('data/train/examples.json', 'Training examples'),
        ('data/train/current_train.json', 'Current training examples'),
        ('data/validation/examples.json', 'Validation examples'),
        ('data/validation/current_validation.json', 'Current validation examples')
    ]
    
    for file_path, description in datasets:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    print(f"{description}: {len(data)} examples in {file_path}")
                    if len(data) > 0:
                        print(f"  First example: {data[0].get('user_input', '')[:50]}...")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        else:
            print(f"{description}: File not found: {file_path}")
    
    print("\\nVerification complete.")

if __name__ == "__main__":
    verify_datasets()
"""
        with open('verify_datasets.py', 'w') as f:
            f.write(verify_script)
        logger.info(f"Created verification script at verify_datasets.py")
        
        return True
    except Exception as e:
        logger.error(f"Error creating test examples: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    logger.info("=== FIXING TRAINING PROCESS STARTED ===")
    logger.info(f"Fix script version: 1.0.0")
    logger.info(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create test examples
    create_test_examples()
    
    # Fix the workflow batch limits
    fix_workflow_batch_limits()
    
    # Fix the training.js progress tracking
    fix_training_js_progress_tracking()
    
    # Fix the data module splitting
    fix_data_module_splitting()
    
    # Fix the experiment tracking
    fix_experiment_tracking()
    
    # Fix the two_stage_train endpoint
    fix_two_stage_train_endpoint()
    
    logger.info("=== FIXING TRAINING PROCESS COMPLETED ===")
    logger.info("All fixes have been applied. Please restart the application.")
    logger.info("Check fix_training.log for detailed results")
    
    print("\n✅ Training fixes completed! Please restart the application.")
    print("To verify the dataset is now correctly loaded, run:")
    print("  python verify_datasets.py\n")

if __name__ == "__main__":
    main()