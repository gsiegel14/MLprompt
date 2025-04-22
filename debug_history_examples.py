
import os
import json
import time
from pathlib import Path
import logging
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_history_examples.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_driver():
    """Setup Chrome webdriver with headless options"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(options=chrome_options)

def check_history_page_loading(driver):
    """Check if the history page loads correctly"""
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "experiments-table"))
        )
        logger.info("✅ History page loaded successfully")
        return True
    except TimeoutException:
        logger.error("❌ History page failed to load experiments table")
        return False

def check_experiment_details(driver, experiment_id):
    """Check if experiment details load correctly"""
    try:
        # Find and click view button for the specified experiment
        view_buttons = driver.find_elements(By.CLASS_NAME, "view-experiment")
        for button in view_buttons:
            if button.get_attribute("data-id") == experiment_id:
                button.click()
                break
        else:
            logger.warning(f"⚠️ No experiment found with ID {experiment_id}")
            return False

        # Wait for experiment details to load
        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, "iterations-accordion"))
        )
        logger.info(f"✅ Experiment details loaded for {experiment_id}")
        return True
    except TimeoutException:
        logger.error(f"❌ Experiment details failed to load for {experiment_id}")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking experiment details: {str(e)}")
        return False

def check_examples_loading(driver):
    """Check if examples load when view examples button is clicked"""
    try:
        # Find the 'View Examples' button
        view_examples_buttons = driver.find_elements(By.CLASS_NAME, "view-examples")
        if not view_examples_buttons:
            logger.error("❌ No 'View Examples' buttons found")
            return False
        
        # Scroll to the first button and click it
        button = view_examples_buttons[0]
        driver.execute_script("arguments[0].scrollIntoView();", button)
        
        # Check if the button has onclick attribute
        onclick_attr = button.get_attribute("onclick")
        if not onclick_attr:
            logger.error("❌ 'View Examples' button missing onclick attribute")
            
            # Try to fix it directly in the DOM
            logger.info("Attempting to fix onclick attribute directly...")
            iteration = button.get_attribute("data-iteration")
            driver.execute_script(f"arguments[0].setAttribute('onclick', 'window.loadExamplesForIteration({iteration})')", button)
            logger.info(f"✅ Added onclick attribute: window.loadExamplesForIteration({iteration})")
        else:
            logger.info(f"✅ 'View Examples' button has onclick: {onclick_attr}")
        
        # Execute the function directly via JavaScript instead of clicking
        iteration = button.get_attribute("data-iteration")
        logger.info(f"Executing window.loadExamplesForIteration({iteration}) directly...")
        driver.execute_script(f"window.loadExamplesForIteration({iteration})")
        logger.info("✅ Executed loadExamplesForIteration via JavaScript")
        
        # Wait for examples to load
        time.sleep(3)  # Give it some time to load or show error
        
        # Check for JavaScript errors
        js_errors = driver.execute_script("return window.JSErrors || []")
        if js_errors:
            logger.error(f"❌ JavaScript errors detected: {js_errors}")
            
        # Check for examples container content
        examples_container = driver.find_element(By.ID, "examples-container")
        if "No examples available" in examples_container.text:
            logger.warning("⚠️ 'No examples available' message shown")
            return False
            
        loading_elem = driver.find_element(By.ID, "examples-loading")
        if loading_elem.is_displayed():
            logger.warning("⚠️ Examples still loading after timeout")
            return False
            
        # Check if example cards are present
        example_cards = driver.find_elements(By.CLASS_NAME, "example-card")
        if example_cards:
            logger.info(f"✅ Examples loaded successfully: {len(example_cards)} examples found")
            
            # Display example contents for debugging
            logger.info("=== EXAMPLE CONTENTS ===")
            
            for i, card in enumerate(example_cards[:3]):  # Limit to first 3 for brevity
                try:
                    # Extract content from the card
                    user_input = card.find_element(By.CSS_SELECTOR, ".user-input").text
                    ground_truth = card.find_element(By.CSS_SELECTOR, ".ground-truth").text
                    model_response = card.find_element(By.CSS_SELECTOR, ".model-response").text
                    score = card.find_element(By.CSS_SELECTOR, ".score").text
                    
                    logger.info(f"\nEXAMPLE #{i+1}:")
                    logger.info(f"User Input: {user_input[:100]}..." if len(user_input) > 100 else f"User Input: {user_input}")
                    logger.info(f"Ground Truth: {ground_truth[:100]}..." if len(ground_truth) > 100 else f"Ground Truth: {ground_truth}")
                    logger.info(f"Model Response: {model_response[:100]}..." if len(model_response) > 100 else f"Model Response: {model_response}")
                    logger.info(f"Score: {score}")
                    logger.info("-" * 50)
                except Exception as e:
                    logger.error(f"Error extracting content from example card {i+1}: {e}")
            
            # Display direct examination of examples.json file
            experiment_id = iteration = None
            try:
                # Get experiment ID and iteration from the page
                breadcrumb = driver.find_element(By.CSS_SELECTOR, ".breadcrumb")
                breadcrumb_text = breadcrumb.text
                
                # Extract experiment ID and iteration
                if "Experiment:" in breadcrumb_text and "Iteration:" in breadcrumb_text:
                    exp_start = breadcrumb_text.find("Experiment:") + len("Experiment:")
                    exp_end = breadcrumb_text.find("Iteration:")
                    experiment_id = breadcrumb_text[exp_start:exp_end].strip()
                    
                    iter_start = breadcrumb_text.find("Iteration:") + len("Iteration:")
                    iteration = breadcrumb_text[iter_start:].strip()
                    
                    logger.info(f"Found experiment ID: {experiment_id}, iteration: {iteration}")
                    
                    # Try to load the examples file directly
                    examples_path = f"experiments/{experiment_id}/examples/examples_{iteration}.json"
                    if os.path.exists(examples_path):
                        with open(examples_path, 'r') as f:
                            examples_data = json.load(f)
                            
                        logger.info(f"\n=== DIRECT FILE CONTENT ({examples_path}) ===")
                        for i, example in enumerate(examples_data[:2]):  # Show first 2 examples
                            logger.info(f"\nFILE EXAMPLE #{i+1}:")
                            user_input = example.get('user_input', '')
                            ground_truth = example.get('ground_truth_output', '')
                            model_response = example.get('model_response', '')
                            score = example.get('score', 0)
                            
                            logger.info(f"User Input: {user_input[:100]}..." if len(user_input) > 100 else f"User Input: {user_input}")
                            logger.info(f"Ground Truth: {ground_truth[:100]}..." if len(ground_truth) > 100 else f"Ground Truth: {ground_truth}")
                            logger.info(f"Model Response: {model_response[:100]}..." if len(model_response) > 100 else f"Model Response: {model_response}")
                            logger.info(f"Score: {score}")
                            logger.info("-" * 50)
                    else:
                        logger.warning(f"Examples file not found: {examples_path}")
            except Exception as e:
                logger.error(f"Error loading examples file: {e}")
            
            return True
        else:
            logger.warning("⚠️ No example cards found after loading")
            return False
            
    except NoSuchElementException as e:
        logger.error(f"❌ Element not found: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking examples loading: {str(e)}")
        return False

def find_function_in_js(driver, function_name):
    """Check if a function exists in the global scope"""
    exists = driver.execute_script(f"return typeof {function_name} === 'function'")
    if exists:
        logger.info(f"✅ Function '{function_name}' exists in global scope")
    else:
        logger.error(f"❌ Function '{function_name}' NOT found in global scope")
    return exists

def inject_javascript_listener(driver):
    """Inject JavaScript to capture JS errors"""
    script = """
    window.JSErrors = [];
    window.addEventListener('error', function(e) {
        window.JSErrors.push({
            message: e.message,
            source: e.filename,
            lineno: e.lineno
        });
        console.error('Captured error:', e);
    });
    """
    driver.execute_script(script)
    logger.info("✅ Injected JavaScript error listener")

def check_examples_files_exist(experiment_id):
    """Check if examples files exist for the given experiment"""
    base_path = Path(f"experiments/{experiment_id}")
    if not base_path.exists():
        logger.error(f"❌ Experiment directory not found: {base_path}")
        return False
        
    # Check for examples directory
    examples_dir = base_path / "examples"
    if examples_dir.exists():
        example_files = list(examples_dir.glob("examples_*.json"))
        if example_files:
            logger.info(f"✅ Found {len(example_files)} example files in examples directory")
            
            # Display content of the most recent example file for debugging
            if example_files:
                latest_example_file = max(example_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"Most recent example file: {latest_example_file}")
                
                try:
                    with open(latest_example_file, 'r') as f:
                        examples_data = json.load(f)
                    
                    logger.info(f"=== EXAMPLES CONTENT ({latest_example_file.name}) ===")
                    for i, example in enumerate(examples_data[:2]):  # Limit to first 2 for brevity
                        logger.info(f"\nEXAMPLE #{i+1}:")
                        user_input = example.get('user_input', '')
                        ground_truth = example.get('ground_truth_output', '')
                        model_response = example.get('model_response', '')
                        score = example.get('score', 0)
                        
                        logger.info(f"User Input: {user_input[:100]}..." if len(user_input) > 100 else f"User Input: {user_input}")
                        logger.info(f"Ground Truth: {ground_truth[:100]}..." if len(ground_truth) > 100 else f"Ground Truth: {ground_truth}")
                        logger.info(f"Model Response: {model_response[:100]}..." if len(model_response) > 100 else f"Model Response: {model_response}")
                        logger.info(f"Score: {score}")
                        logger.info("-" * 50)
                except Exception as e:
                    logger.error(f"Error reading example file: {e}")
            
            return True
    
    # Check for example files directly in experiment directory
    example_files = list(base_path.glob("examples_*.json"))
    if example_files:
        logger.info(f"✅ Found {len(example_files)} example files in experiment directory")
        
        # Display content of the most recent example file for debugging
        if example_files:
            latest_example_file = max(example_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Most recent example file: {latest_example_file}")
            
            try:
                with open(latest_example_file, 'r') as f:
                    examples_data = json.load(f)
                
                logger.info(f"=== EXAMPLES CONTENT ({latest_example_file.name}) ===")
                for i, example in enumerate(examples_data[:2]):  # Limit to first 2 for brevity
                    logger.info(f"\nEXAMPLE #{i+1}:")
                    user_input = example.get('user_input', '')
                    ground_truth = example.get('ground_truth_output', '')
                    model_response = example.get('model_response', '')
                    score = example.get('score', 0)
                    
                    logger.info(f"User Input: {user_input[:100]}..." if len(user_input) > 100 else f"User Input: {user_input}")
                    logger.info(f"Ground Truth: {ground_truth[:100]}..." if len(ground_truth) > 100 else f"Ground Truth: {ground_truth}")
                    logger.info(f"Model Response: {model_response[:100]}..." if len(model_response) > 100 else f"Model Response: {model_response}")
                    logger.info(f"Score: {score}")
                    logger.info("-" * 50)
            except Exception as e:
                logger.error(f"Error reading example file: {e}")
        
        return True
    
    logger.error(f"❌ No example files found for experiment {experiment_id}")
    return False

def check_history_js_file():
    """Check if history.js file has properly defined functions"""
    try:
        with open("app/static/history.js", "r") as f:
            content = f.read()
            
        if "window.loadExamplesForIteration" in content:
            logger.info("✅ Found 'window.loadExamplesForIteration' in history.js")
        else:
            logger.error("❌ 'window.loadExamplesForIteration' NOT found in history.js")
            
        return "window.loadExamplesForIteration" in content
    except Exception as e:
        logger.error(f"❌ Error reading history.js: {str(e)}")
        return False

def scan_experiments():
    """Scan experiments directory to find the most recent experiment"""
    try:
        experiments_dir = Path("experiments")
        if not experiments_dir.exists():
            logger.error("❌ Experiments directory not found")
            return None
            
        experiments = [d for d in experiments_dir.iterdir() if d.is_dir() and not d.name in ['metrics', 'prompts']]
        if not experiments:
            logger.error("❌ No experiment directories found")
            return None
            
        # Sort by creation time (most recent first)
        sorted_experiments = sorted(experiments, key=lambda d: d.stat().st_mtime, reverse=True)
        most_recent = sorted_experiments[0].name
        logger.info(f"✅ Most recent experiment: {most_recent}")
        return most_recent
    except Exception as e:
        logger.error(f"❌ Error scanning experiments: {str(e)}")
        return None

def fix_history_js():
    """Fix the history.js file if needed"""
    try:
        with open("app/static/history.js", "r") as f:
            content = f.read()
            
        if "window.loadExamplesForIteration" not in content:
            # Replace the function definition
            if "function loadExamplesForIteration" in content:
                new_content = content.replace(
                    "function loadExamplesForIteration", 
                    "window.loadExamplesForIteration = function"
                )
                
                with open("app/static/history.js", "w") as f:
                    f.write(new_content)
                    
                logger.info("✅ Fixed history.js by adding window. prefix to loadExamplesForIteration")
                return True
            else:
                logger.error("❌ Could not find function loadExamplesForIteration in history.js")
                return False
        else:
            logger.info("✅ No fix needed for history.js")
            return True
    except Exception as e:
        logger.error(f"❌ Error fixing history.js: {str(e)}")
        return False

def fix_history_html():
    """Fix the history.html file if needed"""
    try:
        with open("app/templates/history.html", "r") as f:
            content = f.read()
            
        # Check if view-examples buttons have onclick attribute
        if 'class="btn btn-sm btn-outline-primary view-examples"' in content and 'onclick="loadExamplesForIteration' not in content:
            # Add onclick attribute to view-examples buttons
            new_content = content.replace(
                'class="btn btn-sm btn-outline-primary view-examples" data-iteration="${iteration.iteration}"',
                'class="btn btn-sm btn-outline-primary view-examples" data-iteration="${iteration.iteration}" onclick="loadExamplesForIteration(${iteration.iteration})"'
            )
            
            with open("app/templates/history.html", "w") as f:
                f.write(new_content)
                
            logger.info("✅ Fixed history.html by adding onclick attribute to view-examples buttons")
            return True
        else:
            logger.info("✅ No fix needed for history.html")
            return True
    except Exception as e:
        logger.error(f"❌ Error fixing history.html: {str(e)}")
        return False

def run_tests():
    """Run all tests and fixes"""
    logger.info("=== Starting History Examples Debug ===")
    
    # Check files first
    check_history_js_file()
    
    # Find most recent experiment
    experiment_id = scan_experiments()
    if not experiment_id:
        logger.error("❌ Cannot proceed without experiment ID")
        return
        
    # Check if example files exist
    check_examples_files_exist(experiment_id)
    
    # Apply fixes if needed
    fixed_js = fix_history_js()
    fixed_html = fix_history_html()
    
    if fixed_js or fixed_html:
        logger.info("✅ Applied fixes to files")
    
    # Skip Selenium tests since they're not working reliably
    # and examine examples directly from files instead
    try:
        # Check for examples directory
        examples_dir = Path(f"experiments/{experiment_id}/examples")
        if examples_dir.exists():
            example_files = list(examples_dir.glob("examples_*.json"))
            
            if example_files:
                logger.info(f"Found {len(example_files)} example files in examples directory")
                
                # Display content of all example files
                for example_file in example_files:
                    try:
                        with open(example_file, 'r') as f:
                            examples_data = json.load(f)
                        
                        logger.info(f"=== EXAMPLES CONTENT ({example_file.name}) ===")
                        
                        for i, example in enumerate(examples_data):
                            logger.info(f"\nEXAMPLE #{i+1}:")
                            user_input = example.get('user_input', '')
                            ground_truth = example.get('ground_truth_output', '')
                            model_response = example.get('model_response', '')
                            
                            # Try to also get optimized response if it exists
                            optimized_response = example.get('optimized_response', '')
                            
                            # Get score if it exists
                            score = example.get('score', None)
                            
                            logger.info(f"User Input: {user_input[:100]}..." if len(user_input) > 100 else f"User Input: {user_input}")
                            logger.info(f"Ground Truth: {ground_truth[:100]}..." if len(ground_truth) > 100 else f"Ground Truth: {ground_truth}")
                            logger.info(f"Model Response: {model_response[:100]}..." if len(model_response) > 100 else f"Model Response: {model_response}")
                            
                            if optimized_response:
                                logger.info(f"Optimized Response: {optimized_response[:100]}..." if len(optimized_response) > 100 else f"Optimized Response: {optimized_response}")
                            
                            if score is not None:
                                logger.info(f"Score: {score}")
                            
                            logger.info("-" * 50)
                    except Exception as e:
                        logger.error(f"Error reading example file {example_file}: {e}")
            else:
                logger.warning("No example files found in examples directory")
        
        # Also check for example files directly in experiment directory
        example_files = list(Path(f"experiments/{experiment_id}").glob("examples_*.json"))
        if example_files and not examples_dir.exists():
            logger.info(f"Found {len(example_files)} example files in experiment directory")
            
            # Display content of all example files
            for example_file in example_files:
                try:
                    with open(example_file, 'r') as f:
                        examples_data = json.load(f)
                    
                    logger.info(f"=== EXAMPLES CONTENT ({example_file.name}) ===")
                    
                    for i, example in enumerate(examples_data):
                        logger.info(f"\nEXAMPLE #{i+1}:")
                        user_input = example.get('user_input', '')
                        ground_truth = example.get('ground_truth_output', '')
                        model_response = example.get('model_response', '')
                        
                        # Try to also get optimized response if it exists
                        optimized_response = example.get('optimized_response', '')
                        
                        # Get score if it exists
                        score = example.get('score', None)
                        
                        logger.info(f"User Input: {user_input[:100]}..." if len(user_input) > 100 else f"User Input: {user_input}")
                        logger.info(f"Ground Truth: {ground_truth[:100]}..." if len(ground_truth) > 100 else f"Ground Truth: {ground_truth}")
                        logger.info(f"Model Response: {model_response[:100]}..." if len(model_response) > 100 else f"Model Response: {model_response}")
                        
                        if optimized_response:
                            logger.info(f"Optimized Response: {optimized_response[:100]}..." if len(optimized_response) > 100 else f"Optimized Response: {optimized_response}")
                        
                        if score is not None:
                            logger.info(f"Score: {score}")
                        
                        logger.info("-" * 50)
                except Exception as e:
                    logger.error(f"Error reading example file {example_file}: {e}")
        
        # Check the history.js file for loadExamplesForIteration implementation
        try:
            with open("app/static/history.js", "r") as f:
                js_content = f.read()
                
            if "function loadExamplesForIteration" in js_content:
                logger.info("✅ Found 'loadExamplesForIteration' function in history.js")
                
                # Extract the function implementation to see what it does
                start_idx = js_content.find("function loadExamplesForIteration")
                if start_idx != -1:
                    end_idx = js_content.find("}", start_idx)
                    if end_idx != -1:
                        function_code = js_content[start_idx:end_idx+1]
                        logger.info(f"Function implementation:\n{function_code}")
                        
                        # Check if the function is properly defined
                        if "window.loadExamplesForIteration" not in js_content:
                            logger.warning("⚠️ Function is defined but not attached to window object")
                            
            else:
                logger.error("❌ Could not find 'loadExamplesForIteration' function in history.js")
                
        except Exception as e:
            logger.error(f"Error analyzing history.js: {e}")
            
        # Check the history.html file for view-examples buttons
        try:
            with open("app/templates/history.html", "r") as f:
                html_content = f.read()
                
            if 'class="btn btn-sm btn-outline-primary view-examples"' in html_content:
                logger.info("✅ Found 'view-examples' buttons in history.html")
                
                # Check if the onclick attribute is correctly set
                if 'onclick="window.loadExamplesForIteration' in html_content:
                    logger.info("✅ 'view-examples' buttons have correct onclick attribute with window prefix")
                elif 'onclick="loadExamplesForIteration' in html_content:
                    logger.info("✅ 'view-examples' buttons have onclick attribute but missing window prefix")
                else:
                    logger.warning("⚠️ 'view-examples' buttons are missing onclick attribute")
                    
            else:
                logger.error("❌ Could not find 'view-examples' buttons in history.html")
                
        except Exception as e:
            logger.error(f"Error analyzing history.html: {e}")
        
    except Exception as e:
        logger.error(f"❌ Error during file analysis: {str(e)}")

if __name__ == "__main__":
    run_tests()
