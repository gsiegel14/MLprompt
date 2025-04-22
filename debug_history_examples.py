
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
        else:
            logger.info(f"✅ 'View Examples' button has onclick: {onclick_attr}")
        
        # Click the button
        button.click()
        logger.info("✅ Clicked 'View Examples' button")
        
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
            return True
    
    # Check for example files directly in experiment directory
    example_files = list(base_path.glob("examples_*.json"))
    if example_files:
        logger.info(f"✅ Found {len(example_files)} example files in experiment directory")
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
        logger.info("✅ Applied fixes to files. Testing with Selenium...")
    
    # Setup and run Selenium tests
    try:
        driver = setup_driver()
        driver.get("http://0.0.0.0:5000/history")
        
        # Inject error listener
        inject_javascript_listener(driver)
        
        # Check page loading
        if not check_history_page_loading(driver):
            logger.error("❌ Basic history page not loading correctly")
            driver.quit()
            return
            
        # Check if loadExamplesForIteration exists
        find_function_in_js(driver, "loadExamplesForIteration")
        
        # Check experiment details
        if not check_experiment_details(driver, experiment_id):
            logger.error("❌ Cannot load experiment details")
            driver.quit()
            return
            
        # Check examples loading
        examples_loaded = check_examples_loading(driver)
        if examples_loaded:
            logger.info("✅ SUCCESS: Examples loaded correctly!")
        else:
            logger.error("❌ FAILURE: Examples did not load correctly")
            
        driver.quit()
        
    except Exception as e:
        logger.error(f"❌ Error during Selenium tests: {str(e)}")

if __name__ == "__main__":
    run_tests()
