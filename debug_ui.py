#!/usr/bin/env python3
"""
Interactive Debug UI for Prompt Engineering Platform

This script provides a Flask-based web interface for testing and debugging
different components of the prompt engineering platform.

Features:
1. Component-by-component testing
2. Visual result display
3. Memory usage monitoring
4. API connection testing
5. Prompt optimization testing
6. NEJM data verification

Usage:
    python debug_ui.py
    Then open http://localhost:5050 in your browser
"""

import gc
import importlib.util
import json
import logging
import os
import psutil
import sys
import time
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify, abort

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_ui.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("debug_ui")

# Add the app directory to the path
sys.path.append("./app")

# Import the debugging script
DEBUG_SCRIPT_PATH = "./debug_platform.py"

spec = importlib.util.spec_from_file_location("debug_platform", DEBUG_SCRIPT_PATH)
if spec and spec.loader:
    debug_platform = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(debug_platform)
    logger.info("Successfully imported debug_platform.py")
else:
    logger.error(f"Failed to import debug_platform.py from {DEBUG_SCRIPT_PATH}")
    debug_platform = None

# Create Flask app
app = Flask(__name__)

# Test cases
TEST_CASES = debug_platform.TEST_CASES if debug_platform else []

# Sample prompts
SAMPLE_SYSTEM_PROMPT = debug_platform.SAMPLE_SYSTEM_PROMPT if debug_platform else ""
SAMPLE_OUTPUT_PROMPT = debug_platform.SAMPLE_OUTPUT_PROMPT if debug_platform else ""

@app.route('/')
def index():
    """Render the main debug UI page."""
    return render_template_string(HTML_TEMPLATE, 
                                  test_cases=TEST_CASES, 
                                  system_prompt=SAMPLE_SYSTEM_PROMPT, 
                                  output_prompt=SAMPLE_OUTPUT_PROMPT)

@app.route('/api/test-api-connection', methods=['POST'])
def test_api_connection():
    """Test the connection to the Gemini API."""
    if not debug_platform:
        return jsonify({'success': False, 'message': 'Debug platform module not loaded'})
    
    try:
        result = debug_platform.test_api_connection()
        return jsonify({
            'success': result,
            'message': 'API connection successful' if result else 'API connection failed'
        })
    except Exception as e:
        logger.error(f"Error testing API connection: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/test-data-module', methods=['POST'])
def test_data_module():
    """Test the data module functionality."""
    if not debug_platform:
        return jsonify({'success': False, 'message': 'Debug platform module not loaded'})
    
    try:
        result = debug_platform.test_data_module()
        return jsonify({
            'success': result,
            'message': 'Data module test successful' if result else 'Data module test failed'
        })
    except Exception as e:
        logger.error(f"Error testing data module: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/test-llm-client', methods=['POST'])
def test_llm_client():
    """Test the LLM client functionality."""
    if not debug_platform:
        return jsonify({'success': False, 'message': 'Debug platform module not loaded'})
    
    try:
        result = debug_platform.test_llm_client()
        return jsonify({
            'success': result,
            'message': 'LLM client test successful' if result else 'LLM client test failed'
        })
    except Exception as e:
        logger.error(f"Error testing LLM client: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/test-evaluator', methods=['POST'])
def test_evaluator():
    """Test the evaluator functionality."""
    if not debug_platform:
        return jsonify({'success': False, 'message': 'Debug platform module not loaded'})
    
    try:
        result = debug_platform.test_evaluator()
        return jsonify({
            'success': result,
            'message': 'Evaluator test successful' if result else 'Evaluator test failed'
        })
    except Exception as e:
        logger.error(f"Error testing evaluator: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/test-optimizer', methods=['POST'])
def test_optimizer():
    """Test the optimizer functionality."""
    if not debug_platform:
        return jsonify({'success': False, 'message': 'Debug platform module not loaded'})
    
    try:
        result = debug_platform.test_optimizer()
        return jsonify({
            'success': result,
            'message': 'Optimizer test successful' if result else 'Optimizer test failed'
        })
    except Exception as e:
        logger.error(f"Error testing optimizer: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/test-nejm-data', methods=['POST'])
def test_nejm_data():
    """Test the NEJM data loading and processing."""
    if not debug_platform:
        return jsonify({'success': False, 'message': 'Debug platform module not loaded'})
    
    try:
        result = debug_platform.test_nejm_data()
        return jsonify({
            'success': result,
            'message': 'NEJM data test successful' if result else 'NEJM data test failed'
        })
    except Exception as e:
        logger.error(f"Error testing NEJM data: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/fix-nejm-data', methods=['POST'])
def fix_nejm_data():
    """Run the fix_nejm_data function."""
    try:
        fix_nejm = importlib.util.spec_from_file_location("fix_nejm_data", "./fix_nejm_data.py")
        if fix_nejm and fix_nejm.loader:
            fix_nejm_module = importlib.util.module_from_spec(fix_nejm)
            fix_nejm.loader.exec_module(fix_nejm_module)
            fix_nejm_module.fix_nejm_data()
            return jsonify({
                'success': True,
                'message': 'NEJM data fixed successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to import fix_nejm_data module'
            })
    except Exception as e:
        logger.error(f"Error fixing NEJM data: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/test-experiment-tracker', methods=['POST'])
def test_experiment_tracker():
    """Test the experiment tracker functionality."""
    if not debug_platform:
        return jsonify({'success': False, 'message': 'Debug platform module not loaded'})
    
    try:
        result = debug_platform.test_experiment_tracker()
        return jsonify({
            'success': result,
            'message': 'Experiment tracker test successful' if result else 'Experiment tracker test failed'
        })
    except Exception as e:
        logger.error(f"Error testing experiment tracker: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/test-memory-management', methods=['POST'])
def test_memory_management():
    """Test memory management."""
    if not debug_platform:
        return jsonify({'success': False, 'message': 'Debug platform module not loaded'})
    
    try:
        result = debug_platform.test_memory_management()
        return jsonify({
            'success': result,
            'message': 'Memory management test successful' if result else 'Memory management test failed'
        })
    except Exception as e:
        logger.error(f"Error testing memory management: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/test-all', methods=['POST'])
def test_all():
    """Run all tests."""
    if not debug_platform:
        return jsonify({'success': False, 'message': 'Debug platform module not loaded'})
    
    results = {}
    success = True
    
    try:
        # Test API connection
        results['API Connection'] = debug_platform.test_api_connection()
        success = success and results['API Connection']
        
        # Test data module
        results['Data Module'] = debug_platform.test_data_module()
        success = success and results['Data Module']
        
        # Test LLM client
        results['LLM Client'] = debug_platform.test_llm_client()
        success = success and results['LLM Client']
        
        # Test evaluator
        results['Evaluator'] = debug_platform.test_evaluator()
        success = success and results['Evaluator']
        
        # Test optimizer
        results['Optimizer'] = debug_platform.test_optimizer()
        success = success and results['Optimizer']
        
        # Test experiment tracker
        results['Experiment Tracker'] = debug_platform.test_experiment_tracker()
        success = success and results['Experiment Tracker']
        
        # Test NEJM data
        results['NEJM Data'] = debug_platform.test_nejm_data()
        success = success and results['NEJM Data']
        
        # Test memory management
        results['Memory Management'] = debug_platform.test_memory_management()
        success = success and results['Memory Management']
        
        return jsonify({
            'success': success,
            'results': results,
            'message': 'All tests completed'
        })
    except Exception as e:
        logger.error(f"Error running all tests: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e), 'results': results})

@app.route('/api/memory-usage', methods=['GET'])
def memory_usage():
    """Get current memory usage."""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return jsonify({
            'success': True,
            'memory_mb': mem_info.rss / 1024 / 1024,
            'memory_formatted': f"{mem_info.rss / 1024 / 1024:.2f} MB"
        })
    except Exception as e:
        logger.error(f"Error getting memory usage: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/run-garbage-collection', methods=['POST'])
def run_garbage_collection():
    """Run garbage collection."""
    try:
        # Record memory before
        process = psutil.Process(os.getpid())
        before = process.memory_info().rss / 1024 / 1024
        
        # Run garbage collection
        gc.collect()
        
        # Record memory after
        after = process.memory_info().rss / 1024 / 1024
        
        return jsonify({
            'success': True,
            'before_mb': before,
            'after_mb': after,
            'difference_mb': before - after,
            'message': f"Garbage collection freed {before - after:.2f} MB"
        })
    except Exception as e:
        logger.error(f"Error running garbage collection: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/test-single-prompt', methods=['POST'])
def test_single_prompt():
    """Test a single prompt with a test case."""
    try:
        data = request.json
        system_prompt = data.get('system_prompt', '')
        output_prompt = data.get('output_prompt', '')
        test_case_index = int(data.get('test_case_index', 0))
        
        if not system_prompt or not output_prompt:
            return jsonify({'success': False, 'message': 'System prompt and output prompt are required'})
        
        if test_case_index < 0 or test_case_index >= len(TEST_CASES):
            return jsonify({'success': False, 'message': f'Invalid test case index: {test_case_index}'})
        
        # Get test case
        test_case = TEST_CASES[test_case_index]
        
        # Import llm_client
        try:
            from app.llm_client import get_llm_response
        except ImportError:
            # Try alternate import method
            llm_client = importlib.util.spec_from_file_location("llm_client", "./app/llm_client.py")
            if not llm_client or not llm_client.loader:
                return jsonify({'success': False, 'message': 'Failed to import llm_client'})
            
            llm_client_module = importlib.util.module_from_spec(llm_client)
            llm_client.loader.exec_module(llm_client_module)
            get_llm_response = llm_client_module.get_llm_response
        
        # Get response
        start_time = time.time()
        response = get_llm_response(
            system_prompt=system_prompt,
            user_input=test_case['user_input'],
            output_prompt=output_prompt
        )
        end_time = time.time()
        
        # Import evaluator
        try:
            from app.evaluator import calculate_score
        except ImportError:
            # Try alternate import method
            evaluator = importlib.util.spec_from_file_location("evaluator", "./app/evaluator.py")
            if not evaluator or not evaluator.loader:
                return jsonify({'success': False, 'message': 'Failed to import evaluator'})
            
            evaluator_module = importlib.util.module_from_spec(evaluator)
            evaluator.loader.exec_module(evaluator_module)
            calculate_score = evaluator_module.calculate_score
        
        # Calculate score
        score = calculate_score(response, test_case['ground_truth_output'])
        
        return jsonify({
            'success': True,
            'response': response,
            'score': score,
            'time_seconds': end_time - start_time,
            'input': test_case['user_input'],
            'expected': test_case['ground_truth_output']
        })
    except Exception as e:
        logger.error(f"Error testing single prompt: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})

# HTML template for the UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prompt Engineering Platform Debugger</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        .status-icon {
            font-size: 1.2rem;
            margin-right: 0.5rem;
        }
        .success { color: #198754; }
        .failure { color: #dc3545; }
        .neutral { color: #6c757d; }
        .prompt-textarea {
            height: 200px;
            font-family: monospace;
        }
        .log-output {
            height: 300px;
            overflow-y: auto;
            background-color: #f8f9fa;
            font-family: monospace;
            padding: 10px;
            border: 1px solid #dee2e6;
        }
        .memory-usage {
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        .test-case-info {
            height: 200px;
            overflow-y: auto;
        }
        .card-title-with-icon {
            display: flex;
            align-items: center;
        }
        .log-message {
            margin: 0;
            padding: 2px 0;
        }
        .log-info { color: #0d6efd; }
        .log-error { color: #dc3545; }
        .log-success { color: #198754; }
        .log-warning { color: #ffc107; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="#">
                <i class="fas fa-bug me-2"></i>
                Prompt Engineering Platform Debugger
            </a>
            <span class="navbar-text">
                <span id="memory-display" class="text-light"></span>
                <button id="gc-button" class="btn btn-sm btn-outline-light ms-2" title="Run garbage collection">
                    <i class="fas fa-trash-alt"></i> GC
                </button>
            </span>
        </div>
    </nav>

    <div class="container my-4">
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h5 class="card-title mb-0">
                            <i class="fas fa-tools me-2"></i>
                            Platform Component Tests
                        </h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="list-group mb-3">
                                    <button id="test-api" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                                        <span>
                                            <i class="status-icon fas fa-circle neutral" data-test="api"></i>
                                            Test API Connection
                                        </span>
                                        <span class="spinner-border spinner-border-sm d-none" id="spinner-test-api" role="status"></span>
                                    </button>
                                    <button id="test-data" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                                        <span>
                                            <i class="status-icon fas fa-circle neutral" data-test="data"></i>
                                            Test Data Module
                                        </span>
                                        <span class="spinner-border spinner-border-sm d-none" id="spinner-test-data" role="status"></span>
                                    </button>
                                    <button id="test-llm" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                                        <span>
                                            <i class="status-icon fas fa-circle neutral" data-test="llm"></i>
                                            Test LLM Client
                                        </span>
                                        <span class="spinner-border spinner-border-sm d-none" id="spinner-test-llm" role="status"></span>
                                    </button>
                                    <button id="test-evaluator" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                                        <span>
                                            <i class="status-icon fas fa-circle neutral" data-test="evaluator"></i>
                                            Test Evaluator
                                        </span>
                                        <span class="spinner-border spinner-border-sm d-none" id="spinner-test-evaluator" role="status"></span>
                                    </button>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="list-group mb-3">
                                    <button id="test-optimizer" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                                        <span>
                                            <i class="status-icon fas fa-circle neutral" data-test="optimizer"></i>
                                            Test Optimizer
                                        </span>
                                        <span class="spinner-border spinner-border-sm d-none" id="spinner-test-optimizer" role="status"></span>
                                    </button>
                                    <button id="test-tracker" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                                        <span>
                                            <i class="status-icon fas fa-circle neutral" data-test="tracker"></i>
                                            Test Experiment Tracker
                                        </span>
                                        <span class="spinner-border spinner-border-sm d-none" id="spinner-test-tracker" role="status"></span>
                                    </button>
                                    <button id="test-nejm" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                                        <span>
                                            <i class="status-icon fas fa-circle neutral" data-test="nejm"></i>
                                            Test NEJM Data
                                        </span>
                                        <span class="spinner-border spinner-border-sm d-none" id="spinner-test-nejm" role="status"></span>
                                    </button>
                                    <button id="test-memory" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                                        <span>
                                            <i class="status-icon fas fa-circle neutral" data-test="memory"></i>
                                            Test Memory Management
                                        </span>
                                        <span class="spinner-border spinner-border-sm d-none" id="spinner-test-memory" role="status"></span>
                                    </button>
                                </div>
                            </div>
                        </div>
                        <div class="d-grid gap-2">
                            <button id="test-all" class="btn btn-primary">
                                <i class="fas fa-play-circle me-2"></i>
                                Run All Tests
                            </button>
                            <button id="fix-nejm" class="btn btn-warning">
                                <i class="fas fa-wrench me-2"></i>
                                Fix NEJM Data
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-info text-white">
                        <h5 class="card-title mb-0">
                            <i class="fas fa-flask me-2"></i>
                            Prompt Tester
                        </h5>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label for="test-case-select" class="form-label">Select Test Case</label>
                            <select id="test-case-select" class="form-select">
                                {% for i, case in enumerate(test_cases) %}
                                <option value="{{ i }}">Case {{ i+1 }}: {{ case.user_input[:50] }}...</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="mb-3">
                            <label for="system-prompt" class="form-label">System Prompt</label>
                            <textarea id="system-prompt" class="form-control prompt-textarea">{{ system_prompt }}</textarea>
                        </div>
                        <div class="mb-3">
                            <label for="output-prompt" class="form-label">Output Prompt</label>
                            <textarea id="output-prompt" class="form-control prompt-textarea">{{ output_prompt }}</textarea>
                        </div>
                        <div class="d-grid">
                            <button id="test-single-prompt" class="btn btn-success">
                                <i class="fas fa-paper-plane me-2"></i>
                                Test Prompt
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-success text-white">
                        <h5 class="card-title mb-0">
                            <i class="fas fa-vial me-2"></i>
                            Test Results
                        </h5>
                    </div>
                    <div class="card-body">
                        <ul class="nav nav-tabs" id="resultTabs" role="tablist">
                            <li class="nav-item" role="presentation">
                                <button class="nav-link active" id="prompt-tab" data-bs-toggle="tab" data-bs-target="#prompt" type="button" role="tab" aria-controls="prompt" aria-selected="true">Prompt Test</button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="case-tab" data-bs-toggle="tab" data-bs-target="#case" type="button" role="tab" aria-controls="case" aria-selected="false">Test Case</button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="logs-tab" data-bs-toggle="tab" data-bs-target="#logs" type="button" role="tab" aria-controls="logs" aria-selected="false">Logs</button>
                            </li>
                        </ul>
                        <div class="tab-content mt-3">
                            <div class="tab-pane fade show active" id="prompt" role="tabpanel" aria-labelledby="prompt-tab">
                                <div id="prompt-result" class="test-case-info">
                                    <div class="text-center text-muted py-5">
                                        <i class="fas fa-arrow-left me-2"></i>
                                        Test a prompt with a test case to see results
                                    </div>
                                </div>
                            </div>
                            <div class="tab-pane fade" id="case" role="tabpanel" aria-labelledby="case-tab">
                                <div id="case-info" class="test-case-info">
                                    <div class="text-center text-muted py-5">
                                        <i class="fas fa-arrow-left me-2"></i>
                                        Select a test case to see details
                                    </div>
                                </div>
                            </div>
                            <div class="tab-pane fade" id="logs" role="tabpanel" aria-labelledby="logs-tab">
                                <div id="log-output" class="log-output"></div>
                                <div class="d-grid mt-2">
                                    <button id="clear-logs" class="btn btn-sm btn-outline-secondary">
                                        <i class="fas fa-trash me-2"></i>
                                        Clear Logs
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Initialize the page
        document.addEventListener('DOMContentLoaded', function() {
            // Initialize test case selection
            updateTestCaseInfo();
            
            // Update memory usage initially and every 5 seconds
            updateMemoryUsage();
            setInterval(updateMemoryUsage, 5000);
            
            // Add event listeners
            document.getElementById('test-api').addEventListener('click', () => runTest('api', '/api/test-api-connection'));
            document.getElementById('test-data').addEventListener('click', () => runTest('data', '/api/test-data-module'));
            document.getElementById('test-llm').addEventListener('click', () => runTest('llm', '/api/test-llm-client'));
            document.getElementById('test-evaluator').addEventListener('click', () => runTest('evaluator', '/api/test-evaluator'));
            document.getElementById('test-optimizer').addEventListener('click', () => runTest('optimizer', '/api/test-optimizer'));
            document.getElementById('test-tracker').addEventListener('click', () => runTest('tracker', '/api/test-experiment-tracker'));
            document.getElementById('test-nejm').addEventListener('click', () => runTest('nejm', '/api/test-nejm-data'));
            document.getElementById('test-memory').addEventListener('click', () => runTest('memory', '/api/test-memory-management'));
            document.getElementById('test-all').addEventListener('click', runAllTests);
            document.getElementById('fix-nejm').addEventListener('click', fixNejmData);
            document.getElementById('gc-button').addEventListener('click', runGarbageCollection);
            document.getElementById('test-single-prompt').addEventListener('click', testSinglePrompt);
            document.getElementById('test-case-select').addEventListener('change', updateTestCaseInfo);
            document.getElementById('clear-logs').addEventListener('click', clearLogs);
        });
        
        // Run a test
        async function runTest(testId, endpoint) {
            const statusIcon = document.querySelector(`.status-icon[data-test="${testId}"]`);
            const spinner = document.getElementById(`spinner-test-${testId}`);
            
            try {
                // Update UI to show loading
                statusIcon.classList.remove('success', 'failure', 'neutral');
                statusIcon.classList.add('neutral');
                spinner.classList.remove('d-none');
                
                logMessage(`Running ${testId} test...`, 'info');
                
                // Call API endpoint
                const response = await fetch(endpoint, {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                // Update UI with result
                if (data.success) {
                    statusIcon.classList.remove('neutral');
                    statusIcon.classList.add('success');
                    logMessage(`✅ ${testId} test: ${data.message}`, 'success');
                } else {
                    statusIcon.classList.remove('neutral');
                    statusIcon.classList.add('failure');
                    logMessage(`❌ ${testId} test: ${data.message}`, 'error');
                }
            } catch (error) {
                console.error(`Error running ${testId} test:`, error);
                statusIcon.classList.remove('neutral');
                statusIcon.classList.add('failure');
                logMessage(`❌ ${testId} test: ${error.message}`, 'error');
            } finally {
                spinner.classList.add('d-none');
            }
        }
        
        // Run all tests
        async function runAllTests() {
            try {
                logMessage('Running all tests...', 'info');
                
                // Reset all status icons
                document.querySelectorAll('.status-icon').forEach(icon => {
                    icon.classList.remove('success', 'failure');
                    icon.classList.add('neutral');
                });
                
                // Show all spinners
                document.querySelectorAll('[id^="spinner-test-"]').forEach(spinner => {
                    spinner.classList.remove('d-none');
                });
                
                // Call API endpoint
                const response = await fetch('/api/test-all', {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                if (data.results) {
                    // Update UI with results
                    for (const [test, result] of Object.entries(data.results)) {
                        const testId = test.toLowerCase().replace(' ', '-');
                        const statusIcon = document.querySelector(`.status-icon[data-test="${testId}"]`);
                        
                        if (statusIcon) {
                            statusIcon.classList.remove('neutral');
                            statusIcon.classList.add(result ? 'success' : 'failure');
                        }
                        
                        logMessage(`${result ? '✅' : '❌'} ${test}: ${result ? 'Success' : 'Failure'}`, 
                                  result ? 'success' : 'error');
                    }
                }
                
                logMessage(`All tests completed: ${data.success ? 'SUCCESS' : 'FAILURE'}`, 
                          data.success ? 'success' : 'error');
            } catch (error) {
                console.error('Error running all tests:', error);
                logMessage(`❌ Error running all tests: ${error.message}`, 'error');
            } finally {
                // Hide all spinners
                document.querySelectorAll('[id^="spinner-test-"]').forEach(spinner => {
                    spinner.classList.add('d-none');
                });
            }
        }
        
        // Fix NEJM data
        async function fixNejmData() {
            try {
                logMessage('Fixing NEJM data...', 'info');
                
                // Call API endpoint
                const response = await fetch('/api/fix-nejm-data', {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                if (data.success) {
                    logMessage(`✅ ${data.message}`, 'success');
                } else {
                    logMessage(`❌ ${data.message}`, 'error');
                }
            } catch (error) {
                console.error('Error fixing NEJM data:', error);
                logMessage(`❌ Error fixing NEJM data: ${error.message}`, 'error');
            }
        }
        
        // Update memory usage
        async function updateMemoryUsage() {
            try {
                const response = await fetch('/api/memory-usage');
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('memory-display').textContent = 
                        `Memory: ${data.memory_formatted}`;
                }
            } catch (error) {
                console.error('Error updating memory usage:', error);
            }
        }
        
        // Run garbage collection
        async function runGarbageCollection() {
            try {
                logMessage('Running garbage collection...', 'info');
                
                const response = await fetch('/api/run-garbage-collection', {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                if (data.success) {
                    logMessage(`✅ ${data.message}`, 'success');
                    updateMemoryUsage();
                } else {
                    logMessage(`❌ ${data.message}`, 'error');
                }
            } catch (error) {
                console.error('Error running garbage collection:', error);
                logMessage(`❌ Error running garbage collection: ${error.message}`, 'error');
            }
        }
        
        // Test a single prompt
        async function testSinglePrompt() {
            try {
                const systemPrompt = document.getElementById('system-prompt').value;
                const outputPrompt = document.getElementById('output-prompt').value;
                const testCaseIndex = document.getElementById('test-case-select').value;
                
                if (!systemPrompt || !outputPrompt) {
                    logMessage('❌ System prompt and output prompt are required', 'error');
                    return;
                }
                
                logMessage('Testing prompt...', 'info');
                
                // Update UI to show loading
                const promptResult = document.getElementById('prompt-result');
                promptResult.innerHTML = `
                    <div class="text-center py-5">
                        <div class="spinner-border" role="status"></div>
                        <p class="mt-2">Testing prompt...</p>
                    </div>
                `;
                
                // Call API endpoint
                const response = await fetch('/api/test-single-prompt', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        system_prompt: systemPrompt,
                        output_prompt: outputPrompt,
                        test_case_index: testCaseIndex
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    const scoreClass = data.score >= 0.8 ? 'text-success' : 
                                      data.score >= 0.5 ? 'text-warning' : 'text-danger';
                    
                    promptResult.innerHTML = `
                        <div class="mb-3">
                            <h6 class="fw-bold">Model Response:</h6>
                            <pre class="p-2 bg-light border rounded">${data.response}</pre>
                        </div>
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <span class="badge bg-${data.score >= 0.8 ? 'success' : 
                                                        data.score >= 0.5 ? 'warning' : 'danger'}">
                                    Score: ${(data.score * 100).toFixed(1)}%
                                </span>
                            </div>
                            <div class="text-muted small">
                                Response time: ${data.time_seconds.toFixed(2)}s
                            </div>
                        </div>
                    `;
                    
                    logMessage(`✅ Prompt test completed (Score: ${(data.score * 100).toFixed(1)}%)`, 'success');
                } else {
                    promptResult.innerHTML = `
                        <div class="alert alert-danger">
                            <i class="fas fa-exclamation-circle me-2"></i>
                            ${data.message}
                        </div>
                    `;
                    
                    logMessage(`❌ Prompt test failed: ${data.message}`, 'error');
                }
            } catch (error) {
                console.error('Error testing prompt:', error);
                document.getElementById('prompt-result').innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-circle me-2"></i>
                        ${error.message}
                    </div>
                `;
                
                logMessage(`❌ Error testing prompt: ${error.message}`, 'error');
            }
        }
        
        // Update test case information
        function updateTestCaseInfo() {
            const testCaseIndex = document.getElementById('test-case-select').value;
            const testCases = {{ test_cases|tojson }};
            
            const testCase = testCases[testCaseIndex];
            const caseInfo = document.getElementById('case-info');
            
            caseInfo.innerHTML = `
                <div class="mb-3">
                    <h6 class="fw-bold">User Input:</h6>
                    <pre class="p-2 bg-light border rounded">${testCase.user_input}</pre>
                </div>
                <div class="mb-3">
                    <h6 class="fw-bold">Expected Output:</h6>
                    <pre class="p-2 bg-light border rounded">${testCase.ground_truth_output}</pre>
                </div>
            `;
        }
        
        // Log a message
        function logMessage(message, type = 'info') {
            const logOutput = document.getElementById('log-output');
            const timestamp = new Date().toLocaleTimeString();
            
            const messageElement = document.createElement('p');
            messageElement.classList.add('log-message', `log-${type}`);
            messageElement.innerHTML = `[${timestamp}] ${message}`;
            
            logOutput.appendChild(messageElement);
            logOutput.scrollTop = logOutput.scrollHeight;
        }
        
        // Clear logs
        function clearLogs() {
            document.getElementById('log-output').innerHTML = '';
            logMessage('Logs cleared', 'info');
        }
    </script>
</body>
</html>
"""

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)